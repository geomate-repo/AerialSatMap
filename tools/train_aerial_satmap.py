#!/usr/bin/env python3
"""
Train the aerial-only SatMap model using SatMap's MMDetection3D training stack.

This is deliberately a thin, config-driven entry point: model construction,
optimizer/runner hooks, checkpointing, logging, and distributed training remain
owned by the repository's MMDetection3D/MMCV APIs. The input discovery follows
the directory conventions used by SD_mapping_modules/train_smp.py:

  <dataset>/<city>/polylines<month>/*.geojson
  <dataset>/<city>/city-images<month>/tiles/<percent>_percent/<tile_size>_<overlap>/*.jp2

Each JP2 is paired with a GeoJSON having the same basename. The configured
AerialMapDataset must accept `img_dir` and `ann_dir` (as in
projects/mmdet3d_plugin/aerial_map/aerial_map_dataset.py) and convert those
annotations into fixed-length normalized polylines.
"""

import argparse
import os
import random
import sys
from pathlib import Path

# from .pipelines import (
#     LoadAerialImage,
#     NormalizeAerialImage,
#     DefaultFormatBundleAerial,
#     CollectAerialMap,
# )

import numpy as np
import torch
from mmcv import Config
from mmcv.utils import DictAction
from mmcv.utils import import_modules_from_strings
from mmcv.runner import get_dist_info, init_dist, set_random_seed

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# from projects.mmdet3d_plugin.aerial_map import pipelines as aerial_pipelines
import projects.mmdet3d_plugin.aerial_map.pipelines

import distutils
import distutils.version

distutils.version = distutils.version

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train aerial-only SatMap with the repository's MMDet3D runner"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="projects/configs/aerial/aerial_satmap_swin_tiny.py",
        help="Aerial SatMap config file (relative to the SatMap repository root).",
    )
    parser.add_argument("--datasets-dir-list", nargs="+", required=True,
                        help="One dataset root per city.")
    parser.add_argument("--city-name-list", nargs="+", required=True,
                        help="City directory names, one per dataset root.")
    parser.add_argument("--month-dir-list", nargs="*", default=None,
                        help="Optional suffix/subdirectory after 'polylines' and 'city-images', e.g. '' or '/2021-06'. Use one value per city.")
    parser.add_argument("--tile-percent-list", nargs="+", type=int, default=None,
                        help="Tile coverage percentages, e.g. 100.")
    parser.add_argument("--tile-size", type=int, default=1250)
    parser.add_argument("--overlap-size", type=int, default=150)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=57)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--load-from", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"],
                        default="none")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction,
                        help="Override config values, e.g. model.backbone.depth=18.")
    return parser.parse_args()


def _month_suffixes(args, n):
    if args.month_dir_list is None:
        return [""] * n
    if len(args.month_dir_list) != n:
        raise ValueError(
            f"--month-dir-list must contain {n} entries (one per city); "
            f"received {len(args.month_dir_list)}."
        )
    # Allow the convenient token 'none' to mean no suffix.
    return ["" if x.lower() in {"none", "null"} else x for x in args.month_dir_list]


def discover_pairs(args):
    n = len(args.datasets_dir_list)
    if len(args.city_name_list) != n:
        raise ValueError("--datasets-dir-list and --city-name-list must have the same length.")

    months = _month_suffixes(args, n)
    percents = args.tile_percent_list or [100] * n
    if len(percents) != n:
        raise ValueError(
            f"--tile-percent-list must contain {n} entries; received {len(percents)}."
        )

    pairs = []
    for root, city, month, percent in zip(
        args.datasets_dir_list, args.city_name_list, months, percents
    ):
        root = Path(root).expanduser()
        label_dir = root / city / f"polylines/{month}"
        image_dir = (
            root / city / f"city-images/{month}" / "tiles"
            / f"{percent}_percent" / f"{args.tile_size}_{args.overlap_size}"
        )
        if not label_dir.is_dir():
            raise FileNotFoundError(f"GeoJSON directory not found: {label_dir}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"JP2 image directory not found: {image_dir}")

        labels = {p.stem: p for p in label_dir.glob("*.geojson")}
        images = {p.stem: p for p in image_dir.glob("*.jp2")}
        common = sorted(labels.keys() & images.keys())
        if not common:
            raise RuntimeError(
                f"No matching JP2/GeoJSON basenames in:\n  {image_dir}\n  {label_dir}"
            )
        missing_labels = sorted(images.keys() - labels.keys())
        missing_images = sorted(labels.keys() - images.keys())
        print(f"[{city}] paired samples: {len(common)}")
        if missing_labels:
            print(f"  warning: {len(missing_labels)} JP2s have no GeoJSON; ignored.")
        if missing_images:
            print(f"  warning: {len(missing_images)} GeoJSONs have no JP2; ignored.")
        pairs.extend((str(images[k].resolve()), str(labels[k].resolve())) for k in common)

    if len(pairs) < 2:
        raise RuntimeError(f"Need at least 2 paired samples to split train/validation; got {len(pairs)}.")
    return pairs


def split_pairs(pairs, val_ratio, seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, min(len(pairs) - 1, int(round(len(pairs) * val_ratio))))
    val_indices = set(indices[:n_val])
    train_pairs = [p for i, p in enumerate(pairs) if i not in val_indices]
    val_pairs = [p for i, p in enumerate(pairs) if i in val_indices]
    return train_pairs, val_pairs


def configure_dataset(dataset_cfg, pairs, pipeline=None):
    dataset_cfg = dataset_cfg.copy()

    dataset_cfg["samples"] = [
        {"image": img, "annotation": ann}
        for img, ann in pairs
    ]

    # These defaults allow the dataset constructor to build without
    # a separate annotation index file.
    dataset_cfg.setdefault("data_root", "")
    dataset_cfg.setdefault("ann_file", None)

    if pipeline is not None:
        dataset_cfg["pipeline"] = pipeline

    return dataset_cfg

def main():
    args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if len(args.datasets_dir_list) != len(args.city_name_list):
        raise ValueError("--datasets-dir-list and --city-name-list must have the same number of entries.")

    ########
    # Ensure the repository root is importable when MMCV loads configs
    # through temporary files under /tmp.
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    # Also expose the repository root to subprocesses and other import mechanisms.
    os.environ["PYTHONPATH"] = (
        REPO_ROOT
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    )

    #############
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)

    # Keep repository registration imports active (the config should also define
    # custom_imports; this import is a defensive fallback for older MMCV configs).
    try:
        import projects.mmdet3d_plugin  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Could not import projects.mmdet3d_plugin. Run this script from the "
            "SatMap repository root and verify the custom aerial modules are registered."
        ) from exc

    pairs = discover_pairs(args)
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.split_seed)
    print(f"Total paired samples: {len(pairs)}")
    print(f"Train samples: {len(train_pairs)} | validation samples: {len(val_pairs)}")
    print("Classes must be configured in this order: divider, ped_crossing, boundary.")

    # Configs for the aerial adaptation should use the repository's dataset class.
    # This script injects explicit paired file lists so that the custom dataset
    # does not need to reconstruct paths from the original nuScenes layout.
    if "data" not in cfg or "train" not in cfg.data:
        raise KeyError("Config must define cfg.data.train for the aerial dataset.")
    # train_cfg = configure_dataset(cfg.data.train, train_pairs)
    cfg.data.train = configure_dataset(
        cfg.data.train,
        train_pairs,
        pipeline=cfg.data.train.get("pipeline"),
    )
    # cfg.data.train = train_cfg

    # if args.validate:
    #     if "val" not in cfg.data:
    #         raise KeyError("--validate was supplied, but cfg.data.val is not defined.")
    #     val_cfg = configure_dataset(cfg.data.val, val_pairs)
    #     cfg.data.val = val_cfg
    if args.validate:
        if cfg.get("data", {}).get("val") is None:
            raise KeyError("--validate was supplied, but cfg.data.val is not defined.")

        val_cfg = configure_dataset(
            cfg.data.val,
            val_pairs,
            pipeline=cfg.data.val.get("pipeline"),
        )
        cfg.data.val = val_cfg

    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get("work_dir"):
        cfg.work_dir = os.path.join("./work_dirs", Path(args.config).stem)
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if args.load_from:
        cfg.load_from = args.load_from

    cfg.seed = args.seed
    if args.launcher != "none":
        cfg.launcher = args.launcher
        init_dist(args.launcher, **cfg.get("dist_params", {}))
    else:
        cfg.launcher = "none"

    set_random_seed(args.seed, deterministic=args.deterministic)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))

    # MMDetection3D owns optimizer, LR schedule, runner, hooks, logging, and
    # checkpointing, just as it does for the repository's standard SatMap model.
    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    if hasattr(model, "init_weights"):
        model.init_weights()


    ###
    from mmdet.datasets.builder import PIPELINES

    required_pipelines = [
        "LoadAerialImage",
        "NormalizeAerialImage",
        "DefaultFormatBundleAerial",
        "CollectAerialMap",
    ]

    missing = [
        name for name in required_pipelines
        if name not in PIPELINES.module_dict
    ]

    if missing:
        raise RuntimeError(
            f"Custom aerial transforms are not registered: {missing}"
        )

    datasets = [build_dataset(cfg.data.train)]
    if cfg.get("workflow") is None:
        cfg.workflow = [("train", 1)]

     ##debug

    ####

    # for i, dataset in enumerate(datasets):
    #     import inspect

    #     print("\n========== DATASET DEBUG ==========")
    #     print("Dataset class:", type(dataset).__name__)
    #     print("Dataset module:", type(dataset).__module__)
    #     # print("Dataset source:", inspect.getfile(type(dataset)))
    #     print("Dataset length:", len(dataset))
    #     print("Dataset test_mode:", getattr(dataset, "test_mode", "unknown"))

    #     # This executes the actual dataset pipeline.
    #     sample = dataset[0]

    #     print("\nDataset sample keys:", sorted(sample.keys()))

    #     for key in ("aerial_img", "gt_bboxes_3d", "gt_labels_3d", "img_metas"):
    #         if key not in sample:
    #             print(f"{key}: MISSING")
    #             continue

    #         value = sample[key]

    #         print(f"\n{key}:")
    #         print("  Outer type:", type(value))

    #         if hasattr(value, "data"):
    #             print("  DataContainer contents type:", type(value.data))
    #             print("  DataContainer stack:", getattr(value, "stack", None))
    #             print("  DataContainer cpu_only:", getattr(value, "cpu_only", None))

    #             if hasattr(value.data, "shape"):
    #                 print("  Contents shape:", value.data.shape)
    #             elif isinstance(value.data, (list, tuple)):
    #                 print("  Contents length:", len(value.data))
    #                 if len(value.data) > 0:
    #                     print("  First element type:", type(value.data[0]))
    #                     if hasattr(value.data[0], "shape"):
    #                         print("  First element shape:", value.data[0].shape)
    #         elif hasattr(value, "shape"):
    #             print("  Shape:", value.shape)

    #     print("===================================\n")

    # from mmdet.datasets import build_dataloader

    # debug_loader = build_dataloader(
    #     datasets[0],
    #     samples_per_gpu=1,
    #     workers_per_gpu=0,
    #     num_gpus=1,
    #     dist=False,
    #     shuffle=False,
    # )

    # batch = next(iter(debug_loader))

    # print("\n========== DATALOADER DEBUG ==========")
    # print("Batch keys:", sorted(batch.keys()))

    # for key, value in batch.items():
    #     print("\n", key)
    #     print("  Type:", type(value))

    #     if isinstance(value, (list, tuple)):
    #         print("  Length:", len(value))

    #         if len(value) > 0:
    #             print("  First element type:", type(value[0]))

    #             if hasattr(value[0], "shape"):
    #                 print("  First element shape:", value[0].shape)

    #     elif hasattr(value, "shape"):
    #         print("  Shape:", value.shape)

    # print("=====================================\n")
    # print("\n========== BATCH CONTENTS ==========")

    # for key in ("gt_bboxes_3d", "gt_labels_3d", "img_metas"):
    #     container = batch[key]

    #     print(f"\n{key}:")
    #     print("Container type:", type(container))
    #     print("Container data type:", type(container.data))
    #     print("Container data:", repr(container.data)[:1500])

    #     if isinstance(container.data, (list, tuple)):
    #         print("Container data length:", len(container.data))

    #         for i, item in enumerate(container.data):
    #             print(f"  [{i}] type:", type(item))
    #             if hasattr(item, "shape"):
    #                 print(f"  [{i}] shape:", item.shape)

    # print("===================================\n")
    ####
    for i, dataset in enumerate(datasets):
        print(
            "[Dataset {}] class={} module={} length={}".format(
                i,
                type(dataset).__name__,
                type(dataset).__module__,
                len(dataset),
            )
        )

    # Keep cfg.data.workers_per_gpu / samples_per_gpu as defined in the config.
    rank, _ = get_dist_info()
    if rank == 0:
        print(f"Config: {args.config}")
        print(f"Work dir: {cfg.work_dir}")
        print(f"Model: {cfg.model.get('type', '<not specified>')}")

    if not hasattr(cfg, "gpu_ids"):
        cfg.gpu_ids = range(1)


    print("Workflow:", cfg.workflow)
    print("Workflow phases:", len(cfg.workflow))
    print("Training datasets:", len(datasets))

    train_model(
        model,
        datasets,
        cfg,
        distributed=(args.launcher != "none"),
        validate=args.validate,
        timestamp=None,
        meta={"seed": args.seed, "config": args.config},
    )


if __name__ == "__main__":
    main()
