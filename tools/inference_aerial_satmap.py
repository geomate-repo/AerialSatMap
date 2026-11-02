#!/usr/bin/env python
"""
AerialSatMap inference for orthographic aerial JP2/TIFF tiles.

Uses the repository's configured AerialMapDataset pipeline and the custom
AerialSatMap.forward_test -> simple_test -> pts_bbox_head.get_bboxes ->
SatMap.pred2result path.

Outputs per sample:
  - predictions_raw.geojson: all decoded predictions
  - predictions_thr_*.geojson: confidence-filtered predictions
  - overlay_thr_*.png: aerial image + predictions (+ GT when available)
  - predictions.npz: raw local-frame points, scores and labels

Example:
  python tools/inference_aerial_satmap.py \
      projects/configs/aerial/aerial_satmap_swin_tiny.py \
      work_dirs/aerial_satmap/epoch_10.pth \
      --out-dir work_dirs/aerial_satmap/inference_epoch10 \
      --score-thrs 0.05 0.10 0.20 0.30 \
      --max-samples 10

For one existing validation sample:
  ... --sample-idx 0

For annotation-free inference, provide --tile-index 0_generated_tiles.geojson
and --image-dir containing the JP2 tiles. The tile index supplies each tile's
EPSG:3857 affine transform and dimensions; no per-tile annotation is needed.
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import build_dataloader

import projects.mmdet3d_plugin.aerial_map.pipelines


CLASS_NAMES = ("divider", "ped_crossing", "boundary")
CLASS_COLORS = {
    0: (0, 220, 80),     # divider: green
    1: (255, 210, 0),    # pedestrian crossing: yellow
    2: (255, 70, 70),    # boundary: red
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AerialSatMap inference on aerial raster tiles."
    )
    parser.add_argument("config", help="AerialSatMap config .py")
    parser.add_argument("checkpoint", help="Trained .pth checkpoint")
    parser.add_argument("--out-dir", default="work_dirs/aerial_satmap/inference")
    parser.add_argument("--sample-idx", type=int, default=None,
                        help="Run only this index from the configured val dataset.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of dataset samples to process.")
    parser.add_argument("--image", default=None,
                        help="Optional standalone JP2/TIF. Uses val pipeline.")
    parser.add_argument("--image-dir", default=None,
                        help="Directory of JP2/TIF tiles listed in --tile-index.")
    parser.add_argument("--tile-index", default=None,
                        help="0_generated_tiles.geojson containing tile fname and meta.transform.")
    parser.add_argument("--samples-json", default=None,
                        help="JSON file containing AerialMapDataset samples "
                             "(list of {image, annotation} objects, or "
                             "{'samples': [...]}). Required when --image is "
                             "not supplied and cfg.data.val has no ann_file/samples.")
    parser.add_argument("--annotation", default=None,
                        help="Optional ground-truth GeoJSON for --image; not required for inference.")
    parser.add_argument("--score-thrs", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20, 0.30],
                        help="Thresholds for filtered outputs. Raw output is always saved.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--plugin-import", default=None,
                        help="Optional Python module to import for custom registration.")
    parser.add_argument("--detector-import", default=None,
                        help="Module containing/registering AerialSatMap. "
                             "Set this if the config import does not register it.")
    parser.add_argument("--no-gt", action="store_true",
                        help="Do not attempt to load/overlay annotations.")
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--max-display-dimension", type=int, default=1800)
    return parser.parse_args()


def import_custom_modules(cfg, args):
    """Trigger custom registry registration before build_model/build_dataset."""
    modules = []

    if args.plugin_import:
        modules.append(args.plugin_import)

    if args.detector_import:
        modules.append(args.detector_import)

    custom_imports = cfg.get("custom_imports", None)
    if custom_imports:
        modules.extend(custom_imports.get("imports", []))

    # Common project package locations. Import only modules that exist.
    candidates = [
        "projects.mmdet3d_plugin",
        "projects.mmdet3d_plugin.satmap",
        "projects.mmdet3d_plugin.satmap.detectors",
        "projects.mmdet3d_plugin.satmap.detectors.aerial_satmap",
        "projects.mmdet3d_plugin.aerial_map",
    ]
    for name in candidates:
        if name not in modules:
            modules.append(name)

    errors = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            # Candidate packages may not exist in every repository layout.
            errors.append("{}: {}".format(module_name, exc))

    return errors


def get_test_dataset_cfg(cfg):
    """Use the configured validation pipeline as the inference pipeline."""
    if "data" not in cfg or "val" not in cfg.data:
        raise KeyError("Config must define cfg.data.val for inference.")

    dataset_cfg = cfg.data.val.copy()
    dataset_cfg["test_mode"] = True
    return dataset_cfg


def load_tile_index(path):
    """Read generated-tile GeoJSON and index metadata by JP2 filename."""
    with open(path, "r") as f:
        document = json.load(f)

    if document.get("type") != "FeatureCollection":
        raise ValueError("Tile index must be a GeoJSON FeatureCollection.")

    index = {}
    for feature in document.get("features", []):
        props = feature.get("properties", {})
        fname = props.get("fname")
        meta = props.get("meta", {})
        transform_text = meta.get("transform")
        if not fname or not transform_text:
            continue
        try:
            values = [float(v.strip()) for v in transform_text.split(",")]
            if len(values) != 6:
                raise ValueError("expected six affine coefficients")
            transform = Affine(*values)
        except Exception as exc:
            raise ValueError("Invalid meta.transform for {}: {}".format(fname, exc))

        width = int(props.get("width", meta.get("width", 0)))
        height = int(props.get("height", meta.get("height", 0)))
        crs = props.get("crs", meta.get("crs", "EPSG:3857"))
        index[os.path.basename(fname)] = {
            "fname": fname,
            "transform": transform,
            "width": width,
            "height": height,
            "crs": crs,
            "properties": props,
            "geometry": feature.get("geometry"),
        }

    if not index:
        raise ValueError("No usable tile entries found in {}".format(path))
    return index


def make_dataset(cfg, args):
    """Build the configured inference dataset, allowing annotation-free tiles."""
    dataset_cfg = get_test_dataset_cfg(cfg)

    if args.image and args.samples_json:
        raise ValueError("Use either --image or --samples-json, not both.")

    tile_index = load_tile_index(args.tile_index) if args.tile_index else None

    if args.image:
        image_path = os.path.abspath(args.image)
        annotation_path = os.path.abspath(args.annotation) if args.annotation else ""
        dataset_cfg["data_root"] = ""
        dataset_cfg["samples"] = [{"image": image_path, "annotation": annotation_path}]
        dataset_cfg.pop("ann_file", None)

    elif args.samples_json:
        samples_path = os.path.abspath(args.samples_json)
        with open(samples_path, "r") as f:
            sample_data = json.load(f)
        samples = sample_data.get("samples") if isinstance(sample_data, dict) else sample_data
        if not isinstance(samples, list) or not samples:
            raise ValueError("--samples-json must contain a non-empty list of samples.")
        dataset_cfg["samples"] = samples
        dataset_cfg.pop("ann_file", None)

    elif tile_index is not None:
        if not args.image_dir:
            raise ValueError("--image-dir is required with --tile-index when --image is not supplied.")
        image_dir = os.path.abspath(args.image_dir)
        samples = []
        missing = []
        for fname in sorted(tile_index):
            image_path = os.path.join(image_dir, fname)
            if os.path.isfile(image_path):
                samples.append({"image": image_path, "annotation": ""})
            else:
                missing.append(image_path)
        if not samples:
            raise FileNotFoundError("No indexed tile files found in {}".format(image_dir))
        if missing:
            print("Tile index entries without matching files:", len(missing))
            for path in missing[:5]:
                print("  missing:", path)
        dataset_cfg["data_root"] = ""
        dataset_cfg["samples"] = samples
        dataset_cfg.pop("ann_file", None)

    elif not dataset_cfg.get("samples") and not dataset_cfg.get("ann_file"):
        raise ValueError(
            "cfg.data.val does not define samples or ann_file. Supply --image, "
            "--samples-json, or --tile-index together with --image-dir."
        )

    dataset = build_dataset(dataset_cfg)
    if len(dataset) == 0:
        raise RuntimeError("Inference dataset is empty.")

    if args.sample_idx is not None:
        if args.sample_idx < 0 or args.sample_idx >= len(dataset):
            raise IndexError("--sample-idx {} outside dataset range [0, {}).".format(args.sample_idx, len(dataset)))
        dataset.data_infos = [dataset.data_infos[args.sample_idx]]
        if hasattr(dataset, "samples") and dataset.samples is not None:
            dataset.samples = dataset.data_infos

    if args.max_samples is not None:
        dataset.data_infos = dataset.data_infos[:args.max_samples]
        if hasattr(dataset, "samples") and dataset.samples is not None:
            dataset.samples = dataset.data_infos

    return dataset, tile_index


def build_aerial_model(cfg, checkpoint_path, device):
    model_cfg = cfg.model.copy()

    # The model is inference-only. Avoid loading any config pretrained weights.
    model_cfg["pretrained"] = None

    model = build_model(
        model_cfg,
        train_cfg=None,
        test_cfg=cfg.get("test_cfg", model_cfg.get("test_cfg", None)),
    )
    checkpoint = load_checkpoint(
        model, checkpoint_path, map_location="cpu"
    )

    # Print checkpoint metadata when available; useful for confirming epoch.
    meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    if meta:
        print("Checkpoint metadata:", {
            k: meta.get(k) for k in ("epoch", "iter", "version")
            if k in meta
        })

    model.eval()
    model.to(device)
    return model


def unwrap_data_container(value):
    """Unwrap MMCV DataContainer recursively where useful."""
    if hasattr(value, "data") and value.__class__.__name__ == "DataContainer":
        value = value.data
    if isinstance(value, (list, tuple)):
        return [unwrap_data_container(v) for v in value]
    return value


def extract_result_dict(batch_result):
    """Model output for this detector is a list of {'pts_bbox': pred_dict}."""
    if not isinstance(batch_result, (list, tuple)) or len(batch_result) != 1:
        raise TypeError(
            "Expected one-sample result list from AerialSatMap.forward_test; "
            "got {}".format(type(batch_result))
        )
    result = batch_result[0]
    if isinstance(result, dict) and "pts_bbox" in result:
        result = result["pts_bbox"]
    if not isinstance(result, dict):
        raise TypeError("Expected pred2result dictionary, got {}".format(type(result)))
    return result


def to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "tensor"):
        value = value.tensor
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def extract_predictions(result):
    """Mirror AerialMapDataset._extract_prediction key handling."""
    points = next(
        (result[k] for k in ("pts_3d", "pts", "polylines", "vectors")
         if k in result), None
    )
    scores = next(
        (result[k] for k in ("scores_3d", "scores", "scores_2d")
         if k in result), None
    )
    labels = next(
        (result[k] for k in ("labels_3d", "labels", "labels_2d")
         if k in result), None
    )

    if points is None:
        raise KeyError(
            "No polyline points in pred2result output. Keys: {}".format(
                list(result.keys())
            )
        )
    if labels is None:
        raise KeyError(
            "No class labels in pred2result output. Keys: {}".format(
                list(result.keys())
            )
        )

    points = to_numpy(points).astype(np.float32)
    if points.ndim == 2 and points.shape[-1] == 2:
        points = points[None, ...]
    if points.ndim != 3 or points.shape[-1] < 2:
        raise ValueError("Expected points [N,P,2+] but got {}".format(points.shape))
    points = points[..., :2]

    labels = to_numpy(labels).reshape(-1).astype(np.int64)
    if scores is None:
        scores = np.ones(len(labels), dtype=np.float32)
    else:
        scores = to_numpy(scores).reshape(-1).astype(np.float32)

    if not (len(points) == len(scores) == len(labels)):
        raise ValueError(
            "Prediction count mismatch: points={}, scores={}, labels={}".format(
                len(points), len(scores), len(labels)
            )
        )
    return points, scores, labels


def read_rgb_raster(path, max_dimension=1800):
    """Read raster for display; downsample at read time to limit memory."""
    with rasterio.open(path) as src:
        width, height = src.width, src.height
        scale = min(1.0, float(max_dimension) / max(width, height))
        out_w = max(1, int(round(width * scale)))
        out_h = max(1, int(round(height * scale)))

        count = src.count
        if count >= 3:
            bands = src.read(
                indexes=[1, 2, 3],
                out_shape=(3, out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            rgb = np.moveaxis(bands, 0, -1)
        else:
            band = src.read(
                1,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            rgb = np.repeat(band[..., None], 3, axis=-1)

        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    rgb = np.asarray(rgb)
    if np.issubdtype(rgb.dtype, np.integer):
        # Percentile stretch is for visualization only, not model input.
        valid = np.isfinite(rgb)
        if nodata is not None:
            valid &= rgb != nodata
        vals = rgb[valid]
        if vals.size:
            lo, hi = np.percentile(vals, [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            rgb = np.clip((rgb.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
        else:
            rgb = np.zeros_like(rgb, dtype=np.float32)
    else:
        rgb = rgb.astype(np.float32)
        finite = np.isfinite(rgb)
        vals = rgb[finite]
        if vals.size:
            lo, hi = np.percentile(vals, [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            rgb = np.clip((rgb - lo) * 255.0 / (hi - lo), 0, 255)
        else:
            rgb = np.zeros_like(rgb)

    return rgb.astype(np.uint8), transform, crs, (width, height)


def local_to_pixel(points_local, width, height, transform):
    """
    Invert AerialMapDataset._world_to_local for north-up rasters.

    The dataset conversion uses transform.a and abs(transform.e), with image
    center as origin. This intentionally mirrors that convention.
    """
    if abs(transform.a) < 1e-12 or abs(transform.e) < 1e-12:
        raise ValueError("Raster transform has zero pixel size.")

    cx = width / 2.0
    cy = height / 2.0
    col = cx + points_local[..., 0] / transform.a
    row = cy - points_local[..., 1] / abs(transform.e)
    return np.stack([col, row], axis=-1)


def local_to_world(points_local, width, height, transform):
    """Convert local metric BEV points to raster CRS coordinates."""
    pixels = local_to_pixel(points_local, width, height, transform)
    world = np.empty_like(pixels, dtype=np.float64)
    for idx in np.ndindex(pixels.shape[:-1]):
        world[idx] = transform * (float(pixels[idx][0]), float(pixels[idx][1]))
    return world


def load_ground_truth(dataset, sample_index, image_path):
    """Load GT using the dataset's own _build_gt implementation."""
    if not hasattr(dataset, "_build_gt"):
        return None, None
    try:
        info = dataset.get_data_info(sample_index)
        # For standalone image mode, get_data_info already resolves the
        # annotation path; an empty path means no annotation was supplied.
        ann_path = info.get("annotation_path")
        if not ann_path or not os.path.isfile(ann_path):
            return None, None
        return dataset._build_gt(info)
    except Exception as exc:
        print("GT overlay unavailable for sample {}: {}".format(sample_index, exc))
        return None, None


def write_geojson(path, points_world, scores, labels, crs=None):
    features = []
    for line, score, label in zip(points_world, scores, labels):
        if label < 0 or label >= len(CLASS_NAMES):
            class_name = "class_{}".format(int(label))
        else:
            class_name = CLASS_NAMES[int(label)]

        coordinates = [
            [float(x), float(y)] for x, y in line
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(coordinates) < 2:
            continue

        features.append({
            "type": "Feature",
            "properties": {
                "class_id": int(label),
                "class_name": class_name,
                "confidence": float(score),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
        })

    document = {
        "type": "FeatureCollection",
        "name": Path(path).stem,
        # Legacy named-CRS member is retained for GIS software such as QGIS.
        "crs": (
            {"type": "name", "properties": {"name": str(crs)}}
            if crs is not None else None
        ),
        "features": features,
    }
    with open(path, "w") as f:
        json.dump(document, f, indent=2)
    return len(features)


def save_overlay(path, rgb, transform, original_size, points_local, scores,
                 labels, threshold, gt_points=None, gt_labels=None,
                 line_width=2.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_width, original_height = original_size
    display_h, display_w = rgb.shape[:2]
    sx = display_w / float(original_width)
    sy = display_h / float(original_height)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb)
    ax.set_axis_off()

    keep = scores >= threshold
    pred = points_local[keep]
    pred_scores = scores[keep]
    pred_labels = labels[keep]

    if gt_points is not None and gt_labels is not None:
        gt_px = local_to_pixel(
            gt_points, original_width, original_height, transform
        )
        gt_px[..., 0] *= sx
        gt_px[..., 1] *= sy
        for line in gt_px:
            ax.plot(
                line[:, 0], line[:, 1],
                color="white", linewidth=max(1.0, line_width),
                alpha=0.85, linestyle="--",
            )

    if len(pred):
        pred_px = local_to_pixel(
            pred, original_width, original_height, transform
        )
        pred_px[..., 0] *= sx
        pred_px[..., 1] *= sy

        for line, score, label in zip(pred_px, pred_scores, pred_labels):
            color = np.asarray(
                CLASS_COLORS.get(int(label), (255, 255, 255)),
                dtype=np.float32
            ) / 255.0
            ax.plot(
                line[:, 0], line[:, 1],
                color=color, linewidth=line_width, alpha=0.9,
            )
            mid = line[len(line) // 2]
            ax.text(
                mid[0], mid[1], "{:.2f}".format(float(score)),
                color="black", fontsize=6,
                bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=1),
            )

    legend_handles = []
    from matplotlib.lines import Line2D
    for class_id, class_name in enumerate(CLASS_NAMES):
        c = np.asarray(CLASS_COLORS[class_id]) / 255.0
        legend_handles.append(
            Line2D([0], [0], color=c, lw=3, label=class_name)
        )
    if gt_points is not None:
        legend_handles.append(
            Line2D([0], [0], color="white", lw=2, ls="--", label="ground truth")
        )
    if legend_handles:
        # Put legend outside image so it does not obscure predictions.
        leg = ax.legend(handles=legend_handles, loc="lower center",
                        bbox_to_anchor=(0.5, -0.015), ncol=4,
                        framealpha=0.9, fontsize=9)
        for text_obj in leg.get_texts():
            text_obj.set_color("black")

    ax.set_title(
        "AerialSatMap predictions | score >= {:.2f} | {} lines".format(
            threshold, int(keep.sum())
        ),
        color="white",
        fontsize=11,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    cfg = Config.fromfile(args.config)
    import_errors = import_custom_modules(cfg, args)
    print("Custom import attempts completed.")
    if import_errors:
        print("Some optional import candidates were unavailable:")
        for item in import_errors:
            print("  ", item)

    dataset, tile_index = make_dataset(cfg, args)
    print("Inference samples:", len(dataset))
    if tile_index is not None:
        print("Tile-index entries:", len(tile_index))

    model = build_aerial_model(cfg, args.checkpoint, args.device)
    model = MMDataParallel(model, device_ids=[0] if args.device.startswith("cuda") else [])

    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
    )

    for sample_index, data in enumerate(loader):
        info = dataset.get_data_info(sample_index)
        image_path = info["aerial_img_path"]
        sample_name = Path(image_path).stem
        sample_dir = os.path.join(args.out_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        # Keep the standard MMDetection3D test-time dispatch. AerialSatMap's
        # forward_test unwraps the augmentation list and calls simple_test.
        with torch.no_grad():
            batch_result = model(return_loss=False, rescale=True, **data)

        result_dict = extract_result_dict(batch_result)
        points, scores, labels = extract_predictions(result_dict)

        print("\n[{}] {}".format(sample_index, image_path))
        print("  result keys:", list(result_dict.keys()))
        print("  points:", points.shape, "scores:", scores.shape, "labels:", labels.shape)
        if len(scores):
            print("  score range: {:.4f} .. {:.4f}".format(
                float(scores.min()), float(scores.max())
            ))
            print("  class counts:", {
                CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c):
                int((labels == c).sum())
                for c in np.unique(labels)
            })

        np.savez_compressed(
            os.path.join(sample_dir, "predictions.npz"),
            points_local=points,
            scores=scores,
            labels=labels,
        )

        rgb, raster_transform, raster_crs, original_size = read_rgb_raster(
            image_path, args.max_display_dimension
        )
        width, height = original_size

        tile_meta = None
        if tile_index is not None:
            tile_meta = tile_index.get(os.path.basename(image_path))
            if tile_meta is None:
                raise KeyError(
                    "Tile {} is absent from --tile-index {}".format(
                        os.path.basename(image_path), args.tile_index
                    )
                )
            transform = tile_meta["transform"]
            crs = tile_meta["crs"]
            if tile_meta["width"] and tile_meta["width"] != width:
                raise ValueError("Tile-index width does not match raster for {}".format(image_path))
            if tile_meta["height"] and tile_meta["height"] != height:
                raise ValueError("Tile-index height does not match raster for {}".format(image_path))
            # The current dataset's local frame assumes a north-up, unrotated raster.
            if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9:
                raise ValueError("Rotated/sheared affine transform is not supported by this local-frame conversion: {}".format(image_path))
            if raster_crs and str(raster_crs) != str(crs):
                print("  note: raster CRS {} differs from tile-index CRS {}".format(raster_crs, crs))
        else:
            transform, crs = raster_transform, raster_crs

        points_world = local_to_world(points, width, height, transform)

        # Always save the unfiltered output.
        raw_path = os.path.join(sample_dir, "predictions_raw.geojson")
        n_raw = write_geojson(raw_path, points_world, scores, labels, crs=crs)
        with open(os.path.join(sample_dir, "raster_metadata.json"), "w") as f:
            json.dump({
                "image": image_path,
                "crs": str(crs) if crs else None,
                "width": width,
                "height": height,
                "transform": list(transform)[:6],
                "tile_index_entry": tile_meta["fname"] if tile_meta else None,
                "coordinate_frame": "AerialMapDataset local metric frame",
                "class_names": list(CLASS_NAMES),
            }, f, indent=2)
        print("  raw GeoJSON features:", n_raw, "| CRS:", crs)

        gt_points, gt_labels = (None, None)
        if not args.no_gt:
            gt_points, gt_labels = load_ground_truth(
                dataset, sample_index, image_path
            )
            if gt_points is not None:
                print("  GT lines:", len(gt_points))

        for threshold in args.score_thrs:
            keep = scores >= threshold
            filtered_world = points_world[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]
            tag = "{:.2f}".format(threshold).replace(".", "p")

            geojson_path = os.path.join(
                sample_dir, "predictions_thr_{}.geojson".format(tag)
            )
            count = write_geojson(
                geojson_path, filtered_world, filtered_scores, filtered_labels,
                crs=crs
            )

            overlay_path = os.path.join(
                sample_dir, "overlay_thr_{}.png".format(tag)
            )
            save_overlay(
                overlay_path, rgb, transform, original_size,
                points, scores, labels, threshold,
                gt_points=gt_points, gt_labels=gt_labels,
                line_width=args.line_width,
            )
            print("  threshold {:.2f}: {} lines -> {}".format(
                threshold, count, overlay_path
            ))

    print("\nInference complete. Outputs:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()