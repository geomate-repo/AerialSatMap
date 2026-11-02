import numpy as np
import mmcv
import torch

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES

from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadAerialImage:

    def __init__(
        self,
        to_float32=True,
    ):
        self.to_float32 = to_float32

    def __call__(self, results):

        path = results["aerial_img_path"]

        img = mmcv.imread(
            path,
            flag="color",
        )

        if img is None:
            raise FileNotFoundError(
                f"Could not load aerial image: {path}"
            )

        if self.to_float32:
            img = img.astype(
                np.float32
            )

        results["aerial_img"] = img

        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape

        return results


@PIPELINES.register_module()
class NormalizeAerialImage:

    def __init__(
        self,
        mean,
        std,
    ):

        self.mean = np.asarray(
            mean,
            dtype=np.float32,
        )

        self.std = np.asarray(
            std,
            dtype=np.float32,
        )

    def __call__(self, results):

        img = results["aerial_img"]

        img = (
            img - self.mean
        ) / self.std

        results["aerial_img"] = img

        return results

@PIPELINES.register_module()
class DefaultFormatBundleAerial:
    def __call__(self, results):
        img = results["aerial_img"]
        img = img.transpose(2, 0, 1)

        results["aerial_img"] = DC(
            torch.from_numpy(np.ascontiguousarray(img)),
            stack=True,
        )

        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = DC(
                results["gt_bboxes_3d"],
                cpu_only=True,
            )

        if "gt_labels_3d" in results:
            results["gt_labels_3d"] = DC(
                torch.as_tensor(
                    results["gt_labels_3d"],
                    dtype=torch.long,
                ),
                stack=False,
            )

        return results

@PIPELINES.register_module()
class CollectAerialMap:

    def __init__(
        self,
        keys,
    ):
        self.keys = keys

    def __call__(self, results):

        data = {}

        for key in self.keys:

            if key in results:
                data[key] = results[key]

        # MapTR requires img_metas.
        data["img_metas"] = DC(
            {
                "sample_idx": results[
                    "sample_idx"
                ],
                "can_bus": np.zeros(
                    18,
                    dtype=np.float32,
                ),
                "scene_token": str(
                    results["sample_idx"]
                ),
                "prev_bev_exists": False,
            },
            cpu_only=True,
        )

        return data