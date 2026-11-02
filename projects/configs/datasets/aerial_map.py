import os
import json
import numpy as np
import rasterio
import torch

from shapely.geometry import LineString, MultiLineString

from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset

from projects.mmdet3d_plugin.satmap.detectors.satmap import (
    SatMap,
)
from projects.mmdet3d_plugin.aerial_map.instance_lines import (
    AerialInstanceLines,
)

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

@DATASETS.register_module()
class AerialMapDataset(Custom3DDataset):
    """
    Dataset for orthographic aerial imagery + vector map annotations.

    Each sample consists of:

        aerial image
        GeoJSON vector annotations

    GeoJSON coordinates are assumed to be in the same CRS as the
    raster unless coordinate_mode='pixel'.
    """

    CLASSES = (
        "divider",
        "ped_crossing",
        "boundary",
    )

    def __init__(
        self,
        data_root="",
        ann_file=None,
        pipeline=None,
        classes=None,
        pc_range=None,
        fixed_ptsnum_per_line=20,
        coordinate_mode="world",
        image_size=None,
        samples=None,
        **kwargs
    ):
        self.data_root = data_root
        self.samples = samples
        self.fixed_ptsnum_per_line = fixed_ptsnum_per_line
        self.pc_range = pc_range
        self.coordinate_mode = coordinate_mode
        self.image_size = image_size

        if classes is not None:
            self.CLASSES = tuple(classes)

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=self.CLASSES,
            **kwargs
        )

    def __getitem__(self, index):
        """Return an aerial-map sample without 3D detection filtering."""
        if self.test_mode:
            return self.prepare_test_img(index)

        return self.prepare_train_img(index)

    def load_annotations(self, ann_file):
        if self.samples is not None:
            return self.samples

        if ann_file is None:
            raise ValueError(
                "AerialMapDataset requires samples or an ann_file."
            )

        with open(ann_file, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data["samples"]

        return data

    def _load_geojson(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _feature_class(self, feature):
        props = feature.get("properties", {})

        if "class" in props:
            return props["class"]

        if "class_name" in props:
            return props["class_name"]

        class_id = props.get("class_id", None)

        if class_id is not None:
            class_map = {
                0: "divider",
                1: "ped_crossing",
                2: "boundary",
            }
            return class_map.get(class_id)

        return None

    def _extract_lines(self, geojson):
        lines = []

        for feature in geojson.get("features", []):

            class_name = self._feature_class(feature)

            if class_name not in self.CLASSES:
                continue

            geometry = feature.get("geometry", {})
            geom_type = geometry.get("type")
            coords = geometry.get("coordinates")

            if geom_type == "LineString":
                lines.append(
                    (
                        class_name,
                        np.asarray(coords, dtype=np.float32),
                    )
                )

            elif geom_type == "MultiLineString":
                for line in coords:
                    lines.append(
                        (
                            class_name,
                            np.asarray(line, dtype=np.float32),
                        )
                    )

        return lines

    def _world_to_local(
        self,
        coords,
        transform,
        width,
        height,
    ):
        """
        Convert raster/world coordinates to a local BEV frame.

        Origin = center of aerial image.

        +X = image right
        +Y = image up
        """

        # Image center in pixel coordinates.
        cx = width / 2.0
        cy = height / 2.0

        # Convert world coordinates to pixel coordinates.
        inv_transform = ~transform

        pixels = []

        for x, y in coords:
            col, row = inv_transform * (x, y)

            local_x = (col - cx) * transform.a
            local_y = (cy - row) * abs(transform.e)

            pixels.append(
                [local_x, local_y]
            )

        return np.asarray(
            pixels,
            dtype=np.float32,
        )

    def _pixel_to_local(
        self,
        coords,
        width,
        height,
        gsd,
    ):

        cx = width / 2.0
        cy = height / 2.0

        result = np.zeros_like(
            coords,
            dtype=np.float32,
        )

        result[:, 0] = (
            coords[:, 0] - cx
        ) * gsd

        result[:, 1] = (
            cy - coords[:, 1]
        ) * gsd

        return result

    def _sample_line(
        self,
        coords,
        n_points,
    ):

        if len(coords) < 2:
            return None

        line = LineString(coords)

        if line.length == 0:
            return None

        distances = np.linspace(
            0,
            line.length,
            n_points,
        )

        sampled = np.asarray(
            [
                line.interpolate(d).coords[0]
                for d in distances
            ],
            dtype=np.float32,
        )

        return sampled

    def get_data_info(self, index):

        info = self.data_infos[index]

        image_path = os.path.join(
            self.data_root,
            info["image"],
        )

        annotation_path = os.path.join(
            self.data_root,
            info["annotation"],
        )

        return {
            "aerial_img_path": image_path,
            "annotation_path": annotation_path,
            "sample_idx": index,
        }

    def _build_gt(self, info):

        with rasterio.open(
            info["aerial_img_path"]
        ) as src:

            width = src.width
            height = src.height
            transform = src.transform

            geojson = self._load_geojson(
                info["annotation_path"]
            )

            lines = self._extract_lines(
                geojson
            )

            gt_points = []
            gt_labels = []

            for class_name, coords in lines:

                if self.coordinate_mode == "world":

                    coords = self._world_to_local(
                        coords,
                        transform,
                        width,
                        height,
                    )

                elif self.coordinate_mode == "pixel":

                    gsd = abs(transform.a)

                    coords = self._pixel_to_local(
                        coords,
                        width,
                        height,
                        gsd,
                    )

                sampled = self._sample_line(
                    coords,
                    self.fixed_ptsnum_per_line,
                )

                if sampled is None:
                    continue

                # Keep only vectors inside the model range.
                if self.pc_range is not None:

                    xmin, ymin = self.pc_range[:2]
                    xmax, ymax = self.pc_range[3:5]

                    if not np.any(
                        (sampled[:, 0] >= xmin)
                        & (sampled[:, 0] <= xmax)
                        & (sampled[:, 1] >= ymin)
                        & (sampled[:, 1] <= ymax)
                    ):
                        continue

                gt_points.append(sampled)

                gt_labels.append(
                    self.CLASSES.index(
                        class_name
                    )
                )

        if len(gt_points) == 0:

            points = np.zeros(
                (
                    0,
                    self.fixed_ptsnum_per_line,
                    2,
                ),
                dtype=np.float32,
            )

            labels = np.zeros(
                (0,),
                dtype=np.int64,
            )

        else:

            points = np.stack(
                gt_points,
                axis=0,
            )

            labels = np.asarray(
                gt_labels,
                dtype=np.int64,
            )

        return points, labels


    ###
    @staticmethod
    def _as_numpy(value):
        """Convert a tensor or array-like value to NumPy."""
        if value is None:
            return None

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()

        return np.asarray(value)

    def _extract_prediction(self, result):

        ## added for non-debug
        if isinstance(result, dict) and "pts_bbox" in result:
            result = result["pts_bbox"]


        if not isinstance(result, dict):
            raise TypeError(
                "Expected each prediction result to be a dictionary, "
                "as returned by pred2result(). Got: {}".format(
                    type(result)
                )
            )

        points = None
        for key in ("pts_3d", "pts", "polylines", "vectors"):
            if key in result:
                points = result[key]
                break

        scores = None
        for key in ("scores_3d", "scores", "scores_2d"):
            if key in result:
                scores = result[key]
                break

        labels = None
        for key in ("labels_3d", "labels", "labels_2d"):
            if key in result:
                labels = result[key]
                break

        if points is None:
            raise KeyError(
                "Could not find predicted polyline points in result. "
                "Available keys: {}".format(list(result.keys()))
            )

        points = self._as_numpy(points)
        labels = self._as_numpy(labels)
        scores = self._as_numpy(scores)

        # Support box-like wrappers exposing a tensor.
        if hasattr(points, "tensor"):
            points = self._as_numpy(points.tensor)

        points = np.asarray(points, dtype=np.float32)

        # Handle a single polyline represented as [N, 2].
        if points.ndim == 2 and points.shape[-1] == 2:
            points = points[None, ...]

        if points.ndim != 3 or points.shape[-1] < 2:
            raise ValueError(
                "Expected predicted points shaped [N, P, 2] "
                "or [N, P, D], got {}".format(points.shape)
            )

        points = points[..., :2]

        if labels is None:
            raise KeyError(
                "Prediction result has no class labels. "
                "Available keys: {}".format(list(result.keys()))
            )

        labels = np.asarray(labels).reshape(-1).astype(np.int64)

        if scores is None:
            scores = np.ones(len(labels), dtype=np.float32)
        else:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if not (
            len(points) == len(labels) == len(scores)
        ):
            raise ValueError(
                "Prediction lengths disagree: points={}, labels={}, "
                "scores={}".format(
                    len(points), len(labels), len(scores)
                )
            )

        return points, scores, labels

    @staticmethod
    def _chamfer_distance(points_a, points_b):
        """
        Symmetric mean Chamfer distance between two polylines.

        Uses distances between sampled points, in the coordinate units
        of the local aerial map frame.
        """
        if len(points_a) == 0 or len(points_b) == 0:
            return float("inf")

        distances = cdist(
            points_a[:, :2],
            points_b[:, :2],
            metric="euclidean",
        )

        return float(
            0.5 * (
                distances.min(axis=1).mean()
                + distances.min(axis=0).mean()
            )
        )

    def evaluate(
        self,
        results,
        metric="chamfer",
        logger=None,
        score_thr=0.0,
        chamfer_threshold=1.5,
        **kwargs
    ):
        """
        Evaluate aerial-map polylines without the inherited 3D evaluator.

        Args:
            results: One prediction dictionary per dataset sample.
            metric: 'chamfer' or 'chamfer_f1'.
            logger: Optional MMCV logger.
            score_thr: Minimum prediction confidence.
            chamfer_threshold: Maximum Chamfer distance for a match,
                expressed in the local map coordinate units.

        Returns:
            dict: Aggregate and per-class validation metrics.
        """
        if isinstance(metric, (list, tuple)):
            if len(metric) != 1:
                raise ValueError(
                    "AerialMapDataset supports one metric at a time."
                )
            metric = metric[0]

        if metric not in ("chamfer", "chamfer_f1"):
            raise KeyError(
                "Unsupported metric {!r}. Use 'chamfer' "
                "or 'chamfer_f1'.".format(metric)
            )

        if len(results) != len(self.data_infos):
            raise ValueError(
                "Got {} predictions for {} dataset samples.".format(
                    len(results), len(self.data_infos)
                )
            )

        if chamfer_threshold <= 0:
            raise ValueError("chamfer_threshold must be positive.")

        num_classes = len(self.CLASSES)

        # Accumulate per-class counts and matched distances.
        stats = {
            class_id: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "distances": [],
            }
            for class_id in range(num_classes)
        }

        for sample_idx, result in enumerate(results):
            info = self.get_data_info(sample_idx)

            # Build GT using exactly the same transformations as training.
            gt_points, gt_labels = self._build_gt(info)

            pred_points, pred_scores, pred_labels = (
                self._extract_prediction(result)
            )

            # Ignore low-confidence predictions.
            keep = pred_scores >= score_thr
            pred_points = pred_points[keep]
            pred_labels = pred_labels[keep]

            for class_id in range(num_classes):
                gt_mask = gt_labels == class_id
                pred_mask = pred_labels == class_id

                gt_class = gt_points[gt_mask]
                pred_class = pred_points[pred_mask]

                n_gt = len(gt_class)
                n_pred = len(pred_class)

                if n_gt == 0:
                    stats[class_id]["fp"] += n_pred
                    continue

                if n_pred == 0:
                    stats[class_id]["fn"] += n_gt
                    continue

                # Pairwise Chamfer distance matrix.
                cost = np.empty(
                    (n_pred, n_gt),
                    dtype=np.float64,
                )

                for i in range(n_pred):
                    for j in range(n_gt):
                        cost[i, j] = self._chamfer_distance(
                            pred_class[i],
                            gt_class[j],
                        )

                # One-to-one minimum-cost matching.
                pred_indices, gt_indices = linear_sum_assignment(
                    cost
                )


                #####################DEBUG
                # Temporary diagnostics: inspect only the first sample.
                # if sample_idx == 0:
                #     print("\n===== COORDINATE DIAGNOSTICS =====")
                #     print("Prediction points shape:", pred_points.shape)
                #     print("Ground truth points shape:", gt_points.shape)

                #     if len(pred_points):
                #         print(
                #             "Prediction X range:",
                #             float(pred_points[..., 0].min()),
                #             float(pred_points[..., 0].max()),
                #         )
                #         print(
                #             "Prediction Y range:",
                #             float(pred_points[..., 1].min()),
                #             float(pred_points[..., 1].max()),
                #         )
                #         print(
                #             "Prediction labels:",
                #             np.unique(pred_labels, return_counts=True),
                #         )
                #         print(
                #             "Prediction scores:",
                #             float(pred_scores.min()),
                #             float(pred_scores.max()),
                #         )

                #     if len(gt_points):
                #         print(
                #             "GT X range:",
                #             float(gt_points[..., 0].min()),
                #             float(gt_points[..., 0].max()),
                #         )
                #         print(
                #             "GT Y range:",
                #             float(gt_points[..., 1].min()),
                #             float(gt_points[..., 1].max()),
                #         )
                #         print(
                #             "GT labels:",
                #             np.unique(gt_labels, return_counts=True),
                #         )

                #     if cost.size:
                #         print(
                #             "Minimum Chamfer distance:",
                #             float(cost.min()),
                #             "m; median:",
                #             float(np.median(cost)),
                #             "m; threshold:",
                #             chamfer_threshold,
                #             "m",
                #         )
                #####################

                matched = 0

                for pred_i, gt_i in zip(
                    pred_indices, gt_indices
                ):
                    distance = cost[pred_i, gt_i]

                    if distance <= chamfer_threshold:
                        matched += 1
                        stats[class_id]["distances"].append(
                            float(distance)
                        )

                stats[class_id]["tp"] += matched
                stats[class_id]["fp"] += n_pred - matched
                stats[class_id]["fn"] += n_gt - matched

        metrics = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_distances = []

        for class_id, class_name in enumerate(self.CLASSES):
            values = stats[class_id]

            tp = values["tp"]
            fp = values["fp"]
            fn = values["fn"]
            distances = values["distances"]

            precision = tp / float(tp + fp) if tp + fp else 0.0
            recall = tp / float(tp + fn) if tp + fn else 0.0

            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            mean_chamfer = (
                float(np.mean(distances))
                if distances else float("nan")
            )

            prefix = class_name + "/"

            metrics[prefix + "precision"] = precision
            metrics[prefix + "recall"] = recall
            metrics[prefix + "f1"] = f1
            metrics[prefix + "chamfer"] = mean_chamfer
            metrics[prefix + "tp"] = tp
            metrics[prefix + "fp"] = fp
            metrics[prefix + "fn"] = fn

            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_distances.extend(distances)

        metrics["precision"] = (
            total_tp / float(total_tp + total_fp)
            if total_tp + total_fp else 0.0
        )

        metrics["recall"] = (
            total_tp / float(total_tp + total_fn)
            if total_tp + total_fn else 0.0
        )

        p = metrics["precision"]
        r = metrics["recall"]

        metrics["f1"] = (
            2.0 * p * r / (p + r)
            if p + r else 0.0
        )

        metrics["chamfer"] = (
            float(np.mean(all_distances))
            if all_distances else float("nan")
        )

        if logger is not None:
            for key, value in metrics.items():
                logger.info(
                    "AerialMap evaluation - %s: %s",
                    key,
                    value,
                )

        return metrics


    ###
    def prepare_train_img(self, index):
        info = self.get_data_info(index)

        results = {
            "aerial_img_path": info["aerial_img_path"],
            "sample_idx": index,
        }

        points, labels = self._build_gt(info)

        # Wrap sampled polylines in the instance-line representation
        # expected by MapTRHead.
        results["gt_bboxes_3d"] = AerialInstanceLines(
            points,
            fixed_num=self.fixed_ptsnum_per_line,
        )

        results["gt_labels_3d"] = torch.as_tensor(
            labels,
            dtype=torch.long,
        )

        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, index):
        info = self.get_data_info(index)

        results = {
            "aerial_img_path": info["aerial_img_path"],
            "sample_idx": index,
        }

        if not self.test_mode:
            points, labels = self._build_gt(info)

            results["gt_bboxes_3d"] = AerialInstanceLines(
                points,
                fixed_num=self.fixed_ptsnum_per_line,
            )

            results["gt_labels_3d"] = torch.as_tensor(
                labels,
                dtype=torch.long,
            )

        self.pre_pipeline(results)

        return self.pipeline(results)