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


import numpy as np
import torch


class AerialInstanceLines:
    """Vector-instance wrapper compatible with SatMap's MapTRHead.

    Args:
        points: Sampled local-coordinate polylines, [N, fixed_num, 2].
        fixed_num: Number of points per polyline.
        padding_value: Sentinel used for invalid shift candidates.
        eps: Tolerance for detecting closed polylines.
    """

    def __init__(
        self,
        points,
        fixed_num=20,
        padding_value=-10000.0,
        eps=1e-5,
    ):
        points = np.asarray(points, dtype=np.float32)

        if points.size == 0:
            points = np.empty((0, fixed_num, 2), dtype=np.float32)

        if points.ndim != 3 or points.shape[1:] != (fixed_num, 2):
            raise ValueError(
                "Expected points shaped [N, {}, 2], got {}".format(
                    fixed_num, points.shape
                )
            )

        self.points = torch.from_numpy(
            np.ascontiguousarray(points)
        ).float()

        self.fixed_num = int(fixed_num)
        self.padding_value = float(padding_value)
        self.eps = float(eps)

    def __len__(self):
        return self.points.shape[0]

    @property
    def bbox(self):
        """Return [xmin, ymin, xmax, ymax] for each polyline: [N, 4]."""
        if len(self) == 0:
            return self.points.new_empty((0, 4))

        minimum = self.points.amin(dim=1)
        maximum = self.points.amax(dim=1)

        return torch.cat([minimum, maximum], dim=-1)

    @property
    def fixed_num_sampled_points(self):
        """Sampled points in the local coordinate system: [N, 20, 2]."""
        return self.points

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """Generate v2 shift candidates: [N, 19, 20, 2].

        Closed lines:
            Rotate the unique sampled vertices in forward cyclic order,
            append the new starting point, and retain 19 candidates.

        Open lines:
            Use the forward and reversed point sequences, then pad the
            remaining candidates with padding_value.

        This follows the MapTR v2 convention of fixed_num - 1 candidates.
        """
        num_shifts = self.fixed_num - 1

        if len(self) == 0:
            return self.points.new_empty(
                (0, num_shifts, self.fixed_num, 2)
            )

        all_shifts = []

        for line in self.points:
            is_closed = torch.allclose(
                line[0],
                line[-1],
                atol=self.eps,
                rtol=0.0,
            )

            candidates = []

            if is_closed:
                # The final point duplicates the first point. Rotate only
                # the unique vertices, then close each rotated sequence.
                unique_points = line[:-1]
                num_unique = unique_points.shape[0]

                for shift in range(min(num_unique, num_shifts)):
                    rotated = torch.roll(
                        unique_points,
                        shifts=shift,
                        dims=0,
                    )
                    candidate = torch.cat(
                        [rotated, rotated[:1]],
                        dim=0,
                    )
                    candidates.append(candidate)

            else:
                # Open polylines have two valid orientations.
                candidates.append(line)
                candidates.append(torch.flip(line, dims=[0]))

            # Ensure every instance has exactly fixed_num - 1 candidates.
            if len(candidates) > num_shifts:
                candidates = candidates[:num_shifts]

            while len(candidates) < num_shifts:
                candidates.append(
                    line.new_full(
                        (self.fixed_num, 2),
                        self.padding_value,
                    )
                )

            all_shifts.append(torch.stack(candidates, dim=0))

        return torch.stack(all_shifts, dim=0)


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

    # def __init__(
    #     self,
    #     data_root,
    #     ann_file,
    #     pipeline,
    #     classes=None,
    #     pc_range=None,
    #     fixed_ptsnum_per_line=20,
    #     coordinate_mode="world",
    #     image_size=None,
    #     **kwargs,
    # ):

    #     self.data_root = data_root
    #     self.fixed_ptsnum_per_line = fixed_ptsnum_per_line
    #     self.pc_range = pc_range
    #     self.coordinate_mode = coordinate_mode
    #     self.image_size = image_size

    #     if classes is not None:
    #         self.CLASSES = tuple(classes)

    #     super().__init__(
    #         data_root=data_root,
    #         ann_file=ann_file,
    #         pipeline=pipeline,
    #         classes=self.CLASSES,
    #         **kwargs,
    #     )
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

    # def load_annotations(self, ann_file):
    #     """
    #     Annotation format:

    #     [
    #         {
    #             "image": "tile_001.jp2",
    #             "annotation": "tile_001.geojson"
    #         },
    #         ...
    #     ]

    #     Alternatively the annotation file can be a JSON object
    #     containing a list under "samples".
    #     """

    #     with open(ann_file, "r") as f:
    #         data = json.load(f)

    #     if isinstance(data, dict):
    #         data = data["samples"]

    #     return data
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

    def prepare_train_img(self, index):

        info = self.get_data_info(index)


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

        # results = {
        #     "aerial_img_path": info[
        #         "aerial_img_path"
        #     ],
        #     "sample_idx": index,
        # }

        # points, labels = self._build_gt(
        #     info
        # )

        # results["gt_vecs_pts_loc"] = points
        # results["gt_vecs_label"] = labels

        # self.pre_pipeline(results)

        # return self.pipeline(results)

    def prepare_test_img(self, index):

        info = self.get_data_info(index)

        results = {
            "aerial_img_path": info[
                "aerial_img_path"
            ],
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

            # points, labels = self._build_gt(
            #     info
            # )

            # results[
            #     "gt_vecs_pts_loc"
            # ] = points

            # results[
            #     "gt_vecs_label"
            # ] = labels

        self.pre_pipeline(results)

        return self.pipeline(results)