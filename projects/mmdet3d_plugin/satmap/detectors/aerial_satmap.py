import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_backbone, build_neck
from mmcv.runner import force_fp32

from projects.mmdet3d_plugin.satmap.detectors.satmap import SatMap


@DETECTORS.register_module()
class AerialSatMap(SatMap):
    """
    Aerial-only adaptation of SatMap.

    Architecture:

        aerial image
            |
        Swin backbone
            |
        GFPN
            |
        BEV feature projection
            |
        MapTR decoder
            |
        vectorized HD map

    No camera branch, BEVFormer/GKT, LiDAR branch, or ConvFuser
    is used.
    """

    def __init__(
        self,
        aerial_backbone=None,
        aerial_neck=None,
        bev_h=80,
        bev_w=40,
        embed_dims=256,
        **kwargs,
    ):

        # Do not construct the original camera backbone.
        kwargs["img_backbone"] = None
        kwargs["img_neck"] = None

        self.align = None

        # We inherit the MapTR head infrastructure from SatMap.
        super().__init__(
            **kwargs,
            prior_map_encoder=None,
            fuser=None,
        )

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims

        if aerial_backbone is None:
            raise ValueError("aerial_backbone must be provided.")

        if aerial_neck is None:
            raise ValueError("aerial_neck must be provided.")

        self.aerial_backbone = build_backbone(aerial_backbone)
        self.aerial_neck = build_neck(aerial_neck)

        self.aerial_projection = nn.Sequential(
            nn.Conv2d(
                embed_dims,
                embed_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
        )


    def extract_aerial_feat(self, aerial_img):
        """
        Extract hierarchical aerial-image features.

        Args:
            aerial_img: [B, 3, H, W]

        Returns:
            Projected aerial features: [B, C, H', W']
        """
        feats = self.aerial_backbone(aerial_img)

        if isinstance(feats, dict):
            feats = list(feats.values())

        feats = self.aerial_neck(feats)

        # The configured neck has one output level.
        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        # Match the input dtype to the projection weights.
        projection_dtype = self.aerial_projection[0].weight.dtype
        if feats.dtype != projection_dtype:
            feats = feats.to(dtype=projection_dtype)

        feats = self.aerial_projection(feats)

        return feats


    def aerial_to_bev(self, aerial_feat):
        """
        Convert the aerial feature map into the spatial BEV grid expected
        by MapTR.

        Since the input itself is already an orthographic/top-down image,
        no perspective-to-BEV transformation is required.
        """

        aerial_feat = F.interpolate(
            aerial_feat,
            size=(self.bev_h, self.bev_w),
            mode="bilinear",
            align_corners=False,
        )

        # [B,C,H,W] -> [B,H*W,C]
        bev_feat = (
            aerial_feat
            .flatten(2)
            .permute(0, 2, 1)
            .contiguous()
        )

        return bev_feat

    @force_fp32(
        apply_to=("aerial_img",)
    )
    def forward_train(
        self,
        aerial_img=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        if aerial_img is None:
            raise ValueError("aerial_img is required.")

        aerial_feat = self.extract_aerial_feat(aerial_img)

        bev_embed = self.aerial_to_bev(aerial_feat)

        outs = self.pts_bbox_head(
            None,
            None,
            img_metas,
            None,
            bev_embed=bev_embed,
        )

        if gt_labels_3d is None:
            raise ValueError(
                "gt_labels_3d is None in AerialSatMap.forward_train. "
                "Check AerialMapDataset and CollectAerialMap."
            )

        if gt_bboxes_3d is None:
            raise ValueError(
                "gt_bboxes_3d is None in AerialSatMap.forward_train. "
                "Check AerialMapDataset and CollectAerialMap."
            )


        ###
        # import inspect

        # print("\n========== HEAD LOSS DEBUG ==========")

        # print("Head class:", type(self.pts_bbox_head))
        # print("Head module:", type(self.pts_bbox_head).__module__)
        # print("Head source:", inspect.getfile(type(self.pts_bbox_head)))

        # print(
        #     "Loss signature:",
        #     inspect.signature(self.pts_bbox_head.loss)
        # )

        # print("gt_bboxes_3d type:", type(gt_bboxes_3d))
        # print("gt_labels_3d type:", type(gt_labels_3d))
        # print("outs type:", type(outs))

        # if isinstance(gt_labels_3d, (list, tuple)):
        #     print("gt_labels_3d length:", len(gt_labels_3d))
        #     for i, item in enumerate(gt_labels_3d):
        #         print(f"  labels[{i}] type:", type(item))
        #         if hasattr(item, "shape"):
        #             print(f"  labels[{i}] shape:", item.shape)
        #         if hasattr(item, "device"):
        #             print(f"  labels[{i}] device:", item.device)
        # elif hasattr(gt_labels_3d, "shape"):
        #     print("gt_labels_3d shape:", gt_labels_3d.shape)
        #     print("gt_labels_3d device:", gt_labels_3d.device)

        # print("=====================================\n")
        ###

        # losses = self.pts_bbox_head.loss(
        #     gt_bboxes_3d,
        #     gt_labels_3d,
        #     outs,
        #     img_metas=img_metas,
        # )

        # print("\n========== PRE-LOSS DEBUG ==========")

        # print("gt_bboxes_3d:", type(gt_bboxes_3d))
        # print("gt_labels_3d:", type(gt_labels_3d))
        # print("outs:", type(outs))

        # assert gt_bboxes_3d is not None
        # assert gt_labels_3d is not None

        # assert isinstance(gt_bboxes_3d, (list, tuple))
        # assert isinstance(gt_labels_3d, (list, tuple))

        # assert len(gt_bboxes_3d) == len(gt_labels_3d)

        # for i, (boxes, labels) in enumerate(zip(gt_bboxes_3d, gt_labels_3d)):
        #     print(
        #         "Sample {}: boxes={}, labels={}, label_shape={}".format(
        #             i,
        #             type(boxes),
        #             type(labels),
        #             getattr(labels, "shape", None),
        #         )
        #     )

        # print("===================================\n")

        losses = self.pts_bbox_head.loss(
            gt_bboxes_list=gt_bboxes_3d,
            gt_labels_list=gt_labels_3d,
            preds_dicts=outs,
            gt_bboxes_ignore=None,
            img_metas=img_metas,
        )

        return losses

    def simple_test(
        self,
        img_metas,
        aerial_img=None,
        rescale=False,
        **kwargs,
    ):

        if aerial_img is None:
            raise ValueError("aerial_img is required.")

        aerial_feat = self.extract_aerial_feat(aerial_img)

        bev_embed = self.aerial_to_bev(aerial_feat)

        outs = self.pts_bbox_head(
            None,
            None,
            img_metas,
            None,
            bev_embed=bev_embed,
        )

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs,
            img_metas,
            rescale=rescale,
        )

        bbox_results = []

        for bboxes, scores, labels, pts in bbox_list:
            bbox_results.append(
                self.pred2result(
                    bboxes,
                    scores,
                    labels,
                    pts,
                )
            )

        return bbox_results

    def forward_test(
        self,
        img_metas,
        aerial_img=None,
        **kwargs,
    ):

        if not isinstance(img_metas, list):
            raise TypeError(
                "img_metas must be a list."
            )

        # MMDetection3D test-time convention.
        if isinstance(aerial_img, list):
            aerial_img = aerial_img[0]

        return [
            dict(
                pts_bbox=result
            )
            for result in self.simple_test(
                img_metas[0],
                aerial_img=aerial_img,
                **kwargs,
            )
        ]