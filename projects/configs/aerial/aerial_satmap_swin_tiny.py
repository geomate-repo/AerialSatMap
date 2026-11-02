_base_ = [
    "../datasets/aerial_map.py",
    "../_base_/default_runtime.py",
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"


# ============================================================
# Geometry
# ============================================================

# point_cloud_range = [
#     -15.0,
#     -30.0,
#     -2.0,
#     15.0,
#     30.0,
#     2.0,
# ]

point_cloud_range = [
    -46.6534, -46.6534, -2.0,
     46.6534,  46.6534,  2.0,
]

voxel_size = [
    0.15,
    0.15,
    4.0,
]


# ============================================================
# Map classes
# ============================================================

map_classes = [
    "divider",
    "ped_crossing",
    "boundary",
]

num_map_classes = len(map_classes)

fixed_ptsnum_per_line = 20

bev_h = 80
bev_w = 40


# ============================================================
# Model dimensions
# ============================================================

embed_dims = 256
pos_dim = embed_dims // 2
ffn_dim = embed_dims * 2

num_feature_levels = 1


# ============================================================
# Model
# ============================================================

model = dict(

    type="AerialSatMap",

    use_grid_mask=False,

    video_test_mode=False,

    bev_h=bev_h,
    bev_w=bev_w,

    embed_dims=embed_dims,


    # --------------------------------------------------------
    # Aerial encoder
    # --------------------------------------------------------

    aerial_backbone=dict(

        type="SwinTransformer",

        pretrained=
            "ckpts/swin_tiny_patch4_window7_224.pth",

        embed_dims=96,

        depths=[
            2,
            2,
            6,
            2,
        ],

        num_heads=[
            3,
            6,
            12,
            24,
        ],

        window_size=7,

        mlp_ratio=4,

        qkv_bias=True,

        drop_rate=0.0,

        attn_drop_rate=0.0,

        drop_path_rate=0.3,

        patch_norm=True,

        out_indices=[
            1,
            2,
            3,
        ],

        with_cp=False,
    ),


    # --------------------------------------------------------
    # Aerial feature pyramid
    # --------------------------------------------------------

    aerial_neck=dict(

        type="GeneralizedLSSFPN",

        in_channels=[
            192,
            384,
            768,
        ],

        out_channels=embed_dims,

        start_level=0,

        num_outs=1,

        norm_cfg=dict(
            type="BN2d",
            requires_grad=True,
        ),

        act_cfg=dict(
            type="ReLU",
            inplace=True,
        ),

        upsample_cfg=dict(
            mode="bilinear",
            align_corners=False,
        ),
    ),


    # --------------------------------------------------------
    # MapTR head
    # --------------------------------------------------------

    pts_bbox_head=dict(

        type="MapTRHead",

        bev_h=bev_h,

        bev_w=bev_w,

        num_query=900,

        num_vec=50,

        num_pts_per_vec=fixed_ptsnum_per_line,

        num_pts_per_gt_vec=fixed_ptsnum_per_line,

        dir_interval=1,

        query_embed_type="instance_pts",

        transform_method="minmax",

        gt_shift_pts_pattern="v2",

        num_classes=num_map_classes,

        in_channels=embed_dims,

        sync_cls_avg_factor=True,

        with_box_refine=True,

        as_two_stage=False,

        code_size=2,

        code_weights=[
            1.0,
            1.0,
            1.0,
            1.0,
        ],

        transformer=dict(

            type="MapTRPerceptionTransformer",

            # IMPORTANT:
            # We do not use BEVFormer for aerial imagery.
            encoder=None,

            decoder=dict(

                type="MapTRDecoder",

                num_layers=6,

                return_intermediate=True,

                transformerlayers=dict(

                    type="DetrTransformerDecoderLayer",

                    attn_cfgs=[

                        dict(
                            type="MultiheadAttention",

                            embed_dims=embed_dims,

                            num_heads=8,

                            dropout=0.1,
                        ),

                        dict(
                            type="CustomMSDeformableAttention",

                            embed_dims=embed_dims,

                            num_levels=1,
                        ),
                    ],

                    feedforward_channels=ffn_dim,

                    ffn_dropout=0.1,

                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),

            embed_dims=embed_dims,

            rotate_prev_bev=False,

            use_shift=False,

            use_can_bus=False,

            num_feature_levels=1,

            num_cams=1,
        ),


        bbox_coder=dict(

            type="MapTRNMSFreeCoder",

            post_center_range= [
                -46.6534, -46.6534, -2.0,
                46.6534,  46.6534,  2.0,
            ],
            
            
            # [
            #     -15,
            #     -30,
            #     -20,
            #     -35,
            #     15,
            #     30,
            #     20,
            #     35,
            # ],

            pc_range=point_cloud_range,

            max_num=50,

            voxel_size=voxel_size,

            num_classes=num_map_classes,
        ),


        positional_encoding=dict(

            type="LearnedPositionalEncoding",

            num_feats=pos_dim,

            row_num_embed=bev_h,

            col_num_embed=bev_w,
        ),


        loss_cls=dict(

            type="FocalLoss",

            use_sigmoid=True,

            gamma=2.0,

            alpha=0.25,

            loss_weight=2.0,
        ),

        loss_bbox=dict(
            type="L1Loss",
            loss_weight=0.0,
        ),

        loss_iou=dict(
            type="GIoULoss",
            loss_weight=0.0,
        ),

        loss_pts=dict(
            type="PtsL1Loss",
            loss_weight=5.0,
        ),

        loss_dir=dict(
            type="PtsDirCosLoss",
            loss_weight=0.005,
        ),
    ),


    train_cfg=dict(

        pts=dict(

            grid_size=[
                512,
                512,
                1,
            ],

            voxel_size=voxel_size,

            point_cloud_range=
                point_cloud_range,

            out_size_factor=4,

            assigner=dict(

                type="MapTRAssigner",

                cls_cost=dict(
                    type="FocalLossCost",
                    weight=2.0,
                ),

                reg_cost=dict(
                    type="BBoxL1Cost",
                    weight=0.0,
                    box_format="xywh",
                ),

                iou_cost=dict(
                    type="IoUCost",
                    iou_mode="giou",
                    weight=0.0,
                ),

                pts_cost=dict(
                    type="OrderedPtsL1Cost",
                    weight=5,
                ),

                pc_range=point_cloud_range,
            ),
        ),
    ),
)


# ============================================================
# Dataset
# ============================================================

train_pipeline = [
    dict(type="LoadAerialImage", to_float32=True),
    dict(
        type="NormalizeAerialImage",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    dict(type="DefaultFormatBundleAerial"),
    dict(
        type="CollectAerialMap",
        keys=[
            "aerial_img",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadAerialImage", to_float32=True),
    dict(
        type="NormalizeAerialImage",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    dict(type="DefaultFormatBundleAerial"),
    dict(
        type="CollectAerialMap",
        keys=[
            "aerial_img",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
    ),
]

data = dict(
    samples_per_gpu=2, #4,
    workers_per_gpu=4, #4,

    train=dict(
        type="AerialMapDataset",
        pipeline=train_pipeline,
        classes=["divider", "ped_crossing", "boundary"],
        fixed_ptsnum_per_line=fixed_ptsnum_per_line,
        coordinate_mode="world",
        pc_range=point_cloud_range,
        test_mode=False,
    ),

    val=dict(
        type="AerialMapDataset",
        pipeline=test_pipeline,
        classes=["divider", "ped_crossing", "boundary"],
        fixed_ptsnum_per_line=fixed_ptsnum_per_line,
        coordinate_mode="world",
        pc_range=point_cloud_range,
        test_mode=False,
    ),
)

# ============================================================
# Optimization
# ============================================================

optimizer = dict(

    type="AdamW",

    lr=2e-4,

    weight_decay=0.01,
)

optimizer_config = dict(

    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    ),
)


lr_config = dict(

    policy="CosineAnnealing",

    warmup="linear",

    warmup_iters=500,

    warmup_ratio=1.0 / 3,

    min_lr_ratio=1e-3,
)


total_epochs = 10


evaluation = dict(

    interval=2,

    pipeline=None,

    metric="chamfer",
)


runner = dict(

    type="EpochBasedRunner",

    max_epochs=total_epochs,
)


fp16 = dict(
    loss_scale=512.0,
)


checkpoint_config = dict(
    interval=1,
)