model = dict(
    type='EProPnPDet',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='DeformPnPHead',
        num_classes=10,
        in_channels=256,
        strides=(8, 16, 32, 64, 128),
        output_stride=8,
        dense_lvl_range=(0, 3),
        det_lvl_range=(0, 5),
        dense_channels=256,
        embed_dims=256,
        num_heads=8,
        num_points=16,
        detector=dict(
            type='FCOSEmbHead',
            feat_channels=256,
            stacked_convs=2,
            emb_channels=256,
            strides=[8, 16, 32, 64, 128],
            regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384),
                            (384, 1e8)),
            bbox_branch=(256, ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=0.5),
            loss_rp=dict(  # reference point loss
                type='SmoothL1LossMod', beta=1.0, loss_weight=1.0),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            offset_cls_agnostic=False),
        attention_sampler=dict(
            type='DeformableAttentionSampler',
            embed_dims=256,
            num_heads=8,
            num_points=16,
            stride=8),
        center_target=dict(
            type='VolumeCenter',
            output_stride=8,
            render_stride=4,
            min_box_size=4.0),
        dim_coder=dict(
            type='MultiClassLogDimCoder',
            target_means=[
                (4.62, 1.73, 1.96),
                (6.94, 2.84, 2.52),
                (12.56, 3.89, 2.94),
                (11.22, 3.50, 2.95),
                (6.68, 3.21, 2.85),
                (1.70, 1.29, 0.61),
                (2.11, 1.46, 0.78),
                (0.73, 1.77, 0.67),
                (0.41, 1.08, 0.41),
                (0.50, 0.99, 2.52)],
            target_stds=[
                (0.46, 0.24, 0.16),
                (2.11, 0.84, 0.45),
                (4.50, 0.77, 0.54),
                (2.06, 0.49, 0.33),
                (3.23, 0.93, 1.07),
                (0.26, 0.35, 0.16),
                (0.33, 0.29, 0.17),
                (0.19, 0.19, 0.14),
                (0.14, 0.27, 0.13),
                (0.17, 0.15, 0.62)]),
        positional_encoding=dict(
            type='SinePositionalEncodingMod',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        pnp=dict(
            type='EProPnP4DoF',
            mc_samples=128,
            num_iter=4,
            normalize=True,
            solver=dict(
                type='LMSolver',
                num_iter=10,
                normalize=True,
                init_solver=dict(
                    type='RSLMSolver',
                    num_points=16,
                    num_proposals=64,
                    num_iter=3))),
        camera=dict(type='PerspectiveCamera'),
        cost_fun=dict(
            type='AdaptiveHuberPnPCost',
            relative_delta=0.5),
        loss_pose=dict(  # main pose loss
            type='MonteCarloPoseLoss',
            loss_weight=0.5,
            momentum=0.01),
        loss_dim=dict(  # dimension (size) loss
            type='SmoothL1LossMod',
            loss_weight=1.0),
        loss_proj=dict(  # auxiliary reprojection-based loss
            type='MVDGaussianMixtureNLLLoss',
            adaptive_weight=True,
            delta=2.0,
            loss_weight=0.5),
        loss_regr=dict(
            type='MVDGaussianMixtureNLLLoss',
            adaptive_weight=True,
            sigma=0.2,
            freeze_sigma=False,
            delta=2.0,
            loss_weight=2.5),
        loss_depth=dict(
            type='MVDGaussianMixtureNLLLoss',
            adaptive_weight=True,
            sigma=0.3,
            freeze_sigma=False,
            delta=0.3,
            loss_weight=1.5),
        loss_score=dict(  # 3D score loss
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_reg_pos=dict(  # derivative regularization loss for position
            type='SmoothL1LossMod',
            beta=1.0,
            loss_weight=0.05),
        loss_reg_orient=dict(  # derivative regularization loss for orientation
            type='CosineAngleLoss',
            loss_weight=0.05),
        loss_velo=dict(
            type='SmoothL1LossMod',
            loss_weight=0.05),
        loss_attr=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.5),
        use_cls_emb=True,
        dim_cls_agnostic=False,
        pred_velo=True,
        pred_attr=True,
        num_attrs=9,
        score_type='te'),
    train_cfg=dict(
        num_obj_samples_per_img=48,
        roi_shape=(14, 14)),  # RoI shape for the reprojection-based auxiliary loss
    test_cfg=dict(
        override_cfg={'pnp.solver.num_iter': 5,
                      'pnp.mc_samples': 512,  # for smooth visualization
                      'pnp.iter_samples': 128},  # for smooth visualization
        mc_scoring_ratio=0.0,  # 1.0 for Monte Carlo scoring
        nms_iou2d=dict(type='nms', iou_threshold=0.8),
        nms_ioubev_thr=0.25))
dataset_type = 'Intersection'
data_root = 'data/nuscenes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3DInt',
         with_bbox_3d=True,
         with_bbox_2d=True,
         with_labels=True,
         with_center=True,
         with_K = True,
         with_pts3d=True,
         with_transform=True,
         with_img_dense_x2d=True),
    dict(type='Crop3DInt', crop_box=(0, 228, 1600, 900), trunc_ignore_thres=0.8),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad3DInt', size_divisor=32),
    dict(type='DefaultFormatBundle3DInt'),
    dict(type='Collect',
         keys=['img',
               'gt_bboxes_3d',
               'gt_bboxes_2d',
               'gt_center_2d',
               'gt_labels',
               'cam_intrinsic',
               'gt_x3d',
               'gt_x2d',
               'img_transform',
               'img_dense_x2d',
               'img_dense_x2d_mask'],
        meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape','img_norm_cfg')),
]
test_pipeline = [
    dict(type='LoadImageFromFile3D',
         with_img_dense_x2d=False),
    dict(type='MultiScaleFlipAug',
         scale_factor=1.0,
         flip=False,
         transforms=[
             dict(type='RandomFlip3D', flip_ratio=0.5),
             dict(type='Crop3D', crop_box=(0, 228, 1600, 900)),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad3D', size_divisor=32),
             dict(type='DefaultFormatBundle3D'),
             dict(type='Collect', keys=[
                 'img', 'cam_intrinsic', 'img_transform']),
         ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file='train.json',
        pipeline=train_pipeline,
        data_root=data_root,
        img_prefix='/simplstor/ypatel/workspace/single-image-pose/external/EPro-PnP-v2/EPro-PnP-Det_v2/data/int_2',
        filter_empty_gt=True),
    val=dict(
        type='NuScenes3DDataset',
        samples_per_gpu=4,
        ann_file='nuscenes_annotations_val.pkl',
        pipeline=test_pipeline,
        data_root='data/nuscenes/',
        filter_empty_gt=False))
evaluation = dict(
    interval=1,
    metric='NDS')
# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'sampling_offsets': dict(lr_mult=0.1)
        }))
optimizer_config = dict(
    type='OptimizerHookMod',
    grad_clip=dict(max_norm=5.0, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=7)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/simplstor/ypatel/workspace/single-image-pose/external/EPro-PnP-v2/checkpoints/epropnp_det_v2.pth'
resume_from = None
workflow = [('train',1)]
custom_hooks = [dict(type='EmptyCacheHook')]
find_unused_parameters = True

