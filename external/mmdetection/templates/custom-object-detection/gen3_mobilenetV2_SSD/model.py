_base_ = [
    './coco_data_pipeline.py'
]
width_mult = 1.0
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(4, 5),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        num_classes=80,
        in_channels=(96, 320),
        anchor_generator=dict(
            type='SSDAnchorGeneratorClustered',
            strides=(16, 32),
            reclustering_anchors=True,
            widths=[[
                38.641007923271076, 92.49516032784699, 271.4234764938237,
                141.53469410876247
            ],
                    [
                        206.04136086566515, 386.6542727907841,
                        716.9892752215089, 453.75609561761405,
                        788.4629155558277
                    ]],
            heights=[[
                48.9243877087132, 147.73088476194903, 158.23569788707474,
                324.14510379107367
            ],
                     [
                         587.6216059488938, 381.60024152086544,
                         323.5988913027747, 702.7486097568518,
                         741.4865860938451
                     ]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2)),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=False),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.0,
        use_giou=False,
        use_focal=False,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        min_bbox_size=0,
        score_thr=0.02,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=200))

cudnn_benchmark = True
evaluation = dict(interval=1, metric='mAP', save_best='mAP')
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='mAP',
    patience=3,
    iteration_patience=600,
    interval=1,
    min_lr=0.00001,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3)

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
runner = dict(type='EpochRunnerWithCancel', max_epochs=300)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'output'
load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-992x736.pth'
resume_from = None
workflow = [('train', 1)]
custom_hooks = [
    dict(type='EarlyStoppingHook', patience=5, iteration_patience=1000, metric='mAP', interval=1, priority=75)
]
