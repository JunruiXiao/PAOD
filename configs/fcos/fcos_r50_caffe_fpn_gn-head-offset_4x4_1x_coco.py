# TODO: Remove this config after benchmarking all related configs
_base_ = 'fcos_r50_caffe_fpn_gn-head_1x_coco.py'

model = dict(bbox_head=dict(
        type='FCOSOHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

data = dict(samples_per_gpu=4, workers_per_gpu=4)

evaluation = dict(interval=1)
checkpoint_config = dict(interval=12)
log_config = dict(interval=100)

work_dir = './work_dirs/fcos_r50_caffe_fpn_gn-head-offset_4x4_1x_coco/'