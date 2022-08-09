_base_ = [
    './paod_r50_fpn_1x_voc.py'
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
work_dir = './work_dirs/paod_r101_fpn_1x_voc'