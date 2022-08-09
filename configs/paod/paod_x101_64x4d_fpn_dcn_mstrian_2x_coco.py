_base_ = './paod_x101_64x4d_fpn_mstrain_2x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

work_dir = './work_dirs/paod_x101_64x4d_fpn_dcn_mstrain_2x_coco.py'