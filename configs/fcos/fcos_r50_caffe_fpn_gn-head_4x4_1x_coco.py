# TODO: Remove this config after benchmarking all related configs
_base_ = 'fcos_r50_caffe_fpn_gn-head_1x_coco.py'

data = dict(samples_per_gpu=4, workers_per_gpu=4)

evaluation = dict(interval=1)
checkpoint_config = dict(interval=12)
log_config = dict(interval=100)