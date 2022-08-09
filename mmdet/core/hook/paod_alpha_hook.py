# Copyright (c) OpenMMLab. All rights reserved.
from math import exp, sqrt
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class PAODAlphaHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        max_epochs = runner.max_epochs
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        
        alpha = model.bbox_head.alpha
        model.bbox_head._alpha = (1 - alpha) * exp(
            -epoch / (1 * sqrt(max_epochs))) + alpha
