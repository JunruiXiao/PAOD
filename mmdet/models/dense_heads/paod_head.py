# Copyright (c) OpenMMLab. All rights reserved.
from cv2 import norm
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32
from mmcv.cnn import (ConvModule, Scale, bias_init_with_prob, constant_init,
                      normal_init)
from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from mmcv.ops import ModulatedDeformConv2d
from mmdet.core.bbox import bbox_overlaps

class FeatureEnhancement(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_shared_convs,
                 conv_cfg,
                 norm_cfg,
                 ratio=16):
        super().__init__()
        self.in_channels = in_channels # n * channel
        self.out_channels = out_channels # channel
        self.num_shared_convs = num_shared_convs
        self.layer_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(self.in_channels, self.in_channels // ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // ratio, num_shared_convs, 1),
            nn.Sigmoid())
        self.reduction_conv = ConvModule(
            in_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

    def init_weights(self):
        for m in self.layer_weights.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat):
        b, _, h, w = feat.size()
        weight = self.layer_weights(feat) # 1,ic,1,1
        conv_weight = weight.reshape(
            b, 1, self.num_shared_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.out_channels, self.num_shared_convs, self.out_channels) # 
        conv_weight = conv_weight.reshape(b, self.out_channels,
                                          self.in_channels)
        
        feat = feat.reshape(b, self.in_channels, h * w) # 1,256,HW
        feat = torch.bmm(conv_weight, feat).reshape(b, self.out_channels, h, w)
        if self.reduction_conv.with_bias:
            conv_bias = self.reduction_conv.conv.bias.view(1, -1, 1, 1)
            feat = feat + conv_bias
        if self.reduction_conv.norm is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)
        return feat


@HEADS.register_module()
class PAODHead(AnchorHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 shared_convs=4,
                 num_point=9,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_decoded_bbox=False,
                 loss_iou=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='atss_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 eps=1e-7,
                 **kwargs):
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_shared_convs = shared_convs
        self.num_specific_convs =  (8 - self.num_shared_convs) // 2
        self.num_point = num_point
        self.eps = eps
        
        super(PAODHead, self).__init__(
            num_classes,
            in_channels,
            reg_decoded_bbox=reg_decoded_bbox,
            init_cfg=init_cfg,
            **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            self.alpha = self.train_cfg.alpha
            self._alpha = self.alpha
        self.loss_iou = build_loss(loss_iou)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.shared_convs = nn.ModuleList()
        for _ in range(self.num_shared_convs):
            self.shared_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.fusion_conv = ConvModule(
            self.feat_channels * 2,
            self.feat_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.offset = nn.Conv2d(
            self.feat_channels, self.num_point * 6, 3, padding=1)

        self.cls_enhance = FeatureEnhancement(
            self.num_shared_convs * self.feat_channels,
            self.feat_channels,
            self.num_shared_convs,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.reg_enhance = FeatureEnhancement(
            self.num_shared_convs * self.feat_channels,
            self.feat_channels,
            self.num_shared_convs,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.num_specific_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))


        self.cls_offset_dcn = ModulatedDeformConv2d(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            deform_groups=1,
            bias=False)

        self.reg_offset_dcn = ModulatedDeformConv2d(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            deform_groups=1,
            bias=False)


        self.atss_reg_conv = ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)
        self.atss_cls_conv = ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)

        self.atss_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.atss_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        self.atss_iou = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def init_weights(self):

        """Initialize weights of the head."""
        for shared_conv in self.shared_convs:
            normal_init(shared_conv.conv, std=0.01)
        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            normal_init(cls_conv.conv, std=0.01)
            normal_init(reg_conv.conv, std=0.01)

        normal_init(self.fusion_conv.conv, std=0.01)
        constant_init(self.offset, val=0.0, bias=0.0)


        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)
        constant_init(self.atss_reg, val=0.0, bias=4.0)
        normal_init(self.atss_iou, std=0.01)
        normal_init(self.atss_cls_conv, std=0.01)
        normal_init(self.atss_reg_conv, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        shared_feats = []
        for shared_conv in self.shared_convs:
            x = shared_conv(x)
            shared_feats.append(x)
    
        shared_feat = torch.cat(shared_feats, dim=1)
        cls_feat = self.cls_enhance(shared_feat)
        reg_feat = self.reg_enhance(shared_feat)


        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)


        fusion_feat = self.fusion_conv(
                torch.cat([cls_feat, reg_feat], dim=1))
        offsetmap = self.offset(fusion_feat)
        cls_offset_map, reg_offset_map = offsetmap.split(3 * self.num_point, dim=1)
        cls_offset, cls_mask = cls_offset_map.split(2*self.num_point, dim=1)
        reg_offset, reg_mask = reg_offset_map.split(2*self.num_point, dim=1)

        cls_feat = self.atss_cls_conv(cls_feat)
        reg_feat = self.atss_reg_conv(reg_feat)

        cls_feat = self.relu(self.cls_offset_dcn(cls_feat, cls_offset.contiguous(), cls_mask.sigmoid()))
        reg_feat = self.relu(self.reg_offset_dcn(reg_feat, reg_offset.contiguous(), reg_mask.sigmoid()))

        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float() #[B, 4, h, w]
        cls_score = self.atss_cls(cls_feat) # [B, 80, h, w]
        iou_pred = self.atss_iou(reg_feat)

        return cls_score, bbox_pred, iou_pred

    def loss_single(self, anchors, cls_score, bbox_pred, iou_pred, labels,
                    label_weights, bbox_targets, norm_metrics_list, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        iou_preds = iou_pred.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        norm_metrics = norm_metrics_list.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_iou_pred = iou_preds[pos_inds]

            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)

            # regression loss
            pos_bbox_weight = norm_metrics[pos_inds]
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)

            pos_iou_targets= bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)

            # centerness loss
            loss_iou = self.loss_iou(
                pos_iou_pred,
                pos_iou_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_iou = iou_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)
            #centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_iou, pos_bbox_weight.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            cls_scores,
            bbox_preds,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         norm_metrics_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_iou,\
            bbox_avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                iou_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                norm_metrics_list,
                num_total_samples=num_total_samples)

        
        bbox_avg_factor = reduce_mean(sum(bbox_avg_factor)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_iou=losses_iou)


    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        
        # comput predicted bbox location
        num_levels = len(cls_scores)
        cls_score_list = []
        bbox_pred_list = []
        for i in range(num_imgs):
            tmp_cls_list = []; tmp_bbox_list = []
            for j in range(num_levels):
                cls_score = cls_scores[j][i].permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                bbox_pred = bbox_preds[j][i].permute(1, 2, 0).reshape(-1, 4)
                tmp_cls_list.append(cls_score); tmp_bbox_list.append(bbox_pred)
            cat_cls_score = torch.cat(tmp_cls_list, dim=0); cat_bbox_pred = torch.cat(tmp_bbox_list, dim=0)
            cls_score_list.append(cat_cls_score); bbox_pred_list.append(cat_bbox_pred)


        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_norm_metrics, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             cls_score_list,
             bbox_pred_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        norm_metrics_list = images_to_levels(all_norm_metrics,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, norm_metrics_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           cls_scores,
                           bbox_preds,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        bbox_preds_valid = bbox_preds[inside_flags, :]
        cls_scores_valid = cls_scores[inside_flags, :]

        bbox_preds_valid = self.bbox_coder.decode(anchors, bbox_preds_valid)
        
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             cls_scores_valid, bbox_preds_valid,
                                             self._alpha, gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)
        assigned_metrics = assign_result.assigned_metrics

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_metrics = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            pos_metrics = assigned_metrics[gt_class_inds]
            """pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()"""
            norm_metrics[gt_class_inds] = pos_metrics  

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            norm_metrics = unmap(norm_metrics,
                                          num_total_anchors, inside_flags)
        
        #print(norm_metrics.sum())
        return (anchors, labels, label_weights, bbox_targets, norm_metrics,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
