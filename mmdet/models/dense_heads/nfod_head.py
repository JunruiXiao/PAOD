import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.ops import modulated_deform_conv2d
from mmcv.runner import force_fp32

from mmdet.core import (bbox_limited, bbox_overlaps, distance2bbox,
                        multi_apply, multiclass_nms, reduce_mean)
from ..builder import HEADS
from .anchor_free_head import AnchorFreeHead

EPS = 1e-12


class TaskEnhancement(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg,
                 norm_cfg,
                 reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.la_conv1 = ConvModule(in_channels, in_channels // reduction_ratio,
                                   1)
        self.la_conv2 = ConvModule(
            in_channels // reduction_ratio,
            in_channels,
            1,
            act_cfg=dict(type='Sigmoid'))
        self.reduction_conv = ConvModule(
            in_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

    def init_weights(self):
        normal_init(self.la_conv1.conv, std=0.01)
        normal_init(self.la_conv2.conv, std=0.01)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, x):
        b, _, h, w = x.size()
        avg_feat = F.adaptive_avg_pool2d(x, (1, 1))
        scale_factors = self.la_conv2(self.la_conv1(avg_feat))
        conv_weight = scale_factors.view(
            b, 1, -1) * self.reduction_conv.conv.weight.view(
                1, self.out_channels, self.in_channels)
        x = x.view(b, -1, h * w)
        x = torch.bmm(conv_weight, x).view(b, -1, h, w)
        if self.reduction_conv.with_bias:
            conv_bias = self.reduction_conv.conv.bias.view(1, -1, 1, 1)
            x = x + conv_bias
        if self.reduction_conv.norm is not None:
            x = self.reduction_conv.norm(x)
        x = self.reduction_conv.activate(x)
        return x


@HEADS.register_module()
class NFODHead(AnchorFreeHead):
    """TOOD: Task-aligned One-stage Object Detection.

    TOOD uses Task-aligned head (T-head) and is optimized by Task Alignment
    Learning (TAL).

    todo: list link of the paper.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_specific_convs=2,
                 offset_kernel_size=3,
                 alpha=0.5,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_cls=dict(
                     type='FocalLossWithProb',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                 **kwargs):
        self.num_shared_convs = 8 - 2 * num_specific_convs
        self.num_specific_convs = num_specific_convs
        self.offset_kernel_size = offset_kernel_size
        self.alpha = alpha
        self.epoch = 0  # which would be update in head hook!
        super(NFODHead, self).__init__(
            num_classes,
            in_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.shared_convs = nn.ModuleList()
        for i in range(self.num_shared_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.shared_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in range(self.num_specific_convs):
            self.cls_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.cls_enhance = TaskEnhancement(
            self.feat_channels * 2,
            self.feat_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.reg_enhance = TaskEnhancement(
            self.feat_channels * 2,
            self.feat_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.nfod_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.nfod_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        self.fusion_conv = ConvModule(
            self.feat_channels * 2,
            self.feat_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.conv_offset = nn.Conv2d(
            self.feat_channels, 3 * self.offset_kernel_size**2, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for shared_conv in self.shared_convs:
            normal_init(shared_conv.conv, std=0.01)

        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            normal_init(cls_conv.conv, std=0.01)
            normal_init(reg_conv.conv, std=0.01)

        self.cls_enhance.init_weights()
        self.reg_enhance.init_weights()

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.nfod_cls, std=0.01, bias=bias_cls)
        normal_init(self.nfod_reg, std=0.01)

        normal_init(self.fusion_conv.conv, std=0.01)
        constant_init(self.conv_offset, val=0)

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
        featmap_sizes = [featmap.size()[-2:] for featmap in feats]
        mlvl_points = self.get_points(featmap_sizes, feats[0].dtype,
                                      feats[0].device)

        return multi_apply(self.forward_single, feats, self.scales,
                           mlvl_points, self.strides)

    def forward_single(self, x, scale, points, stride):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (tuple[Tensor]): Stride of the current scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        b, _, h, w = x.size()

        for shared_conv in self.shared_convs:
            x = shared_conv(x)

        cls_feat = x
        reg_feat = x
        for cls_conv, reg_conv in zip(self.cls_convs, self.reg_convs):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)
        cls_feat = self.cls_enhance(torch.cat([x, cls_feat], dim=1))
        reg_feat = self.reg_enhance(torch.cat([x, reg_feat], dim=1))

        cls_score = self.nfod_cls(cls_feat).sigmoid()
        bbox_pred = self.nfod_reg(reg_feat)
        bbox_pred = F.relu(scale(bbox_pred).float()) * stride
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(b, -1, 4)
        bbox_pred = distance2bbox(points,
                                  bbox_pred).reshape(b, h, w, 4).permute(
                                      0, 3, 1, 2).contiguous()

        fusion_feat = self.fusion_conv(torch.cat([cls_feat, reg_feat], dim=1))
        o1, o2, mask = torch.chunk(self.conv_offset(fusion_feat), 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = F.softmax(mask, dim=1)
        cls_score = self.masked_deform_sampling(cls_score, offset, mask)
        bbox_pred = self.masked_deform_sampling(bbox_pred, offset, mask)

        return cls_score, bbox_pred

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        y, x = super()._get_points_single(
            featmap_size, stride, dtype, device, flatten=flatten)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def masked_deform_sampling(self, feat, offset, mask):
        c = feat.size(1)
        weight = feat.new_ones(c, 1, self.offset_kernel_size,
                               self.offset_kernel_size)
        bias = feat.new_zeros(c)
        return modulated_deform_conv2d(feat, offset, mask, weight, bias, 1,
                                       self.offset_kernel_size // 2, 1, c, 1)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
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
        assert len(cls_scores) == len(bbox_preds)
        labels, bbox_targets = self.get_targets(cls_scores, bbox_preds,
                                                gt_bboxes, gt_labels)
        # flatten cls_scores, bbox_preds
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=0)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=0)
        flatten_labels = torch.cat(labels, dim=0)
        flatten_bbox_targets = torch.cat(bbox_targets, dim=0)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]

        if len(pos_inds) > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            loss_bbox = self.loss_bbox(
                pos_bbox_preds, pos_bbox_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels)

            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            # Always keep topk op for dynamic input in onnx
            if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
                                       or scores.shape[-2] > nms_pre_tensor):
                from torch import _shape_as_tensor

                # keep shape as tensor and get k
                num_anchor = _shape_as_tensor(scores)[-2].to(device)
                nms_pre = torch.where(nms_pre_tensor < num_anchor,
                                      nms_pre_tensor, num_anchor)

                max_scores, _ = scores.max(dim=-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

            bboxes = bbox_limited(bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

        # Set max number of box to be feed into nms in deployment
        deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
        if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
            batch_mlvl_scores, _ = (
                batch_mlvl_scores *
                batch_mlvl_centerness.unsqueeze(2).expand_as(batch_mlvl_scores)
            ).max(-1)
            _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
            batch_inds = torch.arange(batch_mlvl_scores.shape[0]).view(
                -1, 1).expand_as(topk_inds)
            batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
            batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]
            batch_mlvl_centerness = batch_mlvl_centerness[batch_inds,
                                                          topk_inds]

        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def get_targets(self, cls_scores, bbox_preds, gt_bboxes, gt_labels):
        """Compute regression and classification for points in multiple images.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        b = len(gt_bboxes)
        num_levels = len(cls_scores)
        cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(b, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(b, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # concat all levels cls_score, bbox_pred and points
        mlvl_cls_scores = torch.cat(cls_scores, dim=1)
        mlvl_bbox_preds = torch.cat(bbox_preds, dim=1)

        # the number of points per img, per lvl
        num_points_per_lvl = [
            cls_score.size(1) for cls_score in cls_scores
        ]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(self._get_target_single,
                                                     mlvl_cls_scores,
                                                     mlvl_bbox_preds,
                                                     gt_bboxes, gt_labels)

        # split to per img, per level
        labels_list = [
            labels.split(num_points_per_lvl, 0) for labels in labels_list
        ]
        bbox_targets_list = [
            bbox_targets.split(num_points_per_lvl, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list], dim=0))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list], dim=0)
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, cls_score, bbox_pred, gt_bboxes, gt_labels):
        """Compute regression, classification targets for a single image.

        Args:
            cls_score (Tensor): Multi-level cls scores of the image, which are
                concatenated into a single tensor of shape (num_points,)
            bbox_pred (Tensor): Multi level bbox preds of the image,
                which are concatenated into a single tensor of
                    shape (num_points, 4).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).

        Returns:
            tuple:
                labels (Tensor): Labels of all points in the image with shape
                    (num_points,).
                bbox_targets (Tensor): BBox targets of all points in the
                    image with shape (num_points, 4).
        """
        num_gts, num_bboxes = gt_bboxes.size(0), cls_score.size(0)

        # default assignment
        labels = gt_labels.new_full((num_bboxes, ), self.num_classes)
        bbox_targets = gt_bboxes.new_zeros((num_bboxes, 4))

        if num_gts == 0:
            return labels, bbox_targets

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        _, sorted_idxs = torch.sort(areas)
        gt_labels = gt_labels[sorted_idxs]
        gt_bboxes = gt_bboxes[sorted_idxs, :]

        # compute assign metric between all bbox and gt
        # (num_bboxes, num_gts)
        overlaps = bbox_overlaps(bbox_pred, gt_bboxes).detach()
        bbox_scores = cls_score[:, gt_labels].detach()
        assign_metrics = self.alpha * bbox_scores + (1 - self.alpha) * overlaps
        # select the top-1 bbox for each gt
        # (num_gts->topk, num_gts)
        _, candidate_idxs = assign_metrics.topk(num_gts, dim=0, largest=True)
        valid_flag = gt_labels.new_full((num_bboxes, ), True, dtype=torch.bool)
        selected_idxs = []
        for idxs in candidate_idxs.t():
            for idx in idxs:
                idx = idx.item()
                if valid_flag[idx]:
                    selected_idxs.append(idx)
                    valid_flag[idx] = False
                    break
        assert len(selected_idxs) == num_gts
        labels[selected_idxs] = gt_labels
        bbox_targets[selected_idxs, :] = gt_bboxes

        return labels, bbox_targets
