import copy
import math
from typing import Callable
import torch
import numpy as np
import torch.nn.functional as F
from sympy.polys.polyroots import preprocess_roots
from torch import nn
from core.utils import box_ops
from core.utils.misc import (accuracy, get_world_size,
                       is_dist_avail_and_initialized)
from core.dataset.misc import nested_tensor_from_tensor_list, NestedTensor
from core.model.build import build_spatio_backbone, build_temporal_backbone
from core.model.matcher import build_matcher
from core.model.transformer import build_transformer
from core.model import head_helper
from torchvision.ops import generalized_box_iou
from core.model.position_encoding import build_position_encoding
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import generalized_box_iou
from core.dataset.cv2_transform import ava_scale_boxes, detrbox2oribox, adjust_boxes_for_crop, clip_boxes_to_image_tensor
_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "slow_c2d": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "slow_i3d": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}

class LWDETR(nn.Module):
    def __init__(self,
                 cfg,
                 spatio_backbone,
                 temporal_backbone,
                 transformer,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 group_detr=1,
                 two_stage=False,
                 lite_refpoint_refine=False,
                 bbox_reparam=False):
        super().__init__()
        self.num_queries = num_queries  #300
        self.transformer = transformer
        hidden_dim = transformer.d_model    #256
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        query_dim=4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)
        self.two_stage = True
        self.backbone = spatio_backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr
        self.train_crop_size = cfg.DATA.TRAIN_CROP_SIZE
        self.test_crop_size = cfg.DATA.TEST_CROP_SIZE
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.channel_3 = 48
        self.channel_4 = 96
        self.channel_5 = 192
        self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
        self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
        self.threshold = cfg.AVA.DETECTION_SCORE_THRESH
        # self.st_fuse2 = nn.ConvTranspose2d(in_channels=192, out_channels=256, kernel_size=4, stride=4, padding=4, output_padding=0)
        # self.st_fuse3 = nn.ConvTranspose2d(in_channels=192, out_channels=self.channel_3, kernel_size=6, stride=1, padding=0, output_padding=0)
        only_conv = False
        if only_conv:
            # self.st_fuse3 = nn.Conv2d(192, self.channel_3, kernel_size=1, stride=1, padding=0)
            self.st_fuse3 = nn.Linear(192,192)
            # self.st_fuse4_0 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3,stride=1,padding=0)
            # self.st_fuse4_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5,stride=1, padding=0)
            # self.st_fuse4_2 = nn.Conv2d(in_channels=192, out_channels=self.channel_4, kernel_size=4, stride=1, padding=0)
            self.st_fuse4_0 = nn.MaxPool2d(kernel_size=2, stride=2)
            # 池化后尺寸为16×16
            self.st_fuse4_1  = nn.Conv2d(in_channels=192, out_channels=self.channel_4,
                             kernel_size=3, stride=1, padding=3)
            # self.st_fuse5_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=3, padding=3)
            # self.st_fuse5_2 = nn.Conv2d(in_channels=192, out_channels=self.channel_5, kernel_size=1, stride=1, padding=0)  # 32.32 → 12x12
            self.st_fuse5_1 = nn.Conv2d(
                in_channels=192,
                out_channels=self.channel_5,
                kernel_size=5,
                stride=3,
                padding=0,
                bias=True
            )
            # self.st_fuse4 = nn.Conv2d(192, self.channel_4, kernel_size=3, stride=2, padding=1)
            # self.st_fuse5 = nn.Conv2d(192, self.channel_5, kernel_size=4, stride=2, padding=1)
            # self.groupnorm2 = nn.GroupNorm(num_groups=1, num_channels=256)
            self.groupnorm3 = nn.GroupNorm(num_groups=1, num_channels=self.channel_3)
            self.groupnorm4 = nn.GroupNorm(num_groups=1, num_channels=self.channel_4)
            self.groupnorm5 = nn.GroupNorm(num_groups=1, num_channels=self.channel_5)
            # self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(self.channel_3)
            self.bn4 = nn.BatchNorm2d(self.channel_4)
            self.bn5 = nn.BatchNorm2d(self.channel_5)
            self.activate = F.leaky_relu
        self.matcher = build_test_matcher(cfg)
        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])

        self._export = False
        self.temporal_backbone = temporal_backbone

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, action_samples, samples: NestedTensor, bboxes=None, orig_target_sizes=None, postprocess=None ,targets=None, batch_length=None, ori_boxes=None, mode=None):
        # print(samples[0].shape)
        # print(samples[1].shape)
        return_attn = False
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # print(samples[0].mask)
        #----- extract spatio and temporal features
        features, poss, encoder_out_feat= self.backbone(samples)# 1，256，16，16
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            spatio_feat = src
            assert mask is not None
        #--------- spatio temporal features fusion  -------------------
        i = 0
        # for enc_out in encoder_out_feat:
        #     encoder_out_feat[i] = self.adaptive_pool(enc_out)
        #     i += 1
        downsampling_type = " "
        #use downsampling and conv to align feature map
        if downsampling_type == 'use_downsampling_conv':
            s3_fuse = F.interpolate(encoder_out_feat[3], size=(45, 45), mode='bilinear', align_corners=False)
            s3_fuse = self.activate(self.bn3(self.st_fuse3(s3_fuse)))
        elif downsampling_type == "conv_only":
            s2_fuse = encoder_out_feat[3]
            s3_fuse = F.interpolate(encoder_out_feat[3], size=(32, 32), mode='bilinear', align_corners=False)
            s3_fuse = self.st_fuse3(s3_fuse)
            s3_fuse = self.bn3(s3_fuse)
            s3_fuse = self.activate(s3_fuse)
            s4_fuse = self.st_fuse4_0( F.interpolate(encoder_out_feat[3], size=(32, 32), mode='bilinear', align_corners=False))
            s4_fuse = self.st_fuse4_1(s4_fuse)
            # s4_fuse = self.st_fuse4_2(s4_fuse)
            s4_fuse = self.bn4(s4_fuse)
            s4_fuse = self.activate(s4_fuse)
            s5_fuse = self.st_fuse5_1(F.interpolate(encoder_out_feat[3], size=(32, 32), mode='bilinear', align_corners=False))
            s5_fuse = self.bn5(s5_fuse)
            s5_fuse = self.activate(s5_fuse)
        else:   #claculate in temporal backbone
            s2_fuse = encoder_out_feat[3]
            s3_fuse = encoder_out_feat[3]
            s4_fuse = encoder_out_feat[3]
            s5_fuse = encoder_out_feat[3]
        #only use last temporal layer's features
        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight#300*13,4
            query_feat_weight = self.query_feat.weight#300*13,256
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]
        #defornable detr for spatio decoder
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)
        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)   #3，2，3900，4
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat(
                [outputs_coord_cxcy, outputs_coord_wh], dim=-1
            )
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
        # roi_coord = F.pad(outputs_coord, pad=(1, 0), mode='constant', value=0)
        outputs_class = self.class_embed(hs)#3，2，3900，91
        # if mode =="demo":
        #     det_preds = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        #     boxes = postprocess['bbox'](det_preds, orig_target_sizes, type="ava")  # get pred bbox and labels xywh->xyxy
        # else:
        matched_boxes = self.match_ava_boxes(outputs_class, outputs_coord, postprocess, orig_target_sizes, targets, batch_length,
                        bboxes, ori_boxes, mode)     #get matched pred boxes with gt boxes
        if matched_boxes == None:
            return None
        if return_attn:
            temporal_feat, attn_dict = self.temporal_backbone(action_samples, bboxes, matched_boxes, s2_fuse, s3_fuse, s4_fuse, s5_fuse)
        else:
            temporal_feat = self.temporal_backbone(action_samples, bboxes, matched_boxes, s2_fuse, s3_fuse, s4_fuse, s5_fuse)
        fuse_feat = temporal_feat

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': matched_boxes, 'pred_actions': fuse_feat}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, fuse_feat)

        if self.two_stage:
            hs_enc_list = hs_enc.split(self.num_queries, dim=1) #13x2,300,96
            cls_enc = []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])#2,300,91
                cls_enc.append(cls_enc_gidx)
            cls_enc = torch.cat(cls_enc, dim=1)#2,3900,91
            out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc, 'pred_actions': fuse_feat}
        if return_attn:
            return out, attn_dict
        else:
            return out

    def match_ava_boxes(self, outputs_class, outputs_coord, postprocess, orig_target_sizes, targets, batch_length, bboxes, ori_boxes, mode):
        # model will choose 100 bbox，each gt box choose 1 pred box as roi input
        det_preds = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        boxes = postprocess['bbox'](det_preds, orig_target_sizes, type="ava")  # get pred bbox and labels xywh->xyxy
        if boxes == None:
            return None
        unique_box = False
        if unique_box:  # check if there are multiple boxes
            for t in targets:
                if len(t['boxes']) > 1:
                    t['boxes'] = torch.unique(t['boxes'], dim=0)
        batch_length = batch_length
        coco_original_size = (1920, 1080)
        coco_resized_size = (640, 640)
        #for demo, we did not use matcher
        if mode == "demo":
            if boxes is not None:
                boxes = boxes[0]
                scores = boxes['scores']
                labels = boxes['labels']
                boxes = boxes['ori_boxes']
                score_mask = (scores > self.threshold)
                category_mask = (labels == 1)
                valid_mask = category_mask & score_mask
                boxes = boxes[valid_mask]
                scores = scores[valid_mask]
                index_pad = torch.full(
                    size=(boxes.shape[0], 1),
                    fill_value=float(0),
                    device=boxes.device,
                )
                boxes = torch.cat([index_pad, boxes], axis=1)
            boxes = detrbox2oribox(boxes, coco_original_size, coco_resized_size)
            boxes = clip_boxes_to_image_tensor(boxes, 1080, 1920)
            boxes, width, heigth = ava_scale_boxes(self.test_crop_size, boxes, 1080, 1920)
            boxes = clip_boxes_to_image_tensor(boxes, heigth, width)
            return boxes
        # match boxes
        indices = self.matcher(boxes, ori_boxes, batch_length)
        boxes = self.reorder_bboxes_by_gt(boxes, bboxes, indices, batch_length)
        boxes = self.stack_sorted_bboxes_with_batch_id(boxes)
        # boxes = self.process_and_stack_batches(boxes, idx)
        # 0～1 -> 1920,1080
        boxes = detrbox2oribox(boxes, coco_original_size, coco_resized_size)
        boxes = clip_boxes_to_image_tensor(boxes, 1080, 1920)
        # ----- do ava preprocessing -----#
        # 1920,1080 -> 455,256
        if self.training:
            boxes, width, heigth = ava_scale_boxes(self.train_crop_size, boxes, 1080, 1920)
            boxes = adjust_boxes_for_crop(self.train_crop_size, heigth, width, 1, boxes)
            boxes = clip_boxes_to_image_tensor(boxes, self.train_crop_size, self.train_crop_size)
        elif mode == "test":
            boxes, width, heigth = ava_scale_boxes(self.test_crop_size, boxes, 1080, 1920)
        else:   #val
            boxes, width, heigth = ava_scale_boxes(self.test_crop_size, boxes, 1080, 1920)
            boxes = adjust_boxes_for_crop(self.test_crop_size, heigth, width, 1, boxes)
            boxes = clip_boxes_to_image_tensor(boxes, self.test_crop_size, self.test_crop_size)
        # 455,256 -> 256，256 bounding box to the middle
        if mode == "test":
            boxes = clip_boxes_to_image_tensor(boxes, heigth, width)
            return boxes
        # print(boxes)
        # print(bboxes)
        # --------------------------------#
        return boxes



    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        # only use one group in inference
        refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
        query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight)

        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat(
                [outputs_coord_cxcy, outputs_coord_wh], dim=-1
            )
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
        outputs_class = self.class_embed(hs)
        return outputs_coord, outputs_class

    def stack_sorted_bboxes_with_batch_id(self,sorted_boxes):
        all_bboxes = []
        for batch_id, batch in enumerate(sorted_boxes):
            ori_boxes = batch['ori_boxes']
            num_boxes = ori_boxes.shape[0]
            batch_ids = torch.full((num_boxes, 1), batch_id, dtype=ori_boxes.dtype, device=ori_boxes.device)
            # print(batch_ids)
            # print(ori_boxes)
            batch_bboxes = torch.cat([batch_ids, ori_boxes], dim=1)
            all_bboxes.append(batch_bboxes)
        stacked_bboxes = torch.cat(all_bboxes, dim=0)

        return stacked_bboxes

    def process_and_stack_batches(self, outputs, idx):
        processed_boxes = []  # Store processed boxes here

        for batch_id, (batch, (pred_indices, _)) in enumerate(zip(outputs, idx)):
            boxes = batch["boxes"]  # Shape: [num_boxes, 4]
            selected_boxes = boxes[pred_indices]  # Shape: [num_selected_boxes, 4]
            batch_id_column = torch.full(
                (selected_boxes.shape[0], 1), batch_id, device=selected_boxes.device, dtype=torch.float32
            )  # Shape: [num_selected_boxes, 1]
            boxes_with_batch_id = torch.cat([batch_id_column, selected_boxes], dim=1)  # Shape: [num_selected_boxes, 5]
            processed_boxes.append(boxes_with_batch_id)
        stacked_boxes = torch.cat(processed_boxes, dim=0)  # Shape: [total_selected_boxes, 5]
        return stacked_boxes

    def reorder_bboxes_by_gt(self, pred_boxes, gt_bboxes, indices, batch_length):
        reordered_boxes = []
        for batch, (pred_ids, gt_ids) in zip(pred_boxes, indices):
            pred_boxes = batch['ori_boxes']
            sorted_indices = torch.argsort(gt_ids)
            if pred_ids.numel() == 1:
                reordered_pred_boxes = pred_boxes[pred_ids]
                reordered_pred_boxes = reordered_pred_boxes.unsqueeze(0)
            else:
                reordered_pred_boxes = pred_boxes[pred_ids][sorted_indices]
            reordered_boxes.append({'ori_boxes': reordered_pred_boxes})
        # print(indices)
        # print(reordered_boxes)
        return reordered_boxes
        # reordered_boxes = []
        # batch_ids = gt_boxes[:, 0].unique()  # Get unique batch IDs
        # for batch_id in batch_ids:
        #     # Get bboxes for the current batch
        #     pred_batch = pred_boxes[pred_boxes[:, 0] == batch_id][:, 1:]
        #     gt_batch = gt_boxes[gt_boxes[:, 0] == batch_id][:, 1:]
        #     num_preds, num_gts = pred_batch.shape[0], gt_batch.shape[0]
        #     iou_matrix = self.compute_iou_matrix(pred_batch, gt_batch)
        #     matched_indices = torch.argmax(iou_matrix, dim=0)
        #     reordered_batch = pred_batch[matched_indices]
        #     batch_id_column = torch.full((reordered_batch.shape[0], 1), batch_id, device=pred_boxes.device)
        #     reordered_batch = torch.cat([batch_id_column, reordered_batch], dim=1)
        #     reordered_boxes.append(reordered_batch)
        # return torch.cat(reordered_boxes, dim=0)

    def compute_iou_matrix(self, boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # Area of boxes1
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # Area of boxes2
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # Shape: [N, M]
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # Intersection area
        union = area1[:, None] + area2 - inter  # Shape: [N, M]
        iou = inter / union
        return iou

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, fuse_feat):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b ,'fuse_feat': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1],fuse_feat)]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """ """
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder.blocks[i].drop_path, 'drop_prob'):
                self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate

def build_test_matcher(cfg):
    return testHungarianMatcher()

# class testHungarianMatcher(nn.Module):
#     def __init__(self):
#         super().__init__()
#     @torch.no_grad()
#     def forward(self, outputs, targets, num_targets_per_batch):
#         """
#         Args:
#             outputs: List of dictionaries, where each dictionary contains predictions for a batch.
#                      Each dictionary has keys like "ori_boxes" (predicted boxes) and "labels" (predicted labels).
#             targets: Tensor of shape [total_num_targets, 4] (all GT boxes stacked together).
#             num_targets_per_batch: List of integers, indicating the number of targets in each batch.
#
#         Returns:
#             indices: List of tuples (pred_indices, gt_indices) for each batch.
#         """
#         indices = []
#
#         # Split targets into per-batch targets based on the provided sizes
#         batch_targets = torch.split(targets, num_targets_per_batch)
#
#         for batch_idx, (output, target) in enumerate(zip(outputs, batch_targets)):
#             # Extract predictions and GTs for this batch
#             pred_boxes = output["ori_boxes"]  # [num_queries, 4]
#             pred_labels = output["labels"]  # [num_queries]
#             gt_boxes = target[:, 1:]  # [num_target_boxes, 4]
#
#             # Only consider predictions where the label is 1
#             valid_mask = pred_labels == 1
#             if valid_mask.sum() == 0:  # No valid predictions
#                 indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
#                 continue
#
#             valid_boxes = pred_boxes[valid_mask]  # [num_valid_queries, 4]
#
#             # Compute GIoU between valid predictions and GTs
#             giou = generalized_box_iou(
#                 valid_boxes,  # [num_valid_queries, 4]
#                 gt_boxes  # [num_target_boxes, 4]
#             )  # Result: [num_valid_queries, num_target_boxes]
#
#             # For each GT, find the prediction with the highest GIoU
#             pred_indices = []
#             gt_indices = []
#
#             for gt_idx in range(len(gt_boxes)):
#                 giou_for_gt = giou[:, gt_idx]  # GIoU values for this GT
#                 best_pred_idx = giou_for_gt.argmax().item()  # Index of the highest GIoU
#                 pred_indices.append(torch.nonzero(valid_mask)[best_pred_idx].item())  # Map back to original indices
#                 gt_indices.append(gt_idx)
#
#             # Append results for this batch
#             indices.append((
#                 torch.as_tensor(pred_indices, dtype=torch.int64),
#                 torch.as_tensor(gt_indices, dtype=torch.int64)
#             ))
#
#         return indices



class testHungarianMatcher(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets, num_targets_per_batch):
        """
        Args:
            outputs: List of dictionaries, where each dictionary contains predictions for a batch.
                     Each dictionary has keys like "ori_boxes" (predicted boxes) and "labels" (predicted labels).
            targets: Tensor of shape [total_num_targets, 4] (all GT boxes stacked together).
            num_targets_per_batch: List of integers, indicating the number of targets in each batch.

        Returns:
            indices: List of tuples (pred_indices, gt_indices) for each batch.
        """
        indices = []
        # if isinstance(targets, list):
        #     targets = torch.cat(targets, dim=0)
        # Split targets into per-batch GT boxes using the provided sizes
        batch_targets = torch.split(targets, num_targets_per_batch)

        for batch_idx, (output, target) in enumerate(zip(outputs, batch_targets)):
            # Extract predictions and GTs for this batch
            pred_boxes = output["ori_boxes"]  # num_queries, 4
            pred_labels = output["labels"]
            gt_boxes = target[:, 1:]  # num_target_boxes, 4

            # Only consider predictions where the label is 1
            valid_mask = pred_labels == 1
            if valid_mask.sum() == 0:  # No valid predictions
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            valid_boxes = pred_boxes[valid_mask]  # num_valid_queries, 4

            # Compute GIoU between valid predictions and GTs
            giou = generalized_box_iou(
                valid_boxes,  # num_valid_queries, 4
                gt_boxes  # num_target_boxes, 4
            )  #  num_valid_queries, num_target_boxes

            # Convert GIoU to a cost matrix (negative values for minimization)
            cost_matrix = -giou.cpu().numpy()

            # Apply Hungarian algorithm to find the optimal matching
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

            # Map valid indices back to the original prediction indices
            pred_indices = torch.as_tensor(valid_mask.nonzero()[pred_indices].squeeze(), dtype=torch.int64)
            gt_indices = torch.as_tensor(gt_indices, dtype=torch.int64)

            # Append results for this batch
            indices.append((pred_indices, gt_indices))

        return indices

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 focal_alpha,
                 losses,
                 group_detr=1,
                 sum_group_losses=False,
                 use_varifocal_loss=False,
                 use_position_supervised_loss=False,
                 ia_bce_loss=False,):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            group_detr: Number of groups to speed detr training. Default is 1.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher  #hungraianmatcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        # for t, (_, J) in zip(targets, indices):
            # print(t['labels'])
            # print(J)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            #init positive weights and negative weights
            pos_weights = torch.zeros_like(src_logits)
            neg_weights =  prob ** gamma

            pos_ind=[id for id in idx]
            pos_ind.append(target_classes_o)

            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[pos_ind] = t
            neg_weights[pos_ind] = 1 - t
            loss_ce = - pos_weights * prob.log() - neg_weights * (1 - prob).log()
            loss_ce = loss_ce.sum() / num_boxes

        elif self.use_position_supervised_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            # pos_ious_func = pos_ious ** 2
            pos_ious_func = pos_ious

            cls_iou_func_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind=[id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_func_targets[pos_ind] = pos_ious_func
            norm_cls_iou_func_targets = cls_iou_func_targets \
                / (cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8)
            loss_ce = position_supervised_loss(src_logits, norm_cls_iou_func_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        elif self.use_varifocal_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()

            cls_iou_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind=[id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_targets[pos_ind] = pos_ious
            loss_ce = sigmoid_varifocal_loss(src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:,:,:-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def xyxy2xywh(self, target):
        for sample in target:
            if "bbox" in sample:
                bboxes = sample["bbox"]
                x_min = bboxes[:, 0]
                y_min = bboxes[:, 1]
                x_max = bboxes[:, 2]
                y_max = bboxes[:, 3]
                width = x_max - x_min
                height = y_max - y_min
                x_center = x_min + width / 2
                y_center = y_min + height / 2
                sample["bbox"] = torch.stack([x_center, y_center, width, height], dim=1)
        return target

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # targets = self.xyxy2xywh(targets)
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    focal_weight = targets * (targets > 0.0).float() + \
            (1 - alpha) * (prob - targets).abs().pow(gamma) * \
            (targets <= 0.0).float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes, type=None):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        if type == ("ava"):
            batch_size = boxes.shape[0]
            target_sizes = torch.full((batch_size, 2), 255.0)   #256，0～255 不能直接这样缩放回去
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            target_sizes = target_sizes.to(device)
            img_h, img_w = target_sizes.unbind(1)
        else:
            img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        ori_boxes = boxes
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b, 'ori_boxes': o} for s, l, b, o in zip(scores, labels, boxes, ori_boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # print(cfg)
    num_det_classes = 91 # one class + background
    device = torch.device(0)

    spatio_backbone = build_spatio_backbone(cfg)    #lw detr

    temporal_backbone = build_temporal_backbone(cfg)   #slowonly

    matcher = build_matcher(cfg)
    cfg.num_feature_levels = len(cfg.TRANSFORMER.PROJECTOR_SCALE)
    transformer = build_transformer(cfg)

    model = LWDETR(
        cfg,
        spatio_backbone,
        temporal_backbone,
        transformer,
        num_classes=num_det_classes,
        num_queries=cfg.TRANSFORMER.NUM_QUERIES,
        aux_loss=cfg.LOSS.NO_AUX_LOSS,
        group_detr=cfg.TRANSFORMER.GROUP_DETR,
        two_stage= False,
        lite_refpoint_refine=cfg.TRANSFORMER.LITE_REFPOINT_REFINE,
        bbox_reparam=cfg.TRANSFORMER.BBOX_REPARAM,
    )
    # model = temporal_backbone

    weight_dict = {'loss_ce': cfg.LOSS.CLS_LOSS_COEF, 'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF}
    weight_dict['loss_giou'] = cfg.LOSS.GIOU_LOSS_COEF

    if cfg.LOSS.NO_AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.TRANSFORMER.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        if cfg.TRANSFORMER.TWO_STAGE:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    try:
        sum_group_losses = cfg.LOSS.SUM_GROUP_LOSSES
    except:
        sum_group_losses = False
    criterion = SetCriterion(num_det_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=cfg.LOSS.FOCAL_ALPHA, losses=losses,
                             group_detr=cfg.TRANSFORMER.GROUP_DETR, sum_group_losses=sum_group_losses,
                             use_varifocal_loss = cfg.LOSS.USE_VARIFOCAL_LOSS,
                             use_position_supervised_loss=cfg.LOSS.USE_POSITION_SUPERVISED_LOSS,
                             ia_bce_loss=cfg.LOSS.IA_BCE_LOSS)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=cfg.TRANSFORMER.NUM_SELECT)}

    return model, criterion, postprocessors
# def build(cfg):
#     device = torch.device(0)
#     model = build_temporal_backbone(cfg)   #slowonly
#     return model
class conv_fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_3 = 48
        self.channel_4 = 96
        self.channel_5 = 192
        self.st_fuse3 = nn.Conv2d(192, self.channel_3, kernel_size=1, stride=1, padding=0)
        # self.st_fuse3 = nn.Linear(192, 192)
        # self.st_fuse4_0 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3,stride=1,padding=0)
        # self.st_fuse4_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5,stride=1, padding=0)
        # self.st_fuse4_2 = nn.Conv2d(in_channels=192, out_channels=self.channel_4, kernel_size=4, stride=1, padding=0)
        self.st_fuse4_0 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 池化后尺寸为16×16
        self.st_fuse4_1 = nn.Conv2d(in_channels=192, out_channels=self.channel_4,
                                    kernel_size=3, stride=1, padding=3)
        # self.st_fuse5_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=3, padding=3)
        # self.st_fuse5_2 = nn.Conv2d(in_channels=192, out_channels=self.channel_5, kernel_size=1, stride=1, padding=0)  # 32.32 → 12x12
        self.st_fuse5_1 = nn.Conv2d(
            in_channels=192,
            out_channels=self.channel_5,
            kernel_size=5,
            stride=3,
            padding=0,
            bias=True
        )
        # self.st_fuse4 = nn.Conv2d(192, self.channel_4, kernel_size=3, stride=2, padding=1)
        # self.st_fuse5 = nn.Conv2d(192, self.channel_5, kernel_size=4, stride=2, padding=1)
        # self.groupnorm2 = nn.GroupNorm(num_groups=1, num_channels=256)
        self.groupnorm3 = nn.GroupNorm(num_groups=1, num_channels=self.channel_3)
        self.groupnorm4 = nn.GroupNorm(num_groups=1, num_channels=self.channel_4)
        self.groupnorm5 = nn.GroupNorm(num_groups=1, num_channels=self.channel_5)
        # self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(self.channel_3)
        self.bn4 = nn.BatchNorm2d(self.channel_4)
        self.bn5 = nn.BatchNorm2d(self.channel_5)
        self.activate = F.leaky_relu
    def forward(self, encoder_out_feat):
        s2_fuse = encoder_out_feat
        s3_fuse = F.interpolate(encoder_out_feat, size=(32, 32), mode='bilinear', align_corners=False)
        s3_fuse = self.st_fuse3(s3_fuse)
        s3_fuse = self.bn3(s3_fuse)
        s3_fuse = self.activate(s3_fuse)
        s4_fuse = self.st_fuse4_0(F.interpolate(encoder_out_feat, size=(32, 32), mode='bilinear', align_corners=False))
        s4_fuse = self.st_fuse4_1(s4_fuse)
        # s4_fuse = self.st_fuse4_2(s4_fuse)
        s4_fuse = self.bn4(s4_fuse)
        s4_fuse = self.activate(s4_fuse)
        s5_fuse = self.st_fuse5_1(F.interpolate(encoder_out_feat, size=(32, 32), mode='bilinear', align_corners=False))
        s5_fuse = self.bn5(s5_fuse)
        s5_fuse = self.activate(s5_fuse)
# if __name__ == "__main__":
#     from fvcore.nn import FlopCountAnalysis, parameter_count_table
#     module = conv_fuse()
#     x = [torch.randn(2,96,16,20,20)]
#     fuse = torch.randn(2,192,40,40)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x[0] = x[0].to(device)
#     fuse = fuse.to(device)
#     module = module.to(device)
#     flops = FlopCountAnalysis(module, (fuse))
#     params = parameter_count_table(module)
#     print(f"FLOPs: {flops.total() / 1e9:.6f} GFLOPs")
#     print(params)