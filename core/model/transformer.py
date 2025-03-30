# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Transformer class sampling_offsets
"""
import math
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .attention import MultiheadAttention
from .defor_attn.modules import MSDeformAttn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers    #2
        h = [hidden_dim] * (num_layers - 1) #[256]
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor, dim=128):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, unsigmoid=True):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape   #1，1600，256
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        if memory_padding_mask is not None: #mask flatten
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1) #提取某一层级特征图的形状，然后还原回原来的形态
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)  #计算有效行数和列数
        else:
            valid_H = torch.tensor([H_ for _ in range(N_)], device=memory.device)
            valid_W = torch.tensor([W_ for _ in range(N_)], device=memory.device)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))#40，40
    #生成一个二维网格坐标，生成从 0 到 H_ - 1 的 H_ 个等间距的值，表示特征图在x ,  y 轴上的坐标;torch.meshgrid 根据输入的 y 和 x 坐标生成一个二维网格
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2  #40，40，2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)  #2，1，1，2
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale     #对网格做归一化 2，40，40，2

        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)    #为网格生成高度，宽度，宽高会根据层级来调整 2，40，40，2

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)        #生成候选框x,y,w,h 2，1600，4
        proposals.append(proposal)
        _cur += (H_ * W_)

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
#检查每个值是否大于0.01小于0.99，返回的是true false 表示是否符合条件
    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))    #根据mask来去掉不要的proposals
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))  #将无效的区域用0替代

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


class Transformer(nn.Module):

    def __init__(self, d_model=512, sa_nhead=8, ca_nhead=8, num_queries=300,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, group_detr=13,
                 two_stage=False,
                 num_feature_levels=4, dec_n_points=4,
                 lite_refpoint_refine=False,
                 decoder_norm_type='LN',
                 bbox_reparam=False):
        super().__init__()
        self.encoder = None

        decoder_layer = TransformerDecoderLayer(d_model, sa_nhead, ca_nhead, dim_feedforward,
                                                dropout, activation, normalize_before, 
                                                group_detr=group_detr,
                                                num_feature_levels=num_feature_levels,
                                                dec_n_points=dec_n_points,
                                                skip_self_attn=False,)
        assert decoder_norm_type in ['LN', 'Identity']
        norm = {
            "LN": lambda channels: nn.LayerNorm(channels),
            "Identity": lambda channels: nn.Identity(),
        }   #构造一个字典，里面放了两个归一化函数
        decoder_norm = norm[decoder_norm_type](d_model) #layernorm(256)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model,
                                          lite_refpoint_refine=lite_refpoint_refine,
                                          bbox_reparam=bbox_reparam)
        
        
        self.two_stage = True
        if self.two_stage:   #ture
            self.enc_output = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(group_detr)])
            self.enc_output_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(group_detr)])
    #构建两个线性层，数量由group_detr决定
        self._reset_parameters()

        self.num_queries = num_queries #300
        self.d_model = d_model  #256
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels    #1
        self.bbox_reparam = True    #true

        self._export = False
    
    def export(self):
        self._export = True

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
    
    def get_valid_ratio(self, mask):    #计算mask中图片有效区域部分 meshgrid
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)      #计算mask中有几行，几列有效
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W     #计算mask
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)       #堆叠起来，得到每张图片的有效区域mask
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):
        src_flatten = []
        mask_flatten = [] if masks is not None else None
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = [] if masks is not None else None
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape #1，256，40，40
            spatial_shape = (h, w)  #40,40
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c  flatten(2) 从第二个维度开始展开 2,1600,256
            lvl_pos_embed_flatten.append(pos_embed)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c  1，1600，256
            src_flatten.append(src)
            if masks is not None:
                mask = masks[lvl].flatten(1)                    # bs, hw    1,1600
                mask_flatten.append(mask)
        memory = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 2,1600,256
        if masks is not None:
            mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}     #1,1600
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) #2,1,2
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 1,256,40,40
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device)    #1，2
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        #计算每一级特征的索引位置（如果有多级特征，这些特征又被拼接到同一维度）
        
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, unsigmoid=not self.bbox_reparam)  #2，1600，256， 2，1600，4
            # group detr for first stage
            refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory)) #enc output:linear 2，1600，256
    
                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](output_memory_gidx) #linear 2，1600，91
                if self.bbox_reparam:
                    enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](output_memory_gidx)   #mlp 2，1600，4
                    #enc_outputs_coord_delta_gidx : 1,1600,4 xywh偏移量
                    enc_outputs_coord_cxcy_gidx = enc_outputs_coord_delta_gidx[...,
                        :2] * output_proposals[..., 2:] + output_proposals[..., :2] #xy中心
                    #[...:2]: 取出最后一个维度的前两个元素 修正中心点的位置 2，1600，2
                    enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:] #2，1600，2
                    enc_outputs_coord_unselected_gidx = torch.concat(
                        [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1)   #2，1600，4
                else:
                    enc_outputs_coord_unselected_gidx = self.enc_out_bbox_embed[g_idx](
                        output_memory_gidx) + output_proposals # (bs, \sum{hw}, 4) unsigmoid

                topk = self.num_queries #300
                topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1] # bs, nq 找到得分最高的topk个框，然后返回索引
                #2，300
                refpoint_embed_gidx_undetach = torch.gather(#前300索引对应的coord
                    enc_outputs_coord_unselected_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid 取出对应索引的框坐标 2，300，4
                # for decoder layer, detached as initial ones, (bs, nq, 4)
                refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()
                #从计算图中分离，以免影响反向传播
                # get memory tgt
                tgt_undetach_gidx = torch.gather(
                    output_memory_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model))    #2，300，256
                #提取这300个分高的候选框的特征向量
                refpoint_embed_ts.append(refpoint_embed_gidx)   #1，300，4 框加class
                memory_ts.append(tgt_undetach_gidx) #1，300，256 对应特征
                boxes_ts.append(refpoint_embed_gidx_undetach)   #1，300，4 不参加反向传播的
            # concat on dim=1, the nq dimension, (bs, nq, d) --> (bs, nq, d)
            refpoint_embed_ts = torch.cat(refpoint_embed_ts, dim=1) #2，3900，4
            # (bs, nq, d)
            memory_ts = torch.cat(memory_ts, dim=1)#.transpose(0, 1)#2，3900，256
            boxes_ts = torch.cat(boxes_ts, dim=1)#.transpose(0, 1)#2，3900，4
        
        tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)#2，3900，256 复制bs次 query_feat = self.query_feat.weight = nn.Embedding(num_queries * group_detr, hidden_dim).weight
        refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)   #2,3900,4 refpoint_embed=self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        if self.two_stage:
            if self.bbox_reparam:
                refpoint_embed_cxcy = refpoint_embed[..., :2] * refpoint_embed_ts[..., 2:] + refpoint_embed_ts[..., :2] #2,3900,2
                refpoint_embed_wh = refpoint_embed[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                refpoint_embed = torch.concat(
                    [refpoint_embed_cxcy, refpoint_embed_wh], dim=-1
                )   #又偏移中心点，wh变换了一次
            else:
                refpoint_embed = refpoint_embed + refpoint_embed_ts

        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask_flatten,
                          pos=lvl_pos_embed_flatten, refpoints_unsigmoid=refpoint_embed,
                          level_start_index=level_start_index, 
                          spatial_shapes=spatial_shapes,
                          valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else valid_ratios) #3,2,3900,256;1,2,3900,4
        if self.two_stage:
            if self.bbox_reparam:
                return hs, references, memory_ts, boxes_ts
            else:
                return hs, references, memory_ts, boxes_ts.sigmoid()
        return hs, references, None, None


class TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None, 
                 return_intermediate=False,
                 d_model=256,
                 lite_refpoint_refine=False,
                 bbox_reparam=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers    #3
        self.d_model = d_model      #256
        self.norm = norm
        self.return_intermediate = return_intermediate  #true
        self.lite_refpoint_refine = lite_refpoint_refine    #true
        self.bbox_reparam = bbox_reparam    #true

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2) #几层堆叠的线性层

        self._export = False
    
    def export(self):
        self._export = True

    def refpoints_refine(self, refpoints_unsigmoid, new_refpoints_delta):
        if self.bbox_reparam:
            new_refpoints_cxcy = new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:] + refpoints_unsigmoid[..., :2]
            new_refpoints_wh = new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            new_refpoints_unsigmoid = torch.concat(
                [new_refpoints_cxcy, new_refpoints_wh], dim=-1
            )
        else:
            new_refpoints_unsigmoid = refpoints_unsigmoid + new_refpoints_delta
        return new_refpoints_unsigmoid

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,
                # for memory
                level_start_index: Optional[Tensor] = None, # num_levels
                spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]#2,3900,4 refpoint_emb.weight
        
        def get_reference(refpoints):   #todo7
            # [num_queries, batch_size, 4]
            obj_center = refpoints[..., :4] #2,3900,4
            
            if self._export:
                query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model / 2) # bs, nq, 256*2 
                refpoints_input = obj_center[:, :, None] # bs, nq, 1, 4
            else:
                refpoints_input = obj_center[:, :, None] \
                                        * torch.cat([valid_ratios, valid_ratios], -1)[:, None] # bs, nq, nlevel, 4 ；2，3900，1，4
                query_sine_embed = gen_sineembed_for_position(
                    refpoints_input[:, :, 0, :], self.d_model / 2) # bs, nq, 256*2 ；2，3900，512
            query_pos = self.ref_point_head(query_sine_embed)   #mlp 2,3900,256
            return obj_center, refpoints_input, query_pos, query_sine_embed
        
        # always use init refpoints
        if self.lite_refpoint_refine:   #refpoint_unsigmoid:经过两阶段box reparam后的框
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)   #2，3900，4
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

        for layer_id, layer in enumerate(self.layers):
            # iter refine each layer
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
                else:
                    obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

            # For the first decoder layer, we do not apply transformation over p_s
            pos_transformation = 1

            query_pos = query_pos * pos_transformation
            
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed, 
                           is_first=(layer_id == 0),
                           reference_points=refpoints_input,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index) #2,3900,256

            if not self.lite_refpoint_refine:
                # box iterative update
                new_refpoints_delta = self.bbox_embed(output)
                new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self._export:
                # to shape: B, N, C
                hs = intermediate[-1]
                if self.bbox_embed is not None:
                    ref = hs_refpoints_unsigmoid[-1]
                else:
                    ref = refpoints_unsigmoid
                return hs, ref
            # box iterative update
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate),
                    torch.stack(hs_refpoints_unsigmoid),
                ]
            else:
                return [
                    torch.stack(intermediate), 
                    refpoints_unsigmoid.unsqueeze(0)
                ]

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, sa_nhead, ca_nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, group_detr=1, 
                 num_feature_levels=4, dec_n_points=4, 
                 skip_self_attn=False):
        #sa_head 8 ,ca_head 16
        super().__init__()
        # Decoder Self-Attention
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Decoder Cross-Attention
        self.cross_attn = MSDeformAttn(
            d_model, n_levels=num_feature_levels, n_heads=ca_nhead, n_points=dec_n_points)

        self.nhead = ca_nhead   #16

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)    #relu
        self.normalize_before = normalize_before        #false
        self.group_detr = group_detr

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False,
                     reference_points = None,
                     spatial_shapes=None,
                     level_start_index=None,
                     ):
        bs, num_queries, _ = tgt.shape  #2，3900，256
        
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: batch_size x num_queries x 256
        q = k = tgt + query_pos #2，3900，256
        v = tgt #2，3900，256
        if self.training:
            q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)    #26，300，256
            k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)
            v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)

        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask,#todo
                            key_padding_mask=tgt_key_padding_mask)[0]   #26,300,256
        
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)  #2,3900,256
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)   #2，3900，256

        # ========== Begin of Cross-Attention =============
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask
        )
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False,
                reference_points = None,
                spatial_shapes=None,
                level_start_index=None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, 
                                 query_sine_embed, is_first,
                                 reference_points, spatial_shapes, level_start_index)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)]) #深拷贝复制模块


def build_transformer(cfg):
    
    two_stage = cfg.TRANSFORMER.TWO_STAGE
    return Transformer(
        d_model=cfg.TRANSFORMER.HIDDEN_DIM,
        sa_nhead=cfg.TRANSFORMER.SA_NHEADS,
        ca_nhead=cfg.TRANSFORMER.CA_NHEADS,
        num_queries=cfg.TRANSFORMER.NUM_QUERIES,
        dropout=cfg.DROPOUT.DROPOUT_RATE,
        dim_feedforward=cfg.TRANSFORMER.DIM_FEEDFORWARD ,
        num_decoder_layers=cfg.TRANSFORMER.DEC_LAYERS,
        return_intermediate_dec=True,
        group_detr=cfg.TRANSFORMER.GROUP_DETR,
        two_stage=two_stage,
        num_feature_levels= 1,
        dec_n_points=cfg.TRANSFORMER.DEC_N_POINTS,
        lite_refpoint_refine=cfg.TRANSFORMER.LITE_REFPOINT_REFINE,
        decoder_norm_type=cfg.TRANSFORMER.DECODER_NORM,
        bbox_reparam=cfg.TRANSFORMER.BBOX_REPARAM,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
