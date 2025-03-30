from typing import Dict, List
import torch
from torch import nn
from core.dataset.misc import NestedTensor
from core.model.position_encoding import build_position_encoding
from .backbone import *
from core.model.temporal_builder import X3D
from core.model import temporal_builder

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor):
        # print(tensor_list)
        x, encoder_out_feat = self[0](tensor_list)  # self：[Joiner,posembsine]
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos, encoder_out_feat

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

    def forward_export(self, inputs: torch.Tensor):
        feats, masks = self[0](inputs)
        poss = []
        for feat, mask in zip(feats, masks):
            poss.append(self[1](mask, align_dim_orders=False).to(feat.dtype))
        return feats, None, poss


def build_spatio_backbone(cfg):
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(cfg)  # sine

    # if args.encoder in ['vit_tiny', 'vit_small', 'vit_base', 'res18vd', 'res50vd', 'mobilenetv3_large_1.0']:
    spatio_backbone = Backbone( #build spatio_backbone
        cfg.BACKBONE.ENCODER,
        cfg.BACKBONE.VIT_ENCODER_NUM_LAYERS,
        cfg.MODEL.PRETRAINED_ENCODER,
        window_block_indexes=cfg.BACKBONE.WINDOW_BLOCK_INDEXES,
        drop_path=cfg.DROPOUT.DROP_PATH_RATE,
        out_channels=cfg.TRANSFORMER.HIDDEN_DIM,
        out_feature_indexes=cfg.BACKBONE.OUT_FEATURE_INDEXES,
        projector_scale=cfg.TRANSFORMER.PROJECTOR_SCALE ,
    )

    model = Joiner(spatio_backbone, position_embedding)
    return model


def build_temporal_backbone(cfg):
    backbone = "x3d"
    if backbone == 'x3d':
        model = X3D(cfg)
    return model