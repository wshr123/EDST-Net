# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import math
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import core.utils.logging as logging
import core.utils.weight_init_helper as init_helper
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.batchnorm_helper import get_norm
from core.model.utils import round_width
from core.model.defor_attn.modules.ms_deform_attn import MSDeformAttn, tMSDeformAttn, MSDeformAttnfuse
from core.model import head_helper, operators, resnet_helper, stem_helper  # noqa
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None

logger = logging.get_logger(__name__)

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}
_POOL1 = {
    "x3d": [[1, 1, 1]],
}

class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.
    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        pool_size = _POOL1[cfg.MODEL.ARCH]
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = round_width(self.dim_res4, exp_stage, divisor=8)
        # self.proj2 = Projection(24)
        self.proj3 = Projection(48)
        self.proj4 = Projection(96)
        self.proj5 = Projection(192)
        # self.st_fuse2 = STFusion2()
        self.st_fuse3 = STFusion3()
        self.st_fuse4 = STFusion4()
        self.st_fuse5 = STFusion5()
        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        self.head = head_helper.ResNetRoIHead(
            dim_in= [192],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
            resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
            scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            aligned=cfg.DETECTION.ALIGNED,
            detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
        )
    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(stage + 2)  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner] if cfg.X3D.CHANNELWISE_3x3x3 else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            NotImplementedError
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = head_helper.X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
            )

    def forward(self, x, bboxes, matched_boxes=None, s2_fuse=None, s3_fuse=None, s4_fuse=None, s5_fuse=None):
        x = self.s1(x)
        x = self.s2(x)
        # s2_fuse = self.proj2(x, s2_fuse)
        # x = self.st_fuse2(x, s2_fuse)
        x = self.s3(x)
        s3_fuse = self.proj3(x, s3_fuse)
        x = self.st_fuse3(x, s3_fuse)
        x = self.s4(x)
        s4_fuse = self.proj4(x, s4_fuse)
        x = self.st_fuse4(x, s4_fuse)
        x = self.s5(x)
        s5_fuse = self.proj5(x, s5_fuse)
        attn_dict = None
        x = self.st_fuse5(x, s5_fuse)
        self.enable_detection = True
        if isinstance(bboxes, dict):
            bboxes = bboxes['boxes']
        if self.enable_detection:
            if self.training:
                x = self.head(x, bboxes)
            else:
                use_matcher = True
                if use_matcher:
                    x = self.head(x, matched_boxes)
                else:
                    x = self.head(x, bboxes)
        else:
            pass
        if attn_dict is not None:
            return x, attn_dict
        else:
            return x

#align spatio feature with temporal feature
class Projection(nn.Module):
    def __init__(self, channel):
        super(Projection, self).__init__()
        self.in_channel = 192 # encoder output
        self.out_channel = channel
        self.conv = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.bn = nn.BatchNorm2d(self.out_channel)
        self.activate = F.leaky_relu
    def forward(self, x, fuse):
        x = x[0]
        _, _, _, h, w = x.shape
        if fuse.shape[3] != h:
            fuse = F.interpolate(fuse, size=(h, w), mode='bilinear', align_corners=False)
            fuse = self.activate(self.bn(self.conv(fuse)))
        else:
            return fuse
        return fuse


class STFusion2(nn.Module):
    def __init__(self):
        super(STFusion2, self).__init__()
        self.channels = 24
        self.reduction = 1
        self.activate = F.gelu
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(self.channels)
        group_conv = False
        if group_conv == True:
            self.conv = nn.Conv3d(
                in_channels=self.channels + self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                groups=2
            )
        else:
            self.conv = nn.Conv3d(self.channels * 2, self.channels,1,1,0)
        self.fc1 = nn.Linear(self.channels, self.channels // self.reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.channels // self.reduction, self.channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm3d(self.channels)
        self.bn2 = nn.BatchNorm3d(self.channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.Conv = nn.Sequential(
            self.conv,
            self.bn1,
            self.relu
        )
        self.attn = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.sigmoid
        )

    def forward(self, x, fuse=None):
        x = x[0]
        n, c, t, h, w = x.shape
        fuse = fuse.unsqueeze(2)
        fuse = fuse.expand(-1, -1, t, -1, -1)
        # fuse = fuse.repeat(1, 1, t, 1, 1)
        x = torch.cat((x, fuse), dim=1)
        # x = self.Conv(x)
        x = self.conv(x)
        # x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        y = self.pool(x).view(n, c)
        # y = self.attn(y)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(n, c, 1, 1, 1)
        x = x * y.expand_as(fuse)
        x = self.bn2(x)
        return [x]

class STFusion3(nn.Module):
    def __init__(self):
        super(STFusion3, self).__init__()
        self.channels = 48
        self.reduction = 1
        self.activate = F.gelu
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(256)
        self.conv = nn.Conv3d(self.channels * 2, self.channels,1,1,0)
        self.fc = nn.Linear(self.channels * 2, self.channels, bias=False)
        self.fc1 = nn.Linear(self.channels, self.channels // self.reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.channels // self.reduction, self.channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm3d(self.channels)
        self.bn2 = nn.BatchNorm3d(self.channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.Conv = nn.Sequential(
            self.conv,
            self.bn1,
            self.relu
        )
        self.attn = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.sigmoid
        )
    def forward(self, x, fuse=None):
        x = x[0]
        n, c, t, h, w = x.shape
        acc = False
        if acc:
            fuse = fuse.unsqueeze(2).repeat(1, 1, t, 1, 1)
            x = torch.cat((x, fuse), dim=1)
            x = self.Conv(x)
            y = self.pool(x).view(n, c)
            y = self.attn(y)
            y = y.view(n, c, 1, 1, 1)
            x = x * y
            x = self.bn2(x)
        else:
            _, _, _, h, w = x.shape
            if fuse.shape[3] != h:
                fuse = F.interpolate(fuse, size=(h, w), mode='bilinear', align_corners=False)
            fuse = fuse.unsqueeze(2)
            fuse = fuse.expand(-1, -1, t, -1, -1)
            x = torch.cat((x, fuse), dim=1)
            x = self.conv(x)
            # x = self.fc(x)
            x = self.bn1(x)
            x = self.relu(x)
            y = self.pool(x).view(n, c)
            y = self.fc1(y)
            y = self.relu(y)
            y = self.fc2(y)
            y = self.sigmoid(y)
            y = y.view(n, c, 1, 1, 1)
            x = x * y.expand_as(fuse)
            x = self.bn2(x)
        return [x]

class STFusion4(nn.Module):
    def __init__(self):
        super(STFusion4, self).__init__()
        self.channels = 96
        self.reduction = 2
        self.activate = F.gelu
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(256)
        self.conv = nn.Conv3d(self.channels * 2, self.channels, 1, 1, 0)
        self.fc = nn.Linear(self.channels * 2, self.channels, bias=False)
        self.fc1 = nn.Linear(self.channels, self.channels // self.reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.channels // self.reduction, self.channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm3d(self.channels)
        self.bn2 = nn.BatchNorm3d(self.channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.Conv = nn.Sequential(
            self.conv,
            self.bn1,
            self.relu
        )
        self.attn = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.sigmoid
        )
    def forward(self, x, fuse=None):
        x = x[0]
        n, c, t, h, w = x.shape
        acc = False
        if acc:
            fuse = fuse.unsqueeze(2).repeat(1, 1, t, 1, 1)
            x = torch.cat((x, fuse), dim=1)
            x = self.Conv(x)
            y = self.pool(x).view(n, c)
            y = self.attn(y)
            y = y.view(n, c, 1, 1, 1)
            x = x * y
            x = self.bn2(x)
        else:
            _, _, _, h, w = x.shape
            if fuse.shape[3] != h:
                fuse = F.interpolate(fuse, size=(h, w), mode='bilinear', align_corners=False)
            fuse = fuse.unsqueeze(2)
            fuse = fuse.expand(-1, -1, t, -1, -1)
            x = torch.cat((x, fuse), dim=1)
            x = self.conv(x)
            # x = self.fc(x)
            x = self.bn1(x)
            x = self.relu(x)
            y = self.pool(x).view(n, c)
            y = self.fc1(y)
            y = self.relu(y)
            y = self.fc2(y)
            y = self.sigmoid(y)
            y = y.view(n, c, 1, 1, 1)
            x = x * y.expand_as(fuse)
            x = self.bn2(x)
        return [x]

class STFusion5(nn.Module):
    def __init__(self):
        super(STFusion5, self).__init__()
        normal_fuse = False
        if normal_fuse:
            self.channels = 192
            self.reduction = 4
            self.activate = F.gelu
            self.dropout = nn.Dropout(0.5)
            self.norm = nn.LayerNorm(self.channels)
            self.conv = nn.Conv3d(self.channels * 2, self.channels, 1, 1, 0)
            self.fc = nn.Linear(self.channels * 2, self.channels, bias=False)
            self.fc1 = nn.Linear(self.channels, self.channels // self.reduction, bias=False)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(self.channels // self.reduction, self.channels, bias=False)
            self.sigmoid = nn.Sigmoid()
            self.bn1 = nn.BatchNorm3d(self.channels)
            self.bn2 = nn.BatchNorm3d(self.channels)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.Conv = nn.Sequential(
                self.conv,
                self.bn1,
                self.relu
            )
            self.attn = nn.Sequential(
                self.fc1,
                self.relu,
                self.fc2,
                self.sigmoid
            )
        else:
            self.channels = 192
            self.d_model = 192  #192
            self.reduction = 8
            self.activate = F.gelu
            self.s_dropout = nn.Dropout(0.5)
            self.t_dropout = nn.Dropout(0.5)
            self.norm1 = nn.LayerNorm(self.d_model)
            self.norm2 = nn.LayerNorm(self.d_model)
            self.conv = nn.Conv3d(self.channels * 2, self.d_model, 1, 1, 0)
            self.fc = nn.Linear(self.d_model, self.d_model, bias=False)
            self.fc1 = nn.Linear(self.d_model, self.d_model // self.reduction, bias=False)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(self.d_model // self.reduction, self.d_model, bias=False)
            self.sigmoid = nn.Sigmoid()
            self.bn1 = nn.BatchNorm3d(self.d_model)
            self.bn2 = nn.BatchNorm3d(self.d_model)
            self.pool = nn.AdaptiveAvgPool3d(1)
            #deformable attention
            self.spatio_attn = MSDeformAttn(
                self.d_model, n_levels=1, n_heads=16, n_points=2)
            self.temporal_attn = tMSDeformAttn(
                self.d_model, n_levels=1, n_heads=8, n_points=4)
            self.linear1 = nn.Linear(self.d_model * 2, self.d_model)
            self.dropout1 = nn.Dropout(0.5)
            self.linear2 = nn.Linear(self.d_model, self.d_model)
            self.dropout2 = nn.Dropout(0.5)
            self.norm3 = nn.LayerNorm(self.d_model)
            conv = False
            if conv:
                self.spatial_conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=7, stride=1, padding=3)
                self.temporal_conv = nn.Conv1d(in_channels=self.channels, out_channels=self.channels,kernel_size=7,padding=3,stride=1)
                self.fusion_conv = nn.Conv3d(in_channels=self.channels*2,out_channels=self.channels,kernel_size=1,stride=1,padding=0)
            normal_attention = False
            if normal_attention:
                self.d_model = 256
                self.conv = nn.Conv3d(self.channels * 2, self.d_model, 1, 1, 0)
                # normal multi head attention
                self.spatial_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=8, batch_first=True)
                self.temporal_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=8, batch_first=True)
                self.norm1 = nn.LayerNorm(self.d_model)
                self.norm2 = nn.LayerNorm(self.d_model)
                self.fc = nn.Linear(self.d_model, self.d_model, bias=False)
                self.fc1 = nn.Linear(self.d_model, self.d_model // self.reduction, bias=False)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(self.d_model // self.reduction, self.d_model, bias=False)
                self.sigmoid = nn.Sigmoid()
                self.bn1 = nn.BatchNorm3d(self.d_model)
                self.bn2 = nn.BatchNorm3d(self.d_model)
                self.linear1 = nn.Linear(self.d_model * 2, self.d_model)
                self.dropout1 = nn.Dropout(0.5)
                self.linear2 = nn.Linear(self.d_model, self.d_model)
                self.dropout2 = nn.Dropout(0.5)
                self.norm3 = nn.LayerNorm(self.d_model)
    def forward(self, x, fuse=None):
        method = "deformable"
        return_spatio_attn = False
        return_temporal_attn = False
        if method == "attention":
            x = x[0]
            n, c, t, h, w = x.shape
            c = self.d_model
            fuse = fuse.unsqueeze(2)
            fuse = fuse.expand(-1, -1, t, -1, -1)
            x = torch.cat((x, fuse), dim=1) #4096
            x = self.conv(x)
            x_space = x.permute(0, 2, 3, 4, 1).reshape(n * t, h * w, c)
            spatial_out, _ = self.spatial_attention(x_space, x_space, x_space)
            x_temporal = x.permute(0, 3, 4, 1, 2).reshape(n * h * w, t, c)
            temporal_out , _ = self.temporal_attention(x_temporal, x_temporal, x_temporal)
            x_space = x_space + self.s_dropout(spatial_out)
            x_space = self.norm1(x_space).view(h * w, t, n, c).permute(1, 0, 2, 3).contiguous().view(t * h * w, n, c)
            x_temporal = x_temporal + self.t_dropout(x_temporal)
            x_temporal = self.norm2(x_temporal).view(h * w, t, n, c).permute(1, 0, 2, 3).contiguous().view(t * h * w, n, c)
            x_st = torch.cat((x_temporal, x_space), dim=-1)
            x_st = self.linear2(self.dropout1(self.activate(self.linear1(x_st))))
            x_st = x_st.view(t, h, w, n, c).permute(3, 4, 0, 1, 2).contiguous()
            # x = x.permute(0, 2, 3, 4, 1).reshape(t * h * w, n, c)
            x = x + self.dropout2(x_st)
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm3(x)
            x = x.permute(0, 4, 1, 2, 3)
        elif method == "conv":
            x = x[0]
            n, c, t, h, w = x.shape
            channel = 512
            fuse = fuse.unsqueeze(2)
            fuse = fuse.expand(-1, -1, t, -1, -1)
            x = torch.cat((x, fuse), dim=1)  # 4096
            x = self.conv(x)   #4096->2048
            x_space = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
            x_space = self.spatial_conv(x_space)
            x_space = x_space.reshape(n, t, channel, h, w).permute(0, 2, 1, 3, 4)
            x_temporal = x.permute(0, 3, 4, 1, 2).reshape(n * h * w, c, t)
            x_temporal = self.temporal_conv(x_temporal)
            x_temporal = x_temporal.reshape(n, h, w, channel, t).permute(0, 3, 4, 1, 2)
            x_st = torch.cat((x_temporal, x_space), dim=1)
            x_st = self.dropout1(self.activate(self.fusion_conv(x_st)))
            x_st = x_st.view(t, h, w, n, c).permute(3, 4, 0, 1, 2).contiguous()
            x = x + self.dropout2(x_st)
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm3(x)
            x = x.permute(0, 4, 1, 2, 3)
        elif method == "shallow":
            x = x[0]
            n, c, t, h, w = x.shape
            fuse = fuse.unsqueeze(2)
            fuse = fuse.expand(-1, -1, t, -1, -1)
            x = torch.cat((x, fuse), dim=1)
            x = self.conv(x)
            # x = self.fc(x)
            x = self.bn1(x)
            x = self.relu(x)
            y = self.pool(x).view(n, c)
            y = self.fc1(y)
            y = self.relu(y)
            y = self.fc2(y)
            y = self.sigmoid(y)
            y = y.view(n, c, 1, 1, 1)
            x = x * y.expand_as(fuse)
            x = self.bn2(x)
        elif method == "deformable":
            x = x[0]
            _, _, _, h, w = x.shape
            if fuse.shape[3] != h:
                fuse = F.interpolate(fuse, size=(h, w), mode='bilinear', align_corners=False)
            n, c, t, h, w = x.shape
            fuse = fuse.unsqueeze(2).expand(-1, -1, t, -1, -1)
            x = torch.cat((x, fuse), dim=1)
            x = self.conv(x)
            # ===================== Spatial Attention =====================
            x_space = x.permute(0, 2, 3, 4, 1).reshape(n * t, h * w,
                                                       self.d_model)
            spatial_shapes = torch.tensor([[h, w]], device=x.device, dtype=torch.long)
            level_start_index = torch.tensor([0], device=x.device, dtype=torch.long)
            reference_points = torch.stack(torch.meshgrid(
                torch.linspace(0, 1, h, device=x.device),
                torch.linspace(0, 1, w, device=x.device)
            ), dim=-1).reshape(1, h * w, 2).expand(n * t, h * w, 2).unsqueeze(2)
            if return_spatio_attn:
                spatial_out, attn_dict = self.spatio_attn(
                    query=x_space,
                    reference_points=reference_points,
                    input_flatten=x_space,
                    input_spatial_shapes=spatial_shapes,
                    input_level_start_index=level_start_index,
                    return_attn_dict = True
                )
            else:
                spatial_out = self.spatio_attn(
                    query=x_space,
                    reference_points=reference_points,
                    input_flatten=x_space,
                    input_spatial_shapes=spatial_shapes,
                    input_level_start_index=level_start_index,
                )
            # ===================== Temporal Attention =====================
            x_temporal = x.permute(0, 3, 4, 1, 2).reshape(n * h * w, t,
                                                          self.d_model)
            temporal_shapes = torch.tensor([[t, 1]], device=x.device, dtype=torch.long)
            level_start_index = torch.tensor([0], device=x.device, dtype=torch.long)
            reference_points = torch.linspace(0, 1, t, device=x.device).view(1, t, 1).expand(n * h * w, t, 1).unsqueeze(
                2)
            if return_temporal_attn:
                temporal_out, attn_dict = self.temporal_attn(
                    query=x_temporal,
                    reference_points=reference_points,
                    input_flatten=x_temporal,
                    input_spatial_shapes=temporal_shapes,
                    input_level_start_index=level_start_index,
                    return_attn_dict=True
                )
            else:
                temporal_out = self.temporal_attn(
                    query=x_temporal,
                    reference_points=reference_points,
                    input_flatten=x_temporal,
                    input_spatial_shapes=temporal_shapes,
                    input_level_start_index=level_start_index,
                )
            # ===================== Residual Connections =====================
            x_space = x_space + self.s_dropout(spatial_out)
            x_space = self.norm1(x_space).view(h * w, t, n, self.d_model).permute(1, 0, 2, 3).contiguous().view(
                t * h * w, n, self.d_model)

            x_temporal = x_temporal + self.t_dropout(temporal_out)
            x_temporal = self.norm2(x_temporal).view(h * w, t, n, self.d_model).permute(1, 0, 2, 3).contiguous().view(
                t * h * w, n, self.d_model)
            # ===================== Combine Spatial and Temporal Outputs =====================
            x_st = torch.cat((x_temporal, x_space), dim=-1)
            x_st = self.linear2(self.dropout1(F.relu(self.linear1(x_st))))
            x_st = x_st.view(t, h, w, n, self.d_model).permute(3, 4, 0, 1,
                                                               2).contiguous()
            x = x + self.dropout2(x_st)
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm3(x)
            x = x.permute(0, 4, 1, 2, 3)
            attn_dict = x
        if return_temporal_attn or return_spatio_attn:
            return [x], attn_dict
        return [x]


if __name__ == "__main__":
    module = Projection(192)
    x = [torch.randn(2,192,16,10,10)]
    fuse = torch.randn(2,192,40,40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x[0] = x[0].to(device)
    fuse = fuse.to(device)
    module = module.to(device)
    flops = FlopCountAnalysis(module, (x,fuse))
    params = parameter_count_table(module)
    print(f"FLOPs: {flops.total() / 1e9:.6f} GFLOPs")
    print(params)