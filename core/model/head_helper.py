#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

from functools import partial

import core.utils.logging as logging
import torch
import torch.nn as nn
from detectron2.layers import ROIAlign
# from core.model.attention import MultiScaleBlock
from core.model.batchnorm_helper import (
    NaiveSyncBatchNorm1d as NaiveSyncBatchNorm1d,
)
from core.model.nonlocal_helper import Nonlocal

logger = logging.get_logger(__name__)


class ResNetRoIHead(nn.Module):

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
        detach_final_fc=False,
    ):

        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.detach_final_fc = detach_final_fc

        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d([pool_size[pathway][0], 1, 1], stride=1)
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        # print(bboxes)
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        if self.detach_final_fc:
            x = x.detach()
        x = self.projection(x)
        x = self.act(x)
        return x



class MLPHead(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        mlp_dim,
        num_layers,
        bn_on=False,
        bias=True,
        flatten=False,
        xavier_init=True,
        bn_sync_num=1,
        global_sync=False,
    ):
        super(MLPHead, self).__init__()
        self.flatten = flatten
        b = False if bn_on else bias
        # assert bn_on or bn_sync_num=1
        mlp_layers = [nn.Linear(dim_in, mlp_dim, bias=b)]
        mlp_layers[-1].xavier_init = xavier_init
        for i in range(1, num_layers):
            if bn_on:
                if global_sync or bn_sync_num > 1:
                    mlp_layers.append(
                        NaiveSyncBatchNorm1d(
                            num_sync_devices=bn_sync_num,
                            global_sync=global_sync,
                            num_features=mlp_dim,
                        )
                    )
                else:
                    mlp_layers.append(nn.BatchNorm1d(num_features=mlp_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            if i == num_layers - 1:
                d = dim_out
                b = bias
            else:
                d = mlp_dim
            mlp_layers.append(nn.Linear(mlp_dim, d, bias=b))
            mlp_layers[-1].xavier_init = xavier_init
        self.projection = nn.Sequential(*mlp_layers)

    def forward(self, x):
        if x.ndim == 5:
            x = x.permute((0, 2, 3, 4, 1))
        if self.flatten:
            x = x.reshape(-1, x.shape[-1])

        return self.projection(x)


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        detach_final_fc=False,
        cfg=None,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            detach_final_fc (bool): if True, detach the fc layer from the
                gradient graph. By doing so, only the final fc layer will be
                trained.
            cfg (struct): The config for the current experiment.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        self.detach_final_fc = detach_final_fc
        self.cfg = cfg
        self.local_projection_modules = []
        self.predictors = nn.ModuleList()
        self.l2norm_feats = False

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        if cfg.CONTRASTIVE.NUM_MLP_LAYERS == 1:
            self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        else:
            self.projection = MLPHead(
                sum(dim_in),
                num_classes,
                cfg.CONTRASTIVE.MLP_DIM,
                cfg.CONTRASTIVE.NUM_MLP_LAYERS,
                bn_on=cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=(
                    cfg.BN.NUM_SYNC_DEVICES if cfg.CONTRASTIVE.BN_SYNC_MLP else 1
                ),
                global_sync=(cfg.CONTRASTIVE.BN_SYNC_MLP and cfg.BN.GLOBAL_SYNC),
            )

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "none":
            self.act = None
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

        if cfg.CONTRASTIVE.PREDICTOR_DEPTHS:
            d_in = num_classes
            for i, n_layers in enumerate(cfg.CONTRASTIVE.PREDICTOR_DEPTHS):
                local_mlp = MLPHead(
                    d_in,
                    num_classes,
                    cfg.CONTRASTIVE.MLP_DIM,
                    n_layers,
                    bn_on=cfg.CONTRASTIVE.BN_MLP,
                    flatten=False,
                    bn_sync_num=(
                        cfg.BN.NUM_SYNC_DEVICES if cfg.CONTRASTIVE.BN_SYNC_MLP else 1
                    ),
                    global_sync=(cfg.CONTRASTIVE.BN_SYNC_MLP and cfg.BN.GLOBAL_SYNC),
                )
                self.predictors.append(local_mlp)

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if self.detach_final_fc:
            x = x.detach()
        if self.l2norm_feats:
            x = nn.functional.normalize(x, dim=1, p=2)

        if (
            x.shape[1:4] == torch.Size([1, 1, 1])
            and self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        ):
            x = x.view(x.shape[0], -1)

        x_proj = self.projection(x)

        time_projs = []
        if self.predictors:
            x_in = x_proj
            for proj in self.predictors:
                time_projs.append(proj(x_in))

        if not self.training:
            if self.act is not None:
                x_proj = self.act(x_proj)
            # Performs fully convlutional inference.
            if x_proj.ndim == 5 and x_proj.shape[1:4] > torch.Size([1, 1, 1]):
                x_proj = x_proj.mean([1, 2, 3])

        x_proj = x_proj.view(x_proj.shape[0], -1)

        if time_projs:
            return [x_proj] + time_projs
        else:
            return x_proj


class X3DHead(nn.Module):
    """
    X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_inner,
        dim_out,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        bn_lin5_on=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        X3DHead takes a 5-dim feature tensor (BxCxTxHxW) as input.

        Args:
            dim_in (float): the channel dimension C of the input.
            num_classes (int): the channel dimensions of the output.
            pool_size (float): a single entry list of kernel size for
                spatiotemporal pooling for the TxHxW dimensions.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            bn_lin5_on (bool): if True, perform normalization on the features
                before the classifier.
        """
        super(X3DHead, self).__init__()
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        self._construct_head(dim_in, dim_inner, dim_out, norm_module)

    def _construct_head(self, dim_in, dim_inner, dim_out, norm_module):

        self.conv_5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.conv_5_bn = norm_module(
            num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt
        )
        self.conv_5_relu = nn.ReLU(self.inplace_relu)

        if self.pool_size is None:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AvgPool3d(self.pool_size, stride=1)

        self.lin_5 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        if self.bn_lin5_on:
            self.lin_5_bn = norm_module(
                num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
            )
        self.lin_5_relu = nn.ReLU(self.inplace_relu)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_out, self.num_classes, bias=True)

        # Softmax for evaluation and testing.
        if self.act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif self.act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(self.act_func)
            )

    def forward(self, inputs):
        # In its current design the X3D head is only useable for a single
        # pathway input.
        assert len(inputs) == 1, "Input tensor does not contain 1 pathway"
        x = self.conv_5(inputs[0])
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)
        x = self.avg_pool(x)

        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cfg=None,
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        if cfg.CONTRASTIVE.NUM_MLP_LAYERS == 1:
            self.projection = nn.Linear(dim_in, num_classes, bias=True)
        else:
            self.projection = MLPHead(
                dim_in,
                num_classes,
                cfg.CONTRASTIVE.MLP_DIM,
                cfg.CONTRASTIVE.NUM_MLP_LAYERS,
                bn_on=cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=(
                    cfg.BN.NUM_SYNC_DEVICES if cfg.CONTRASTIVE.BN_SYNC_MLP else 1
                ),
                global_sync=(cfg.CONTRASTIVE.BN_SYNC_MLP and cfg.BN.GLOBAL_SYNC),
            )
        self.detach_final_fc = cfg.MODEL.DETACH_FINAL_FC

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "none":
            self.act = None
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if self.detach_final_fc:
            x = x.detach()
        x = self.projection(x)

        if not self.training:
            if self.act is not None:
                x = self.act(x)
            # Performs fully convlutional inference.
            if x.ndim == 5 and x.shape[1:4] > torch.Size([1, 1, 1]):
                x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)

        return x


