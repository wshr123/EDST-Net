/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#pragma once        /*！具体实现在cuda内核文件中 .cu */

#include "cpu/ms_deform_attn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif

at::Tensor ms_deform_attn_forward(
        const at::Tensor &value,    /*! 把要用的参数传进来 im2col_step：分块处理的步长，通常用于减少内存消耗*/
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const int im2col_step) {
    if (value.type().is_cuda()) {
#ifdef WITH_CUDA        /*！ 送入前向传播 */
        return ms_deform_attn_cuda_forward(
                value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> ms_deform_attn_backward(
        const at::Tensor &value,
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const at::Tensor &grad_output,
        const int im2col_step) {
    if (value.type().is_cuda()) {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(
                value,
                spatial_shapes,
                level_start_index,
                sampling_loc,
                attn_weight,
                grad_output,
                im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}
