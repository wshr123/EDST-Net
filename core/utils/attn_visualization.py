import core.utils.distribute as du
import core.utils.logging as logging
import core.utils.misc as misc
import core.dataset.loader as loader
from torch.utils.data import Dataset, DataLoader
from core.dataset import build_dataset, get_coco_api_from_dataset
import core.utils.checkpoint as cu
from core.dataset import loader
from core.utils.env import pathmgr
from core.utils.meters import AVAMeter, TestMeter
import core.visualization.tensorboard_vis as tb
import core.model.losses as losses
import numpy as np
import torch
from core.model.my_model import build
import os
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import pickle
import timeit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
logger = logging.get_logger(__name__)


def visualize_feature_activation_simple(x, inputs, output_dir='visualizations', iter_num=0, max_frames=None,
                                        colormap='inferno', vertical_offset=0):
    """
    Visualization with improved alignment and offset correction.

    Args:
        x: Output tensor from the model (or dict containing output)
        inputs: Original input tensor with shape [N, C_in, T, H, W]
        output_dir: Base directory to save visualizations
        iter_num: Current iteration number (for naming folders)
        max_frames: Maximum number of frames to visualize (None=all frames)
        colormap: Matplotlib colormap to use for heatmaps (default: 'inferno')
        vertical_offset: Vertical offset correction in pixels (negative = move up, positive = move down)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    import os
    import traceback
    from scipy.ndimage import shift

    # 为当前批次创建单独的目录
    batch_dir = os.path.join(output_dir, f'batch_{iter_num:04d}')
    frames_dir = os.path.join(batch_dir, 'frames')

    # 创建必要的目录
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    try:
        # 处理x可能是字典的情况
        if isinstance(x, dict):
            print(f"Input is a dictionary with keys: {list(x.keys())}")

            # 尝试找到可能的特征图
            if 'output' in x:
                x = x['output']
            elif 'features' in x:
                x = x['features']
            elif 'attn' in x:
                x = x['attn']
            elif len(x) > 0:
                first_key = list(x.keys())[0]
                print(f"Using value from key: {first_key}")
                x = x[first_key]
            else:
                print("Warning: Empty dictionary passed as x. Cannot visualize.")
                return None

        # 确保x是一个列表的第一个元素
        if isinstance(x, list):
            print(f"Input is a list with {len(x)} elements")
            x = x[0]
        inputs = inputs[0]
        # 打印形状信息以便调试
        print(f"Feature shape: {x.shape}")
        print(f"Input shape: {inputs.shape}")

        # 检查x的维度
        if len(x.shape) < 3:
            print(f"Error: Feature tensor has fewer than 3 dimensions: {x.shape}")
            return None

        # 获取形状
        if len(x.shape) == 5:  # [N, C, T, H, W]
            N, C, T, H, W = x.shape
        elif len(x.shape) == 4:  # [N, C, H, W]
            N, C, H, W = x.shape
            T = 1  # 没有时间维度
        else:
            print(f"Unexpected feature tensor shape: {x.shape}")
            return None

        # 获取输入形状
        _, C_in, T_in, H_in, W_in = inputs.shape

        print(f"T={T}, T_in={T_in}, H={H}, H_in={H_in}, W={W}, W_in={W_in}")

        # 选择一个批次用于可视化
        batch_idx = 0

        # 将输入帧转换为可视化格式
        input_frames = inputs[batch_idx].permute(1, 2, 3, 0).cpu().numpy()

        # 对输入帧进行标准化用于可视化
        input_frames = (input_frames - input_frames.min()) / (input_frames.max() - input_frames.min() + 1e-9)

        # 处理不同维度的特征图
        if len(x.shape) == 5:  # [N, C, T, H, W]
            # 计算特征图的激活强度（通道维度的L2范数）
            feature_activation = torch.norm(x[batch_idx], dim=0)
        elif len(x.shape) == 4:  # [N, C, H, W]
            # 对于2D特征图，计算L2范数后复制到时间维度
            feature_activation = torch.norm(x[batch_idx], dim=0)
            # 创建与时间维度匹配的张量
            temp = torch.zeros((T_in, H, W), device=feature_activation.device)
            for t in range(T_in):
                temp[t] = feature_activation
            feature_activation = temp

        print(f"Feature activation shape after initial processing: {feature_activation.shape}")

        # 调整特征激活的大小（使用更精确的对齐方法）
        try:
            # 如果时间维度不匹配，则调整为输入的时间维度
            if feature_activation.shape[0] != T_in:
                print(f"Time dimension mismatch. Resizing from {feature_activation.shape[0]} to {T_in}")
                # 创建目标大小的张量
                resized_activation = torch.zeros((T_in, feature_activation.shape[1], feature_activation.shape[2]),
                                                 device=feature_activation.device)

                # 复制或插值时间维度
                t_indices = torch.linspace(0, feature_activation.shape[0] - 1, T_in).long()
                for i, t in enumerate(t_indices):
                    resized_activation[i] = feature_activation[t]

                feature_activation = resized_activation

            # 使用更精确的空间维度调整方法
            if feature_activation.shape[1] != H_in or feature_activation.shape[2] != W_in:
                print(f"Spatial dimensions mismatch. Resizing from {feature_activation.shape[1:]} to ({H_in}, {W_in})")

                # 使用更精确的插值方法
                resized_activation = torch.zeros((T_in, H_in, W_in), device=feature_activation.device)

                for t in range(T_in):
                    # 更精确的空间插值，使用align_corners=True以保持边界对齐
                    temp = feature_activation[t].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    temp = F.interpolate(
                        temp,
                        size=(H_in, W_in),
                        mode='bicubic',  # 使用bicubic插值以获得更平滑的结果
                        align_corners=True  # 确保边界对齐
                    )
                    resized_activation[t] = temp.squeeze(0).squeeze(0)  # [H_in, W_in]

                feature_activation = resized_activation
        except Exception as e:
            print(f"Error during feature resizing: {e}")
            traceback.print_exc()
            return None

        print(f"Feature activation shape after resizing: {feature_activation.shape}")

        # 转换为numpy并标准化
        feature_activation = feature_activation.cpu().numpy()
        feature_activation = (feature_activation - feature_activation.min()) / (
                    feature_activation.max() - feature_activation.min() + 1e-9)

        # 应用垂直偏移校正（如果指定）
        if vertical_offset != 0:
            print(f"Applying vertical offset correction: {vertical_offset} pixels")
            corrected_activation = np.zeros_like(feature_activation)
            for t in range(T_in):
                # 使用scipy的shift函数应用垂直偏移
                # 负值表示向上移动，正值表示向下移动
                corrected_activation[t] = shift(feature_activation[t], (vertical_offset, 0), mode='constant', cval=0)

            # 替换为校正后的激活图
            feature_activation = corrected_activation

        # 确定要处理的帧数
        if max_frames is None:
            # 处理所有帧
            frames_to_process = range(T_in)
        else:
            # 选择特定帧
            frames_to_process = np.linspace(0, T_in - 1, min(max_frames, T_in), dtype=int)

        # 保存校正前和校正后的版本，用于比较
        if vertical_offset != 0:
            # 保存原始和校正后的激活图
            with open(os.path.join(batch_dir, 'alignment_info.txt'), 'w') as f:
                f.write(f"Vertical offset applied: {vertical_offset} pixels\n")
                f.write("Negative value = move up, positive value = move down\n")

        # 保存每一帧的单独可视化
        for t in range(T_in):
            # 为当前帧创建可视化
            plt.figure(figsize=(15, 5))

            # 布局: 原始帧 | 热力图 | 叠加图
            # 显示原始输入帧
            plt.subplot(1, 3, 1)
            if C_in == 1:
                plt.imshow(input_frames[t], cmap='gray')
            elif input_frames.shape[3] >= 3:
                plt.imshow(input_frames[t, :, :, :3])
            else:
                plt.imshow(input_frames[t])
            plt.title(f'Original Frame {t}')
            plt.axis('off')

            # 显示激活热力图
            plt.subplot(1, 3, 2)
            plt.imshow(feature_activation[t], cmap=colormap)
            plt.title(f'Activation Map {t}')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')

            # 显示叠加图
            plt.subplot(1, 3, 3)
            if C_in == 1:
                plt.imshow(input_frames[t], cmap='gray')
            elif input_frames.shape[3] >= 3:
                plt.imshow(input_frames[t, :, :, :3])
            else:
                plt.imshow(input_frames[t])
            plt.imshow(feature_activation[t], cmap=colormap, alpha=0.6)
            plt.title(f'Overlay {t}')
            plt.axis('off')

            # 保存当前帧的可视化
            plt.tight_layout()
            plt.savefig(os.path.join(frames_dir, f'frame_{t:03d}.png'), dpi=200)
            plt.close()

            # 单独保存热力图（方便后续分析）
            plt.figure(figsize=(8, 6))
            plt.imshow(feature_activation[t], cmap=colormap)
            plt.colorbar(label='Activation Intensity')
            plt.title(f'Frame {t} Attention Heatmap')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(frames_dir, f'heatmap_{t:03d}.png'), dpi=200)
            plt.close()

        # 创建概览可视化（选择关键帧）
        # 选择几个关键帧进行可视化
        n_frames = min(8, T_in)  # 最多显示8帧，以免图像过于拥挤
        overview_indices = np.linspace(0, T_in - 1, n_frames, dtype=int)

        plt.figure(figsize=(20, 12))
        for i, t in enumerate(overview_indices):
            # 显示原始输入帧
            plt.subplot(3, n_frames, i + 1)
            if C_in == 1:
                plt.imshow(input_frames[t], cmap='gray')
            elif input_frames.shape[3] >= 3:
                plt.imshow(input_frames[t, :, :, :3])
            else:
                plt.imshow(input_frames[t])
            plt.title(f'Frame {t}')
            plt.axis('off')

            # 显示激活图
            plt.subplot(3, n_frames, n_frames + i + 1)
            plt.imshow(feature_activation[t], cmap=colormap)
            plt.title(f'Activation Map {t}')
            plt.axis('off')

            # 显示叠加图
            plt.subplot(3, n_frames, 2 * n_frames + i + 1)
            if C_in == 1:
                plt.imshow(input_frames[t], cmap='gray')
            elif input_frames.shape[3] >= 3:
                plt.imshow(input_frames[t, :, :, :3])
            else:
                plt.imshow(input_frames[t])
            plt.imshow(feature_activation[t], cmap=colormap, alpha=0.6)
            plt.title(f'Overlay {t}')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, 'overview.png'), dpi=200)
        plt.close()

        # 可视化时间维度上的激活强度
        plt.figure(figsize=(12, 8))

        # 计算每个时间步的平均激活强度
        temporal_activation = np.mean(feature_activation, axis=(1, 2))

        print(f"Temporal activation shape: {temporal_activation.shape}, expected: ({T_in},)")

        # 确保维度匹配
        if len(temporal_activation) != T_in:
            print(f"Warning: Temporal activation length ({len(temporal_activation)}) doesn't match T_in ({T_in})")
            # 如果不匹配，则调整到正确的长度
            new_temporal = np.zeros(T_in)
            for i in range(min(len(temporal_activation), T_in)):
                new_temporal[i] = temporal_activation[i]
            temporal_activation = new_temporal

        # 绘制原始的非归一化版本
        plt.subplot(2, 1, 1)
        plt.plot(range(T_in), temporal_activation, 'r-', linewidth=2, marker='o')
        plt.xlabel('Frame Index')
        plt.ylabel('Average Activation')
        plt.title('Temporal Activation Profile (Non-normalized)')
        plt.grid(True)

        # 标记出激活最强的帧
        max_activation_frame = np.argmax(temporal_activation)
        plt.axvline(x=max_activation_frame, color='blue', linestyle='--',
                    label=f'Peak Activation Frame ({max_activation_frame})')
        plt.legend()

        # 计算归一化版本
        normalized_activation = temporal_activation / np.sum(temporal_activation)

        # 绘制归一化版本
        plt.subplot(2, 1, 2)
        plt.plot(range(T_in), normalized_activation, 'g-', linewidth=2, marker='o')
        plt.xlabel('Frame Index')
        plt.ylabel('Normalized Activation')
        plt.title('Temporal Activation Profile (Normalized)')
        plt.grid(True)

        # 标记出激活最强的帧
        plt.axvline(x=max_activation_frame, color='blue', linestyle='--',
                    label=f'Peak Activation Frame ({max_activation_frame})')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, 'temporal_profile.png'), dpi=200)
        plt.close()

        # 可视化空间维度上的激活强度
        plt.figure(figsize=(12, 10))

        # 计算每个空间位置在所有时间步上的平均激活强度
        spatial_activation = np.mean(feature_activation, axis=0)

        plt.imshow(spatial_activation, cmap=colormap)
        plt.colorbar(label='Average Activation')
        plt.title('Spatial Activation Map (Averaged Over Time)')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, 'spatial_activation.png'), dpi=200)
        plt.close()

        # 保存原始视频帧和热力图的数据（方便后续分析）
        np.save(os.path.join(batch_dir, 'input_frames.npy'), input_frames)
        np.save(os.path.join(batch_dir, 'feature_activation.npy'), feature_activation)
        np.save(os.path.join(batch_dir, 'temporal_activation.npy'), temporal_activation)

        # 保存峰值帧信息
        with open(os.path.join(batch_dir, 'peak_info.txt'), 'w') as f:
            f.write(f"Peak activation frame: {max_activation_frame}\n")
            f.write(f"Peak activation value: {temporal_activation[max_activation_frame]}\n")
            f.write(f"Normalized peak value: {normalized_activation[max_activation_frame]}\n")

            # 添加前5个最高激活帧信息
            top_indices = np.argsort(temporal_activation)[-5:][::-1]
            f.write("\nTop 5 activation frames:\n")
            for i, idx in enumerate(top_indices):
                f.write(
                    f"{i + 1}. Frame {idx}: {temporal_activation[idx]} (normalized: {normalized_activation[idx]})\n")

        print(f"Visualization complete for batch {iter_num}. Files saved to {batch_dir}")
        return batch_dir

    except Exception as e:
        print(f"Error during visualization: {e}")
        traceback.print_exc()
        return None
# def visualize_feature_activation_simple(x, inputs, output_dir='visualizations', iter_num=0, max_frames=None,
#                                         colormap='inferno'):
#     """
#     Simplified visualization with enhanced file organization:
#     - Creates a separate folder for each batch
#     - Saves individual frame heatmaps separately
#     - Maintains overview visualizations
#
#     Args:
#         x: Output tensor from the model (or dict containing output)
#         inputs: Original input tensor with shape [N, C_in, T, H, W]
#         output_dir: Base directory to save visualizations
#         iter_num: Current iteration number (for naming folders)
#         max_frames: Maximum number of frames to visualize (None=all frames)
#         colormap: Matplotlib colormap to use for heatmaps (default: 'inferno')
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import torch
#     import torch.nn.functional as F
#     import os
#     import traceback
#
#     # 为当前批次创建单独的目录
#     batch_dir = os.path.join(output_dir, f'batch_{iter_num:04d}')
#     frames_dir = os.path.join(batch_dir, 'frames')
#
#     # 创建必要的目录
#     os.makedirs(batch_dir, exist_ok=True)
#     os.makedirs(frames_dir, exist_ok=True)
#
#     try:
#         # 处理x可能是字典的情况
#         if isinstance(x, dict):
#             print(f"Input is a dictionary with keys: {list(x.keys())}")
#
#             # 尝试找到可能的特征图
#             if 'output' in x:
#                 x = x['output']
#             elif 'features' in x:
#                 x = x['features']
#             elif 'attn' in x:
#                 x = x['attn']
#             elif len(x) > 0:
#                 first_key = list(x.keys())[0]
#                 print(f"Using value from key: {first_key}")
#                 x = x[first_key]
#             else:
#                 print("Warning: Empty dictionary passed as x. Cannot visualize.")
#                 return None
#
#         # 确保x是一个列表的第一个元素
#         if isinstance(x, list):
#             print(f"Input is a list with {len(x)} elements")
#             x = x[0]
#         inputs = inputs[0]
#         # 打印形状信息以便调试
#         print(f"Feature shape: {x.shape}")
#         print(f"Input shape: {inputs.shape}")
#
#         # 检查x的维度
#         if len(x.shape) < 3:
#             print(f"Error: Feature tensor has fewer than 3 dimensions: {x.shape}")
#             return None
#
#         # 获取形状
#         if len(x.shape) == 5:  # [N, C, T, H, W]
#             N, C, T, H, W = x.shape
#         elif len(x.shape) == 4:  # [N, C, H, W]
#             N, C, H, W = x.shape
#             T = 1  # 没有时间维度
#         else:
#             print(f"Unexpected feature tensor shape: {x.shape}")
#             return None
#
#         # 获取输入形状
#         _, C_in, T_in, H_in, W_in = inputs.shape
#
#         print(f"T={T}, T_in={T_in}, H={H}, H_in={H_in}, W={W}, W_in={W_in}")
#
#         # 选择一个批次用于可视化
#         batch_idx = 0
#
#         # 将输入帧转换为可视化格式
#         input_frames = inputs[batch_idx].permute(1, 2, 3, 0).cpu().numpy()
#
#         # 对输入帧进行标准化用于可视化
#         input_frames = (input_frames - input_frames.min()) / (input_frames.max() - input_frames.min() + 1e-9)
#
#         # 处理不同维度的特征图
#         if len(x.shape) == 5:  # [N, C, T, H, W]
#             # 计算特征图的激活强度（通道维度的L2范数）
#             feature_activation = torch.norm(x[batch_idx], dim=0)
#         elif len(x.shape) == 4:  # [N, C, H, W]
#             # 对于2D特征图，计算L2范数后复制到时间维度
#             feature_activation = torch.norm(x[batch_idx], dim=0)
#             # 创建与时间维度匹配的张量
#             temp = torch.zeros((T_in, H, W), device=feature_activation.device)
#             for t in range(T_in):
#                 temp[t] = feature_activation
#             feature_activation = temp
#
#         print(f"Feature activation shape after initial processing: {feature_activation.shape}")
#
#         # 调整特征激活的大小
#         try:
#             # 如果时间维度不匹配，则调整为输入的时间维度
#             if feature_activation.shape[0] != T_in:
#                 print(f"Time dimension mismatch. Resizing from {feature_activation.shape[0]} to {T_in}")
#                 # 创建目标大小的张量
#                 resized_activation = torch.zeros((T_in, feature_activation.shape[1], feature_activation.shape[2]),
#                                                  device=feature_activation.device)
#
#                 # 复制或插值时间维度
#                 t_indices = torch.linspace(0, feature_activation.shape[0] - 1, T_in).long()
#                 for i, t in enumerate(t_indices):
#                     resized_activation[i] = feature_activation[t]
#
#                 feature_activation = resized_activation
#
#             # 如果空间维度不匹配，使用插值调整空间维度
#             if feature_activation.shape[1] != H_in or feature_activation.shape[2] != W_in:
#                 print(f"Spatial dimensions mismatch. Resizing from {feature_activation.shape[1:]} to ({H_in}, {W_in})")
#                 resized_activation = torch.zeros((T_in, H_in, W_in), device=feature_activation.device)
#
#                 for t in range(T_in):
#                     # 使用插值调整空间维度
#                     temp = feature_activation[t].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
#                     temp = F.interpolate(temp, size=(H_in, W_in), mode='bilinear', align_corners=False)
#                     resized_activation[t] = temp.squeeze(0).squeeze(0)  # [H_in, W_in]
#
#                 feature_activation = resized_activation
#         except Exception as e:
#             print(f"Error during feature resizing: {e}")
#             traceback.print_exc()
#             return None
#
#         print(f"Feature activation shape after resizing: {feature_activation.shape}")
#
#         # 转换为numpy并标准化
#         feature_activation = feature_activation.cpu().numpy()
#         feature_activation = (feature_activation - feature_activation.min()) / (
#                     feature_activation.max() - feature_activation.min() + 1e-9)
#
#         # 确定要处理的帧数
#         if max_frames is None:
#             # 处理所有帧
#             frames_to_process = range(T_in)
#         else:
#             # 选择特定帧
#             frames_to_process = np.linspace(0, T_in - 1, min(max_frames, T_in), dtype=int)
#
#         # 保存每一帧的单独可视化
#         for t in range(T_in):
#             # 为当前帧创建可视化
#             plt.figure(figsize=(15, 5))
#
#             # 布局: 原始帧 | 热力图 | 叠加图
#             # 显示原始输入帧
#             plt.subplot(1, 3, 1)
#             if C_in == 1:
#                 plt.imshow(input_frames[t], cmap='gray')
#             elif input_frames.shape[3] >= 3:
#                 plt.imshow(input_frames[t, :, :, :3])
#             else:
#                 plt.imshow(input_frames[t])
#             plt.title(f'Original Frame {t}')
#             plt.axis('off')
#
#             # 显示激活热力图
#             plt.subplot(1, 3, 2)
#             plt.imshow(feature_activation[t], cmap=colormap)
#             plt.title(f'Activation Map {t}')
#             plt.colorbar(fraction=0.046, pad=0.04)
#             plt.axis('off')
#
#             # 显示叠加图
#             plt.subplot(1, 3, 3)
#             if C_in == 1:
#                 plt.imshow(input_frames[t], cmap='gray')
#             elif input_frames.shape[3] >= 3:
#                 plt.imshow(input_frames[t, :, :, :3])
#             else:
#                 plt.imshow(input_frames[t])
#             plt.imshow(feature_activation[t], cmap=colormap, alpha=0.6)
#             plt.title(f'Overlay {t}')
#             plt.axis('off')
#
#             # 保存当前帧的可视化
#             plt.tight_layout()
#             plt.savefig(os.path.join(frames_dir, f'frame_{t:03d}.png'), dpi=200)
#             plt.close()
#
#             # 单独保存热力图（方便后续分析）
#             plt.figure(figsize=(8, 6))
#             plt.imshow(feature_activation[t], cmap=colormap)
#             plt.colorbar(label='Activation Intensity')
#             plt.title(f'Frame {t} Attention Heatmap')
#             plt.axis('off')
#             plt.tight_layout()
#             plt.savefig(os.path.join(frames_dir, f'heatmap_{t:03d}.png'), dpi=200)
#             plt.close()
#
#         # 创建概览可视化（选择关键帧）
#         # 选择几个关键帧进行可视化
#         n_frames = min(8, T_in)  # 最多显示8帧，以免图像过于拥挤
#         overview_indices = np.linspace(0, T_in - 1, n_frames, dtype=int)
#
#         plt.figure(figsize=(20, 12))
#         for i, t in enumerate(overview_indices):
#             # 显示原始输入帧
#             plt.subplot(3, n_frames, i + 1)
#             if C_in == 1:
#                 plt.imshow(input_frames[t], cmap='gray')
#             elif input_frames.shape[3] >= 3:
#                 plt.imshow(input_frames[t, :, :, :3])
#             else:
#                 plt.imshow(input_frames[t])
#             plt.title(f'Frame {t}')
#             plt.axis('off')
#
#             # 显示激活图
#             plt.subplot(3, n_frames, n_frames + i + 1)
#             plt.imshow(feature_activation[t], cmap=colormap)
#             plt.title(f'Activation Map {t}')
#             plt.axis('off')
#
#             # 显示叠加图
#             plt.subplot(3, n_frames, 2 * n_frames + i + 1)
#             if C_in == 1:
#                 plt.imshow(input_frames[t], cmap='gray')
#             elif input_frames.shape[3] >= 3:
#                 plt.imshow(input_frames[t, :, :, :3])
#             else:
#                 plt.imshow(input_frames[t])
#             plt.imshow(feature_activation[t], cmap=colormap, alpha=0.6)
#             plt.title(f'Overlay {t}')
#             plt.axis('off')
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(batch_dir, 'overview.png'), dpi=200)
#         plt.close()
#
#         # 可视化时间维度上的激活强度
#         plt.figure(figsize=(12, 8))
#
#         # 计算每个时间步的平均激活强度
#         temporal_activation = np.mean(feature_activation, axis=(1, 2))
#
#         print(f"Temporal activation shape: {temporal_activation.shape}, expected: ({T_in},)")
#
#         # 确保维度匹配
#         if len(temporal_activation) != T_in:
#             print(f"Warning: Temporal activation length ({len(temporal_activation)}) doesn't match T_in ({T_in})")
#             # 如果不匹配，则调整到正确的长度
#             new_temporal = np.zeros(T_in)
#             for i in range(min(len(temporal_activation), T_in)):
#                 new_temporal[i] = temporal_activation[i]
#             temporal_activation = new_temporal
#
#         # 绘制原始的非归一化版本
#         plt.subplot(2, 1, 1)
#         plt.plot(range(T_in), temporal_activation, 'r-', linewidth=2, marker='o')
#         plt.xlabel('Frame Index')
#         plt.ylabel('Average Activation')
#         plt.title('Temporal Activation Profile (Non-normalized)')
#         plt.grid(True)
#
#         # 标记出激活最强的帧
#         max_activation_frame = np.argmax(temporal_activation)
#         plt.axvline(x=max_activation_frame, color='blue', linestyle='--',
#                     label=f'Peak Activation Frame ({max_activation_frame})')
#         plt.legend()
#
#         # 计算归一化版本
#         normalized_activation = temporal_activation / np.sum(temporal_activation)
#
#         # 绘制归一化版本
#         plt.subplot(2, 1, 2)
#         plt.plot(range(T_in), normalized_activation, 'g-', linewidth=2, marker='o')
#         plt.xlabel('Frame Index')
#         plt.ylabel('Normalized Activation')
#         plt.title('Temporal Activation Profile (Normalized)')
#         plt.grid(True)
#
#         # 标记出激活最强的帧
#         plt.axvline(x=max_activation_frame, color='blue', linestyle='--',
#                     label=f'Peak Activation Frame ({max_activation_frame})')
#         plt.legend()
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(batch_dir, 'temporal_profile.png'), dpi=200)
#         plt.close()
#
#         # 可视化空间维度上的激活强度
#         plt.figure(figsize=(12, 10))
#
#         # 计算每个空间位置在所有时间步上的平均激活强度
#         spatial_activation = np.mean(feature_activation, axis=0)
#
#         plt.imshow(spatial_activation, cmap=colormap)
#         plt.colorbar(label='Average Activation')
#         plt.title('Spatial Activation Map (Averaged Over Time)')
#         plt.axis('off')
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(batch_dir, 'spatial_activation.png'), dpi=200)
#         plt.close()
#
#         # 保存原始视频帧和热力图的数据（方便后续分析）
#         np.save(os.path.join(batch_dir, 'input_frames.npy'), input_frames)
#         np.save(os.path.join(batch_dir, 'feature_activation.npy'), feature_activation)
#         np.save(os.path.join(batch_dir, 'temporal_activation.npy'), temporal_activation)
#
#         # 保存峰值帧信息
#         with open(os.path.join(batch_dir, 'peak_info.txt'), 'w') as f:
#             f.write(f"Peak activation frame: {max_activation_frame}\n")
#             f.write(f"Peak activation value: {temporal_activation[max_activation_frame]}\n")
#             f.write(f"Normalized peak value: {normalized_activation[max_activation_frame]}\n")
#
#             # 添加前5个最高激活帧信息
#             top_indices = np.argsort(temporal_activation)[-5:][::-1]
#             f.write("\nTop 5 activation frames:\n")
#             for i, idx in enumerate(top_indices):
#                 f.write(
#                     f"{i + 1}. Frame {idx}: {temporal_activation[idx]} (normalized: {normalized_activation[idx]})\n")
#
#         print(f"Visualization complete for batch {iter_num}. Files saved to {batch_dir}")
#         return batch_dir
#
#     except Exception as e:
#         print(f"Error during visualization: {e}")
#         traceback.print_exc()
#         return None

def visualize_feature_activation(x, inputs):
    """
    Visualize feature activations in the output tensor as a proxy for attention.

    Args:
        x: Output tensor from the model with shape [N, C, T, H, W]
        inputs: Original input tensor with shape [N, C_in, T, H, W]
    """
    if isinstance(x, list):
        x = x[0]
    inputs = inputs[0]
    N, C, T, H, W = x.shape
    _, C_in, T_in, H_in, W_in = inputs.shape
    batch_idx = 0
    input_frames = inputs[batch_idx].permute(1, 2, 3, 0).cpu().numpy()
    input_frames = (input_frames - input_frames.min()) / (input_frames.max() - input_frames.min() + 1e-9)

    # 计算特征图的激活强度（通道维度的L2范数）
    # 这可以视为每个时空位置的"注意力"或重要性
    feature_activation = torch.norm(x[batch_idx], dim=0).cpu().numpy()

    # 如果输入和输出的空间尺寸不同，将激活图调整为与输入相同的尺寸
    if T != T_in or H != H_in or W != W_in:
        # 创建用于调整大小的numpy数组
        resized_activation = np.zeros((T_in, H_in, W_in))

        # 对每个时间步进行调整
        for t in range(min(T, T_in)):
            # 使用简单的双线性插值调整空间尺寸
            resized_activation[t] = np.array(Image.fromarray(feature_activation[t]).resize((W_in, H_in)))

        feature_activation = resized_activation

    # 将激活图标准化到0-1范围
    feature_activation = (feature_activation - feature_activation.min()) / (
                feature_activation.max() - feature_activation.min() + 1e-9)

    # 创建一个时空激活可视化
    plt.figure(figsize=(20, 12))

    # 选择几个关键帧进行可视化
    n_frames = min(8, T)
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    for i, t in enumerate(frame_indices):
        # 显示原始输入帧
        plt.subplot(3, n_frames, i + 1)
        if C_in == 1:
            plt.imshow(input_frames[t], cmap='gray')
        else:
            plt.imshow(input_frames[t, :, :, :3] if input_frames.shape[3] >= 3 else input_frames[t])
        plt.title(f'Frame {t}')
        plt.axis('off')

        # 显示激活图
        plt.subplot(3, n_frames, n_frames + i + 1)
        plt.imshow(feature_activation[t], cmap='hot')
        plt.title(f'Activation Map {t}')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        # 显示叠加图
        plt.subplot(3, n_frames, 2 * n_frames + i + 1)
        if C_in == 1:
            plt.imshow(input_frames[t], cmap='gray')
        else:
            plt.imshow(input_frames[t, :, :, :3] if input_frames.shape[3] >= 3 else input_frames[t])
        plt.imshow(feature_activation[t], cmap='hot', alpha=0.5)
        plt.title(f'Overlay {t}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('feature_activation.png', dpi=200)
    plt.close()

    # 可视化时间维度上的激活强度
    plt.figure(figsize=(12, 6))

    # 计算每个时间步的平均激活强度
    temporal_activation = np.mean(feature_activation, axis=(1, 2))

    plt.plot(range(T), temporal_activation, 'r-', linewidth=2, marker='o')
    plt.xlabel('Frame Index')
    plt.ylabel('Average Activation')
    plt.title('Temporal Activation Profile')
    plt.grid(True)

    # 标记出激活最强的帧
    max_activation_frame = np.argmax(temporal_activation)
    plt.axvline(x=max_activation_frame, color='blue', linestyle='--',
                label=f'Peak Activation Frame ({max_activation_frame})')
    plt.legend()

    plt.tight_layout()
    plt.savefig('temporal_activation_profile.png', dpi=200)
    plt.close()

    # 可视化空间维度上的激活强度
    plt.figure(figsize=(12, 10))

    # 计算每个空间位置在所有时间步上的平均激活强度
    spatial_activation = np.mean(feature_activation, axis=0)

    plt.imshow(spatial_activation, cmap='hot')
    plt.colorbar(label='Average Activation')
    plt.title('Spatial Activation Map (Averaged Over Time)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('spatial_activation_map.png', dpi=200)
    plt.close()

    # 创建一个动态时空可视化（保存多个帧作为序列）
    for t in range(T):
        plt.figure(figsize=(12, 5))

        # 显示原始输入帧
        plt.subplot(1, 2, 1)
        if C_in == 1:
            plt.imshow(input_frames[t], cmap='gray')
        else:
            plt.imshow(input_frames[t, :, :, :3] if input_frames.shape[3] >= 3 else input_frames[t])
        plt.title(f'Input Frame {t}')
        plt.axis('off')

        # 显示带有激活叠加的帧
        plt.subplot(1, 2, 2)
        if C_in == 1:
            plt.imshow(input_frames[t], cmap='gray')
        else:
            plt.imshow(input_frames[t, :, :, :3] if input_frames.shape[3] >= 3 else input_frames[t])
        plt.imshow(feature_activation[t], cmap='hot', alpha=0.5)
        plt.title(f'Activation Overlay (Frame {t})')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'frame_activation_{t:03d}.png', dpi=150)
        plt.close()

    print(
        f"Visualizations saved. Main overview in 'feature_activation.png', 'temporal_activation_profile.png', and 'spatial_activation_map.png'.")
    print(f"Frame-by-frame visualizations saved as 'frame_activation_XXX.png'.")

    # 额外：如果你想分析不同通道的贡献
    plt.figure(figsize=(12, 6))

    # 计算每个通道的平均激活强度
    channel_activation = torch.mean(x[batch_idx], dim=(1, 2, 3)).cpu().numpy()

    # 找出最活跃的前10个通道
    top_channels = np.argsort(channel_activation)[-10:][::-1]

    # 绘制通道激活柱状图
    plt.bar(top_channels, channel_activation[top_channels])
    plt.xlabel('Channel Index')
    plt.ylabel('Average Activation')
    plt.title('Top 10 Most Active Channels')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('top_channels.png', dpi=200)
    plt.close()

    print(f"Channel analysis saved as 'top_channels.png'.")

def visualize_temporal_attention_on_tensor_frames(attn_dict, inputs):
    """
    Visualize temporal deformable attention on video frames.

    Args:
        attn_dict: Dictionary containing attention weights and sampling locations
        inputs: Input tensor with shape [N, C, T, H, W]
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn.functional as F

    # Get attention weights and sampling locations
    attention_weights = attn_dict['attention_weights']  # [N, Len_q, n_heads, n_levels, n_points]
    sampling_locations = attn_dict['sampling_locations']  # [N, Len_q, n_heads, n_levels, n_points, 2]
    inputs = inputs[0]
    # Get shapes
    N, C, T, H, W = inputs.shape

    # Select batch item 0 for visualization
    batch_idx = 0

    # Extract frames from the input
    frames = inputs[batch_idx].permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]

    # Normalize frames for visualization
    frames = (frames - frames.min()) / (frames.max() - frames.min())

    # Average attention weights across heads and points
    avg_attention = attention_weights[batch_idx].mean(dim=(1, 2, 3)).cpu().numpy()  # [Len_q]

    # The attention map needs to be interpolated to match the time dimension
    # Reshape averaged attention to temporal dimension
    attention_timeline = np.zeros(T)
    query_length = avg_attention.shape[0]

    # Map the query positions to temporal dimension (assuming Len_q = T*H*W/H/W = T)
    # If your query organization is different, adjust accordingly
    if query_length == T:
        attention_timeline = avg_attention
    else:
        # Interpolate if query length doesn't match temporal dimension
        attention_timeline = np.interp(
            np.linspace(0, 1, T),
            np.linspace(0, 1, query_length),
            avg_attention
        )

    # Normalize attention for visualization
    attention_timeline = (attention_timeline - attention_timeline.min()) / (
                attention_timeline.max() - attention_timeline.min() + 1e-9)

    # Create visualization with attention heatmap
    plt.figure(figsize=(20, 10))

    # Plot attention timeline
    plt.subplot(2, 1, 1)
    plt.plot(attention_timeline, 'r-', linewidth=2)
    plt.xlabel('Frame Index')
    plt.ylabel('Attention Weight')
    plt.title('Temporal Attention Distribution')
    plt.grid(True)

    # Create a grid of frames with attention overlay
    n_cols = min(8, T)
    n_rows = (T + n_cols - 1) // n_cols
    plt.subplot(2, 1, 2)

    for t in range(T):
        plt.subplot(2 + n_rows, n_cols, n_cols + t + 1)

        # Get the frame and convert to RGB if needed
        frame = frames[t]
        if C == 1:
            frame = np.repeat(frame[..., np.newaxis], 3, axis=2)
        elif C == 3:
            pass  # Already RGB
        else:
            # Take first 3 channels or average them
            frame = frames[t, :, :, :3]

        # Create a red overlay based on attention weight
        overlay = np.zeros_like(frame)
        overlay[:, :, 0] = attention_timeline[t]  # Red channel

        # Blend original frame with attention overlay
        alpha = 0.5  # Blend factor
        blended = (1 - alpha) * frame + alpha * overlay

        plt.imshow(blended)
        plt.title(f'Frame {t}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'temporal_attention_on_frames_batch{batch_idx}.png')
    plt.close()
    print(f"Visualization saved as temporal_attention_on_frames_batch{batch_idx}.png")
def tensor_to_image(tensor_input, denormalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(tensor_input, list):
        tensor = tensor_input[0]
    else:
        tensor = tensor_input
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, but got {type(tensor)}")
    if tensor.dim() == 5:  # (B, C, T, H, W)
        tensor = tensor[0, :, 0]
    elif tensor.dim() == 4:  # (B, C, H, W)
        tensor = tensor[0]
    elif tensor.dim() != 3:  # 不是(C, H, W)
        raise ValueError(f"Unsupported tensor dimensions: {tensor.dim()}")
    img_tensor = tensor.clone().detach().cpu()
    if denormalize:
        mean = torch.tensor(mean, device='cpu').view(-1, 1, 1)
        std = torch.tensor(std, device='cpu').view(-1, 1, 1)
        img_tensor = img_tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_array = img_tensor.numpy()
    img_array = np.transpose(img_array, (1, 2, 0))
    return img_array


def visualize_complete_deformable_attention(attn_dict, image_tensor, batch_idx=0, time_idx=0, head_idx=0, level_idx=0,
                                            figsize=(10, 10), grid_density=6, point_indices=None, denormalize=True,
                                            cmap='jet', alpha=0.6):
    """
    完整可视化可变形注意力，包括原始图像和特征图注意力

    参数:
    - attn_dict: 注意力的字典
    - image_tensor: 原始图像张量
    - batch_idx: 要可视化的批次索引
    - time_idx: 要可视化的时间步索引
    - head_idx: 要可视化的注意力头索引
    - level_idx: 要可视化的特征层级索引
    - figsize: 图像大小
    - grid_density: 采样网格的密度，值越大采样点越少
    - point_indices: 要可视化的采样点索引，如果为None则自动选择部分采样点
    - denormalize: 是否反归一化图像张量
    - cmap: 热图颜色映射
    - alpha: 热图透明度
    """
    image_tensor = image_tensor[0]
    # 从张量中提取原始图像
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.dim() == 5:  # (B, C, T, H, W)
            original_tensor = image_tensor[batch_idx, :, time_idx]
        elif image_tensor.dim() == 4:  # (B, C, H, W)
            original_tensor = image_tensor[batch_idx]
        else:
            original_tensor = image_tensor

        # 转换为numpy图像
        original_image = tensor_to_image(original_tensor, denormalize=denormalize)
    else:
        # 如果不是张量，假设它已经是numpy数组
        original_image = image_tensor

    # 获取原始图像尺寸
    orig_h, orig_w = original_image.shape[:2]

    # 获取注意力信息
    attention_weights = attn_dict['attention_weights']  # (N, Len_q, n_heads, n_levels*n_points)
    sampling_locations = attn_dict['sampling_locations']  # (N, Len_q, n_heads, n_levels, n_points, 2)
    input_spatial_shapes = attn_dict['input_spatial_shapes']  # (n_levels, 2)

    # 获取特征图尺寸
    h, w = input_spatial_shapes[level_idx].cpu().numpy()

    # 计算原始图像与特征图之间的缩放比例
    scale_y = orig_h / h
    scale_x = orig_w / w

    # 获取当前批次、头部的注意力权重和采样位置
    n_levels = sampling_locations.shape[3]
    n_points = sampling_locations.shape[4]

    global_idx = batch_idx * h * w + time_idx

    # 重塑注意力权重
    attn_weights = attention_weights[global_idx, :, head_idx].reshape(h, w, n_levels * n_points).cpu().numpy()

    # 获取采样位置
    sample_locs = sampling_locations[global_idx, :, head_idx].reshape(h, w, n_levels, n_points, 2).cpu().numpy()

    # 如果没有指定要可视化的采样点，则自动选择一些
    if point_indices is None:
        # 如果采样点太多，只选择部分
        if n_points > 4:
            point_indices = list(range(0, n_points, n_points // 4))[:4]  # 最多选择4个
        else:
            point_indices = list(range(n_points))

    # 为特定层级计算平均注意力权重
    level_start = level_idx * n_points
    level_end = (level_idx + 1) * n_points
    avg_attn_level = np.mean(attn_weights[:, :, level_start:level_end], axis=2)

    # 调整热图大小以匹配原始图像
    avg_attn_resized = cv2.resize(avg_attn_level, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 创建可视化 - 2x3布局
    fig = plt.figure(figsize=figsize)

    # 1. 原始图像
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    ax1.axis('on')

    # 2. 特征图尺寸上的注意力热图
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    im2 = ax2.imshow(avg_attn_level, cmap=cmap)
    ax2.set_title(f'Attention Map (Feature Size, Level {level_idx}, Head {head_idx})')
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Height')
    plt.colorbar(im2, ax=ax2)

    # 3. 注意力热图覆盖在原始图像上
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax3.imshow(original_image)
    im3 = ax3.imshow(avg_attn_resized, cmap=cmap, alpha=alpha)
    ax3.set_title('Attention Heatmap Overlay')
    ax3.set_xlabel('Width')
    ax3.set_ylabel('Height')
    plt.colorbar(im3, ax=ax3)

    # 4. 各采样点的独立注意力图
    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=2)

    # 创建子区域来显示各采样点的注意力
    num_points = len(point_indices)
    grid_size = int(np.ceil(np.sqrt(num_points)))
    inner_grid = ax4.inset_axes([0.05, 0.05, 0.9, 0.9])
    inner_grid.set_axis_off()

    for i, p_idx in enumerate(point_indices):
        # 计算当前点在网格中的位置
        row = i // grid_size
        col = i % grid_size

        # 创建子图
        inner_ax = inner_grid.inset_axes(
            [col / grid_size, 1 - 1 / grid_size - row / grid_size, 1 / grid_size, 1 / grid_size]
        )

        # 获取特定采样点的注意力权重
        point_attn = attn_weights[:, :, level_idx * n_points + p_idx]
        inner_ax.imshow(point_attn, cmap=cmap)
        inner_ax.set_title(f'Point {p_idx}')
        inner_ax.set_xticks([])
        inner_ax.set_yticks([])

    ax4.set_title(f'Individual Sampling Points Attention (Level {level_idx}, Head {head_idx})')
    ax4.set_axis_off()

    # 5. 在原始图像上可视化采样点和连接线
    ax5 = plt.subplot2grid((2, 3), (1, 2))
    ax5.imshow(original_image)

    # 为了清晰起见，只选择网格上的某些位置
    grid_step = max(1, min(h, w) // grid_density)
    y_indices = np.arange(0, h, grid_step)
    x_indices = np.arange(0, w, grid_step)

    # 为每个采样点分配不同颜色
    point_cmap = plt.cm.get_cmap('hsv', len(point_indices))

    # 获取注意力权重的最大值，用于归一化
    max_weight = np.max(attn_weights[:, :, [level_idx * n_points + p for p in point_indices]])

    # 绘制采样点和连接线
    legend_handles = []
    legend_labels = []

    for i, y_idx in enumerate(y_indices):
        for j, x_idx in enumerate(x_indices):
            # 只绘制一部分查询点，使可视化更清晰
            if (i + j) % 2 != 0:  # 棋盘模式，减少密度
                continue

            # 将特征图坐标转换为原始图像坐标
            img_y = int(y_idx * scale_y)
            img_x = int(x_idx * scale_x)

            # 参考点（查询位置）
            ax5.plot(img_x, img_y, 'wo', markersize=5, alpha=0.7)

            # 对于该查询位置的选定采样点
            for p_i, p_idx in enumerate(point_indices):
                # 获取归一化的采样位置，转换为原始图像坐标
                norm_y, norm_x = sample_locs[y_idx, x_idx, level_idx, p_idx]
                pixel_x = int(norm_x * (orig_w - 1))
                pixel_y = int(norm_y * (orig_h - 1))

                # 获取该采样点的权重
                weight = attn_weights[y_idx, x_idx, level_idx * n_points + p_idx]

                # 根据权重调整点的大小和透明度
                weight_norm = weight / max_weight
                size = 1 + 7 * weight_norm
                alpha_val = 0.3 + 0.7 * weight_norm

                # 绘制采样点和连接线
                point_color = point_cmap(p_i)
                point = ax5.plot(pixel_x, pixel_y, 'o', color=point_color, markersize=size, alpha=alpha_val)[0]
                ax5.plot([img_x, pixel_x], [img_y, pixel_y], '-', color=point_color, linewidth=0.5,
                         alpha=0.5 * alpha_val)

                # 只添加一次到图例
                if i == y_indices[0] and j == x_indices[0]:
                    legend_handles.append(point)
                    legend_labels.append(f'Point {p_idx}')

    ax5.set_title('Sampling Points on Image')
    ax5.set_xlim(0, orig_w - 1)
    ax5.set_ylim(orig_h - 1, 0)  # 反转y轴，使得(0,0)在左上角

    # 添加图例
    ax5.legend(legend_handles, legend_labels, loc='lower right', fontsize='small')

    plt.tight_layout()
    plt.show()

    return fig


def visualize_temporal_attention(attn_dict, inputs):
    """
    Visualize temporal deformable attention.

    Args:
        attn_dict: Dictionary containing attention weights and sampling locations
        inputs: Input tensor with shape [N, C, T, H, W]
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get attention weights and sampling locations
    attention_weights = attn_dict['attention_weights']  # [N, Len_q, n_heads, n_levels, n_points]
    sampling_locations = attn_dict['sampling_locations']  # [N, Len_q, n_heads, n_levels, n_points, 2]
    reference_points = attn_dict['reference_points']  # [N, Len_q, n_levels, 2]
    inputs = inputs[0]
    # Get shapes
    N, C, T, H, W = inputs.shape

    # Select a batch item, head, and level to visualize
    batch_idx = 0
    head_idx = 0
    level_idx = 0

    # Extract the temporal dimension attention (assuming the first dimension of sampling_locations[..., 0] is time)
    temporal_attn = attention_weights[batch_idx, :, head_idx, level_idx, :]  # [Len_q, n_points]
    temporal_sampling = sampling_locations[batch_idx, :, head_idx, level_idx, :, 0]  # [Len_q, n_points]

    # Create a visualization
    plt.figure(figsize=(15, 10))

    # Create a heatmap of attention weights across query positions and sampling points
    plt.subplot(2, 1, 1)
    plt.imshow(temporal_attn.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Sampling Point Index')
    plt.ylabel('Query Position (Time)')
    plt.title(f'Temporal Attention Weights (Batch {batch_idx}, Head {head_idx})')

    # Visualize sampling locations
    plt.subplot(2, 1, 2)
    for point_idx in range(temporal_sampling.shape[1]):
        plt.scatter(
            np.arange(len(temporal_sampling)),  # Query positions
            temporal_sampling[:, point_idx].cpu().numpy() * T,  # Convert normalized coords to actual time indices
            label=f'Point {point_idx}',
            alpha=0.7,
            s=50
        )
    plt.xlabel('Query Position (Time)')
    plt.ylabel('Sampled Time Position')
    plt.title(f'Temporal Sampling Locations (Batch {batch_idx}, Head {head_idx})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'temporal_attention_batch{batch_idx}_head{head_idx}.png')
    plt.close()

    # Also visualize attention across all heads
    plt.figure(figsize=(15, 10))
    avg_attn_across_heads = attention_weights[batch_idx, :, :, level_idx, :].mean(
        dim=-1).cpu().numpy()  # [Len_q, n_heads]
    plt.imshow(avg_attn_across_heads, aspect='auto', cmap='viridis')
    plt.colorbar(label='Average Attention Weight')
    plt.xlabel('Attention Head')
    plt.ylabel('Query Position (Time)')
    plt.title(f'Temporal Attention Across Heads (Batch {batch_idx})')
    plt.tight_layout()
    plt.savefig(f'temporal_attention_across_heads_batch{batch_idx}.png')
    plt.close()

    print(
        f"Visualizations saved as temporal_attention_batch{batch_idx}_head{head_idx}.png and temporal_attention_across_heads_batch{batch_idx}.png")
def visualize_top_deformable_attention(attn_dict, video_frames, batch_idx=0, figsize=(15, 6)):
    """
    可视化时间deformable attention，自动选择最具信息量的注意力头

    参数:
    - attn_dict: 注意力字典
    - video_frames: 视频帧张量或列表
    - batch_idx: 批次索引
    - figsize: 图表大小

    返回:
    - fig: 图表对象
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # 处理视频帧
    processed_frames = []

    # 如果是5D张量 (B, C, T, H, W)
    if isinstance(video_frames, torch.Tensor) and video_frames.dim() == 5:
        num_frames = video_frames.shape[2]
        for i in range(num_frames):
            frame = video_frames[batch_idx, :, i].cpu()
            # 转换为numpy数组并调整为(H,W,C)格式
            if frame.shape[0] == 3:  # 如果是RGB
                frame = frame.permute(1, 2, 0)
            # 缩放到[0,1]范围
            if frame.max() > 1:
                frame = frame / 255.0
            processed_frames.append(frame.numpy())
    # 如果是帧列表
    elif isinstance(video_frames, list):
        for frame in video_frames:
            if isinstance(frame, torch.Tensor):
                if frame.dim() == 4:  # (B, C, H, W)
                    frame = frame[batch_idx]
                # 转换为numpy数组并调整为(H,W,C)格式
                if frame.shape[0] == 3:  # 如果是RGB
                    frame = frame.permute(1, 2, 0)
                # 缩放到[0,1]范围
                if frame.max() > 1:
                    frame = frame / 255.0
                processed_frames.append(frame.cpu().numpy())
            elif isinstance(frame, np.ndarray):
                processed_frames.append(frame)

    num_frames = len(processed_frames)
    print(f"处理了 {num_frames} 帧视频")

    # 获取注意力权重
    attention_weights = attn_dict['attention_weights']  # 通常形状为(B*H*W, T, n_heads, n_points)

    # 获取注意力头和特征层级信息
    n_heads = attention_weights.shape[2] if len(attention_weights.shape) > 2 else 1

    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"注意力头数量: {n_heads}")

    # 计算每个注意力头的信息量
    head_information = []

    # 获取特征图尺寸 (通常是 spatial_shapes)
    if 'input_spatial_shapes' in attn_dict:
        input_spatial_shapes = attn_dict['input_spatial_shapes']
        if len(input_spatial_shapes.shape) > 1 and input_spatial_shapes.shape[1] == 2:
            h, w = input_spatial_shapes[0].cpu().numpy()
            print(f"特征图尺寸: h={h}, w={w}")
            num_spatial_positions = h * w
        else:
            print("无法确定特征图尺寸，使用默认值")
            num_spatial_positions = 100  # 默认值
    else:
        print("注意力字典中没有空间形状信息，使用默认值")
        num_spatial_positions = 100  # 默认值

    # 选择一个代表性的空间位置（中心附近）
    center_pos = num_spatial_positions // 2

    try:
        # 计算每个头的注意力分布方差（作为信息量的代理）
        for head_idx in range(n_heads):
            head_attentions = []

            # 收集几个不同空间位置的注意力
            sample_positions = [center_pos]  # 可以添加更多位置

            for pos_idx in sample_positions:
                global_idx = batch_idx * num_spatial_positions + pos_idx

                # 检查索引有效性
                if global_idx >= attention_weights.shape[0]:
                    continue

                # 提取该位置、该头的时间注意力
                try:
                    if len(attention_weights.shape) == 4:  # (B*HW, T, n_heads, n_points)
                        # 对采样点维度取平均
                        attn = attention_weights[global_idx, :, head_idx].mean(dim=-1).cpu().numpy()
                    elif len(attention_weights.shape) == 3:  # (B*HW, T, n_heads)
                        attn = attention_weights[global_idx, :, head_idx].cpu().numpy()
                    else:
                        attn = attention_weights[global_idx].cpu().numpy()

                    # 确保长度匹配帧数
                    if len(attn) != num_frames and len(attn) > 0:
                        # 如果注意力和帧数不匹配，尝试调整
                        if len(attn) > num_frames:
                            attn = attn[:num_frames]
                        else:
                            temp_attn = np.zeros(num_frames)
                            temp_attn[:len(attn)] = attn
                            attn = temp_attn

                    head_attentions.append(attn)
                except Exception as e:
                    print(f"提取头 {head_idx} 的注意力时出错: {e}")

            # 如果成功收集了注意力
            if head_attentions:
                # 计算平均注意力
                avg_attn = np.mean(head_attentions, axis=0)

                # 计算方差作为信息量
                variance = np.var(avg_attn)

                # 计算最大最小差异作为对比度
                contrast = np.max(avg_attn) - np.min(avg_attn)

                # 总信息分数
                info_score = variance * contrast

                head_information.append((head_idx, info_score, avg_attn))

                print(f"头 {head_idx} - 方差: {variance:.6f}, 对比度: {contrast:.6f}, 信息分数: {info_score:.6f}")

    except Exception as e:
        print(f"计算头信息量时出错: {e}")
        # 创建一个默认头
        default_attn = np.ones(num_frames) / num_frames
        head_information = [(0, 0.0, default_attn)]

    # 根据信息量排序
    head_information.sort(key=lambda x: x[1], reverse=True)

    # 选择信息量最大的头
    top_head_idx, top_info_score, top_attention = head_information[0]

    print(f"选择了信息量最大的头: {top_head_idx}, 信息分数: {top_info_score:.6f}")

    # 创建可视化
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})

    # 1. 上方显示视频帧
    frame_width = 1.0 / num_frames

    for i in range(num_frames):
        # 为每一帧创建子坐标系
        frame_ax = fig.add_axes([i * frame_width, 0.4, frame_width, 0.5])

        # 显示帧
        frame_ax.imshow(processed_frames[i])
        frame_ax.set_title(f"t={i}")
        frame_ax.axis('off')

    # 2. 下方显示注意力条形图
    axes[1].bar(range(num_frames), top_attention, color='skyblue', alpha=0.7)

    # 添加数值标签
    for i, v in enumerate(top_attention):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)

    axes[1].set_xticks(range(num_frames))
    axes[1].set_xticklabels([f"t={i}" for i in range(num_frames)])
    axes[1].set_xlabel('时间步')
    axes[1].set_ylabel('注意力权重')
    axes[1].set_title(f'头 {top_head_idx} 的时间注意力分布 (信息分数: {top_info_score:.4f})')

    # 调整布局
    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3)

    # 隐藏上方子图的坐标轴
    axes[0].axis('off')

    plt.show()

    return fig
