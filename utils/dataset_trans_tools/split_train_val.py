# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# # 读取 CSV 文件
# csv_file = '/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/ava_v2.2-beifen/ava_train_set.csv'  # 替换为你的 CSV 文件名
# df = pd.read_csv(csv_file, header=None)
#
# # 给列命名，方便后续操作
# df.columns = ['video_name', 'frame_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'action_id', 'track_id']
#
# # 打印数据的前几行，检查是否正确读取
# print("Original dataset preview:")
# print(df.head())
#
# # Step 1: 按视频名称进行分组
# video_names = df['video_name'].unique()  # 获取所有唯一的视频名称
#
# # Step 2: 按 6.5:1.5 比例划分视频名称
# train_videos, val_videos = train_test_split(video_names, test_size=0.1875, random_state=42)
#
# print(f"\nNumber of training videos: {len(train_videos)}")
# print(f"Number of validation videos: {len(val_videos)}")
#
# # Step 3: 将数据按划分的视频名称分别放入训练集和验证集
# train_df = df[df['video_name'].isin(train_videos)]
# val_df = df[df['video_name'].isin(val_videos)]
#
# # Step 4: 检查训练集和验证集中的动作分布，确保每个动作类别都存在
# train_action_distribution = train_df['action_id'].value_counts()
# val_action_distribution = val_df['action_id'].value_counts()
#
# print("\nTraining set action distribution:")
# print(train_action_distribution)
#
# print("\nValidation set action distribution:")
# print(val_action_distribution)
#
#
#
# # Step 6: 保存新的训练集和验证集到 CSV 文件
# train_df.to_csv('/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/ava_v2.2/ava_train_set.csv', index=False, header=False)
# val_df.to_csv('/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/ava_v2.2/ava_val_set.csv', index=False, header=False)
#
# print("\nNew training and validation sets have been saved as 'train_split.csv' and 'val_split.csv'.")

import pandas as pd
import numpy as np

# 加载 CSV 文件
input_file = "/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/cvat.csv"  # 替换为你的 CSV 文件路径
output_file_10_percent = "/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/test.csv"   # 保存 10% 数据的文件
output_file_remaining =  "/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/train.csv"  # 保存剩余数据的文件

# 加载数据
df = pd.read_csv(input_file, header=None)

# 确保列的顺序正确，假设列定义如下：
# [视频名字, 帧id, x1, y1, x2, y2, action id, person id]

# 获取所有视频名字
video_names = df[0].unique()

# 随机选择 10% 的视频
np.random.seed(42)  # 固定随机种子，确保结果可复现
num_videos_10_percent = max(1, int(len(video_names) * 0.2))  # 至少选择 1 个视频
selected_videos = np.random.choice(video_names, num_videos_10_percent, replace=False)

# 按视频划分数据
data_10_percent = df[df[0].isin(selected_videos)]  # 视频名字在选中列表中的数据
data_remaining = df[~df[0].isin(selected_videos)]  # 剩余的视频数据

# 保存数据到新的 CSV 文件
data_10_percent.to_csv(output_file_10_percent, index=False, header=False)
data_remaining.to_csv(output_file_remaining, index=False, header=False)

print(f"10% 的数据（按视频划分）已保存到: {output_file_10_percent}")
print(f"剩余 90% 的数据已保存到: {output_file_remaining}")

