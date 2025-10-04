import csv
import pickle
import numpy as np
from collections import defaultdict

# 输入的 CSV 标注文件路径
csv_file_path = "/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/test.csv"

# 输出的 pkl 文件路径
output_pkl_path = "/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/test.pkl"

# 初始化存储结构
dense_proposals = defaultdict(list)

# 读取 CSV 文件
with open(csv_file_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # 从 CSV 文件中读取字段
        video_id, timestamp, x1, y1, x2, y2, action_id, _= row
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

        frame_number = int(float(timestamp))  # 假设视频是 30 FPS
        frame_number_str = f"{frame_number:04d}"  # 将帧数格式化为 4 位字符串
        key = f"{video_id},{frame_number_str}"

        # 将候选框和置信度添加到对应的键中
        # 置信度默认设置为 1.0（可以根据需求修改）
        dense_proposals[key].append([x1, y1, x2, y2, 0.99])

# 转换为 NumPy 数组格式
for key in dense_proposals:
    dense_proposals[key] = np.array(dense_proposals[key], dtype=np.float32)

# 保存为 .pkl 文件
with open(output_pkl_path, "wb") as f:
    pickle.dump(dense_proposals, f)

print(f"Dense proposals saved to {output_pkl_path}")