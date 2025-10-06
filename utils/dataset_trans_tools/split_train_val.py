import pandas as pd
import numpy as np

input_file = "/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/cvat.csv"
output_file_10_percent = "/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/test.csv" 
output_file_remaining =  "/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/train.csv"  

df = pd.read_csv(input_file, header=None)

video_names = df[0].unique()

np.random.seed(42)
num_videos_10_percent = max(1, int(len(video_names) * 0.2))
selected_videos = np.random.choice(video_names, num_videos_10_percent, replace=False)
data_10_percent = df[df[0].isin(selected_videos)] 
data_remaining = df[~df[0].isin(selected_videos)]  

data_10_percent.to_csv(output_file_10_percent, index=False, header=False)
data_remaining.to_csv(output_file_remaining, index=False, header=False)

print(f"10% 的数据（按视频划分）已保存到: {output_file_10_percent}")
print(f"剩余 90% 的数据已保存到: {output_file_remaining}")

