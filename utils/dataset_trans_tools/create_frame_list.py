import pandas as pd

#todo:need to modify frame number for each videos
input_file = '/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/train.csv' 

output_file = '/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/train_frame.csv'  

df = pd.read_csv(input_file, header=None)

df.columns = ['video_name', 'frame_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'action_id', 'track_id']

unique_videos = df['video_name'].unique()

output_df = pd.DataFrame()

for idx, video in enumerate(unique_videos):
    temp_df = pd.DataFrame({
        'original_video_id': [video] * 300,
        'video_id': [idx] * 300,
        'frame_id': range(300),
        'path': [f"{video}/img_{i + 1:05d}.jpg" for i in range(300)],
        'labels': "'"
    })

    output_df = pd.concat([output_df, temp_df], ignore_index=True)

# output_df['merged'] = (
#     output_df['original_video_id'].astype(str) + ' ' +
#     output_df['video_id'].astype(str) + ' ' +
#     output_df['frame_id'].astype(str) + ' ' +
#     output_df['path'].astype(str) + ' ' +
#     output_df['labels'].astype(str)
# )


print(output_df)
output_df.to_csv(output_file, index=False, header=False, sep=' ')  # 使用空格作为分隔符

print(f"新的 CSV 文件已保存到 {output_file}")
