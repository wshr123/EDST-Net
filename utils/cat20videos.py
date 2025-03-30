import cv2
import os
import glob

input_root_folder = "/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/raw_frames"
output_video_path = "/media/zhong/1.0T/zhong_work/zhong_detr/3000.mp4"
fps = 30  # 你希望的视频帧率

# 读取所有子文件夹
subfolders = sorted([os.path.join(input_root_folder, d) for d in os.listdir(input_root_folder) if os.path.isdir(os.path.join(input_root_folder, d))])

# 存储所有图片路径
image_files = []

# 遍历所有子文件夹，收集图片
for folder in subfolders:
    images = sorted(glob.glob(os.path.join(folder, "*.jpg")))  # 你可以修改为 "*.png" 或其他格式
    image_files.extend(images)

# 确保有图片
if not image_files:
    print("未找到任何图片，请检查文件路径！")
    exit()

# 读取第一张图片，确定视频尺寸
first_image = cv2.imread(image_files[0])
height, width, _ = first_image.shape

# 设置视频编码格式（MP4 使用 'mp4v'，AVI 使用 'XVID'）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 逐帧写入视频
for idx, image_path in enumerate(image_files):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"警告：无法读取 {image_path}，跳过该帧")
        continue
    video_writer.write(frame)
    if idx % 500 == 0:
        print(f"已处理 {idx}/{len(image_files)} 张图片...")
    if idx == 1350:
        break
# 释放资源
video_writer.release()
print(f"视频已保存至 {output_video_path}")