import cv2
import os
from natsort import natsorted

import os
import cv2
from natsort import natsorted

def images_to_video(image_folder, output_video, fps=30):
    """ 将指定文件夹中的图片合成为视频 """
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = natsorted(images)

    if not images:
        print(f"❌ 文件夹 {image_folder} 中没有找到图片！")
        return

    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"❌ 无法读取第一张图片: {first_image_path}")
        return

    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' 可用于 AVI
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"⚠️ 跳过无法读取的图片: {image_name}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"✅ 视频已保存到: {output_video}")

# 获取当前目录的上级目录
parent_directory = "/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/raw_frames"  # 替换为你的图片文件夹路径

# 遍历上级目录中的所有文件夹
for folder_name in os.listdir(parent_directory):
    folder_path = os.path.join(parent_directory, folder_name)

    if os.path.isdir(folder_path):  # 仅处理文件夹
        output_video = os.path.join("/media/zhong/1.0T/zhong_work/zhong_detr/test_videos", f"{folder_name}.mp4")  # 以文件夹名命名视频文件
        images_to_video(folder_path, output_video, fps=30)

# def images_to_video(image_folder, output_video, fps=30):
#     # 获取所有图片文件（按自然排序，确保顺序正确）
#     images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     images = natsorted(images)
#     if not images:
#         print("❌ 文件夹中没有找到图片！")
#         return
#     first_image_path = os.path.join(image_folder, images[0])
#     first_image = cv2.imread(first_image_path)
#     height, width, _ = first_image.shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' 可用于 AVI
#     video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
#     for image_name in images:
#         image_path = os.path.join(image_folder, image_name)
#         frame = cv2.imread(image_path)
#         if frame is None:
#             print(f"⚠️ 跳过无法读取的图片: {image_name}")
#             continue
#         video_writer.write(frame)
#     video_writer.release()
#     print(f"✅ 视频已保存到: {output_video}")
# image_folder = "/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/raw_frames/0002_arm01_gopro1_20200322_222554_beh7_ani1_ins1_cut_1"  # 替换为你的图片文件夹路径
# output_video = "output.mp4"  # 输出视频文件名
# images_to_video(image_folder, output_video, fps=30)

# import cv2
# import csv
# import os
#
# # 配置参数
# csv_file = "/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/ava_v2.2/test.csv"  # 替换为你的 CSV 文件路径
# output_video = "all_test.mp4"  # 生成的视频文件
# fps = 30  # 你想要的视频帧率
#
# # 读取 CSV 文件并提取图片路径
# image_paths = []
#
# with open(csv_file, "r", encoding="utf-8") as file:
#     reader = csv.reader(file, delimiter=" ")  # 以空格分割列
#     for row in reader:
#         if len(row) > 3:  # 确保行数据足够
#             image_path = row[3].strip("'")  # 去除可能的单引号
#             image_path = "/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/raw_frames/" + image_path
#             if os.path.exists(image_path):  # 确保路径有效
#                 image_paths.append(image_path)
#
# # 确保有图片
# if not image_paths:
#     print("未找到有效的图片路径")
#     exit()
#
# # 获取第一张图片的尺寸
# first_image = cv2.imread(image_paths[0])
# height, width, layers = first_image.shape
#
# # 初始化视频写入对象
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 mp4 编码
# video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
#
# # 逐帧写入图片
# for image_path in image_paths:
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"警告: 无法加载图片 {image_path}")
#         continue
#     video_writer.write(img)
#
# # 释放资源
# video_writer.release()
# print(f"视频已生成: {output_video}")