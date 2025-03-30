import os
import shutil

source_folder = "/media/zhong/1.0T/zhong_work/archive/ava/frames"

destination_folder = "/media/zhong/1.0T/zhong_work/archive/ava/coco/train2017"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历源文件夹中的所有子文件夹
for folder_name in os.listdir(source_folder):
    folder_path = os.path.join(source_folder, folder_name)

    # 确保是文件夹而不是文件
    if os.path.isdir(folder_path):
        # 遍历子文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            # 检查文件是否是图片（根据扩展名）
            if file_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                # 构造新文件名：文件夹名_原文件名
                new_file_name = f"{folder_name}_{file_name}"

                # 构造原文件路径和目标文件路径
                source_file_path = os.path.join(folder_path, file_name)
                destination_file_path = os.path.join(destination_folder, new_file_name)

                # 移动并重命名文件
                shutil.move(source_file_path, destination_file_path)

print("所有图片已重命名并移动到目标文件夹。")