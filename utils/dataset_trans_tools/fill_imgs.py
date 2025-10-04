#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量补齐子文件夹中的图片数量到 450 张，并规范命名为 img_00001.jpg ~ img_00450.jpg。
"""

from pathlib import Path
import shutil
import sys

def normalize_and_pad(folder: Path, target_count: int = 450) -> None:
    """将 folder 下的 JPG 图片重新命名为指定格式，并将数量补齐到 target_count。"""
    # 收集 JPG/JPEG 图片
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
    if not images:
        print(f"[跳过] {folder} 内没有 JPG/JPEG 图片。")
        return

    images.sort()

    # 为防止命名冲突，先改成临时文件名
    temp_files = []
    for idx, img_path in enumerate(images, start=1):
        temp_name = folder / f"__tmp__{idx:05d}{img_path.suffix.lower()}"
        if temp_name.exists():
            temp_name.unlink()
        img_path.rename(temp_name)
        temp_files.append(temp_name)

    # 统一改为 img_00001.jpg ~
    for idx, temp_path in enumerate(temp_files, start=1):
        final_name = folder / f"img_{idx:05d}.jpg"
        if final_name.exists():
            final_name.unlink()
        temp_path.rename(final_name)

    current_count = len(temp_files)
    if current_count >= target_count:
        print(f"[完成] {folder} 已有 {current_count} 张图片，无需补齐。")
        return

    # 复制最后一张补齐
    last_image = folder / f"img_{current_count:05d}.jpg"
    if not last_image.exists():
        print(f"[警告] {folder} 无法找到最后一张图片，跳过补齐。")
        return

    for idx in range(current_count + 1, target_count + 1):
        target_path = folder / f"img_{idx:05d}.jpg"
        shutil.copy2(last_image, target_path)

    print(f"[完成] {folder} 已补齐到 {target_count} 张图片。")

def main():
    if len(sys.argv) != 2:
        print("用法: python fill_images.py <根目录>")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"错误: {root} 不是有效的目录。")
        sys.exit(1)

    for subfolder in root.iterdir():
        if subfolder.is_dir():
            normalize_and_pad(subfolder)

if __name__ == "__main__":
    main()
    #python fill_images.py /path/to/your_folder
