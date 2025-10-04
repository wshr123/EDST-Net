#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-normalize the number of images in each subfolder to 450 and rename them to img_00001.jpg ~ img_00450.jpg.
"""

from pathlib import Path
import shutil
import sys

def normalize_and_pad(folder: Path, target_count: int = 450) -> None:
    """Rename JPG images in 'folder' to a unified format and pad the count to 'target_count'."""
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
    if not images:
        print(f"[Skip] No JPG/JPEG images found in {folder}.")
        return

    images.sort()

    temp_files = []
    for idx, img_path in enumerate(images, start=1):
        temp_name = folder / f"__tmp__{idx:05d}{img_path.suffix.lower()}"
        if temp_name.exists():
            temp_name.unlink()
        img_path.rename(temp_name)
        temp_files.append(temp_name)

    for idx, temp_path in enumerate(temp_files, start=1):
        final_name = folder / f"img_{idx:05d}.jpg"
        if final_name.exists():
            final_name.unlink()
        temp_path.rename(final_name)

    current_count = len(temp_files)
    if current_count >= target_count:
        print(f"[Done] {folder} already has {current_count} images, no padding needed.")
        return

    last_image = folder / f"img_{current_count:05d}.jpg"
    if not last_image.exists():
        print(f"[Warning] Unable to locate the last image in {folder}, skipping padding.")
        return

    for idx in range(current_count + 1, target_count + 1):
        target_path = folder / f"img_{idx:05d}.jpg"
        shutil.copy2(last_image, target_path)

    print(f"[Done] {folder} has been padded to {target_count} images.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python fill_images.py <root_directory>")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a valid directory.")
        sys.exit(1)

    for subfolder in root.iterdir():
        if subfolder.is_dir():
            normalize_and_pad(subfolder)

if __name__ == "__main__":
    main()
    # python fill_images.py /path/to/your_folder