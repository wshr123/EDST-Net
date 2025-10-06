import pandas as pd
import json
from PIL import Image
import time
import os
from tqdm import tqdm
import itertools
import argparse

def csv2COCOJson_dynamic(csv_path, movie_list, img_root, json_path, min_json_path):

    # 读取 CSV
    ann_df = pd.read_csv(csv_path, header=None)
    """
    0: movie_name
    1: timestamp
    2: x1
    3: y1
    4: x2
    5: y2
    6: action_id
    7: person_id
    """

    movie_ids = {}
    with open(movie_list, 'r') as f:
        for idx, line in enumerate(f):
            # 假设文件名格式为: something.mp4
            name = line.strip()
            dot_pos = name.rfind('.')
            if dot_pos > 0:
                name = name[:dot_pos]
            movie_ids[name] = idx

    movie_infos = {}

    global_ann_id = 1

    for row in tqdm(ann_df.itertuples(), total=len(ann_df), desc="Processing CSV"):
        _, movie_name, timestamp, x1, y1, x2, y2, action_id, person_id = row
        movie_name = str(movie_name)
        timestamp = timestamp * 30

        if movie_name not in movie_infos:
            movie_infos[movie_name] = {
                'img_infos': {},
                'size': None  # 后面第一次读到对应帧图像时再确定
            }

        if timestamp not in movie_infos[movie_name]['img_infos']:
            img_fname = f"img_{timestamp:05d}.jpg"
            img_full_path = os.path.join(img_root, movie_name, img_fname)

            if movie_infos[movie_name]['size'] is None:
                with Image.open(img_full_path) as img:
                    width, height = img.size
                movie_infos[movie_name]['size'] = (width, height)
            else:
                width, height = movie_infos[movie_name]['size']

            image_id = movie_ids[movie_name] * 1000000 + timestamp

            movie_infos[movie_name]['img_infos'][timestamp] = {
                'id': image_id,
                'file_name': os.path.join(movie_name, img_fname),
                'height': height,
                'width': width,
                'movie': movie_name,
                'timestamp': timestamp,
                'annotations': []  
            }

        img_info = movie_infos[movie_name]['img_infos'][timestamp]
        width = img_info['width']
        height = img_info['height']

        box_w = (x2 - x1) * width
        box_h = (y2 - y1) * height
        real_x1 = x1 * width
        real_y1 = y1 * height
        area = box_w * box_h

        existing_annotation = None
        for annotation in img_info['annotations']:
            if annotation['bbox'] == [
                round(real_x1, 2), round(real_y1, 2),
                round(box_w, 2), round(box_h, 2)
            ]:
                existing_annotation = annotation
                break

        if existing_annotation:
            if action_id not in existing_annotation['action_ids']:
                existing_annotation['action_ids'].append(action_id)
        else:
            ann_id = global_ann_id
            global_ann_id += 1

            img_info['annotations'].append({
                'id': ann_id,
                'image_id': img_info['id'],
                'img_path': os.path.join(movie_name, img_fname),
                'category_id': 1,  
                'action_ids': [action_id],
                'person_id': person_id,
                'bbox': [round(real_x1, 2), round(real_y1, 2),
                         round(box_w, 2), round(box_h, 2)],
                'area': round(area, 2),
                'iscrowd': 0
            })

    tic = time.time()
    print("Writing into json file...")

    jsondata = {
        'categories': [
            {
                'supercategory': 'person',
                'id': 1,
                'name': 'person'
            }
        ],
        'images': [],
        'annotations': []
    }

    for movie_name, minfo in movie_infos.items():
        for timestamp, info in minfo['img_infos'].items():
            jsondata['images'].append({
                'id': info['id'],
                'file_name': info['file_name'],
                'height': info['height'],
                'width': info['width'],
                'movie': movie_name,
                'timestamp': timestamp
            })
            jsondata['annotations'].extend(info['annotations'])

    with open(json_path, 'w') as f:
        json.dump(jsondata, f, indent=4)
    print(f"Write json dataset into json file {json_path} successfully.")

    with open(min_json_path, 'w') as fmin:
        json.dump(jsondata, fmin)
    print(f"Write json dataset with no indent into json file {min_json_path} successfully.")
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

def main():
    parser = argparse.ArgumentParser(description="Generate coco format json for AVA.")
    parser.add_argument(
        "--csv_path",
        default="/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/test.csv",
        help="path to csv file",
        type=str,
    )
    parser.add_argument(
        "--movie_list",
        required=False,
        default="/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/movies.txt",
        help="path to movie list",
        type=str,
    )
    parser.add_argument(
        "--img_root",
        required=False,
        default="/media/zhong/ORICO/rawframess",
        help="root directory of extracted key frames",
        type=str,
    )
    parser.add_argument(
        "--json_path",
        default="/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/test.json",
        help="path of output json",
        type=str,
    )
    parser.add_argument(
        "--min_json_path",
        default="/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/test_min.json",
        help="path of output minimized json",
        type=str,
    )
    args = parser.parse_args()

    if args.json_path=="":
        if args.csv_path == "":
            json_path = "test.json"
        else:
            dotpos = args.csv_path.rfind('.')
            if dotpos < 0:
                csv_name = args.csv_path
            else:
                csv_name = args.csv_path[:dotpos]
            json_path = csv_name + '.json'
    else:
        json_path = args.json_path

    if args.min_json_path=="":
        dotpos = json_path.rfind('.')
        if dotpos < 0:
            json_name = json_path
        else:
            json_name = json_path[:dotpos]
        min_json_path = json_name + '_min.json'
    else:
        min_json_path = args.min_json_path

    if args.csv_path == "":
        genCOCOJson(args.movie_list, args.img_root, json_path, min_json_path)
    else:
        csv2COCOJson_dynamic(args.csv_path, args.movie_list, args.img_root, json_path, min_json_path)

if __name__ == '__main__':
    main()
