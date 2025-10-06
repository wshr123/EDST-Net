import csv
import pickle
import numpy as np
from collections import defaultdict

csv_file_path = "/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/test.csv"

output_pkl_path = "/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/test.pkl"

dense_proposals = defaultdict(list)

with open(csv_file_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        video_id, timestamp, x1, y1, x2, y2, action_id, _= row
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

        frame_number = int(float(timestamp)) 
        frame_number_str = f"{frame_number:04d}"  
        key = f"{video_id},{frame_number_str}"

        dense_proposals[key].append([x1, y1, x2, y2, 0.99])

for key in dense_proposals:
    dense_proposals[key] = np.array(dense_proposals[key], dtype=np.float32)

with open(output_pkl_path, "wb") as f:
    pickle.dump(dense_proposals, f)

print(f"Dense proposals saved to {output_pkl_path}")
