import cv2
import pandas as pd
import os
video_path = "/media/zhong/1.0T/zhong_work/zhong_detr/test_results/1352_arm01_gopro4_20200324_040135_beh3_ani4_ins1_cut_00033.mp4"
csv_path = "/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/ava_v2.2/ava_train_set.csv"
video_name = "1352_arm01_gopro4_20200324_040135_beh3_ani4_ins1_cut_00033"
output_path = '/media/zhong/1.0T/zhong_work/zhong_detr/test_gt/1352_arm01_gopro4_20200324_040135_beh3_ani4_ins1_cut_00033.mp4'
def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    return img
try:
    df = pd.read_csv(csv_path)
    required_columns = ['video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2','id','action']
    assert all(col in df.columns for col in required_columns)
except:
    print("CSV格式错误，尝试无表头读取...")
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2'] + list(df.columns[6:])
video_data = df[df['video_id'] == video_name].copy()
cap = cv2.VideoCapture(video_path)
fps = 30
width = 1920
height = 1080
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_data = video_data.dropna(subset=['x1', 'y1', 'x2', 'y2'])
video_data[['x1', 'y1', 'x2', 'y2']] = video_data[['x1', 'y1', 'x2', 'y2']].astype(float)
video_data['x1'] = (video_data['x1'] * width).astype(int)
video_data['y1'] = (video_data['y1'] * height).astype(int)
video_data['x2'] = (video_data['x2'] * width).astype(int)
video_data['y2'] = (video_data['y2'] * height).astype(int)
gt_boxes = {}
for _, row in video_data.iterrows():
    timestamp = row['timestamp']
    target_frame = int(round(timestamp * fps))  # 1秒标记对应1*fps的帧号
    start_frame = max(0, target_frame - 15)
    end_frame = min(total_frames - 1, target_frame + 15)
    for frame_num in range(start_frame, end_frame + 1):
        if frame_num not in gt_boxes:
            gt_boxes[frame_num] = []
        gt_boxes[frame_num].append((
            row['x1'], row['y1'], row['x2'], row['y2']
        ))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if current_frame in gt_boxes:
        for (x1, y1, x2, y2) in gt_boxes[current_frame]:
            draw_dashed_rectangle(
                img=frame,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(0, 0, 255),
                thickness=2,
                dash_length=8
            )
    if current_frame % 100 == 0:
        print(f"Processing frame {current_frame}/{total_frames}")
    out.write(frame)
    current_frame += 1
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"finish process video, save in：{os.path.abspath(output_path)}")