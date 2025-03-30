import cv2
video_path = '/media/zhong/1.0T/zhong_work/zhong_detr/3000.mp4'
output_path = '/media/zhong/1.0T/zhong_work/zhong_detr/3000_640.mp4'
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_width = 640
new_height = 480
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (new_width, new_height))
    out.write(frame)

cap.release()
out.release()