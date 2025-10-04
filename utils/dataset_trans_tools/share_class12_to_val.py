import pandas as pd

# 加载 CSV 文件
csv_file = "/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/cvb_in_ava_format/ava_train_set.csv"  # 替换为你的 CSV 文件路径
df = pd.read_csv(csv_file, header=None)

# 确定列的含义
# 假设列顺序是：[视频名字, 帧id, x1, y1, x2, y2, action id, person id]

# 筛选出 action id 为 12 的行
action_id_12_rows = df[df[6] == 12]  # 第 7 列 (索引从 0 开始) 是 action id

# 找到所有对应的视频名字（去重）
videos_with_action_12 = action_id_12_rows[0].unique()

# 输出结果
print("所有包含 action id 为 12 的视频：")
print(videos_with_action_12)

# 如果需要保存到文件
output_file = "videos_with_action_12.txt"
with open(output_file, "w") as f:
    for video in videos_with_action_12:
        f.write(video + "\n")

print(f"结果已保存到 {output_file}")