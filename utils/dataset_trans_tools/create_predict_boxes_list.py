import pandas as pd

# 读取 CSV 文件
input_file = '/media/zhong/1.0T/zhong_work/zhong_detr/inference_result/cvat.csv'  # 将此替换为你的 CSV 文件路径
df = pd.read_csv(input_file)

# 将最后一列的所有值替换为 0.99
df.iloc[:, -1] = 0.99

# 保存修改后的数据到新 CSV 文件
output_file = '/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/ava_v2.2/ava_test_predit_boxes.csv'  # 将此替换为你想保存的文件名
df.to_csv(output_file, index=False)

print(f"已将最后一列替换为 0.99，并保存到 {output_file}")