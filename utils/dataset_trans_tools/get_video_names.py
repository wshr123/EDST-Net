import csv


input_csv_path = "/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/ava_v2.2/ava_test_set.csv"
output_txt_path = "/media/zhong/1.0T/zhong_work/CVB/000058916v001/data/ava_v2.2/ava_file_names_test_v2.1.txt"


video_names = set()


with open(input_csv_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row_number, row in enumerate(csv_reader, start=1):
        if row:
            video_name = row[0].strip()
            if video_name:
                video_names.add(video_name)
            else:
                print(f"警告：第 {row_number} 行的第一列为空，忽略此行。")
        else:
            print(f"警告：第 {row_number} 行为空，忽略此行。")

print(f"总共读取了 {row_number} 行视频名称。")
print(f"去重后的视频名称数量：{len(video_names)}")

with open(output_txt_path, mode='w', encoding='utf-8') as txt_file:
    for video_name in sorted(video_names):
        txt_file.write(video_name + '\n')

print(f"去重后的视频名字已写入到 {output_txt_path}")