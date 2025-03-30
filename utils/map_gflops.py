import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据准备
data = {
    'Model': ['Slowonly_R50', 'SlowFast_R50', 'X3D', 'NonLocal_R50', 'SlowFast_R101',
              'Acrn', 'NonLocal_R101', 'VideoMae(todo)', 'Ours'],
    'Gflops': [90.9, 85.1, 73.8, 94.4, 154.9, 130.4, 162.9, 646, 31.5],
    'Params': [53, 55.6, 27.4, 82.2, 74.1, 54.8, 80.3, 326.2, 19.4],
    'mAP': [59.4, 64.5, 65.8, 65.9, 70.3, 70.9, 71.3, 73.2, 72.2]
}
df = pd.DataFrame(data)

# ================== 自定义配置区域 ==================
# 设置每个模型的标签偏移量 (水平偏移, 垂直偏移)
label_offsets = {
    'Slowonly_R50': (20, 15),
    'SlowFast_R50': (-15, 30),
    'X3D': (10, -25),
    'NonLocal_R50': (40, 10),
    'SlowFast_R101': (30, -20),
    'Acrn': (-30, -15),
    'NonLocal_R101': (20, 25),
    'VideoMae(todo)': (-55, 20),
    'Ours': (0, -40)
}
# ================== 配置结束 ==================

# 可视化设置
plt.figure(figsize=(14, 8))
colors = plt.cm.tab20(range(len(df)))
sizes = df['Gflops'] * 5  # 基础缩放系数

# 设置Our模型特殊样式
our_index = df[df['Model'] == 'Ours'].index[0]
colors[our_index] = (1, 0, 0, 0.9)  # 红色
markers = ['o'] * len(df)
markers[our_index] = '*'  # 星形标记
sizes[our_index] = df.loc[our_index, 'Gflops'] * 5  # 保持比例一致

# 设置对数坐标轴
plt.xscale("log")

# 绘制散点图
for i, row in df.iterrows():
    plt.scatter(row['Params'], row['mAP'],
                s=sizes[i],
                c=[colors[i]],
                marker=markers[i],
                edgecolor='w',
                linewidth=1.5 if i == our_index else 1,
                alpha=0.85)

# 添加自定义标签
for i, row in df.iterrows():
    model_name = row['Model']
    offset = label_offsets.get(model_name, (0, 0))  # 获取自定义偏移量

    plt.annotate(
        text=f"{model_name}\n({row['Gflops']}G)",
        xy=(row['Params'], row['mAP']),
        xytext=offset,
        textcoords='offset points',
        ha='center',
        va='center' if offset[1] == 0 else 'bottom' if offset[1] > 0 else 'top',
        fontsize=15,
        alpha=0.9,
        arrowprops=dict(
            arrowstyle='-',
            color='gray',
            alpha=0.6,
            linewidth=0.8
        ),
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor='white',
            edgecolor='lightgray',
            alpha=0.8
        )
    )

# **设置对数刻度**
xticks = [10, 20, 50, 100, 200, 500]  # 选择合适的刻度
plt.xticks(xticks, [str(x) for x in xticks])  # 显示刻度值

# 坐标轴设置
plt.xlim(10, 500)  # 避免 log(0) 计算错误
plt.ylim(55, 75)
plt.xlabel("Model Parameters (M)", fontsize=15, labelpad=10)
plt.ylabel("mAP", fontsize=15, labelpad=10)
plt.grid(color='lightgray', linestyle='--', alpha=0.4)

# 添加图例
legend_elements = [
    plt.scatter([], [], s=50, c='gray', marker='o', label='GFLOPs'),
    plt.scatter([], [], s=50, c='red', marker='*', label='Our Model')
]
plt.legend(handles=legend_elements,
           title="Computation Reference",
           loc='upper left',
           bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()