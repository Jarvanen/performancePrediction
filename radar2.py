import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
df = pd.DataFrame({
    'group': ['A', 'B', 'C', 'D'],
    'var1': [38, 1.5, 30, 4],
    'var2': [29, 10, 9, 34],
    'var3': [8, 39, 23, 24],
    'var4': [7, 31, 33, 14],
    'var5': [28, 15, 32, 14]
})


# ---------- 步骤1 创建背景
def make_spider(row, title, color):
    # number of variable
    # 变量类别
    categories = list(df)[1:]
    # 变量类别个数
    N = len(categories)

    # 设置每个点的角度值
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    # 分图
    ax = plt.subplot(2, 2, row + 1, polar=True, )

    # If you want the first axis to be on top:
    # 设置角度偏移
    ax.set_theta_offset(pi / 2)
    # 设置顺时针还是逆时针，1或者-1
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    # 设置x轴的标签
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    # 画标签
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    plt.ylim(0, 40)

    # Ind
    # 填充数据
    values = df.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    # 设置标题
    plt.title(title, size=11, color=color, y=1.1)


# ---------- 步骤2 绘制图形
my_dpi = 96
plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

# Create a color palette:
# 设定颜色
my_palette = plt.cm.get_cmap("Set2", len(df.index))

# Loop to plot
for row in range(0, len(df.index)):
    make_spider(row=row, title='group ' + df['group'][row], color=my_palette(row))

plt.show();
