import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 用于正常显示中文
plt.rcParams['font.sans-serif'] = 'SimHei'
# 用于正常显示符号
plt.rcParams['axes.unicode_minus'] = False

# 使用ggplot的绘图风格，这个类似于美化了，可以通过plt.style.available查看可选值
plt.style.use('ggplot')
#student = pd.read_csv('demo.csv',usecols=[0,13,17,21,25,29],nrows=1)

# 构造数据
#values2 = [student['jichurate'], student['panduanrate'], student['xunhuanrate'], student['shuzurate'], student['hanshurate']]
values2 = [0.3, 0.7, 0.4, 0.5, 0.5]
values3 = [0.5, 0.8, 0.6, 0.5, 0.6]
values4 = [0.4, 0.2, 0.7, 0.2, 0.4]
values1 = [0.47, 0.36, 0.32, 0.27, 0.42]
feature = ['基本', '分支', '循环', '数组', '函数']


# 设置每个数据点的显示位置，在雷达图上用角度表示
angles = np.linspace(0, 2 * np.pi, len(values1), endpoint=False)

# 拼接数据首尾，使图形中线条封闭
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate([values2, [values2[0]]])
values3 = np.concatenate([values3, [values3[0]]])
values4 = np.concatenate([values4, [values4[0]]])
angles = np.concatenate((angles, [angles[0]]))
feature = np.concatenate((feature, [feature[0]]))

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values1, 'o-', linewidth=2, label='avg', color='black')
#ax.fill(angles, values1, alpha=0.25)

ax.plot(angles, values2, 'o-', linewidth=2, label='stuA', linestyle="-.", color='black')
#ax.fill(angles, values2, alpha=0.25)

ax.plot(angles, values3, 'o-', linewidth=2, label='stuB', linestyle="--", color='black')
#ax.fill(angles, values3, alpha=0.25)

ax.plot(angles, values4, 'o-', linewidth=2, label='stuC', linestyle=":", color='black')
#ax.fill(angles, values4, alpha=0.25)

ax.set_thetagrids(angles * 180 / np.pi, feature, fontsize=16)
ax.set_ylim(0, 1)
# 添加标题
# 添加网格线
plt.legend()
ax.grid(True)

plt.show()