import numpy as np
import matplotlib.pyplot as plt

# 用于正常显示中文
plt.rcParams['font.sans-serif'] = 'SimHei'
# 用于正常显示符号
plt.rcParams['axes.unicode_minus'] = False

# 使用ggplot的绘图风格，这个类似于美化了，可以通过plt.style.available查看可选值，你会发现其它的风格真的丑。。。
plt.style.use('ggplot')

values = [2.6, 2.1, 3.4, 3, 4.1]
values_2 = [1.7, 4.1, 3.3, 2.6, 3.8]
values_2 = np.concatenate([values_2, [values_2[0]]])
values_3 = [1.9, 4.4, 3.9, 4.6, 1.8]
values_3 = np.concatenate([values_3, [values_3[0]]])
feature = ['个人能力', 'QC知识', '解决问题能力', '服务质量意识', '团队精神']

# 设置每个数据点的显示位置，在雷达图上用角度表示
angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)

# 拼接数据首尾，使图形中线条封闭
values = np.concatenate((values, [values[0]]))
angles = np.concatenate((angles, [angles[0]]))
feature = np.concatenate((feature, [feature[0]]))

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values, 'o-', linewidth=2, label='活动前')
ax.fill(angles, values, alpha=0.25)

ax.plot(angles, values_2, 'o-', linewidth=2, label='活动后')
ax.fill(angles, values_2, alpha=0.25)

ax.plot(angles, values_3, 'o-', linewidth=2, label='活动')
ax.fill(angles, values_3, alpha=0.25)

ax.set_thetagrids(angles * 180 / np.pi, feature)
ax.set_ylim(0, 5)
# 添加标题
plt.title('活动前后员工状态表现')
# 添加网格线


plt.legend()
ax.grid(True)

plt.show()