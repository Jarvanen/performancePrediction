import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import *
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import scipy
import pickle


# 初始化数据
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题
student = pd.read_csv('C:/Users/Jarvan/Desktop/data/final.csv')
# student = pd.read_csv('C:/Users/Jarvan/Desktop/PPSCD.csv')
# student = pd.read_csv('demo.csv')

# 分析total_score数据属性
# print(student['total'].describe())

# # 根据人数多少统计各分数段的学生人数
# grade_counts = student['final'].value_counts().sort_values().plot.barh(width=.9,color=sns.color_palette('inferno',40))
# grade_counts.axes.set_title('各分数值的学生分布',fontsize=30)
# grade_counts.set_xlabel('学生数量', fontsize=30)
# grade_counts.set_ylabel('总分数', fontsize=30)
# plt.show()

# # 从低到高展示成绩分布图
# grade_distribution = sns.countplot(student['final'])
# # grade_distribution.set_title('成绩分布图', fontsize=30)
# grade_distribution.set_xlabel('成绩', fontsize=20)
# grade_distribution.set_ylabel('人数统计', fontsize=20)
# plt.show()

# # 从低到高展示成绩分布图
# grade_distribution = sns.countplot(student['total'])
# grade_distribution.set_title('成绩分布图', fontsize=30)
# grade_distribution.set_xlabel('成绩', fontsize=20)
# grade_distribution.set_ylabel('人数统计', fontsize=20)
# plt.show()

# grade_distribution = sns.lineplot(x="contest",y="scoreas10",data=student)
# grade_distribution.set_xlabel('实验', fontsize=20)
# grade_distribution.set_ylabel('成绩', fontsize=20)
# plt.show()

# grade_distribution = sns.lineplot(x="contest",y="rate",data=student)
# grade_distribution.set_xlabel('实验', fontsize=20)
# grade_distribution.set_ylabel('正确率', fontsize=20)
# plt.show()


# # 分析成绩分布比例（曲线图）
# age_distribution = sns.kdeplot(student['rate'], shade=True)
# age_distribution.axes.set_title('实验正确率分布图', fontsize=30)
# age_distribution.set_xlabel('实验正确率', fontsize=20)
# age_distribution.set_ylabel('比例', fontsize=20)
# plt.show()

# # 分析性别比例
# male_studs = len(student[student['sex'] == 'm'])
# female_studs = len(student[student['sex'] == 'f'])
# print('男同学数量:',male_studs)
# print('女同学数量:',female_studs)

# # 分性别成绩分布图（柱状图）
# age_distribution_sex = sns.countplot('final', hue='sex', data=student)
# age_distribution_sex.axes.set_title('不同成绩段的学生人数', fontsize=30)
# age_distribution_sex.set_xlabel('成绩', fontsize=30)
# age_distribution_sex.set_ylabel('人数', fontsize=30)
# plt.show()

# # 分析成绩分布比例（曲线图）
# age_distribution = sns.kdeplot(student['total'], shade=True)
# age_distribution.axes.set_title('成绩分布图', fontsize=30)
# age_distribution.set_xlabel('成绩', fontsize=20)
# age_distribution.set_ylabel('比例', fontsize=20)
# plt.show()
#
# # 分析成绩分布比例（曲线图）
# age_distribution = sns.kdeplot(student['final'], shade=True)
# age_distribution.axes.set_title('成绩分布图', fontsize=30)
# age_distribution.set_xlabel('成绩', fontsize=20)
# age_distribution.set_ylabel('比例', fontsize=20)
# plt.show()

# # 分析基础题型正确率分布比例（曲线图）
# age_distribution = sns.distplot(student['jichurate'], color='g', fit=norm)
# age_distribution.axes.set_title('基础题型正确率分布图', fontsize=30)
# age_distribution.set_xlabel('基础题型正确率', fontsize=20)
# age_distribution.set_ylabel('密度', fontsize=20)
# plt.show()

# 各题型正确率核密度估计图
'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
sns.kdeplot(student['jichurate'], color='black', label='基本题型', linestyle="-.")
sns.kdeplot(student['panduanrate'], color='black', label='分支题型', linestyle="--")
sns.kdeplot(student['xunhuanrate'], color='black', label='循环题型', linestyle=":")
sns.kdeplot(student['shuzurate'], color='black', label='数组题型', linestyle="-")
sns.kdeplot(student['hanshurate'], color='black', label='函数题型', marker='.')
# plt.title('各题型正确率核密度估计图', fontsize=30)
plt.xlabel('正确率', fontsize=20)
plt.ylabel('密度', fontsize=20)
plt.legend()
plt.show()

# # 分析成绩分布比例（曲线图）
# age_distribution = sns.kdeplot(student['final'], shade=True)
# age_distribution.axes.set_title('学生成绩分布图', fontsize=30)
# age_distribution.set_xlabel('成绩', fontsize=20)
# age_distribution.set_ylabel('比例', fontsize=20)
# plt.show()
#
# # 第三次实验的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='c3score', y='final', data=student)
# age_grade_swarmplot.axes.set_title('第三次实验与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('第三次实验', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 第四次实验的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='c4score', y='final', data=student)
# age_grade_swarmplot.axes.set_title('第四次实验与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('第四次实验', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 第五次实验的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='c5score', y='final', data=student)
# age_grade_swarmplot.axes.set_title('第五次实验与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('第五次实验', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 第六次实验的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='c6score', y='final', data=student)
# age_grade_swarmplot.axes.set_title('第六次实验与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('第六次实验', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 第七次实验的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='c7score', y='final', data=student)
# age_grade_swarmplot.axes.set_title('第七次实验与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('第七次实验', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 第十次实验的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='c10score', y='final', data=student)
# age_grade_swarmplot.axes.set_title('第十次实验与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('第十次实验', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 基础题型正确率的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='jichurate', y='final', data=student)
# age_grade_swarmplot.axes.set_title('基础题型正确率与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('基础题型正确率', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 判断题型正确率的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='panduanrate', y='final', data=student)
# age_grade_swarmplot.axes.set_title('判断题型正确率与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('判断题型正确率', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 循环题型正确率的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='xunhuanrate', y='final', data=student)
# age_grade_swarmplot.axes.set_title('循环题型正确率与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('循环题型正确率', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 数组题型正确率的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='shuzurate', y='final', data=student)
# age_grade_swarmplot.axes.set_title('数组题型正确率与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('数组题型正确率', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 函数题型正确率的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='hanshurate', y='final', data=student)
# age_grade_swarmplot.axes.set_title('函数题型正确率与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('函数题型正确率', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()

# # 课外分数的成绩箱型图
# age_grade_boxplot = sns.boxplot(x='total_score', y='final', data=student)
# age_grade_boxplot.axes.set_title('课外分数与成绩', fontsize = 30)
# age_grade_boxplot.set_xlabel('课外分数', fontsize = 20)
# age_grade_boxplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 课外分数的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='total_score', y='final', data=student)
# age_grade_swarmplot.axes.set_title('课外分数与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('课外分数', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 课外提交次数的成绩箱型图
# age_grade_boxplot = sns.boxplot(x='submission_number', y='final', data=student)
# age_grade_boxplot.axes.set_title('课外提交次数与成绩', fontsize = 30)
# age_grade_boxplot.set_xlabel('提交次数', fontsize = 20)
# age_grade_boxplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 课外提交次数的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='submission_number', y='final', data=student)
# age_grade_swarmplot.axes.set_title('课外提交次数与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('提交次数', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 课外正确提交次数的成绩箱型图
# age_grade_boxplot = sns.boxplot(x='accepted_number', y='final', data=student)
# age_grade_boxplot.axes.set_title('课外正确提交次数与成绩', fontsize = 30)
# age_grade_boxplot.set_xlabel('正确提交次数', fontsize = 20)
# age_grade_boxplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 课外正确提交次数的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='accepted_number', y='final', data=student)
# age_grade_swarmplot.axes.set_title('课外正确提交次数与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('正确提交次数', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 课外正确率的成绩箱型图
# age_grade_boxplot = sns.boxplot(x='rate', y='final', data=student)
# age_grade_boxplot.axes.set_title('正确率与成绩', fontsize = 30)
# age_grade_boxplot.set_xlabel('正确率', fontsize = 20)
# age_grade_boxplot.set_ylabel('成绩', fontsize = 20)
# plt.show()
#
# # 课外正确率的成绩分布图
# age_grade_swarmplot = sns.swarmplot(x='rate', y='final', data=student)
# age_grade_swarmplot.axes.set_title('正确率与成绩', fontsize = 30)
# age_grade_swarmplot.set_xlabel('正确率', fontsize = 20)
# age_grade_swarmplot.set_ylabel('成绩', fontsize = 20)
# plt.show()

# # 选取final属性值
# labels = student['final']
# # 删除属性
# # student = student.drop(['username', 'pass', 'variableScope', 'unusedVariable', 'unreadVariable', 'invalidscanf', 'knownConditionTrueFalse', 'getsCalled', 'selfAssignment'], axis='columns')
# student = student.drop(['username'], axis='columns')
# # student = student.drop(['username','T1', 'T2', 'T3','c3score','c4score','c5score','c6score','c7score','c10score','jichutotal','jichuaccept','jichurate','panduantotal','panduanaccept','panduanrate','xunhuantotal','xunhuanaccept','xunhuanrate','shuzutotal','shuzuaccept','shuzurate','hanshutotal','hanshuaccept','hanshurate'], axis='columns')
# # 对离散变量进行独热编码
# # student = pd.get_dummies(student)
# # 选取相关性最强的7个
# most_correlated = student.corr().abs()['final'].sort_values(ascending=False)
# most_correlated = most_correlated[:8]
# print(most_correlated)
#
#
# # 分割数据集
# X_train, X_test, y_train, y_test = train_test_split(student, labels, test_size=0.25, random_state=42)
#
#
# # 计算平均绝对误差和均方根误差
# # MAE-平均绝对误差
# # RMSE-均方根误差
# def evaluate_predictions(predictions, true):
#     mae = np.mean(abs(predictions - true))
#     rmse = np.sqrt(np.mean((predictions - true) ** 2))
#
#     return mae, rmse
#
#
# # 求中位数
# median_pred = X_train['final'].median()
#
# # 所有中位数的列表
# median_preds = [median_pred for _ in range(len(X_test))]
#
# # 存储真实的final值以传递给函数
# true = X_test['final']
#
# # 展示基准
# mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
# print('Median Baseline  MAE: {:.2f}'.format(mb_mae))
# print('Median Baseline RMSE: {:.2f}'.format(mb_rmse))
#
#
# # 通过训练集训练和测试集测试来生成多个线性模型
# def evaluate(X_train, X_test, y_train, y_test):
#     # 模型名称
#     # model_name_list = ['Linear Regression', 'ElasticNet Regression',
#     #                    'Random Forest', 'Extra Trees', 'SVM',
#     #                    'Gradient Boosted', 'BayesianRidge', 'Baseline']
#     model_name_list = ['SVM', 'BayesianRidge', 'Random Forest', 'Extra Trees',
#                        'Gradient Boosted', 'DecisionTree', 'XGboost']
#     X_train = X_train.drop('final', axis='columns')
#     X_test = X_test.drop('final', axis='columns')
#
#     # 实例化模型
#     model1 = LinearRegression()
#     model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
#     model3 = RandomForestRegressor(n_estimators=100)
#     model4 = ExtraTreesRegressor(n_estimators=100)
#     model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
#     model6 = GradientBoostingRegressor(n_estimators=50)
#     model7 = DecisionTreeRegressor(max_depth=1)
#     model8 = BayesianRidge()
#     # model9 = xgboost()
#
#     # 结果数据框
#     results = pd.DataFrame(columns=['mae', 'rmse'], index=model_name_list)
#
#     # 每种模型的训练和预测
#     for i, model in enumerate([model5, model8, model3, model4, model6, model7]):
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         # print(predictions)
#         # 误差标准
#         mae = np.mean(abs(predictions - y_test))
#         rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
#         #acc = 100*accuracy_score(y_test, predictions)
#         #precision
#
#         # 将结果插入结果框
#         model_name = model_name_list[i]
#         results.loc[model_name, :] = [mae, rmse]
#
#     # 中值基准度量
#     baseline = np.median(y_train)
#     baseline_mae = np.mean(abs(baseline - y_test))
#     baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
#
#     results.loc['Baseline', :] = [baseline_mae, baseline_rmse]
#
#     return results
#
#
# results = evaluate(X_train, X_test, y_train, y_test)
# print(results)
# print('testacc:{:.2f}%'.format(100*accuracy_score(y_test, y_predict_on_test)))

# # 找出最合适的模型
# plt.figure(figsize=(12, 8))

# # 平均绝对误差
# ax = plt.subplot(1, 2, 1)
# results.sort_values('mae', ascending=True).plot.bar(y='mae', color='b', ax=ax, fontsize=20)
# plt.title('平均绝对误差', fontsize=20)
# plt.ylabel('MAE', fontsize=20)
#
# # 均方根误差
# ax = plt.subplot(1, 2, 2)
# results.sort_values('rmse', ascending=True).plot.bar(y='rmse', color='r', ax=ax, fontsize=20)
# plt.title('均方根误差', fontsize=20)
# plt.ylabel('RMSE', fontsize=20)
# plt.tight_layout()
# plt.show()