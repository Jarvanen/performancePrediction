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
student = pd.read_csv('E:/data/dataset/PerCD.csv')


# 选取final属性值
labels = student['final']
# 删除属性
# student = student.drop(['username', 'pass', 'variableScope', 'unusedVariable', 'unreadVariable', 'invalidscanf', 'knownConditionTrueFalse', 'getsCalled', 'selfAssignment'], axis='columns')
student = student.drop(['username'], axis='columns')
print(student)
# student = student.drop(['username','T1', 'T2', 'T3','c3score','c4score','c5score','c6score','c7score','c10score','jichutotal','jichuaccept','jichurate','panduantotal','panduanaccept','panduanrate','xunhuantotal','xunhuanaccept','xunhuanrate','shuzutotal','shuzuaccept','shuzurate','hanshutotal','hanshuaccept','hanshurate'], axis='columns')
# 对离散变量进行独热编码
# student = pd.get_dummies(student)
# 选取相关性最强的7个
most_correlated = student.corr().abs()['final'].sort_values(ascending=False)
most_correlated = most_correlated[:8]
print(most_correlated)


# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(student, labels, test_size=0.25, random_state=42)


# 通过训练集训练和测试集测试来生成多个线性模型
def evaluate(X_train, X_test, y_train, y_test):
    # 模型名称
    # model_name_list = ['Linear Regression', 'ElasticNet Regression',
    #                    'Random Forest', 'Extra Trees', 'SVM',
    #                    'Gradient Boosted', 'BayesianRidge', 'Baseline']
    model_name_list = ['SVM', 'BayesianRidge', 'Random Forest', 'Extra Trees',
                       'Gradient Boosted', 'DecisionTree']
    X_train = X_train.drop('final', axis='columns')
    X_test = X_test.drop('final', axis='columns')

    # 实例化模型
    # model1 = LinearRegression()
    # model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model3 = RandomForestRegressor(n_estimators=100)
    model4 = ExtraTreesRegressor(n_estimators=100)
    model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
    model6 = GradientBoostingRegressor(n_estimators=100)
    model7 = DecisionTreeRegressor(max_depth=3)
    model8 = BayesianRidge()

    # 结果数据框
    results = pd.DataFrame(columns=['mae', 'rmse'], index=model_name_list)

    # 每种模型的训练和预测
    for i, model in enumerate([model5, model8, model3, model4, model6, model7]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # print(predictions)
        # 误差标准
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

        # 将结果插入结果框
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]

    # 中值基准度量
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))

    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]

    return results


results = evaluate(X_train, X_test, y_train, y_test)
print(results)

