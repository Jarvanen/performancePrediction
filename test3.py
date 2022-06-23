import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

student = pd.read_csv('C:/Users/Jarvan/Desktop/final2_refine3.csv')
label = student['pass']
# student = student.drop(['username', 'final', 'pass'], axis='columns')
student = student.drop(['username', 'final', 'pass'], axis='columns')

X_train, X_test, y_train, y_test = train_test_split(student, label, test_size=0.25, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 10))
model1 = LinearRegression()
model3 = RandomForestClassifier(n_estimators=100)
model4 = ExtraTreesClassifier(n_estimators=100)
model5 = SVC(kernel='rbf', degree=3, C=1.0, gamma='auto')
model6 = GradientBoostingClassifier(n_estimators=50)
model6.fit(X_train, y_train)

#y_predict_on_train = model.predict(X_train)
y_predict_on_test = model6.predict(X_test)
# for i in range(len(y_predict_on_test)):
#   if y_predict_on_test[i] >= 0.6:
#       y_predict_on_test[i]=1
#   else:
#       y_predict_on_test[i]=0
print(y_predict_on_test)
#print('trainacc:{:.2f}%'.format(100*accuracy_score(y_train, y_predict_on_train)))
print('acc:{:.2f}%'.format(100*accuracy_score(y_test, y_predict_on_test)))
print('pre:{:.2f}%'.format(100*precision_score(y_test, y_predict_on_test)))
print('recall:{:.2f}%'.format(100*recall_score(y_test, y_predict_on_test)))
print('f1:{:.2f}%'.format(100*f1_score(y_test, y_predict_on_test)))



