import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


dataset = pd.read_csv('E:/data/dataset/5.csv')
dataset = dataset.drop(['username'], axis='columns')
print(dataset)

# 拆分训练数据集和测试数据集
train_dataset = dataset.sample(frac=0.75, random_state=42)
test_dataset = dataset.drop(train_dataset.index)
# 数据检查
# 快速查看训练集中几对列的联合分布。
# sns.pairplot(train_dataset[["final"]], diag_kind="kde")

# 也可以查看总体的数据统计:
train_stats = train_dataset.describe()
train_stats.pop("final")
train_stats = train_stats.transpose()
#print(train_stats)

# 从标签中分离特征
train_labels = train_dataset.pop('final')
test_labels = test_dataset.pop('final')

# 数据规范化
# def norm(x):
#   return (x - train_stats['mean']) / train_stats['std']
# normed_train_data = norm(train_dataset)
# normed_test_data = norm(test_dataset)

# 构建模型
def build_model():
  model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_absolute_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model
model = build_model()

# 检查模型
model.summary()

# # 现在试用下这个模型。从训练数据中批量获取‘10’条例子并对这些例子调用 model.predict 。
# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)
# # print(example_result)

# 训练模型
# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split=0.2, verbose=0,
  callbacks=[PrintDot()])

# 使用 history 对象中存储的统计信息可视化模型的训练进度。
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [final]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,30])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$final^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,30])
  plt.legend()
  plt.show()


# plot_history(history)

# 该图表显示在约100个 epochs 之后误差非但没有改进，反而出现恶化。 让我们更新 model.fit 调用，当验证值没有提高上是自动停止训练。
model = build_model()

# patience 值用来检查改进 epochs 的数量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_dataset, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

# plot_history(history)

loss, mean_absolute_error, mean_squared_error = model.evaluate(train_dataset, train_labels, verbose=2)

print("Training set Mean Abs Error: {:5.2f} final".format(mean_absolute_error))
print("Training set Root Mean Squ Error: {:5.2f} final".format(np.sqrt(mean_squared_error)))

loss, mean_absolute_error, mean_squared_error = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} final".format(mean_absolute_error))
print("Testing set Root Mean Squ Error: {:5.2f} final".format(np.sqrt(mean_squared_error)))

# 最后，使用测试集中的数据预测 final 值:
test_predictions = model.predict(test_dataset).flatten()

# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [final]')
# plt.ylabel('Predictions [final]')
# plt.axis('equal')
# plt.axis('square')
# plt.xlim([0,plt.xlim()[1]])
# plt.ylim([0,plt.ylim()[1]])
# _ = plt.plot([-100, 100], [-100, 100])
# plt.show()
#
# error = test_predictions - test_labels
# plt.hist(error, bins = 25)
# plt.xlabel("Prediction Error [final]")
# _ = plt.ylabel("Count")
# plt.show()
