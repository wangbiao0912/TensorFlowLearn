from __future__ import absolute_import, division, print_function, unicode_literals


import pandas as pd
import tensorflow as tf
import sys,os

# csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')
# csv_file = "/Users/wangbiao/github/my/PythonFile/TensorFlowDemo/dataFile/pandasDataframes/heart.csv"
csv_file = sys.path[2]+"/dataFile/pandasDataframes/heart.csv"
df = pd.read_csv(csv_file)
print(df.iloc[:2]) # head # 显示前5条数据
# 将 thal 列（数据帧（dataframe）中的 object ）转换为离散数值。
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
print("======================================================")
# print(df.head())

target = df.pop('target')  # 获取当列的数据
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))