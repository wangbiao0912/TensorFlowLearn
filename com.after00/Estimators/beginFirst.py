from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

import pandas as pd
# 本文档中的示例程序构建并测试了一个模型，该模型根据花萼和花瓣的大小将鸢尾花分成三种物种。
#
# 您将使用鸢尾花数据集训练模型。该数据集包括四个特征和一个标签。这四个特征确定了单个鸢尾花的以下植物学特征：
#
# 花萼长度
# 花萼宽度
# 花瓣长度
# 花瓣宽度
# 根据这些信息，您可以定义一些有用的常量来解析数据：
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# 使用 Keras 与 Pandas 下载并解析鸢尾花数据集。注意为训练和测试保留不同的数据集。
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# 通过检查数据您可以发现有四列浮点型特征和一列 int32 型标签。
print(train.head())

# 对于每个数据集都分割出标签，模型将被训练来预测这些标签。
train_y = train.pop('PetalWidth')
test_y = test.pop('PetalWidth')
test.pop('Species')

# 标签列现已从数据中删除
print(test.head())


print(train.describe())