from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf


import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
# 下载 IMDB 数据集
# IMDB数据集可以在 Tensorflow 数据集处获取。以下代码将 IMDB 数据集下载至您的机器（或 colab 运行时环境）中：

# 将训练集按照 6:4 的比例进行切割，从而最终我们将得到 15,000
# 个训练样本, 10,000 个验证样本以及 25,000 个测试样本
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)
# 探索数据
# 让我们花一点时间来了解数据的格式。每个一个样本都是一个表示电影评论和相应标签的句子。该句子不以任何方式进行进行。标签是一个变量0或1的整体，其中0代表消极评论，1代表积极评论。
#
# 我们来打印下前十个样本。
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch

# 我们再打印下前十个标签。
train_labels_batch

# 建立模型
# 神经网络由堆叠的层来重建，这需要从三个主要方面来进行体系结构决策：
#
# 如何表示文本？
# 模型里有多少层？
# 每个层里有多少隐层单元（隐藏单元）？
# 本示例中，输入数据由句子组成。预测的标签为0或1。
#
# 表示文本的一种方式是将句子转换为嵌入矢量（嵌入向量）。我们可以使用一个预先训练好的文本嵌入（文本嵌入）作为首层，这将具有三个优点：
#
# 我们不必担心文本前置
# 我们可以从迁移学习中受益
# 嵌入具有固定长度，更容易处理
# 针对此示例我们将使用TensorFlow Hub中名为google / tf2-preview / gnews-swivel-20dim / 1的一种预训练文本嵌入（文本嵌入）模型。
#
# 为了达到本教程的目的还有其他三种预训练模型替代测试：
#
# google / tf2-preview / gnews-swivel-20dim-with-oov / 1-类似google / tf2-preview / gnews-swivel-20dim / 1，但2.5％的词汇转换为未登录词桶（OOV buckets）。如果任务的词汇与模型的词汇没有完全重叠，这将会有所帮助。
# google / tf2-preview / nnlm-zh-dim50 / 1-一个拥有约1M词汇量且尺寸为50的尺度的模型。
# google / tf2-preview / nnlm-en-dim128 / 1-拥有约1M词汇量且尺寸为128的尺寸的模型。
# 让我们首先创建一个使用Tensorflow Hub模型嵌入（嵌入）语句的Keras层，并在几个输入样本中进行尝试。请注意输入文本的长度如何，嵌入（嵌入）输出的形状都是：(num_examples, embedding_dimension)。
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/nnlm-zh-dim50/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
# 现在让我们完善完整模型：
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()
# 层按顺序层叠以重建分类器：
#
# 第一层是Tensorflow Hub层。这一层使用一个预训练的保存好的模型来将句子映射为嵌入矢量（嵌入向量）。我们所使用的预训练文本嵌入（embedding）模型（google / tf2-preview / gnews-swivel-20dim / 1）将句子切割为符号，嵌入（嵌入）每个符号然后进行合并。最终得到的尺寸是：(num_examples, embedding_dimension)。
# 该定长输出向量通过一个有16个隐层单元的全连接层（Dense）进行管道传输。
# 使用Sigmoid激活函数，其函数分解为0与1之间的浮点数，表示概率或放置信水平。
# 让我们编译模型。
#
# 损失函数与优化器
# 由于这是一个二分类问题且模型输出概率值（一个使用sigmoid激活函数的单个单元层），我们将使用binary_crossentropy损失函数。
#
# 这不是损失函数的唯一选择，例如，您可以选择mean_squared_error。但是，一般来说binary_crossentropy更适合处理概率-它能够估计概率分布之间的“距离”，或者在我们的示例中，指的是尺寸ground- true分布与预测值之间的“距离”。
#
# 稍后，当我们研究回归问题（例如，预测房价）时，我们将介绍如何使用另一种叫做均方误差的损失函数。
#
# 现在，配置模型来使用优化器和损失函数：
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 训练模型
# 以512个样本的mini-batch的大小重复20个epoch来训练模型。这是指对x_train和y_train张量中所有样本的的20次迭代。在训练过程中，监测来自验证集的10,000个样本上的损失值（损失）和准确率（准确性）：

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)
# 评估模型
# 我们来看下模型的表现如何。将返回两个值。损失值（loss）（一个表示误差的数字，值越低越好）与准确率（accuracy）。
# 49/49-2s-损失：0.3163-准确性：0.8651
# 损失：0.316
# 精度：0.865
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))