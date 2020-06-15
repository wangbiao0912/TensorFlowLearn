# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)
# 下载IMDB数据集
# IMDB数据集已经打包在Tensorflow中。该数据集已经经过预先，评论（单词序列）已经被转换为序列序列，其中每个都是表示字典中的特定单词。
#
# 以下代码将下载IMDB数据集到您的机器上（如果您已经下载过重新缓存中复制）：
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#探索数据
# 让我们花一点时间来了解数据格式。该数据集是经过预先的：每个样本都是一个表示影评中词汇的整体细分。每个标签都是一个数值0或1的整数值，其中0代表消极评论，1代表积极评论。
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# 评论文本被转换为整数值，其中每个每个整体代表词典中的一个单词。首条评论是这样的
print("==========")
train_data[0]
# 电影评论可能具有不同的长度。以下代码显示了第一条和第二条评论的中单词数量。由于神经网络的输入必须是统一的长度，我们以后需要解决这个问题。
len(train_data[0]), len(train_data[1])


# 将孪生转换回单词
# 这里我们将创建一个辅助函数来查询一个包含了整体到串行映射的字典对象：
# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()
# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# 现在我们可以使用decode_review函数来显示首条评论的文本：
print(decode_review(train_data[0]))

# 准备数据
# 影评-即整体整数必须在输入神经网络之前转换为张量。这种转换可以通过以下两种方式来完成：
#
# 例如，序列[3，5]将转换为一个10,000维的向量，该向量除索引为3和5的位置是1以外，其他都为0。然后，将其作为网络的首层-一个可以处理浮点型矢量数据的稠密层。不过，这种方法需要大量的内存，需要一个大小为num_words * num_reviews的矩阵。
#
# 或者，我们可以max_length * num_reviews填充薄片来保证输入数据具有相同的长度，然后创建一个大小为的整型张量。我们可以使用能够处理此形状数据的嵌入层作为网络中的第一层。
#
# 在本教程中，我们将使用第二种方法。
#
# 由于电影评论长度必须相同，我们将使用pad_sequences函数来使长度标准化：
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# 现在让我们看下样本的长度：
print(len(train_data[0]))
print(len(train_data[1]))
# 并检查一下首条评论（当前已经填充）：
print(train_data[0])

# 建立模型
# 神经网络由堆叠的层来重建，这需要从两个主要方面来进行体系结构决策：
#
# 模型里有多少层？
# 每个层里有多少隐层单元（隐藏单元）？
# 在此样本中，输入数据包含一个单词索引的副本。要预测的标签为0或1。让我们来为该问题生成一个模型：
# 输入形状是用于电影评论的词汇数目（10,000 词）
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 创建验证集
# 在从原始训练数据中分离10,000个样本到创建一个验证集。（为什么现在不使用测试集？我们的目标？在训练时，我们想要检查模型在未见过的数据上的准确率（accuracy）。是只使用训练数据来开发和调整模型，然后只使用一次测试数据来评估准确率（accuracy））。
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
# 训练模型
# 以512个样本的mini-batch大小重复40个大纪元来训练模型。这是指对x_train和y_train张量中所有样本的的40次迭代。在训练过程中，监测来自验证集的10,000个样本上的损失值（损失）和准确率（准确性）：
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 评估模型
# 我们来看一下模型的性能如何。将返回两个值。损失值（loss）（一个表示误差的数字，值越低越好）与准确率（accuracy）。
results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)
#创建一个准确率（accuracy）和损失值（loss）随时间变化的图表
# model.fit()返回一个History对象，该对象包含一个字典，其中包含训练阶段所发生的一切事件：
history_dict = history.history
history_dict.keys()

# 有四个分量：在训练和验证期间，每个对应对应一个监控指标。我们可以使用这些合并来定位训练与验证过程的损失值（损失）和准确率（准确度），制动进行比较。

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 清除数字
plt.clf()   # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
# 在该图中，点代表训练损失值（loss）与准确率（accuracy），实线代表验证损失值（loss）与准确率（accuracy）。
#
# 注意在训练损失值随每一个epoch 下降而训练准确率（accuracy）随每一个epoch 上升。这在使用梯度下降优化时是可预期的-理应在每次迭代中最小化期望值。
#
# 验证过程的损失值（损失）与准确率（accuracy）的情况却并非如此-它们似乎在20个纪元后达到目标。这是过拟合的一个实例：模型在训练数据上的表现比在以前从未见过的数据上的表现要更好。在此之后，模型过度优化并学习特定于训练数据的表示，而不能够泛化到测试数据。
#
# 对于这种特殊情况，我们可以通过在20个左右的纪元后停止训练来避免过拟合。稍后，您将看到如何通过更改自动执行此操作。