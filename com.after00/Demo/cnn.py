# CNN简单实现手写数字集识别
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
# 载入mnist数据集

train_images = train_images.reshape((-1,28,28,1))
test_images = test_images.reshape((-1,28,28,1))
# 对数据格式进行改造

model = keras.Sequential([
        # layers.Conv2D(input_shape=(train_images.shape[1], train_images.shape[2], train_images.shape[3]), filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
        layers.Conv2D(input_shape=(train_images.shape[1], train_images.shape[2], train_images.shape[3]),filters=32, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=(2,2)),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
])
# 搭建神经网络层
model.compile(optimizer=keras.optimizers.Adam(),
             # loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
             loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
# 装配，进行配参

model.summary()
# 打印神经网络层
history = model.fit(train_images, train_labels, batch_size=64, epochs=5, validation_split=0.1)
# 开始训练
model.evaluate(test_images, test_labels)
# 训练之后进行预测
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()
# 取得训练过程中的参数，并用plt作图展示出来
model.save('model.h5')
# 保存模型

del model
# 删除模型
print("del model")

network = tf.keras.models.load_model('model.h5', compile=False)
# 载入模型
network.compile(optimizer=keras.optimizers.Adam(),
             # loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
             loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

network.evaluate(test_images, test_labels)
# 这里利用测试数据集进行预测，最好将测试集切片或者新建另一个测试集进行预测