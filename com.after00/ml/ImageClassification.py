from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
# 导入文件集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 每个图像都映射到一个标签。由于类名不包含在数据集中，因此将它们存储在此处以供以后在绘制图像时使用：
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))


# 预处理数据
# 在训练网络之前，必须对数据进行预处理。如果检查训练集中的第一张图像，您将看到像素值落在0到255的范围内：
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 在将它们输入神经网络模型之前，将这些值缩放到0到1的范围。为此，将值除以255。以相同的方式预处理训练集和测试集非常重要：
train_images = train_images / 255.0
test_images = test_images / 255.0

# 为了验证数据的格式正确，并准备好构建和训练网络，让我们显示训练集中的前25张图像，并在每个图像下方显示类别名称。
plt.figure(figsize=(5,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


#设置图层
#神经网络的基本组成部分是层。图层从输入到其中的数据中提取表示。希望这些表示对于当前的问题有意义。

# 深度学习的大部分内容是将简单的层链接在一起。大多数层（例如tf.keras.layers.Dense）具有在训练期间学习的参数。
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# 编译模型
# 在准备训练模型之前，需要进行一些其他设置。这些是在模型的编译步骤中添加的：
#
# 损失函数 -衡量训练期间模型的准确性。您希望最小化此功能，以在正确的方向上“引导”模型。
# 优化器 -这是基于模型看到的数据及其损失函数来更新模型的方式。
# 指标 -用于监视培训和测试步骤。以下示例使用precision，即正确分类的图像比例。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
# 训练神经网络模型需要执行以下步骤：
#
# 将训练数据输入模型。在此示例中，训练数据在train_images和train_labels数组中。
# 该模型学习关联图像和标签。
# 您要求模型对测试集（在本例中为test_images数组）做出预测。验证预测是否与test_labels阵列中的标签匹配。
# 要开始训练，请调用该model.fit方法，之所以这么称呼是因为该方法使模型“适合”训练数据：
print(model.fit(train_images, train_labels, epochs=10))
# 评估准确性
# 接下来，比较模型在测试数据集上的表现：10000/1-1s-损耗：0.2934-精度：0.8830
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)




# 作出预测
# 通过训练模型，您可以使用它来预测某些图像。
predictions = model.predict(test_images)
# 在这里，模型已经预测了测试集中每个图像的标签。让我们看一下第一个预测：


print("==============")
predictions[0]

# 图片分类最高标准的
print(np.argmax(predictions[0]));

img = test_images[0]

print(img.shape)

# 以图形方式查看完整的10个类预测。

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#让我们看一下第0张图片，预测和预测数组。正确的预测标签为蓝色，错误的预测标签为红色。该数字给出了预测标签的百分比（满分为100）。
print("---------华丽分割线------------")

j = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(j, predictions[j], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(j,predictions[j], test_labels)
plt.show()

# 分析数据  显示15张

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()