from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

print("程序开始了====================")
import numpy as np
import PIL.Image as Image

grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
print(grace_hopper)

grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape
# 添加批次尺寸，然后将图像传递给模型。
result = classifier.predict(grace_hopper[np.newaxis, ...])
result.shape
# 结果是logit的1001元素向量，对图像的每个类别的概率进行评级。
# 因此，可以使用argmax找到顶级类ID：
predicted_class = np.argmax(result[0], axis=-1)
predicted_class
# 解码预测
# 我们具有预测的类ID，获取ImageNet标签并解码预测
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())

# 数据集
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
# 结果对象是返回image_batch, label_batch对的迭代器。
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)
for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

  # 现在在图像批处理上运行分类器。
  result_batch = classifier.predict(image_batch)
  result_batch.shape
  predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
  predicted_class_names

  # 现在检查这些预测如何与图像对齐：
  plt.figure(figsize=(10, 9))
  plt.subplots_adjust(hspace=0.5)
  for n in range(30):
      plt.subplot(6, 5, n + 1)
      plt.imshow(image_batch[n])
      plt.title(predicted_class_names[n])
      plt.axis('off')
  _ = plt.suptitle("ImageNet predictions")