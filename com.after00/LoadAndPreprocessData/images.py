from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import keras_applications
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

# 下载并检查数据集
# 检索图片
import os
import tarfile
import zipfile

# 解压文件夹
def extract_file(path, to_directory='.'):
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else:
        raise ValueError("Could not extract `%s` as no appropriate extractor is found" % path)

    cwd = os.getcwd()
    os.chdir(to_directory)

    try:
        file = opener(path, mode)
        try: file.extractall()
        finally: file.close()
    finally:
        os.chdir(data_root_orig)

# data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
#                                          fname='flower_photos', untar=True)
data_root_orig = "/Users/wangbiao/github/my/PythonFile/TensorFlowDemo/dataFile/images/flower_photos/"
# extract_file("/Users/wangbiao/github/my/PythonFile/TensorFlowDemo/dataFile/images/flower_photos.tgz",data_root_orig)
data_root = pathlib.Path(data_root_orig)

print(data_root)
# 获取图片地址
for item in data_root.iterdir():
  print(item)



import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print("总共有图片：{}",image_count)
print(all_image_paths[:10])
# 查看图片
import os

attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

import IPython.display as display


def caption_image(image_path):
    print("=========华丽的分割线============{}",image_path)
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])
for n in range(3):
  image_path = random.choice(all_image_paths)
  display.display(display.Image(image_path))
  print(caption_image(image_path))
  print()


#列出标签
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)
# 为每个标签分配索引：

label_to_index = dict((name, index) for index, name in enumerate(label_names))
print(label_to_index)

# 创建一个列表，包含每个文件的标签索引：

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])

# 加载和格式化图片
# TensorFlow 包含加载和处理图片时你需要的所有工具：
img_path = all_image_paths[0]
print(img_path)
# 以下是原始数据：
img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100]+"...")
# 将它解码为图像 tensor（张量）：
img_tensor = tf.image.decode_image(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)

# 根据你的模型调整其大小：
img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())
# 将这些包装在一个简单的函数里，以备后用。
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image
def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)
import matplotlib.pyplot as plt

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(img_path))
plt.title(label_names[label].title())
print()


# 将字符串数组切片，得到一个字符串数据集：
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# shapes（维数） 和 types（类型） 描述数据集里每个数据项的内容。在这里是一组标量二进制字符串。
print(path_ds)
# 现在创建一个新的数据集，通过在路径数据集上映射 preprocess_image 来动态加载和格式化图片。
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
for n, image in enumerate(image_ds.take(4)):
  plt.subplot(2,2,n+1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(caption_image(all_image_paths[n]))
  plt.show()

# 一个(图片, 标签)对数据集
# 使用同样的 from_tensor_slices 方法你可以创建一个标签数据集：
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
for label in label_ds.take(10):
  print(label_names[label.numpy()])

# 由于这些数据集顺序相同，你可以将他们打包在一起得到一个(图片, 标签)对数据集：
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
# 这个新数据集的 shapes（维数） 和 types（类型） 也是维数和类型的元组，用来描述每个字段：
print(image_label_ds)
# 注意：当你拥有形似 all_image_labels 和 all_image_paths 的数组，tf.data.dataset.Dataset.zip 的替代方法是将这对数组切片。
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
image_label_ds

# 训练的基本方法
# 要使用此数据集训练模型，你将会想要数据：
#
# 被充分打乱。
# 被分割为 batch。
# 永远重复。
# 尽快提供 batch。
# 使用 tf.data api 可以轻松添加这些功能。
BATCH_SIZE = 32

# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

# 顺序很重要。
#
# 在 .repeat 之后 .shuffle，会在 epoch 之间打乱数据（当有些数据出现两次的时候，其他数据还没有出现过）。
#
# 在 .batch 之后 .shuffle，会打乱 batch 的顺序，但是不会在 batch 之间打乱数据。
#
# 你在完全打乱中使用和数据集大小一样的 buffer_size（缓冲区大小）。较大的缓冲区大小提供更好的随机化，但使用更多的内存，直到超过数据集大小。
#
# 在从随机缓冲区中拉取任何元素前，要先填满它。所以当你的 Dataset（数据集）启动的时候一个大的 buffer_size（缓冲区大小）可能会引起延迟。
#
# 在随机缓冲区完全为空之前，被打乱的数据集不会报告数据集的结尾。Dataset（数据集）由 .repeat 重新启动，导致需要再次等待随机缓冲区被填满。
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

# 从 tf.keras.applications 取得 MobileNet v2 副本。
#
# 该模型副本会被用于一个简单的迁移学习例子。
#
# 设置 MobileNet 的权重为不可训练：
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False
# 该模型期望它的输出被标准化至 [-1,1] 范围内：
help(keras_applications.mobilenet_v2.preprocess_input)
def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)
# 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

# 构建一个包装了 MobileNet 的模型并在 tf.keras.layers.Dense 输出层之前使用 tf.keras.layers.GlobalAveragePooling2D 来平均那些空间向量：
model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names), activation = 'softmax')])
# 现在它产出符合预期 shape(维数)的输出：
logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)
# 编译模型以描述训练过程：
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
# 此处有两个可训练的变量 —— Dense 层中的 weights（权重） 和 bias（偏差）：
len(model.trainable_variables)
model.summary()
steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
# print(steps_per_epoch+"：：：：：：：")


# 上面使用的简单 pipeline（管道）在每个 epoch 中单独读取每个文件。在本地使用 CPU 训练时这个方法是可行的，但是可能不足以进行 GPU 训练并且完全不适合任何形式的分布式训练。
#
# 要研究这点，首先构建一个简单的函数来检查数据集的性能：
import time
default_timeit_steps = 2*steps_per_epoch+1

def timeit(ds, steps=default_timeit_steps):
  overall_start = time.time()
  # 在开始计时之前
  # 取得单个 batch 来填充 pipeline（管道）（填充随机缓冲区）
  it = iter(ds.take(steps+1))
  next(it)

  start = time.time()
  for i,(images,labels) in enumerate(it):
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
  print("Total time: {}s".format(end-overall_start))

# 当前数据集的性能是：
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
print(ds)
print(timeit(ds))
# 缓存
# 使用 tf.data.Dataset.cache 在 epoch 之间轻松缓存计算结果。这是非常高效的，特别是当内存能容纳全部数据时。
# 在被预处理之后（解码和调整大小），图片在此被缓存了：

ds = image_label_ds.cache()
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
print(ds)
print(timeit(ds))
# 如果内存不够容纳数据，使用一个缓存文件：
ds = image_label_ds.cache(filename='./cache.tf-data')
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(1)
print(ds)
print(timeit(ds))

# 原始图片数据
# TFRecord 文件是一种用来存储一串二进制 blob 的简单格式。通过将多个示例打包进同一个文件内，TensorFlow 能够一次性读取多个示例，当使用一个远程存储服务，如 GCS 时，这对性能来说尤其重要。
#
# 首先，从原始图片数据中构建出一个 TFRecord 文件：
image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(image_ds)

# 接着，构建一个从 TFRecord 文件读取的数据集，并使用你之前定义的 preprocess_image 函数对图像进行解码/重新格式化：
image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)
# 压缩该数据集和你之前定义的标签数据集以得到期望的 (图片,标签) 对：
ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
print(ds)
print(timeit(ds))
# 序列化的 Tensor（张量）
# 要为 TFRecord 文件省去一些预处理过程，首先像之前一样制作一个处理过的图片数据集：
paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = paths_ds.map(load_and_preprocess_image)
print(image_ds)
# 现在你有一个 tensor（张量）数据集，而不是一个 .jpeg 字符串数据集。
#
# 要将此序列化至一个 TFRecord 文件你首先将该 tensor（张量）数据集转化为一个字符串数据集：
ds = image_ds.map(tf.io.serialize_tensor)
print(ds)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(ds)
# 有了被缓存的预处理，就能从 TFrecord 文件高效地加载数据——只需记得在使用它之前反序列化：
ds = tf.data.TFRecordDataset('images.tfrec')

def parse(x):
  result = tf.io.parse_tensor(x, out_type=tf.float32)
  result = tf.reshape(result, [192, 192, 3])
  return result

ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
print(ds)
# 现在，像之前一样添加标签和进行相同的标准操作：
ds = tf.data.Dataset.zip((ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
print(ds)