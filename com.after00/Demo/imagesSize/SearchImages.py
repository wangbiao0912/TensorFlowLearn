import os
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from six.moves.urllib.request import urlretrieve
from keras_applications.vgg16 import VGG16

# 获取图片
images = []
img_dir = 'data'
for idx, img_name in enumerate(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, img_name)
    print('fetching pic:', img_path)
    img = PIL.Image.open(img_path)
    img = img.resize((299, 299))
    images.append(img)
NUM = len(images)


# 下载 inceptionV3 模型
data_dir = '.'
checkpoint_filename = os.path.join(data_dir, 'inception_v3.ckpt')
if not os.path.exists(checkpoint_filename):
    print('downloading inceptionV3 model...')
    inception_tarball, _ = urlretrieve(
        'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
    tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)

# 计算流 （注意：299,299,3,1001是inceptionV3的标准参数
image = tf.Variable(tf.zeros((299, 299, 3)))
preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
with slim.arg_scope(arg_scope):
    logits, end_points = nets.inception.inception_v3(
        preprocessed, 1001, is_training=False, reuse=False)
    # logits = logits[:, 1:]  # ignore background class
    # probs = tf.nn.softmax(logits)  # probabilities
