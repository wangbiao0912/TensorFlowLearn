from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib as mpl
from IPython.display import clear_output
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image

url = 'https://ss0.bdstatic.com/94oJfD_bAAcT8t7mm9GUKT-xh_/timg?image&quality=100&size=b4000_4000&sec=1576136828&di=92dee6fdb1f43f75b633aa503418397c&src=http://img.pconline.com.cn/images/upload/upc/tx/wallpaper/1209/26/c0/14139494_1348624365103.jpg'

# Download an image and read it into a NumPy array.
def download(url, target_size=None):
  name = url.split('/')[-1]
  image_path = tf.keras.utils.get_file(name, origin=url)
  img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
  return img

# Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)


# Display an image
def show(img):
  plt.figure(figsize=(12,12))
  plt.grid(False)
  plt.axis('off')
  plt.imshow(img)
  plt.show()

# Downsizing the image makes it easier to work with.
original_img = download(url, target_size=[225, 375])
original_img = np.array(original_img)

show(original_img)


print("下载数据完成-----")
#准备特征提取模型
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
# InceptionV3架构非常大（有关模型架构的图表，请参见TensorFlow的研究报告）。对于DeepDream，感兴趣的层是将卷积串联在一起的层。InceptionV3中有11层，名为“ mixed0”至“ mixed10”。使用不同的图层将产生不同的梦幻图像。较深的层对较高级的特征（例如，眼睛和面部）做出响应，而较早的层对较简单的特征（例如，边缘，形状和纹理）做出响应。请随意尝试以下选择的图层，但请记住，由于渐变计算的深度较大，较深的图层（具有较高索引的图层）将需要较长的训练时间。
# Maximize the activations of these layers
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
# 计算损失
# 损失是所选层中激活的总和。损耗在每一层均经过归一化处理，因此较大层的贡献不会超过较小层。通常，损耗是您希望通过梯度下降最小化的数量。在DeepDream中，您将通过梯度上升来最大化此损失。
def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.reduce_sum(losses)

# 渐变上升
# 一旦计算出所选图层的损耗，剩下的就是计算相对于图像的渐变，并将其添加到原始图像中。
#
# 将渐变添加到图像可以增强网络看到的图案。在每个步骤中，您将创建一个图像，该图像将越来越激发网络中某些层的激活。

@tf.function
def deepdream(model, img, step_size):
    with tf.GradientTape() as tape:
        # This needs gradients relative to `img`
        # `GradientTape` only watches `tf.Variable`s by default
        tape.watch(img)
        loss = calc_loss(img, model)

    # Calculate the gradient of the loss with respect to the pixels of the input image.
    gradients = tape.gradient(loss, img)

    # Normalize the gradients.
    gradients /= tf.math.reduce_std(gradients) + 1e-8

    # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
    # You can update the image by directly adding the gradients (because they're the same shape!)
    img = img + gradients * step_size
    img = tf.clip_by_value(img, -1, 1)

    return loss, img


def run_deep_dream_simple(model, img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    for step in range(steps):
        loss, img = deepdream(model, img, step_size)
        print()

        # if step% 10==0:
        #     clear_output(wait=True)
        #     show(deprocess(img))
        #     print("没调整10色，显示图片，Step {}, loss {}".format(step, loss))

        if step % 100 == 0:
            clear_output(wait=True)
            show(deprocess(img))
            print("退出了，Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    clear_output(wait=True)
    show(result)

    return result

print("最后数据是++++++++++++++++++++++++++++++++++")
dream_img = run_deep_dream_simple(model=dream_model, img=original_img,
                                  steps=100, step_size=0.01)

print("数据炫彩完了。。。。。。。")
# 调整照片质量
# 调高八度
# 很好，但是第一次尝试有一些问题：
#
# 输出有噪声（可能会tf.image.total_variation造成损失）。
# 图像分辨率低。
# 这些模式看起来好像都发生在相同的粒度上。
# 解决所有这些问题的一种方法是在不同规模上应用梯度上升。这将允许将较小比例尺生成的图案合并到较高比例尺的图案中，并附加其他细节。
#
# 为此，您可以执行以前的渐变上升方法，然后增加图像的大小（以八度表示），并对多个八度重复此过程。

OCTAVE_SCALE = 1.3

img = tf.constant(np.array(original_img))
base_shape = tf.cast(tf.shape(img)[:-1], tf.float32)

for n in range(3):
  new_shape = tf.cast(base_shape*(OCTAVE_SCALE**n), tf.int32)

  img = tf.image.resize(img, new_shape).numpy()

  img = run_deep_dream_simple(model=dream_model, img=img, steps=200, step_size=0.001)

clear_output(wait=True)
show(img)

# 照片移位置
# 放大瓷砖
# 要考虑的一件事是，随着图像尺寸的增加，执行梯度计算所需的时间和内存也将随之增加。上面的倍频程实现不适用于非常大的图像或许多倍频程。
#
# 为避免此问题，您可以将图像拆分为图块并计算每个图块的梯度。
#
# 在每次平铺计算之前对图像应用随机移位可防止出现平铺接缝。
#
# 首先实现随机移位：
def random_roll(img, maxroll):
  # Randomly shift the image to avoid tiled boundaries.
  shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
  shift_down, shift_right = shift[0],shift[1]
  img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)
  return shift_down, shift_right, img_rolled

shift_down, shift_right, img_rolled = random_roll(np.array(original_img), 512)
show(img_rolled)

# 平铺
# 这是deepdream先前定义的函数的平铺等效项：

@tf.function
def get_tiled_gradients(model, img, tile_size=512):
  shift_down, shift_right, img_rolled = random_roll(img, tile_size)

  # Initialize the image gradients to zero.
  gradients = tf.zeros_like(img_rolled)

  for x in tf.range(0, img_rolled.shape[0], tile_size):
    for y in tf.range(0, img_rolled.shape[1], tile_size):
      # Calculate the gradients for this tile.
      with tf.GradientTape() as tape:
        # This needs gradients relative to `img_rolled`.
        # `GradientTape` only watches `tf.Variable`s by default.
        tape.watch(img_rolled)

        # Extract a tile out of the image.
        img_tile = img_rolled[x:x+tile_size, y:y+tile_size]
        loss = calc_loss(img_tile, model)

      # Update the image gradients for this tile.
      gradients = gradients + tape.gradient(loss, img_rolled)

  # Undo the random shift applied to the image and its gradients.
  gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)

  # Normalize the gradients.
  gradients /= tf.math.reduce_std(gradients) + 1e-8

  return gradients

# 将它们放在一起可提供可扩展的，倍频感知的Deepdream实现：
def run_deep_dream_with_octaves(model, img, steps_per_octave=100, step_size=0.01,
                                num_octaves=3, octave_scale=1.3):
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    for octave in range(num_octaves):
        # Scale the image based on the octave
        if octave > 0:
            new_size = tf.cast(tf.convert_to_tensor(img.shape[:2]), tf.float32) * octave_scale
            img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(model, img)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

            if step % 10 == 0:
                clear_output(wait=True)
                show(deprocess(img))
                print("Octave {}, Step {}".format(octave, step))

    clear_output(wait=True)
    result = deprocess(img)
    show(result)

    return result


dream_img = run_deep_dream_with_octaves(model=dream_model, img=original_img, step_size=0.01)

clear_output()
show(original_img)
show(dream_img)
print("所有数据全部跑完了")