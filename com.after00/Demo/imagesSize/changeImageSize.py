import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 获取当前路径
data_root = pathlib.Path.cwd()
# 获取指定目录下的文件路径（返回是一个列表，每一个元素是一个PosixPath对象）
all_image_paths = list(data_root.glob('data/*.jpg'))
print(type(all_image_paths[0]))
# 将PosixPath对象转为字符串
all_image_paths = [str(path) for path in all_image_paths]
print(all_image_paths[0])
print(data_root)
# 获取所有图片的类标
# label_names = sorted(item.name for item in data_root.glob('*/*/') if item.is_dir())
# label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = pathlib.Path(all_image_paths[0]).name
print(all_image_labels)

# 测试使用
data_root = pathlib.Path('/Users/wangbiao/github/my/PythonFile/TensorFlowDemo/data/')
# 列出所以的图片的路径
train_dataset = tf.data.Dataset.list_files('data/6.jpg')
for x in iter(train_dataset):
    # for x in data_root.iterdir():
    print(x)
    # 读入图片
    image = tf.io.read_file(x)
    # 解码为tensor格式
    img_data = tf.image.decode_jpeg(image)
    print('shape:', img_data.shape, 'dtype:', img_data.dtype)
    plt.figure()
    plt.imshow(img_data)
    # plt.show()

    # 将图片转换问灰度图片，即最后一维只有1
    img_data = tf.image.resize(img_data, [100, 800], method=0)
        # resize_images(img_data, [100, 800], method=0)
    img_data = tf.image.rgb_to_grayscale(img_data)
    # 剪裁图片， 高度和宽度
    img_data = tf.image.resize_with_crop_or_pad(img_data, 3100, 2300)
    img_data = tf.image.adjust_brightness(img_data, -0.5)  # 亮度减少0.5
    # 图像对比度减少到0.5倍
    img_data = tf.image.adjust_contrast(img_data, 0.5)


    # 图像标准化，具体操作是将图像的亮度均值变为0，方差变为1
    img_data = tf.image.per_image_standardization(img_data)
    testName = tf.image.decode_image.__name__
    print(testName+"测试数据名称")

    fileName = str(pathlib.Path(str(x)).name)
    newFilePathName = "./newData/" + fileName
    print("》》》》》》》新路径" + newFilePathName)
    # 写入./221.jpg这个文件里面
    with tf.io.gfile.GFile("./newData/2.jpg", 'wb') as file:
        file.write(img_data.numpy())
    print("数据处理完----")



