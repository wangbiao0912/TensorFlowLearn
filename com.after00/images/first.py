from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import psutil
print(psutil.cpu_count())# CPU逻辑数量
print(psutil.cpu_count(logical=False))# CPU物理核心
print(psutil.cpu_times()) #统计CPU的用户／系统／空闲时间：
# 再实现类似top命令的CPU使用率，每秒刷新一次，累计10次：
for x in range(10):
    psutil.cpu_percent(interval=1, percpu=True)


# 获取root权限  psutil.net_connections()
#获取物理内存和交换内存
print(psutil.virtual_memory())
print(psutil.swap_memory())
print(psutil.disk_partitions())# 磁盘分区信息

psutil.disk_usage('/') # 磁盘使用情况
psutil.disk_io_counters() # 磁盘IO
psutil.net_io_counters() # 获取网络读写字节／包的个数
psutil.net_if_addrs() # 获取网络接口信息
psutil.net_if_stats() # 获取网络接口状态
psutil.pids() # 所有进程ID
psutil.cpu_times() # 进程使用的CPU时间
psutil.test()


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# 验证数据集是正确的,我们先画出25从训练集图像,显示每幅图下面的类名。
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

print(model.summary())