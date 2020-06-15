from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = "/Users/wangbiao/github/my/PythonFile/TensorFlowDemo/dataFile/csv/train.csv"
# tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
test_file_path = "/Users/wangbiao/github/my/PythonFile/TensorFlowDemo/dataFile/csv/eval.csv"
# 让 numpy 数据更易读。
np.set_printoptions(precision=3, suppress=True)
# 加载数据
# 开始的时候，我们通过打印 CSV 文件的前几行来了解文件的格式。
# !head {train_file_path}
print(train_file_path + "》》》》》》")

# 对于包含模型需要预测的值的列是你需要显式指定的。
LABEL_COLUMN = 'survived'
LABELS = [0, 1]


# 现在从文件中读取 CSV 数据并且创建 dataset。
#
# (完整的文档，参考 tf.data.experimental.make_csv_dataset)
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,  # 为了示例更容易展示，手动设置较小的值
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True)
    return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

# dataset 中的每个条目都是一个批次，用一个元组（多个样本，多个标签）表示。样本中的数据组织形式是以列为主的张量（而不是以行为主的张量），每条数据中包含的元素个数就是批次大小（这个示例中是 12）。
#
# 阅读下面的示例有助于你的理解。
examples, labels = next(iter(raw_train_data))  # 第一个批次
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

# 数据预处理
# 分类数据
# CSV 数据中的有些列是分类的列。也就是说，这些列只能在有限的集合中取值。
#
# 使用 tf.feature_column API 创建一个 tf.feature_column.indicator_column 集合，每个 tf.feature_column.indicator_column 对应一个分类的列。
CATEGORIES = {
    'sex': ['male', 'female'],
    'class': ['First', 'Second', 'Third'],
    'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone': ['y', 'n']
}
categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    # 你刚才创建的内容
    categorical_columns


#   连续数据
# 连续数据需要标准化。
#
# 写一个函数标准化这些值，然后将这些值改造成 2 维的张量。

def process_continuous_data(mean, data):
    # 标准化数据
    data = tf.cast(data, tf.float32) * 1 / (2 * mean)
    return tf.reshape(data, [-1, 1])


# 现在创建一个数值列的集合。tf.feature_columns.numeric_column API 会使用 normalizer_fn 参数。在传参的时候使用 functools.partial，functools.partial 由使用每个列的均值进行标准化的函数构成。
MEANS = {
    'age': 29.631308,
    'n_siblings_spouses': 0.545455,
    'parch': 0.379585,
    'fare': 34.385399
}

numerical_columns = []

for feature in MEANS.keys():
    num_col = tf.feature_column.numeric_column(feature,
                                               normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
    numerical_columns.append(num_col)

    # 你刚才创建的内容。
    numerical_columns

    # 创建预处理层
    # 将这两个特征列的集合相加，并且传给
    # tf.keras.layers.DenseFeatures
    # 从而创建一个进行预处理的输入层。
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numerical_columns)
# 构建模型
# 从 preprocessing_layer 开始构建 tf.keras.Sequential。
model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# 训练、评估和预测
# 现在可以实例化和训练模型。
train_data = raw_train_data.shuffle(500)
test_data = raw_test_data
model.fit(train_data, epochs=20)
# 当模型训练完成的时候，你可以在测试集 test_data 上检查准确性。
test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
# 使用 tf.keras.Model.predict 推断一个批次或多个批次的标签。

predictions = model.predict(test_data)

# 显示部分结果
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print("Predicted survival: {:.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("SURVIVED" if bool(survived) else "DIED"))
