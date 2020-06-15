from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from IPython.display import clear_output
import matplotlib.pyplot as plt

# 加载数据集。
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

import tensorflow as tf
tf.random.set_seed(123)
print(dftrain.head())
print(dftrain.describe())
# dftrain.shape[0], dfeval.shape[0]

dftrain.age.hist(bins=100)
plt.show()
#
dftrain.sex.value_counts().plot(kind='barh')
plt.show()
dftrain['class'].value_counts().plot(kind='barh')
plt.show()
dftrain['embark_town'].value_counts().plot(kind='barh')
plt.show()
print(plt.style.core)
