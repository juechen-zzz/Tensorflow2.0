"""
    调用TensorFlow2.0来实现mnist识别
"""

import os
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers, optimizers, datasets

# load_data返回两个元组，第一个是训练集，第二个是测试集
# 训练集大小为（60000，28，28）(60000,10)
# 测试集大小为（10000，28，28）
(x, y), (x_val, y_val) = datasets.mnist.load_data()

x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255.-1       # 转化为张量，缩放到（-1，1）

y = tf.convert_to_tensor(y, dtype=tf.int32)

"""
# one_hot编码是非常稀疏的，相对于数字编码来说，占用较多的存储空间，一般在存储时使用数字编码，计算时需要就转换为one_hot编码
indices = [[0, 2], [1, -1]]
depth = 3
tf.one_hot(indices, depth,
           on_value=1.0, off_value=0.0,
           axis=-1)  # output: [2 x 2 x 3]
# [[[1.0, 0.0, 0.0],   # one_hot(0)
#   [0.0, 0.0, 1.0]],  # one_hot(2)
#  [[0.0, 1.0, 0.0],   # one_hot(1)
#   [0.0, 0.0, 0.0]]]  # one_hot(-1)
y = tf.one_hot(y, depth=10)
"""

print(x.shape, y.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x, y))      # 构建数据集对象

train_dataset = train_dataset.batch(512)                        # 批量训练

model = keras.Sequential([
    # Dense：全连接层
    layers.Dense(256, activation='relu')
    layers.Dense(128, activation='relu')
    layers.Dense(10)
])

with tf.GradientTape() as tape:
    x = tf.reshape(x, (-1, 28 * 28))                        # 打平
    out = model(x)                                          # 得到模型输出output

# grads = tape.gradient(loss, model.trainable_variables)
#
# optimizer.apply_gradients(zip(grads, model.trainable_variables))


