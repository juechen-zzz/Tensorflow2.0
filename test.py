"""
    简单测试1.0和2.0的差别
    TensorFlow 2 和 PyTorch 都是采用动态图(优先)模式开发，调试方便，所见即所得。
    一般来说，动态图模型开发效率高，但是运行效率可能不如静态图模式，TensorFlow 2 也支持通过 tf.function 将动态图优先模式的代码转化为静态图模式，实现开发 和运行效率的双赢。
"""

import tensorflow as tf
import numpy as np

#############################################################################

# 1.0版本

# a = tf.compat.v1.placeholder(tf.float32, name='variable_a')
# b = tf.compat.v1.placeholder(tf.float32, name='variable_b')
#
# c = tf.add(a, b, name='variable_c')
#
# sess = tf.compat.v1.InteractiveSession()
# init = tf.compat.v1.global_variables_initializer()
# sess.run(init)
#
# c_numpy = sess.run(c, feed_dict={a: 2., b: 2.})
# print(c_numpy)
#############################################################################

# 2.0版本

a = tf.constant(1.)                 # 创建标量
b = tf.constant(2.)
print('a+b:', a+b)

c = tf.constant(3.)
w = tf.constant(4.)

with tf.GradientTape() as tape:     # 构建梯度环境
    tape.watch([w])                 # 将w加入梯度跟踪列表
    y = a * w ** 2 + b * w + c

# 求导
[dy_dw] = tape.gradient(y, [w])
print(dy_dw)

# 精度转换
z = tf.constant(np.pi, dtype=tf.float16)
zz = tf.cast(z, tf.double)
print(zz)

# 创建全0或者全1张量,  tf.ones()  tf.zeros
q = tf.ones([3, 2])
qq = tf.ones_like(q)

# 创建指定元素值的张量 tf.fill()
w = tf.fill([2, 2], 99)



