"""
    简单测试1.0和2.0的差别
    TensorFlow 2 和 PyTorch 都是采用动态图(优先)模式开发，调试方便，所见即所得。
    一般来说，动态图模型开发效率高，但是运行效率可能不如静态图模式，TensorFlow 2 也支持通过 tf.function 将动态图优先模式的代码转化为静态图模式，实现开发 和运行效率的双赢。
"""

import tensorflow as tf

#############################################################################
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

a = tf.constant(2.)
b = tf.constant(4.)
print(a+b)