"""
    写一个单输入神经元线性模型，运用梯度下降
"""

import numpy as np
import tensorflow as tf

data = []
for i in range(100):
    x = np.random.uniform(-10., 10.)        # 从-10到10的均匀分布中采样
    eps = np.random.normal(0., 0.1)         # 从高斯分布N(0,0.1)中采样
    y = 1.4 * x + 0.2 + eps
    data.append([x, y])
data = np.array(data)                       # 转换为2D Numpy数组
print("data尺寸：", data.size)

# 计算误差，使用交叉熵（MSE）
def mse(b, w ,points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

# 计算梯度
def step_gradient(b_current, w_current, points, lr):
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        b_gradient += (2/M) * (w_current * x + b_current - y)
        w_gradient += (2/M) * (w_current * x + b_current - y) * x

    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]

# 梯度更新
def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    b = starting_b
    w = starting_w

    for step in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)
        if step % 50 == 0:
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [b, w]

if __name__ == '__main__':
    lr = 0.01
    initial_b = 0
    initial_w = 0
    num_iterations = 100

    [b, w]= gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)
    print(f'Final loss:{loss}, w:{w}, b:{b}')
