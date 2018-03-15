#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Matplotlib模块绘制图像示例脚本
    Created on 2017-11-16
    author: denglelai
"""
import numpy as np
import matplotlib.pyplot as plt


def func(variable_x):
    """
    构建目标函数实现
    args:
        variable_x: 自变量
    return:
        np.square(variable_x): 目标函数
    """
    return np.square(variable_x)


def dfunc(variable_x):
    """
    目标函数一阶导数也即是偏导数实现
    args:
        variable_x: 目标函数
    return:
        2 * variable_x: 目标函数一阶导数
    """
    return 2 * variable_x


def gradient_descent(x_start, func_deri, epochs, learning_rate):
    """
    梯度下降法函数
    args:
        x_start: x的起始点
        func_deri: 目标函数的导函数
        epochs: 迭代周期
        learning_rate: 学习率
    return:
        xs: 求在epochs次迭代后x的更新值
    """
    theta_x = np.zeros(epochs + 1)
    temp_x = x_start
    theta_x[0] = temp_x
    for i in range(epochs):
        deri_x = func_deri(temp_x)
        # delta表示x要改变的幅度
        delta = - deri_x * learning_rate
        temp_x = temp_x + delta
        theta_x[i + 1] = temp_x
    return theta_x


def mat_plot():
    """
    使用matplot lib 绘制图像函数
    """
    line_x = np.linspace(- 5, 5, 100)
    line_y = func(line_x)
    x_start = - 5
    epochs = 5
    learning_rate = 0.3
    dot_x = gradient_descent(x_start, dfunc, epochs,
                             learning_rate=learning_rate)
    color = 'r'
    # plot实现绘制的主功能
    plt.plot(line_x, line_y, c='b')
    plt.plot(dot_x, func(dot_x), c=color,
             label='learning_rate={}'.format(learning_rate))
    plt.scatter(dot_x, func(dot_x), c=color, )
    # legend函数显示图例
    plt.legend()
    # show函数显示
    plt.show()


if __name__ == "__main__":
    mat_plot()
