#!/usr/bin/env python2.7
# -*- encoding:utf-8 -*-
"""
Matplotlib模块绘制图像示例脚本
Created on 2017-11-16
@author: denglelai@baidu.com
@copyright: www.baidu.com
"""
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    """
    构建目标函数实现
    args:
        x: 自变量
    reurn:
        np.square(x): 目标函数
    """
    return np.square(x)


def dfunc(x):
    """
    目标函数一阶导数也即是偏导数实现
    args:
        x: 目标函数
    reurn:
        2 * x: 目标函数一阶导数
    """
    return 2 * x


def gradient_descent(x_start, df, epochs, lr):
    """
    梯度下降法函数
    args:
        x_start: x的起始点
        df: 目标函数的一阶导函数
        epochs: 迭代周期
        lr: 学习率
        x在每次迭代后的位置（包括起始点），长度为epochs+1
    retun:
        xs: 求在epochs次迭代后x的更新值
    """
    xs = np.zeros(epochs + 1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
        # v表示x要改变的幅度
        v = - dx * lr
        x = x + v
        xs[i + 1] = x
    return xs


def mat_plot():
    """
    Matplotlib绘制图像函数    
    """
    line_x = np.linspace(- 5, 5, 100)
    line_y = func(line_x)
    x_start = - 5
    epochs = 5
    lr = 0.3
    x = gradient_descent(x_start, dfunc, epochs, lr=lr)
    color = 'r'
    # plot实现绘制的主功能
    plt.plot(line_x, line_y, c='b')
    plt.plot(x, func(x), c=color, label='lr={}'.format(lr))
    plt.scatter(x, func(x), c=color, )
    # legend函数显示图例
    plt.legend()
    # show函数显示
    plt.show()


if __name__ == "__main__":
    mat_plot()

