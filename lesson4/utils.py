#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Authors: Jiahui Liu(2505774110@qq.com)
    Date:    2017/11/17 17:27:06
    based on http://cs231n.github.io/neural-networks-case-study/
"""
import random

import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, X, Y):
    """
    把数据和分界展示在图上
    Args:
        model: 模型
        X: 数据集的X
        Y: 数据集的标签
    Return:

    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)


def load_data_sets():
    """
    加载数据
    Args：
    Return:
        train_x:训练集的X
        train_y:训练集的label
        X:测试集的X
        Y:测试集的label
    """
    N = 200  # number of points per class
    D = 2  # dimensionality
    K = 2  # number of classes
    number = N * K
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + \
            np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    X = X.T
    Y = y.reshape(1, number)

    # 训练集
    train_num = random.sample(range(number), 320)  # 共400组数据，训练集取其中80%，即320组
    train_x = X[:, train_num]
    train_y = Y[:, train_num]

    result = [train_x, train_y, X, Y]
    return result
