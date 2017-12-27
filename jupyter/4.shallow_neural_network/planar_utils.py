# -*- coding:utf-8 -*-
#!/usr/bin/env python
#  -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
Authors: Jiahui Liu(897744517@qq.com)
Date:    2017/11/19 17:23:06

用于载入数据
load_planar_dataset()函数返回训练集和测试集
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn
import sklearn.datasets
import sklearn.linear_model

#绘制分类结果边界
def plot_decision_boundary(model, X, y):
    """
    绘制分类边界
    model: 模型
    X: 输入值
    y: 真实值
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, : ].min() - 1, X[0, : ].max() + 1
    y_min, y_max = X[1, : ].min() - 1, X[1, : ].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, : ], X[1, : ], c=y, cmap=plt.cm.Spectral)
    
#定义Sigmoid函数
def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    # Sigmoid计算
    s = 1 / (1 + np.exp(-x))
    return s


#加载数据
def load_planar_dataset():
    """
    加载数据
    返回值：
        train_x:训练集的输入
        train_y:训练集的真实值
        X:测试集的输入
        Y:测试集的真实值
    """
    np.random.seed(1) #设置随机种子
    m = 400 # number of examples
    N = int(m / 2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m, D)) # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2 # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2 # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    #测试集，取全部数据
    X = X.T
    Y = Y.T

    #训练集
    train_num = random.sample(range(400),320)#共400组数据，训练集取其中80%，即320组
    train_x = X[: , train_num]
    train_y = Y[: , train_num]

    dataset = [train_x, train_y, X, Y]
    return dataset


#加载其他数据
def load_extra_datasets():
    """
    加载其它数据
    return:
        result:包含5种不同类型数据的集合
    """
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(
mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    result = [noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure]
    return result