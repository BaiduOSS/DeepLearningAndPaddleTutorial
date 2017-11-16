#!/usr/bin/env python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: yin xiaoting(y_tink@163.com)
Date:    2017/11/16

使用matplotlib.pyplot输出房屋价格与房屋面积的分布以及参数a，b生成的线性回归结果。
对比线性回归结果和真实分布，发现该回归结果可以大致根据房屋面积大小预测房屋价格。
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_data(data, a, b):
    """
    展示房屋价格与房屋面积的分布以及参数a，b生成的线性回归结果

    Args:
        data -- 房屋价格与房屋面积数据，存储在data.txt中
        a -- 线性回归拟合结果，斜率a
        b -- 线性回归拟合结果，截距b

    Return:
    """
    x = data[:, 0]
    y = data[:, 1]
    y_predict = x * a + b
    plt.scatter(x, y, marker='.', c='r', label='True')
    plt.title('House Price Distributions of Beijing Beiyuan Area in 2016/12')
    plt.xlabel('House Area ')
    plt.ylabel('House Price ')
    plt.xlim(0, 250)
    plt.ylim(0, 2500)
    predict = plt.plot(x, y_predict, label='Predict')
    plt.legend(loc='upper left')
    plt.savefig('result.png')
    plt.show()



data = np.loadtxt('data.txt', delimiter=',')
X_RAW = data.T[0].copy()
plot_data(data, 7.1, -62.3)
