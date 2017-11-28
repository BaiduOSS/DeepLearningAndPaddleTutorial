#!/usr/bin/env python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: Jiahui Liu(2505774110@qq.com)
Date:    2017/11/17 17:27:06

使用python及numpy库来实现深层神经网络识别猫案例，关键步骤如下：
1.载入数据和预处理：load_data()
2.初始化模型参数（Parameters）
3.循环：
    a)	计算成本（Cost）
    b)	计算梯度（Gradient）
    c)	更新参数（Gradient Descent）
4.利用模型进行预测
5.分析预测结果
6.定义model函数来按顺序将上述步骤合并
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import h5py
import dnn_app_utils_v2
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


def main():
    # 数据加载
    train_x_orig, train_y, test_x, test_y, classes = dnn_app_utils_v2.load_data()

    # 数据预处理
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x.reshape(test_x.shape[0], -1).T
    # 归一化
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    # 网络结构定义
    layers_dims = [12288, 20, 7, 5, 1]

    # 参数计算
    parameters = dnn_app_utils_v2.L_layer_model(train_x, train_y, layers_dims,
                                                num_iterations=2500, print_cost=True)

    # 准确率输出
    print('Train accuracy:')
    pred_train = dnn_app_utils_v2.predict(train_x, train_y, parameters)
    print('Test accuracy:')
    pred_test = dnn_app_utils_v2.predict(test_x, test_y, parameters)

if __name__=='__main__':
    main()