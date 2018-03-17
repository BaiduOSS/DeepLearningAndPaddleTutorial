#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Authors: Jiahui Liu(2505774110@qq.com)
    Date:    2017/11/17 17:27:06
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np


def initialize_parameters(layer):
    """
    初始化参数

    Args:
        layer:各层所包含的节点数
    Return:
        parameters：参数，包括w和b
    """
    np.random.seed(2)
    parameters = {}
    # 随机初始化参数w，b初始化为0
    for i in range(len(layer) - 1):
        parameters['w' + str(i)] = np.random.randn(
            layer[i + 1], layer[i]) / np.sqrt(layer[i])
        parameters['b' + str(i)] = np.random.randn(layer[i + 1], 1) * 0

    return parameters


def forward_calculate(X, parameters):
    """
    前向计算
    Args:
        X: features
        parameters: 参数w和b
    Return:
        A: 包含输入和各层输出值
        Z: 包含隐藏层和输出层的中间值
    """

    A = []
    A.append(X)
    Z = []
    length = int(len(parameters) / 2)

    # 计算隐藏层
    for i in range(length - 1):
        # 加权、偏移
        z = np.dot(parameters['w' + str(i)], A[i]) + parameters['b' + str(i)]
        Z.append(z)
        # 激活
        a = np.maximum(0, z)
        A.append(a)

    # 计算输出层
    z = np.dot(parameters['w' + str(length - 1)], A[length - 1]) \
        + parameters['b' + str(length - 1)]
    Z.append(z)
    a = 1. / (1 + np.exp(-z))
    A.append(a)

    return A, Z


def calculate_cost(A, Y):
    """
    计算Cost

    Args:
        A: 存储输入值和各层输出值
        Y: 真实值
    Return:
        cost: 成本cost
    """

    m = Y.shape[1]  # 样本个数
    Y_out = A[len(A) - 1]  # 取模型输出值

    # 计算成本
    probability = np.multiply(
        np.log(Y_out), Y) + np.multiply(np.log(1 - Y_out), 1 - Y)
    cost = -1. / m * np.sum(probability)

    cost = np.squeeze(cost)  # 确保维度的正确性
    return cost


def update_parameters(p, dp, learning_rate):
    """
    更新参数
    Args:
        p: 参数
        dp: 该参数的梯度
        learning_rate: 学习步长
    Return:
        更新后的参数
    """
    return p - learning_rate * dp


def backward_calculate(A, Z, parameters, Y, learning_rate):
    """
    后向计算
    Args:
        A: 存储输入值和各层输出值
        Z: 存储各层中间值
        parameters: 参数包括w，b
        Y: 标签
        learning_rate: 学习步长
    Return:
        parameters: 更新后的参数
    """

    m = A[0].shape[1]
    length = int(len(parameters) / 2)

    # 反向计算：计算输出层
    da = - (np.divide(Y, A[length]) - np.divide(1 - Y, 1 - A[length]))
    dz = A[length] - Y

    # 反向计算：计算隐藏层
    for i in range(1, length):
        da = np.dot(parameters['w' + str(length - i)].T, dz)
        dz = da
        dz[Z[length - i - 1] <= 0] = 0

        # 更新参数
        dw = 1. / m * np.dot(dz, A[length - i - 1].T)
        db = 1. / m * np.sum(dz, axis=1, keepdims=True)
        parameters['w' + str(length - i - 1)] = update_parameters(
            parameters['w' + str(length - i - 1)], dw, learning_rate)
        parameters['b' + str(length - i - 1)] = update_parameters(
            parameters['b' + str(length - i - 1)], db, learning_rate)

    return parameters


def plot_costs(costs, learning_rate):
    """
    把cost图形化输出
    Args:
        costs: 训练迭代过程中的cost
        learning_rate: 学习步长
    Return:
    """
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("learning rate =" + str(learning_rate))
    plt.show()
    plt.savefig('costs.png')


def deep_neural_network(X, Y, layer, iteration_nums, learning_rate=0.0075):
    """
    深层神经网络模型计算(包含前向计算和后向计算)

    Args:
        X: 输入值
        Y: 真实值
        layer: 各层大小
        iteration_nums: 迭代次数
        learning_rate: 学习率
    Return:
        parameters: 模型训练所得参数，用于预测
    """

    # np.random.seed(1)
    costs = []

    # 参数初始化
    parameters = initialize_parameters(layer)

    # 训练
    for i in range(0, iteration_nums):
        # 正向计算
        A, Z = forward_calculate(X, parameters)

        # 计算成本函数
        Cost = calculate_cost(A, Y)

        # 反向计算并更新参数
        parameters = backward_calculate(A, Z, parameters, Y, learning_rate)

        # 每100次训练打印一次成本函数
        if i % 100 == 0:
            print "Cost after iteration %i: %f" % (i, Cost)
            costs.append(Cost)

    # plot_costs(costs, learning_rate)

    return parameters


def calc_accuracy(predictions, Y):
    """
    准确率计算
    Args:
        predictions: 预测结果
        Y: 标签即label
    Return:
        accuracy: 计算准确率
    """
    Y = np.squeeze(Y)
    right = 0
    for i in range(len(predictions)):
        if predictions[i] == Y[i]:
            right += 1
    accuracy = (right / float(len(predictions))) * 100
    return accuracy


def predict_image(parameters, X, Y):
    """
    使用模型进行预测来预测图片是否为猫（1 cat or 0 non-cat）

    Args:
        parameters: 包含权值和偏移量
        X: 数据，形状为(px_num * px_num * 3, number of examples)
        Y: 标签
    Return:
        accuracy: 准确率
    """

    # m为数据个数
    m = X.shape[1]

    A = []
    A.append(X)
    Z = []
    predictions = []

    # 预测结果,即前向传播过程
    A, Z = forward_calculate(X, parameters)

    # 取输出值Y_out，即A的最后一组数
    Y_out = A[len(A) - 1]

    # 将连续值Y_out转化为二分类结果0或1
    for i in range(m):
        if Y_out[0, i] >= 0.5:
            predictions.append(1)
        elif Y_out[0, i] < 0.5:
            predictions.append(0)

    return calc_accuracy(predictions, Y)


def load_data_sets():
    """
    用于从两个.h5文件中分别加载训练数据和测试数据

    Args:
    Return:
        train_x_ori: 原始训练数据集
        train_y: 原始训练数据标签
        test_x_ori: 原始测试数据集
        test_y: 原始测试数据标签
        classes(cat/non-cat): 分类list
    """
    train_data = h5py.File('datasets/train_images.h5', "r")
    # train images features
    train_x_ori = np.array(train_data["train_set_x"][:])
    # train images labels
    train_y_ori = np.array(train_data["train_set_y"][:])

    test_data = h5py.File('datasets/test_images.h5', "r")
    # test images features
    test_x_ori = np.array(test_data["test_set_x"][:])
    # test images labels
    test_y_ori = np.array(test_data["test_set_y"][:])
    # the list of classes
    classes = np.array(test_data["list_classes"][:])

    train_y_ori = train_y_ori.reshape((1, train_y_ori.shape[0]))
    test_y_ori = test_y_ori.reshape((1, test_y_ori.shape[0]))

    result = [train_x_ori, train_y_ori, test_x_ori,
              test_y_ori, classes]
    return result
