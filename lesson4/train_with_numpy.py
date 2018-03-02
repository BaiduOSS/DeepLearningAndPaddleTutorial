#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Authors: Jiahui Liu(2505774110@qq.com)
Date:    2017/11/17 17:27:06
使用python及numpy库来实现浅层神经网络识别螺旋图案，关键步骤如下：
1.载入数据和预处理：load_data_sets
2.初始化模型参数（Parameters）
3.循环：
    a)	计算成本（Cost）
    b)	计算梯度（Gradient）
    c)	更新参数（Gradient Descent）
4.搭建双层神经网络，利用模型进行预测
5.分析预测结果
6.定义neural_network函数来按顺序将上述步骤合并
"""
import numpy as np

import utils


def initialize_parameters(layer):
    """
    初始化参数
    Args:
        layer: 各层所包含的节点数
    Return:
        parameters: 存储权值和偏移量
    """

    np.random.seed(2)  # 设置随机种子
    parameters = {}
    # 随机初始化参数，偏移量初始化为0
    parameters['w0'] = np.random.randn(layer[1], layer[0]) * 0.01
    parameters['b0'] = np.random.randn(layer[1], 1) * 0
    parameters['w1'] = np.random.randn(layer[2], layer[1]) * 0.01
    parameters['b1'] = np.random.randn(layer[2], 1) * 0

    return parameters


def forward_calculate(X, parameters):
    """
    前向计算
    Args:
        X: 输入值
        parameters: 参数w，b
    Return:
        A: 包含输入和各层输出值
        Z: 包含隐藏层和输出层的中间值
    """

    A = []
    Z = []
    # 将输入值存进A中
    A.append(X)

    # 计算隐藏层
    z1 = np.dot(parameters['w0'], A[0]) + parameters['b0']
    Z.append(z1)
    a1 = np.tanh(z1)
    A.append(a1)

    # 计算输出层
    z2 = np.dot(parameters['w1'], A[1]) + parameters['b1']
    Z.append(z2)
    a2 = 1 / (1 + np.exp(-z2))
    A.append(a2)

    return A, Z


def calculate_cost(A, Y):
    """
    计算成本

    Args:
        A: 存储输入值和各层输出值
        Y: 标签
    Return:
        cost: 成本函数
    """

    m = Y.shape[1]  # 样本个数
    Y_out = A[len(A) - 1]  # 取模型输出值

    # 计算成本
    cost = -1. / m * np.sum(np.multiply(np.log(Y_out), Y)
                            + np.multiply(np.log(1 - Y_out), 1 - Y))
    cost = np.squeeze(cost)  # 确保维度的正确性
    return cost


def update_parameters(p, dp, learning_rate):
    """
    更新参数
    Args:
        p: 参数
        dp: 参数的梯度
        learning_rate:学习步长
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
        parameters: 包含权值和偏移量
        Y: 真实值
        learning_rate: 学习率
    Return:
        parameters: 更新后的参数
    """

    m = A[0].shape[1]

    # 后向传播
    # 计算dz2
    dz2 = A[2] - Y

    # 计算dw2
    dw2 = 1. / m * np.dot(dz2, A[1].T)

    # 计算db2
    db2 = 1. / m * np.sum(dz2, axis=1, keepdims=True)

    # 计算dz1
    dz1 = np.dot(parameters['w1'].T, dz2) * (1 - np.power(A[1], 2))

    # 计算dw1
    dw1 = 1. / m * np.dot(dz1, A[0].T)

    # 计算db1
    db1 = 1. / m * np.sum(dz1, axis=1, keepdims=True)

    parameters['w1'] = update_parameters(parameters['w1'], dw2, learning_rate)
    parameters['w0'] = update_parameters(parameters['w0'], dw1, learning_rate)
    parameters['b1'] = update_parameters(parameters['b1'], db2, learning_rate)
    parameters['b0'] = update_parameters(parameters['b0'], db1, learning_rate)
    return parameters


def neural_network(X, Y, layer, iteration_nums, learning_rate=1.2):
    """
    定义函数：神经网络模型

    Args:
        X: features
        Y: 标签label
        layer: 各层大小
        learning_rate: 学习步长
        iteration_nums: 迭代次数

    Return:
        parameters: 模型训练所得参数，用于预测
    """
    # 初始化参数
    parameters = initialize_parameters(layer)

    for i in range(0, iteration_nums):
        A = []
        Z = []
        A, Z = forward_calculate(X, parameters)

        Cost = calculate_cost(A, Y)

        parameters = backward_calculate(A, Z, parameters, Y, learning_rate)

        # 每1000次训练打印一次成本函数值
        if i % 1000 == 0:
            print "Cost after iteration %i: %f" % (i, Cost)
    return parameters


def calc_accuracy(predictions, Y):
    """
    计算准确率
    Args:
        predictions:预测值
        Y:标签
    Return:
        accuracy:准确率
    """
    Y = np.squeeze(Y)
    right = 0
    for i in range(len(predictions)):
        if predictions[i] == Y[i]:
            right += 1
    accuracy = (right / float(len(predictions))) * 100
    return accuracy


def predict_result(parameters, X, Y):
    """
    用模型来进行预测

    Args:
        parameters: 包含权值和偏移量
        X: 数据，形状为(px_num * px_num * 3, number of examples)
        Y: 标签
    Return:
        accuracy:预测准确率
    """
    data_dim = X.shape[0]
    # m为数据个数
    m = X.shape[1]

    A = []
    A.append(X)
    Z = []
    predictions = []

    # 预测结果,即前向计算过程
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


def main():
    """
    main entry
    """
    train_x, train_y, test_x, test_y = utils.load_data_sets()

    layer = [2, 4, 1]

    parameters = neural_network(train_x, train_y, layer, 10000)

    print('train:', predict_result(parameters, train_x, train_y))
    print('test:', predict_result(parameters, test_x, test_y))


if __name__ == '__main__':
    main()
