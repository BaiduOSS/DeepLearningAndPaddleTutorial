#!/usr/bin/env python
# -*- coding:gbk -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: Jiahui Liu(2505774110@qq.com)
Date:    2017/11/17 17:27:06
使用python及numpy库来实现浅层神经网络识别螺旋图案，关键步骤如下：
1.载入数据和预处理：load_planar_dataset
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
import planar_utils

<<<<<<< HEAD
def initialize_parameters(layer):
=======

# 定义函数：设置网络结构
def layer_sizes(X, Y):
    """
    参数含义:
    X -- 输入的数据
    Y -- 输出值

    返回值:
    n_x -- 输入层节点数
    n_h -- 隐藏层节点数
    n_y -- 输出层节点数
    """
    n_x = X.shape[0]  # 输入层大小（节点数）
    n_h = 4
    n_y = Y.shape[0]  # 输出层大小（节点数）
    return (n_x, n_h, n_y)


# 定义函数：初始化参数
def initialize_parameters(n_x, n_h, n_y):
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
    """
    参数:
    layer -- 各层所包含的节点数

    返回值:
    parameters -- 存储权值和偏移量
    """

<<<<<<< HEAD
    np.random.seed(2) # 设置随机种子
    parameters = {}
    #随机初始化参数，偏移量初始化为0
    parameters['w0'] = np.random.randn(layer[1], layer[0]) * 0.01
    parameters['b0'] = np.random.randn(layer[1], 1) * 0
    parameters['w1'] = np.random.randn(layer[2], layer[1]) * 0.01
    parameters['b1'] = np.random.randn(layer[2], 1) * 0

    return parameters

def forward_propagate(X, parameters):
=======
    np.random.seed(2)  # 设置随机种子

    # 随机初始化参数
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# 定义函数：前向传播
def forward_propagation(X, parameters):
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
    """
    参数:
    X -- 输入值
    parameters -- 包含权值和偏移量
    返回值:
    A -- 包含输入和各层输出值
    Z -- 包含隐藏层和输出层的中间值
    """

    A = []
    Z = []
    #将输入值存进A中
    A.append(X)

    #计算隐藏层
    z1 = np.dot(parameters['w0'], A[0]) + parameters['b0']
    Z.append(z1)
    a1 = np.tanh(z1)
    A.append(a1)

<<<<<<< HEAD
    #计算输出层
    z2 = np.dot(parameters['w1'], A[1]) + parameters['b1']
    Z.append(z2)
    a2 = 1/(1+np.exp(-z2))
    A.append(a2)
=======
    assert (A2.shape == (1, X.shape[1]))
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

    return A, Z

<<<<<<< HEAD
def calculate_cost(A, Y):
=======
    return A2, cache


# 定义函数：成本函数
def compute_cost(A2, Y, parameters):
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
    """
   根据第三章给出的公式计算成本

    参数:
    A -- 存储输入值和各层输出值
    Y -- 真实值

    返回值:
    cost -- 成本函数
    """

<<<<<<< HEAD
    m = Y.shape[1] #样本个数
    Y_out = A[len(A)-1] #取模型输出值

    #计算成本
    cost =  -1. / m * np.sum(np.multiply(np.log(Y_out), Y) + np.multiply(np.log(1 - Y_out), 1 - Y))
    cost = np.squeeze(cost)     # 确保维度的正确性
    return cost


def update_parameters(p, dp, learning_rate):
    return p - learning_rate * dp


def backward_propagate(A, Z, parameters, Y,learning_rate):
=======
    m = Y.shape[1]  # 样本个数

    # 计算成本
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -1. / m * np.sum(logprobs)

    cost = np.squeeze(cost)  # 确保维度的正确性
    assert (isinstance(cost, float))

    return cost


# 定义函数：后向传播
def backward_propagation(parameters, cache, X, Y):
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
    """
    参数:
    A -- 存储输入值和各层输出值
    Z -- 存储各层中间值
    parameters -- 包含权值和偏移量
    Y -- 真实值
    learning_rate -- 学习率
    返回值:
    parameters -- 更新后的参数
    """

<<<<<<< HEAD
    m = A[0].shape[1]
=======
    # 首先从"parameters"获取W1,W2
    W1 = parameters["W1"]
    W2 = parameters["W2"]
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

    #后向传播
    #计算dz2
    dz2 = A[2] - Y

<<<<<<< HEAD
    #计算dw2
    dw2 = 1. / m * np.dot(dz2, A[1].T)
=======
    # 后向传播: 计算dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5


    #计算db2
    db2 = 1. / m * np.sum(dz2, axis = 1, keepdims = True)

<<<<<<< HEAD
    #计算dz1
    dz1 = np.dot(parameters['w1'].T, dz2) * (1 - np.power(A[1], 2))
=======

# 定义函数：参数更新
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    使用梯度更新参数
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

    #计算dw1
    dw1 = 1. / m * np.dot(dz1, A[0].T)

<<<<<<< HEAD
    #计算db1
    db1 = 1. / m * np.sum(dz1, axis = 1, keepdims = True)
=======
    返回值:
    parameters -- 包含更新后参数的python
    """
    # 从"parameters"中读取全部参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 从"grads"中读取全部梯度
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # 更新参数
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

    parameters['w1'] = update_parameters(parameters['w1'], dw2, learning_rate)
    parameters['w0'] = update_parameters(parameters['w0'], dw1, learning_rate)
    parameters['b1'] = update_parameters(parameters['b1'], db2, learning_rate)
    parameters['b0'] = update_parameters(parameters['b0'], db1, learning_rate)
    return parameters

<<<<<<< HEAD
#定义函数：神经网络模型
def neural_network(X, Y, layer, times, learning_rate = 1.2):
=======

# 定义函数：神经网络模型
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
    """
    参数:
    X -- 输入值
    Y -- 真实值
    layer -- 各层大小
    learning_rate -- 学习率
    times -- 训练次数

    返回值:
    parameters -- 模型训练所得参数，用于预测
    """
    #初始化参数
    parameters = initialize_parameters(layer)

    for i in range(0, times):
        A = []
        Z = []
        #前向传播
        A, Z = forward_propagate(X, parameters)

<<<<<<< HEAD
        #成本计算
        Cost = calculate_cost(A, Y)

        #后向传播(含参数更新)
        parameters = backward_propagate(A, Z, parameters, Y, learning_rate)

        #每1000次训练打印一次成本函数值
        if (i % 1000 == 0):
            print ("Cost after iteration %i: %f" %(i, Cost))
    return parameters

#准确率计算
def calc_accuracy(predictions, Y):
    Y = np.squeeze(Y)
    right = 0
    for i in range(len(predictions)):
        if predictions[i] == Y[i]:
            right += 1
    accuracy = (right / float(len(predictions))) * 100
    return accuracy

# 使用模型进行预测
def predict_result(parameters, X, Y):
    """
       用学习到的逻辑回归模型来预测点的类型

    Args:
        parameters -- 包含权值和偏移量
        X -- 数据，形状为(px_num * px_num * 3, number of examples)
        Y -- 真实值

    Returns:
        accuracy -- 准确率
    """
    data_dim = X.shape[0]
    # m为数据个数
    m = X.shape[1]

    A = []
    A.append(X)
    Z = []
    predictions = []
=======
    # 根据n_x, n_h, n_y初始化参数，并取出W1,b1,W2,b2
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        # 前向传播， 输入: "X, parameters". 输出: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # 成本计算. 输入: "A2, Y, parameters". 输出: "cost".
        cost = compute_cost(A2, Y, parameters)

        # 后向传播， 输入: "parameters, cache, X, Y". 输出: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # 参数更新. 输入: "parameters, grads". 输出: "parameters".
        parameters = update_parameters(parameters, grads)

        # 每1000次训练打印一次成本函数值
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

    # 预测结果,即前向传播过程
    A, Z = forward_propagate(X,parameters)

<<<<<<< HEAD
    #取输出值Y_out，即A的最后一组数
    Y_out = A[len(A)-1]
=======

# 定义函数：预测
def predict(parameters, X):
    """
    使用训练所得参数，对每个训练样本进行预测
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

    # 将连续值Y_out转化为二分类结果0或1
    for i in range(m):
        if Y_out[0, i] >= 0.5:
            predictions.append(1)
        elif Y_out[0, i] < 0.5:
            predictions.append(0)
    return calc_accuracy(predictions, Y)

#载入数据
train_x, train_y, test_x, test_y = planar_utils.load_planar_dataset()

<<<<<<< HEAD
layer=[2,4,1]
=======
    # 使用训练所得参数进行前向传播计算，并将模型输出值转化为预测值（大于0.5视作1，即True）
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

parameters = neural_network(train_x, train_y, layer, 10000)

<<<<<<< HEAD
print('train:',predict_result(parameters, train_x, train_y))
print('test:',predict_result(parameters, test_x, test_y))
=======

# 定义函数：main函数
def main():
    """
        参数:
        返回值:
    """
    # 加载数据
    train_x, train_y, test_x, test_y = planar_utils.load_planar_dataset()
    # 训练模型
    parameters = nn_model(train_x, train_y, n_h=4, num_iterations=10000, print_cost=True)
    # 预测训练集
    predictions = predict(parameters, train_x)
    # 输出准确率
    print('Train Accuracy: %d' % float((np.dot(train_y, predictions.T) +
                                        np.dot(1 - train_y, 1 - predictions.T)) /
                                       float(train_y.size) * 100) + '%')
    # 预测测试集
    predictions = predict(parameters, test_x)
    print('Test Accuracy: %d' % float((np.dot(test_y, predictions.T) +
                                       np.dot(1 - test_y, 1 - predictions.T)) /
                                      float(test_y.size) * 100) + '%')


if __name__ == '__main__':
    main()
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
