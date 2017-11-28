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

使用python及numpy库来实现浅层神经网络识别花型图案，关键步骤如下：
1.载入数据和预处理：load_planar_dataset
2.初始化模型参数（Parameters）
3.循环：
    a)	计算成本（Cost）
    b)	计算梯度（Gradient）
    c)	更新参数（Gradient Descent）
4.搭建双层神经网络，利用模型进行预测
5.分析预测结果
6.定义model函数来按顺序将上述步骤合并
"""
import numpy as np

import planar_utils

#定义函数：设置网络结构
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
    n_x = X.shape[0] #输入层大小（节点数）
    n_h = 4
    n_y = Y.shape[0] #输出层大小（节点数）
    return (n_x, n_h, n_y)

# 定义函数：初始化参数
def initialize_parameters(n_x, n_h, n_y):
    """
    参数:
    n_x -- 输入层大小
    n_h -- 隐藏层大小
    n_y -- 输出层大小

    返回值:
    params -- 一个包含所有参数的python字典:
                    W1 -- （隐藏层）权重，维度是 (n_h, n_x)
                    b1 -- （隐藏层）偏移量，维度是 (n_h, 1)
                    W2 -- （输出层）权重，维度是 (n_y, n_h)
                    b2 -- （输出层）偏移量，维度是 (n_y, 1)
    """

    np.random.seed(2) # 设置随机种子

   #随机初始化参数
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
    """
    参数:
    X -- 输入值
    parameters -- 一个python字典，包含计算所需全部参数（是initialize_parameters函数的输出）
    返回值:
    A2 -- 模型输出值
    cache -- 一个字典，包含 "Z1", "A1", "Z2" and "A2"
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

# 定义函数：成本函数
def compute_cost(A2, Y, parameters):
    """
   根据第三章给出的公式计算成本

    参数:
A2 -- 模型输出值
Y -- 真实值
    parameters -- 一个python字典包含参数 W1, b1, W2和b2

    返回值:
    cost -- 成本函数
    """

    m = Y.shape[1] #样本个数

    #计算成本
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost =  -1. / m * np.sum(logprobs)

    cost = np.squeeze(cost)     # 确保维度的正确性
    assert(isinstance(cost, float))

    return cost

# 定义函数：后向传播
def backward_propagation(parameters, cache, X, Y):
    """
    参数:
    parameters -- 一个python字典，包含所有参数
    cache -- 一个python字典包含"Z1", "A1", "Z2"和"A2".
    X -- 输入值
    Y -- 真实值

    返回值:
    grads -- 一个pyth字典包含所有参数的梯度
    """
    m = X.shape[1]

    #首先从"parameters"获取W1,W2
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # 从"cache"中获取A1,A2
    A1 = cache["A1"]
    A2 = cache["A2"]

    # 后向传播: 计算dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

#定义函数：参数更新
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    使用梯度更新参数

    参数:
    parameters -- 包含所有参数的python字典
    grads -- 包含所有参数梯度的python字典

    返回值:
    parameters -- 包含更新后参数的python
    """
    #从"parameters"中读取全部参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 从"grads"中读取全部梯度
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    #更新参数
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


#定义函数：神经网络模型
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    参数:
    X -- 输入值
    Y -- 真实值
    n_h -- 隐藏层大小/节点数
    num_iterations -- 训练次数
    print_cost -- 设置为True，则每1000次训练打印一次成本函数值

    返回值:
    parameters -- 训练结束，更新后的参数值
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    #根据n_x, n_h, n_y初始化参数，并取出W1,b1,W2,b2
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]


    for i in range(0, num_iterations):

        #前向传播， 输入: "X, parameters". 输出: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        #成本计算. 输入: "A2, Y, parameters". 输出: "cost".
        cost = compute_cost(A2, Y, parameters)

        #后向传播， 输入: "parameters, cache, X, Y". 输出: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        #参数更新. 输入: "parameters, grads". 输出: "parameters".
        parameters = update_parameters(parameters, grads)

        #每1000次训练打印一次成本函数值
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters

#定义函数：预测
def predict(parameters, X):
    """
    使用训练所得参数，对每个训练样本进行预测

    参数:
    parameters -- 保安所有参数的python字典
    X -- 输入值

    返回值：
    predictions -- 模型预测值向量(红色: 0 / 蓝色: 1)
    """

    #使用训练所得参数进行前向传播计算，并将模型输出值转化为预测值（大于0.5视作1，即True）
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5

    return predictions

#定义函数：main函数
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
    print('Accuracy: %d' % float((np.dot(train_y, predictions.T) +
                                  np.dot(1 - train_y, 1 - predictions.T)) / float(train_y.size) * 100) + '%')
    #预测测试集
    predictions = predict(parameters, test_x)
    print('Accuracy: %d' % float((np.dot(test_y, predictions.T) +
                                  np.dot(1 - test_y, 1 - predictions.T)) / float(test_y.size) * 100) + '%')

if __name__=='__main__':
    main()
