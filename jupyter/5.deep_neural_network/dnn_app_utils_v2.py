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

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import h5py


# 定义Sigmoid激活函数
def sigmoid(Z):
    """
    使用Sigmoid函数激活

    Arguments:
    Z -- numpy数组

    Returns:
    A -- sigmoid(z)的计算值, 与Z维度相同
    cache -- 返回Z，用于后向传播
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

# Sigmoid后向传播计算
def sigmoid_backward(dA, cache):
    """
    使用Sigmoid函数激活的反向计算

    Arguments:
    dA -- 激活后的数值的梯度
    cache -- 包含Z值

    Returns:
    dZ -- Z的梯度
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


# 定义Relu激活函数
def relu(Z):
    """
    使用Relu函数激活

    Arguments:
    Z -- 线性计算的输出值

    Returns:
    A -- Relu激活后的值，维度与Z相同
    cache -- 包含A的python字典，用于后向传播计算
    """
    # Relu激活
    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache

#Relu后向传播计算
def relu_backward(dA, cache):
    """
    使用Relu函数激活的后向计算

    Arguments:
    dA -- 激活后的数值的梯度
    cache -- 包含Z值

    Returns:
    dZ -- Z的梯度
    """

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # Z<0，置0即可
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


#加载数据
def load_data():
    """
    加载数据

    Arguments:
    Returns:

    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 训练特征
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 训练标签

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 测试特征
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 测试标签

    classes = np.array(test_dataset["list_classes"][:]) # class列表

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    dataset = [train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes]
    return dataset


#初始化参数
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- 输入层节点数
    n_h -- 隐藏层节点数
    n_y -- 输出层节点数

    Returns:
    parameters -- 一个字典，包含:
                    W1 -- 权重矩阵，维度是 (n_h, n_x)
                    b1 -- 偏移值向量，维度是 (n_h, 1)
                    W2 -- 权重矩阵，维度是 (n_y, n_h)
                    b2 -- 偏移值向量，维度是 (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

#深层网络的参数初始化
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- 一个list，包含神经网络各层节点数

    Returns:
    parameters -- 一个字典，包含所有参数： "W1", "b1", ..., "WL", "bL":
                    Wl -- 权重矩阵，维度是 (layer_dims[l], layer_dims[l-1])
                    bl -- 偏移值向量，维度是 (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # 网络层数

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


#前向传播的线性计算
def linear_forward(A, W, b):
    """
   完成前向传播各层的线性计算

    Arguments:
    A -- 上一层的输出值，作为本层输入值
    W -- 权重矩阵
    b -- 偏移值向量

    Returns:
    Z -- 激活计算的输入
    cache -- 一个字典，存储了"A", "W" and "b" 用作后向计算
    """

    Z = W.dot(A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


#线性计算与激活
def linear_activation_forward(A_prev, W, b, activation):
    """
    完成前向传播中各层的线性计算和激活计算

    Arguments:
    A_prev -- 上一层的输出值，作为本层输入
    W -- 权重矩阵
    b -- 偏移值向量
    activation -- 激活函数选择

    Returns:
    A -- 激活后的输出值
    cache -- 一个字典存储"linear_cache" 和"activation_cache";用作后向传播计算
    """

    if activation == "sigmoid":
        # 输入: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # 输入: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


#L层模型前向传播
def L_model_forward(X, parameters):
    """
    完成L层模型的前向传播的计算

    Arguments:
    X -- 输入值
    parameters -- 参数

    Returns:
    AL -- 模型最终输出
    caches -- 包含linear_relu_forward()和linear_sigmoid_forward()的每个cache
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # 神经网络层数
    # L层网络的前向传播计算
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)],
                                             parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    # 在输出层完成线性计算和激活
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)],
                                          parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches


#计算成本函数
def compute_cost(AL, Y):
    """
   完成成本函数计算

    Arguments:
    AL -- 模型输出值
    Y -- 真实值

    Returns:
    cost -- 成本值
    """

    m = Y.shape[1]

    # 根据AL和Y计算成本值
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost


#线性计算的后向传播
def linear_backward(dZ, cache):
    """
    完成某一层的后向传播的线性计算部分

    Arguments:
    dZ -- 线性计算值的梯度
    cache -- 包含前向传播的(A_prev, W, b)值

    Returns:
    dA_prev --前一层输出值的梯度
    dW -- 本层权重梯度
    db -- 本层偏移值梯度
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


#线性计算和激活的后向传播计算
def linear_activation_backward(dA, cache, activation):
    """
    完成线性计算和激活的后向传播计算

    Arguments:
    dA -- 本层输出值梯度
    cache --包含 (linear_cache, activation_cache)
    activation -- 记录激活函数

    Returns:
    dA_prev -- 上一层输出值的梯度
    dW -- 本层权重梯度
    db -- 本层偏移值梯度
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


#L层神经网络模型后向传播
def L_model_backward(AL, Y, caches):
    """
    完成L层神经网络模型后向传播计算

    Arguments:
    AL -- 模型输出值
    Y -- 真实值
    caches -- 包含Relu和Sigmoid激活函数的linear_activation_forward()中每一个cache

    Returns:
    grads -- 包含所有梯度的字典
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # 初始化后向传播计算
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # L层神经网络梯度. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(L - 1)):
        # 第L层: (RELU -> LINEAR) 梯度
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
            linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


#参数更新
def update_parameters(parameters, grads, learning_rate):
    """
    #根据梯度完成参数更新

    Arguments:
    parameters -- 包含所有参数的字典
    grads -- 包含梯度的字典

    Returns:
    parameters -- 一个字典，包含所有更新后的参数
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural network

    # 使用循环更新每个参数
    for l in range(L):
        parameters["W" + str(l + 1)] = \
            parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = \
            parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


#预测
def predict(X, y, parameters):
    """
    用训练后的L层模型预测结果

    Arguments:
    X -- 输入值
    parameters -- 训练后的参数

    Returns:
    p -- 对输入值X的预测
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1, m))

    # 前向传播计算
    probas, caches = L_model_forward(X, parameters)

    # 将预测值转换为0/1
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # 打印结果
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print(str(float(np.sum(p == y)) / float(m)))

    return p


#打印未识别的图片
def print_mislabeled_images(classes, X, y, p):
    """
    画出未识别的图片
    X -- 数据
    y -- 真实值
    p -- 预测值
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0, index])]
                  .decode("utf-8") + " \n Class: " + classes[y[0, index]].decode("utf-8"))


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# L层神经网络模型
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    完成L层神经网络模型（包含前向以及后向传播）

    Arguments:
    X -- 输入值
    Y -- 真实值
    layers_dims -- 存有各层节点数的list
    learning_rate -- 学习率
    num_iterations -- 训练次数
    print_cost -- 如果为真，每100次打印一次cost值

    Returns:
    parameters -- 训练后的参数
    """

    np.random.seed(1)
    costs = []                         # keep track of cost

    # 初始化参数
    parameters = initialize_parameters_deep(layers_dims)

    # 训练
    for i in range(0, num_iterations):

        # 前向传播计算
        AL, caches = L_model_forward(X, parameters)

        # 计算成本
        cost = compute_cost(AL, Y)

        # 后向传播
        grads = L_model_backward(AL, Y, caches)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 每100次训练打印cost
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # 绘制cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    plt.savefig("costs.png")
    return parameters

