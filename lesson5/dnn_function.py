# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import dnn_utils

def initialize_parameters(layer):
    """
    参数:
    layer -- 各层所包含的节点数

    返回值:
    parameters -- 存储权值和偏移量
    """

    np.random.seed(2) # 设置随机种子
    parameters = {}
    #随机初始化参数，偏移量初始化为0
    for i in range(len(layer)-1):
        parameters['w' + str(i)] = np.random.randn(layer[i+1], layer[i]) / np.sqrt(layer[i])
        parameters['b' + str(i)] = np.random.randn(layer[i+1], 1) * 0

    return parameters

def forward_propagate(X, parameters):
    """
    参数:
    X -- 输入值
    parameters -- 包含权值和偏移量
    返回值:
    A -- 包含输入和各层输出值
    Z -- 包含隐藏层和输出层的中间值
    """

    A = []
    A.append(X)
    Z = []
    l = int(len(parameters) / 2)

    #计算隐藏层
    for i in range(l-1):
        #加权、偏移
        z = np.dot(parameters['w' + str(i)],A[i])+parameters['b' + str(i)]
        Z.append(z)
        #激活
        a = np.maximum(0,z)
        A.append(a)

    #计算输出层
    z = np.dot(parameters['w' + str(l-1)], A[l-1]) + parameters['b' + str(l-1)]
    Z.append(z)
    a = 1. / (1 + np.exp(-z))
    A.append(a)

    return A, Z

def calculate_cost(A, Y):
    """
   根据第三章给出的公式计算成本

    参数:
    A -- 存储输入值和各层输出值
    Y -- 真实值

    返回值:
    cost -- 成本函数
    """

    m = Y.shape[1] #样本个数
    Y_out = A[len(A)-1] #取模型输出值

    #计算成本
    probability = np.multiply(np.log(Y_out), Y) + np.multiply(np.log(1 - Y_out), 1 - Y)
    cost =  -1. / m * np.sum(probability)

    cost = np.squeeze(cost)     # 确保维度的正确性
    return cost


def update_parameters(p, dp, learning_rate):
    return p - learning_rate * dp

def backward_propagate(A, Z, parameters, Y,learning_rate):
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

    m = A[0].shape[1]
    l = int(len(parameters) / 2)

    #反向传播：计算输出层
    da = - (np.divide(Y, A[l]) - np.divide(1 - Y, 1 - A[l]))
    dz = A[l] - Y

    #反向传播：计算隐藏层
    for i in range(1, l):
        da = np.dot(parameters['w' + str(l - i)].T, dz)
        dz = da
        dz[Z[l-i-1]<=0]=0

        #更新参数
        dw = 1. / m * np.dot(dz, A[l-i-1].T)
        db = 1. / m * np.sum(dz, axis=1, keepdims=True)
        parameters['w' + str(l - i - 1)] = update_parameters(parameters['w' + str(l - i - 1)], dw, learning_rate)
        parameters['b' + str(l - i - 1)] = update_parameters(parameters['b' + str(l - i - 1)], db, learning_rate)

    return parameters


def plot_costs(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("learning rate =" + str(learning_rate))
    plt.show()
    plt.savefig('costs.png')

#定义函数：深层神经网络模型(包含前向传播和后向传播)
def deep_neural_network(X, Y, layer, times, learning_rate = 0.0075):
    """
    参数:
    X -- 输入值
    Y -- 真实值
    layer -- 各层大小
    times -- 训练次数
    learning_rate -- 学习率

    返回值:
    parameters -- 模型训练所得参数，用于预测
    """

    np.random.seed(1)
    costs = []

    #参数初始化
    parameters = initialize_parameters(layer)

    #训练
    for i in range(0, times):
        #初始化A并添加输入X
        #正向传播
        A, Z = forward_propagate(X, parameters)

        #计算成本函数
        Cost = calculate_cost(A, Y)

        #反向传播(含更新参数)
        parameters = backward_propagate(A, Z, parameters, Y, learning_rate)

        #每100次训练打印一次成本函数
        if(i % 100 == 0):
            print ("Cost after iteration %i: %f" %(i, Cost))
            costs.append(Cost)

    #plot_costs(costs, learning_rate)

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
       用学习到的逻辑回归模型来预测图片是否为猫（1 cat or 0 non-cat）

    Args:
        parameters -- 包含权值和偏移量
        X -- 数据，形状为(px_num * px_num * 3, number of examples)
        Y -- 真实值

    Returns:
        accuracy -- 准确率
    """

    # m为数据个数
    m = X.shape[1]

    A = []
    A.append(X)
    Z = []
    predictions = []

    # 预测结果,即前向传播过程
    A, Z = forward_propagate(X, parameters)

    #取输出值Y_out，即A的最后一组数
    Y_out = A[len(A)-1]

    # 将连续值Y_out转化为二分类结果0或1
    for i in range(m):
        if Y_out[0, i] >= 0.5:
            predictions.append(1)
        elif Y_out[0, i] < 0.5:
            predictions.append(0)

    return calc_accuracy(predictions, Y)

