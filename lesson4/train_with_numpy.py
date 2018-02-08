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
ʹ��python��numpy����ʵ��ǳ��������ʶ������ͼ�����ؼ��������£�
1.�������ݺ�Ԥ����load_planar_dataset
2.��ʼ��ģ�Ͳ�����Parameters��
3.ѭ����
    a)	����ɱ���Cost��
    b)	�����ݶȣ�Gradient��
    c)	���²�����Gradient Descent��
4.�˫�������磬����ģ�ͽ���Ԥ��
5.����Ԥ����
6.����neural_network��������˳����������ϲ�
"""
import numpy as np
import planar_utils

<<<<<<< HEAD
def initialize_parameters(layer):
=======

# ���庯������������ṹ
def layer_sizes(X, Y):
    """
    ��������:
    X -- ���������
    Y -- ���ֵ

    ����ֵ:
    n_x -- �����ڵ���
    n_h -- ���ز�ڵ���
    n_y -- �����ڵ���
    """
    n_x = X.shape[0]  # ������С���ڵ�����
    n_h = 4
    n_y = Y.shape[0]  # ������С���ڵ�����
    return (n_x, n_h, n_y)


# ���庯������ʼ������
def initialize_parameters(n_x, n_h, n_y):
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
    """
    ����:
    layer -- �����������Ľڵ���

    ����ֵ:
    parameters -- �洢Ȩֵ��ƫ����
    """

<<<<<<< HEAD
    np.random.seed(2) # �����������
    parameters = {}
    #�����ʼ��������ƫ������ʼ��Ϊ0
    parameters['w0'] = np.random.randn(layer[1], layer[0]) * 0.01
    parameters['b0'] = np.random.randn(layer[1], 1) * 0
    parameters['w1'] = np.random.randn(layer[2], layer[1]) * 0.01
    parameters['b1'] = np.random.randn(layer[2], 1) * 0

    return parameters

def forward_propagate(X, parameters):
=======
    np.random.seed(2)  # �����������

    # �����ʼ������
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


# ���庯����ǰ�򴫲�
def forward_propagation(X, parameters):
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
    """
    ����:
    X -- ����ֵ
    parameters -- ����Ȩֵ��ƫ����
    ����ֵ:
    A -- ��������͸������ֵ
    Z -- �������ز���������м�ֵ
    """

    A = []
    Z = []
    #������ֵ���A��
    A.append(X)

    #�������ز�
    z1 = np.dot(parameters['w0'], A[0]) + parameters['b0']
    Z.append(z1)
    a1 = np.tanh(z1)
    A.append(a1)

<<<<<<< HEAD
    #���������
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


# ���庯�����ɱ�����
def compute_cost(A2, Y, parameters):
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
    """
   ���ݵ����¸����Ĺ�ʽ����ɱ�

    ����:
    A -- �洢����ֵ�͸������ֵ
    Y -- ��ʵֵ

    ����ֵ:
    cost -- �ɱ�����
    """

<<<<<<< HEAD
    m = Y.shape[1] #��������
    Y_out = A[len(A)-1] #ȡģ�����ֵ

    #����ɱ�
    cost =  -1. / m * np.sum(np.multiply(np.log(Y_out), Y) + np.multiply(np.log(1 - Y_out), 1 - Y))
    cost = np.squeeze(cost)     # ȷ��ά�ȵ���ȷ��
    return cost


def update_parameters(p, dp, learning_rate):
    return p - learning_rate * dp


def backward_propagate(A, Z, parameters, Y,learning_rate):
=======
    m = Y.shape[1]  # ��������

    # ����ɱ�
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -1. / m * np.sum(logprobs)

    cost = np.squeeze(cost)  # ȷ��ά�ȵ���ȷ��
    assert (isinstance(cost, float))

    return cost


# ���庯�������򴫲�
def backward_propagation(parameters, cache, X, Y):
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
    """
    ����:
    A -- �洢����ֵ�͸������ֵ
    Z -- �洢�����м�ֵ
    parameters -- ����Ȩֵ��ƫ����
    Y -- ��ʵֵ
    learning_rate -- ѧϰ��
    ����ֵ:
    parameters -- ���º�Ĳ���
    """

<<<<<<< HEAD
    m = A[0].shape[1]
=======
    # ���ȴ�"parameters"��ȡW1,W2
    W1 = parameters["W1"]
    W2 = parameters["W2"]
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

    #���򴫲�
    #����dz2
    dz2 = A[2] - Y

<<<<<<< HEAD
    #����dw2
    dw2 = 1. / m * np.dot(dz2, A[1].T)
=======
    # ���򴫲�: ����dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5


    #����db2
    db2 = 1. / m * np.sum(dz2, axis = 1, keepdims = True)

<<<<<<< HEAD
    #����dz1
    dz1 = np.dot(parameters['w1'].T, dz2) * (1 - np.power(A[1], 2))
=======

# ���庯������������
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    ʹ���ݶȸ��²���
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

    #����dw1
    dw1 = 1. / m * np.dot(dz1, A[0].T)

<<<<<<< HEAD
    #����db1
    db1 = 1. / m * np.sum(dz1, axis = 1, keepdims = True)
=======
    ����ֵ:
    parameters -- �������º������python
    """
    # ��"parameters"�ж�ȡȫ������
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # ��"grads"�ж�ȡȫ���ݶ�
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # ���²���
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
#���庯����������ģ��
def neural_network(X, Y, layer, times, learning_rate = 1.2):
=======

# ���庯����������ģ��
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
    """
    ����:
    X -- ����ֵ
    Y -- ��ʵֵ
    layer -- �����С
    learning_rate -- ѧϰ��
    times -- ѵ������

    ����ֵ:
    parameters -- ģ��ѵ�����ò���������Ԥ��
    """
    #��ʼ������
    parameters = initialize_parameters(layer)

    for i in range(0, times):
        A = []
        Z = []
        #ǰ�򴫲�
        A, Z = forward_propagate(X, parameters)

<<<<<<< HEAD
        #�ɱ�����
        Cost = calculate_cost(A, Y)

        #���򴫲�(����������)
        parameters = backward_propagate(A, Z, parameters, Y, learning_rate)

        #ÿ1000��ѵ����ӡһ�γɱ�����ֵ
        if (i % 1000 == 0):
            print ("Cost after iteration %i: %f" %(i, Cost))
    return parameters

#׼ȷ�ʼ���
def calc_accuracy(predictions, Y):
    Y = np.squeeze(Y)
    right = 0
    for i in range(len(predictions)):
        if predictions[i] == Y[i]:
            right += 1
    accuracy = (right / float(len(predictions))) * 100
    return accuracy

# ʹ��ģ�ͽ���Ԥ��
def predict_result(parameters, X, Y):
    """
       ��ѧϰ�����߼��ع�ģ����Ԥ��������

    Args:
        parameters -- ����Ȩֵ��ƫ����
        X -- ���ݣ���״Ϊ(px_num * px_num * 3, number of examples)
        Y -- ��ʵֵ

    Returns:
        accuracy -- ׼ȷ��
    """
    data_dim = X.shape[0]
    # mΪ���ݸ���
    m = X.shape[1]

    A = []
    A.append(X)
    Z = []
    predictions = []
=======
    # ����n_x, n_h, n_y��ʼ����������ȡ��W1,b1,W2,b2
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        # ǰ�򴫲��� ����: "X, parameters". ���: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # �ɱ�����. ����: "A2, Y, parameters". ���: "cost".
        cost = compute_cost(A2, Y, parameters)

        # ���򴫲��� ����: "parameters, cache, X, Y". ���: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # ��������. ����: "parameters, grads". ���: "parameters".
        parameters = update_parameters(parameters, grads)

        # ÿ1000��ѵ����ӡһ�γɱ�����ֵ
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

    # Ԥ����,��ǰ�򴫲�����
    A, Z = forward_propagate(X,parameters)

<<<<<<< HEAD
    #ȡ���ֵY_out����A�����һ����
    Y_out = A[len(A)-1]
=======

# ���庯����Ԥ��
def predict(parameters, X):
    """
    ʹ��ѵ�����ò�������ÿ��ѵ����������Ԥ��
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

    # ������ֵY_outת��Ϊ��������0��1
    for i in range(m):
        if Y_out[0, i] >= 0.5:
            predictions.append(1)
        elif Y_out[0, i] < 0.5:
            predictions.append(0)
    return calc_accuracy(predictions, Y)

#��������
train_x, train_y, test_x, test_y = planar_utils.load_planar_dataset()

<<<<<<< HEAD
layer=[2,4,1]
=======
    # ʹ��ѵ�����ò�������ǰ�򴫲����㣬����ģ�����ֵת��ΪԤ��ֵ������0.5����1����True��
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5

parameters = neural_network(train_x, train_y, layer, 10000)

<<<<<<< HEAD
print('train:',predict_result(parameters, train_x, train_y))
print('test:',predict_result(parameters, test_x, test_y))
=======

# ���庯����main����
def main():
    """
        ����:
        ����ֵ:
    """
    # ��������
    train_x, train_y, test_x, test_y = planar_utils.load_planar_dataset()
    # ѵ��ģ��
    parameters = nn_model(train_x, train_y, n_h=4, num_iterations=10000, print_cost=True)
    # Ԥ��ѵ����
    predictions = predict(parameters, train_x)
    # ���׼ȷ��
    print('Train Accuracy: %d' % float((np.dot(train_y, predictions.T) +
                                        np.dot(1 - train_y, 1 - predictions.T)) /
                                       float(train_y.size) * 100) + '%')
    # Ԥ����Լ�
    predictions = predict(parameters, test_x)
    print('Test Accuracy: %d' % float((np.dot(test_y, predictions.T) +
                                       np.dot(1 - test_y, 1 - predictions.T)) /
                                      float(test_y.size) * 100) + '%')


if __name__ == '__main__':
    main()
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
