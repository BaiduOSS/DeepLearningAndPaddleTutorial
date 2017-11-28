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

ʹ��python��numpy����ʵ��ǳ��������ʶ����ͼ�����ؼ��������£�
1.�������ݺ�Ԥ����load_planar_dataset
2.��ʼ��ģ�Ͳ�����Parameters��
3.ѭ����
    a)	����ɱ���Cost��
    b)	�����ݶȣ�Gradient��
    c)	���²�����Gradient Descent��
4.�˫�������磬����ģ�ͽ���Ԥ��
5.����Ԥ����
6.����model��������˳����������ϲ�
"""
import numpy as np

import planar_utils

#���庯������������ṹ
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
    n_x = X.shape[0] #������С���ڵ�����
    n_h = 4
    n_y = Y.shape[0] #������С���ڵ�����
    return (n_x, n_h, n_y)

# ���庯������ʼ������
def initialize_parameters(n_x, n_h, n_y):
    """
    ����:
    n_x -- ������С
    n_h -- ���ز��С
    n_y -- ������С

    ����ֵ:
    params -- һ���������в�����python�ֵ�:
                    W1 -- �����ز㣩Ȩ�أ�ά���� (n_h, n_x)
                    b1 -- �����ز㣩ƫ������ά���� (n_h, 1)
                    W2 -- ������㣩Ȩ�أ�ά���� (n_y, n_h)
                    b2 -- ������㣩ƫ������ά���� (n_y, 1)
    """

    np.random.seed(2) # �����������

   #�����ʼ������
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
    """
    ����:
    X -- ����ֵ
    parameters -- һ��python�ֵ䣬������������ȫ����������initialize_parameters�����������
    ����ֵ:
    A2 -- ģ�����ֵ
    cache -- һ���ֵ䣬���� "Z1", "A1", "Z2" and "A2"
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

# ���庯�����ɱ�����
def compute_cost(A2, Y, parameters):
    """
   ���ݵ����¸����Ĺ�ʽ����ɱ�

    ����:
A2 -- ģ�����ֵ
Y -- ��ʵֵ
    parameters -- һ��python�ֵ�������� W1, b1, W2��b2

    ����ֵ:
    cost -- �ɱ�����
    """

    m = Y.shape[1] #��������

    #����ɱ�
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost =  -1. / m * np.sum(logprobs)

    cost = np.squeeze(cost)     # ȷ��ά�ȵ���ȷ��
    assert(isinstance(cost, float))

    return cost

# ���庯�������򴫲�
def backward_propagation(parameters, cache, X, Y):
    """
    ����:
    parameters -- һ��python�ֵ䣬�������в���
    cache -- һ��python�ֵ����"Z1", "A1", "Z2"��"A2".
    X -- ����ֵ
    Y -- ��ʵֵ

    ����ֵ:
    grads -- һ��pyth�ֵ�������в������ݶ�
    """
    m = X.shape[1]

    #���ȴ�"parameters"��ȡW1,W2
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # ��"cache"�л�ȡA1,A2
    A1 = cache["A1"]
    A2 = cache["A2"]

    # ���򴫲�: ����dW1, db1, dW2, db2.
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

#���庯������������
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    ʹ���ݶȸ��²���

    ����:
    parameters -- �������в�����python�ֵ�
    grads -- �������в����ݶȵ�python�ֵ�

    ����ֵ:
    parameters -- �������º������python
    """
    #��"parameters"�ж�ȡȫ������
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # ��"grads"�ж�ȡȫ���ݶ�
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    #���²���
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


#���庯����������ģ��
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    ����:
    X -- ����ֵ
    Y -- ��ʵֵ
    n_h -- ���ز��С/�ڵ���
    num_iterations -- ѵ������
    print_cost -- ����ΪTrue����ÿ1000��ѵ����ӡһ�γɱ�����ֵ

    ����ֵ:
    parameters -- ѵ�����������º�Ĳ���ֵ
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    #����n_x, n_h, n_y��ʼ����������ȡ��W1,b1,W2,b2
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]


    for i in range(0, num_iterations):

        #ǰ�򴫲��� ����: "X, parameters". ���: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        #�ɱ�����. ����: "A2, Y, parameters". ���: "cost".
        cost = compute_cost(A2, Y, parameters)

        #���򴫲��� ����: "parameters, cache, X, Y". ���: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        #��������. ����: "parameters, grads". ���: "parameters".
        parameters = update_parameters(parameters, grads)

        #ÿ1000��ѵ����ӡһ�γɱ�����ֵ
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters

#���庯����Ԥ��
def predict(parameters, X):
    """
    ʹ��ѵ�����ò�������ÿ��ѵ����������Ԥ��

    ����:
    parameters -- �������в�����python�ֵ�
    X -- ����ֵ

    ����ֵ��
    predictions -- ģ��Ԥ��ֵ����(��ɫ: 0 / ��ɫ: 1)
    """

    #ʹ��ѵ�����ò�������ǰ�򴫲����㣬����ģ�����ֵת��ΪԤ��ֵ������0.5����1����True��
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5

    return predictions

#���庯����main����
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
    print('Accuracy: %d' % float((np.dot(train_y, predictions.T) +
                                  np.dot(1 - train_y, 1 - predictions.T)) / float(train_y.size) * 100) + '%')
    #Ԥ����Լ�
    predictions = predict(parameters, test_x)
    print('Accuracy: %d' % float((np.dot(test_y, predictions.T) +
                                  np.dot(1 - test_y, 1 - predictions.T)) / float(test_y.size) * 100) + '%')

if __name__=='__main__':
    main()
