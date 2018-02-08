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

def initialize_parameters(layer):
    """
    ����:
    layer -- �����������Ľڵ���

    ����ֵ:
    parameters -- �洢Ȩֵ��ƫ����
    """

    np.random.seed(2) # �����������
    parameters = {}
    #�����ʼ��������ƫ������ʼ��Ϊ0
    parameters['w0'] = np.random.randn(layer[1], layer[0]) * 0.01
    parameters['b0'] = np.random.randn(layer[1], 1) * 0
    parameters['w1'] = np.random.randn(layer[2], layer[1]) * 0.01
    parameters['b1'] = np.random.randn(layer[2], 1) * 0

    return parameters

def forward_propagate(X, parameters):
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

    #���������
    z2 = np.dot(parameters['w1'], A[1]) + parameters['b1']
    Z.append(z2)
    a2 = 1/(1+np.exp(-z2))
    A.append(a2)

    return A, Z

def calculate_cost(A, Y):
    """
   ���ݵ����¸����Ĺ�ʽ����ɱ�

    ����:
    A -- �洢����ֵ�͸������ֵ
    Y -- ��ʵֵ

    ����ֵ:
    cost -- �ɱ�����
    """

    m = Y.shape[1] #��������
    Y_out = A[len(A)-1] #ȡģ�����ֵ

    #����ɱ�
    cost =  -1. / m * np.sum(np.multiply(np.log(Y_out), Y) + np.multiply(np.log(1 - Y_out), 1 - Y))
    cost = np.squeeze(cost)     # ȷ��ά�ȵ���ȷ��
    return cost


def update_parameters(p, dp, learning_rate):
    return p - learning_rate * dp


def backward_propagate(A, Z, parameters, Y,learning_rate):
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

    m = A[0].shape[1]

    #���򴫲�
    #����dz2
    dz2 = A[2] - Y

    #����dw2
    dw2 = 1. / m * np.dot(dz2, A[1].T)


    #����db2
    db2 = 1. / m * np.sum(dz2, axis = 1, keepdims = True)

    #����dz1
    dz1 = np.dot(parameters['w1'].T, dz2) * (1 - np.power(A[1], 2))

    #����dw1
    dw1 = 1. / m * np.dot(dz1, A[0].T)

    #����db1
    db1 = 1. / m * np.sum(dz1, axis = 1, keepdims = True)

    parameters['w1'] = update_parameters(parameters['w1'], dw2, learning_rate)
    parameters['w0'] = update_parameters(parameters['w0'], dw1, learning_rate)
    parameters['b1'] = update_parameters(parameters['b1'], db2, learning_rate)
    parameters['b0'] = update_parameters(parameters['b0'], db1, learning_rate)
    return parameters

#���庯����������ģ��
def neural_network(X, Y, layer, times, learning_rate = 1.2):
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

    # Ԥ����,��ǰ�򴫲�����
    A, Z = forward_propagate(X,parameters)

    #ȡ���ֵY_out����A�����һ����
    Y_out = A[len(A)-1]

    # ������ֵY_outת��Ϊ��������0��1
    for i in range(m):
        if Y_out[0, i] >= 0.5:
            predictions.append(1)
        elif Y_out[0, i] < 0.5:
            predictions.append(0)
    return calc_accuracy(predictions, Y)

#��������
train_x, train_y, test_x, test_y = planar_utils.load_planar_dataset()

layer=[2,4,1]

parameters = neural_network(train_x, train_y, layer, 10000)

print('train:',predict_result(parameters, train_x, train_y))
print('test:',predict_result(parameters, test_x, test_y))
