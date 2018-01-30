#!/usr/bin/env python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: weixing(wx_crome@163.com)
Date:    2017/01/30 17:23:06

使用python及numpy库来实现逻辑回归识别猫案例，关键步骤如下：
1.载入数据和预处理：load_data()
2.初始化模型参数（Parameters）
3.循环：
    a)	计算成本（Cost）
    b)	计算梯度（Gradient）
    c)	更新参数（Gradient Descent）
4.计算准确度
5.展示学习曲线plot_costs()
6.利用模型进行预测
"""


import matplotlib.pyplot as plt
import numpy as np

import utils


def load_data():
    """
        载入数据，数据项包括：
            X_train：原始训练数据集
            Y_train：原始训练数据标签
            X_test：原始测试数据集
            Y_test：原始测试数据标签
            classes(cat/non-cat)：分类list

        Args:
        Return:
    """
    X_train, Y_train, X_test, Y_test, classes = utils.load_dataset()

    train_num = X_train.shape[0]
    test_num = X_test.shape[0]
    px_num = X_train.shape[1]

    data_dim = px_num * px_num * 3
    X_train = X_train.reshape(train_num, data_dim).T
    X_test = X_test.reshape(test_num, data_dim).T

    X_train = X_train / 255.
    X_test = X_test / 255.

    data = [X_train, Y_train, X_test, Y_test, classes, px_num]

    return data


def sigmoid(x):
    """
        利用sigmoid计算z的激活值
    """
    return 1 / (1 + np.exp(-x))


def initialize(data_dim):
    """

        初始化w为形状(data_dim, 1)的向量并初始化b为0

        Args:
            data_dim -- w向量的纬度

        Returns:
            w -- (dim, 1)维向量
            b -- 标量，代表偏置bias
    """
    w = np.zeros((data_dim, 1), dtype=np.float)
    b = 0.1

    return w, b


def propagate(X, Y, w, b):
    """
        计算成本cost和梯度grads

        Args:
            w -- 权重， (num_px * num_px * 3, 1)维的numpy数组
            b -- 偏置bias，标量
            X -- 数据，形状为(num_px * num_px * 3, number of examples)
            Y -- 数据的真实标签(包含值 0 if non-cat, 1 if cat) ，形状为 (1, number of examples)

        Return:
            cost -- 逻辑回归的损失函数
            dw -- cost对参数w的梯度，形状与参数w一致
            db -- cost对参数b的梯度，形状与参数b一致
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum(-(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    cost = np.squeeze(cost)

    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost


def update(X, Y, w, b, lr):
    """
        一次梯度下降更新参数
    """
    grads, cost = propagate(X, Y, w, b)

    w = w - lr * grads['dw']
    b = b - lr * grads['db']

    return w, b, cost


def optimize(X, Y, w, b, iteration_nums, lr):
    """
        使用梯度下降算法优化参数w和b

        Args:
            X -- 数据，形状为(num_px * num_px * 3, number of examples)
            Y -- 数据的真实标签(包含值 0 if non-cat, 1 if cat) ，形状为 (1, number of examples)
            w -- 权重， (num_px * num_px * 3, 1)维的numpy数组
            b -- 偏置bias，标量
            iteratino_nums -- 优化的迭代次数
            lr -- 梯度下降的学习率，可控制收敛速度和效果

        Returns:
            params -- 包含参数w和b的python字典
            costs -- 保存了优化过程cost的list，可以用于输出cost变化曲线
    """
    costs = []
    for i in range(iteration_nums):
        w, b, cost = update(X, Y, w, b, lr)

        if i % 100 == 0:
            costs.append(cost)
            print("Iteration %d, cost %f" % (i, cost))

    params = {
        "w": w,
        "b": b
    }

    return params, costs


def predict(X, w, b):
    """
        用学习到的逻辑回归模型来预测图片是否为猫（1 cat or 0 non-cat）

        Args:
            X -- 数据，形状为(num_px * num_px * 3, number of examples)
            w -- 权重， (num_px * num_px * 3, 1)维的numpy数组
            b -- 偏置bias，标量

        Returns:
            predictions -- 包含了对X数据集的所有预测结果，是一个numpy数组或向量

    """
    data_dim = X.shape[0]

    m = X.shape[1]

    predictions = []

    w = w.reshape(data_dim, 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        if A[0, i] > 0.5:
            predictions.append(1)
        elif A[0, i] < 0.5:
            predictions.append(0)

    return predictions


def calc_accuracy(predictions, Y):
    """
        计算准确度
    """
    Y = np.squeeze(Y)
    right = 0
    for i in range(len(predictions)):
        if predictions[i] == Y[i]:
            right += 1
    accuracy = (right / float(len(predictions))) * 100
    return accuracy


def plot_costs(costs, lr):
    """
        利用costs展示模型的学习曲线
    """
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("learning rate =" + str(lr))
    # plt.show()
    plt.savefig('costs.png')


def main():
    """
        训练过程
    """
    X_train, Y_train, X_test, Y_test, classes, px_num = load_data()

    iteration_nums = 2000

    lr = 0.005

    data_dim = X_train.shape[0]

    w, b = initialize(data_dim)

    params, costs = optimize(X_train, Y_train, w, b, iteration_nums, lr)

    predictions_train = predict(X_train, params['w'], params['b'])
    predictions_test = predict(X_test, params['w'], params['b'])

    print("Accuracy on train set: {} %".format(calc_accuracy(predictions_train, Y_train)))
    print("Accuracy on test set: {} %".format(calc_accuracy(predictions_test, Y_test)))

    plot_costs(costs, lr)

    index = 12
    cat_img = X_test[:, index].reshape((px_num, px_num, 3))
    plt.imshow(cat_img)
    plt.axis('off')
    plt.show()
    print ("The label of this picture is " + str(Y_test[0, index]) + ", which means it's a cat picture. "
                                                                     "and you predict that it's a " + classes[
               int(predictions_test[index])].decode("utf-8") + " picture. Congrats!")


if __name__ == "__main__":
    main()
