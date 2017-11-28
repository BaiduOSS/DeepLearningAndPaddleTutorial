#!/usr/bin/env python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: weixing(wx_crome@163.com)
Date:    2017/11/12 17:23:06

使用python及numpy库来实现逻辑回归识别猫案例，关键步骤如下：
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

import lr_utils


# 下载数据集(cat/non-cat)
def load_data():
    """
    载入数据，数据项包括：
        train_set_x_orig：原始训练数据集
        train_set_y：原始训练数据标签
        test_set_x_orig：原始测试数据集
        test_set_y：原始测试数据标签
        classes(cat/non-cat)：分类list

    Args:
    Return:
    """
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    # Reshape训练数据集和测试数据集
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

    # 归一化数据
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    data = [train_set_x, train_set_y, test_set_x, test_set_y]
    return data


# sigmoid激活函数
def sigmoid(z):
    """
    利用sigmoid计算z的激活值

    Args:
        z -- 一个标量或者numpy数组

    Return:
        s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s


# 初始化w和b
def initialize_with_zeros(dim):
    """

    初始化w为形状(dim, 1)的向量并初始化b为0

    Args:
        dim -- w向量的纬度

    Returns:
        w -- (dim, 1)维向量
        b -- 标量，代表偏置bias
    """

    w = np.zeros((dim, 1), dtype=np.float)
    b = 0.1

    assert w.shape == (dim, 1)
    assert isinstance(b, float) or isinstance(b, int)

    return w, b


# 前向和后向传播
def propagate(w, b, X, Y):
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

    # m个特征
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum(-(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum((A - Y)) / m

    assert dw.shape == w.shape
    assert db.dtype == float
    cost = np.squeeze(cost)
    assert cost.shape == ()

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# 梯度下降
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    使用梯度下降算法优化参数w和b

    Args:
        w -- 权重， (num_px * num_px * 3, 1)维的numpy数组
        b -- 偏置bias，标量
        X -- 数据，形状为(num_px * num_px * 3, number of examples)
        Y -- 数据的真实标签(包含值 0 if non-cat, 1 if cat) ，形状为 (1, number of examples)
        num_iterations -- 优化的迭代次数
        learning_rate -- 梯度下降的学习率，可控制收敛速度和效果
        print_cost -- 每一百次迭代输出一次cost

    Returns:
        params -- 包含参数w和b的python字典
        grads -- 包含梯度dw和db的python字典
        costs -- 保存了优化过程cost的list，可以用于输出cost变化曲线
    """
    costs = []
    dw = []
    db = 0
    for i in range(int(num_iterations)):
        # 获取梯度grads和成本cost
        grads, cost = propagate(w, b, X, Y)

        # 取出梯度dw和db
        dw = grads["dw"]
        db = grads["db"]

        # 更新规则
        w -= learning_rate * dw
        b -= learning_rate * db

        # 记录cost
        if i % 100 == 0:
            costs.append(cost)

        # 每一百次迭代输出一次cost
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# 预测
def predict(w, b, X):
    """
    用学习到的逻辑回归模型来预测图片是否为猫（1 cat or 0 non-cat）

    Args:
        w -- 权重， (num_px * num_px * 3, 1)维的numpy数组
        b -- 偏置bias，标量
        X -- 数据，形状为(num_px * num_px * 3, number of examples)

    Returns:
        Y_prediction -- 包含了对X数据集的所有预测结果，是一个numpy数组或向量

    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    assert Y_prediction.shape == (1, m)

    return Y_prediction


# 综合model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000,
          learning_rate=0.05, print_cost=False):
    """
    按顺序调用上述方法，构建整体逻辑回归模型model

    Args:
        X_train -- 训练数据，形状为(num_px * num_px * 3, m_train)
        Y_train -- 训练数据的真实标签(包含值 0 if non-cat, 1 if cat) ，形状为 (1, m_train)
        X_test -- 测试数据，形状为(num_px * num_px * 3, m_test)
        Y_test -- 测试数据的真实标签(包含值 0 if non-cat, 1 if cat) ，形状为 (1, m_test)
        w -- 权重， (num_px * num_px * 3, 1)维的numpy数组
        b -- 偏置bias，标量
        X -- 数据，形状为(num_px * num_px * 3, number of examples)
        Y -- 数据的真实标签(包含值 0 if non-cat, 1 if cat) ，形状为 (1, number of examples)
        num_iterations -- 优化的迭代次数
        learning_rate -- 梯度下降的学习率，可控制收敛速度和效果
        print_cost -- 每一百次迭代输出一次cost

    Returns:
       d -- 包含模型信息的python字典
    """
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations,
                                        learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# 绘制训练曲线
def plot_costs(d):
    """
    利用costs展示模型的训练曲线

    Args:
        costs -- 记录了训练过程的cost变化的list，每一百次迭代记录一次
    Return:
    """
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()
    plt.savefig('costs.png')


def main():
    """
    载入数据、训练并展示学习曲线
    Args:
    Return:
    """
    train_set_x, train_set_y, test_set_x, test_set_y = load_data()

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000,
              learning_rate=0.01, print_cost=True)
    plot_costs(d)


if __name__ == '__main__':
    main()
