#!/usr/bin/env python
#  -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: weixing(wx_crome@163.com)
Date:    2017/11/12 17:23:06

用于单独对学习到的模型进行检测，考察其训练准确率train_accuracy和测试准确率test_accuracy，
与train-with-paddle.py不同，这里不需要重新训练模型，只需要加载训练生成的parameters.tar
文件来获取模型参数，对这组参数也就是训练完的模型进行检测。
1.载入数据和预处理：load_data()
2.初始化
3.配置网络结构
4.获取训练和测试数据
5.从parameters.tar文件直接获取模型参数
6.根据模型参数和训练/测试数据来预测结果
7.计算训练准确率train_accuracy和测试准确率test_accuracy
"""

import os

import numpy as np
import paddle.v2 as paddle

from lr_utils import load_dataset

TRAINING_SET = None
TEST_SET = None
PARAMETERS = None
DATADIM = None


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
    global TRAINING_SET, TEST_SET, DATADIM

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    # 定义纬度
    DATADIM = num_px * num_px * 3
    # 数据展开,注意此处为了方便处理，没有加上.T的转置操作
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1)
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1)

    # 归一化
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    TRAINING_SET = np.hstack((train_set_x, train_set_y.T))
    TEST_SET = np.hstack((test_set_x, test_set_y.T))


# 训练数据集
def train():
    """
    定义一个reader来获取训练数据集及其标签

    Args:
    Return:
        reader -- 用于获取训练数据集及其标签的reader
    """
    global TRAINING_SET

    def reader():
        """
        一个reader
        Args:
        Return:
            data[:-1], data[-1:] -- 使用yield返回生成器(generator)，
                    data[:-1]表示前n-1个元素，也就是训练数据，data[-1:]表示最后一个元素，也就是对应的标签
        """
        for data in TRAINING_SET:
            yield data[:-1], data[-1:]

    return reader


# 测试数据集
def test():
    """
    定义一个reader来获取测试数据集及其标签

    Args:
    Return:
        reader -- 用于获取测试数据集及其标签的reader
    """
    global TEST_SET

    def reader():
        """
            一个reader
            Args:
            Return:
                data[:-1], data[-1:] -- 使用yield返回生成器(generator)，
                        data[:-1]表示前n-1个元素，也就是测试数据，data[-1:]表示最后一个元素，也就是对应的标签
            """
        for data in TEST_SET:
            yield data[:-1], data[-1:]

    return reader


# 获取train_data
def get_train_data():
    """
    使用train()来获取训练数据

    Args:
    Return:
        result -- 包含训练数据(image)和标签(label)的python字典
    """
    train_data_creator = train()
    train_data_image = []
    train_data_label = []

    for item in train_data_creator():
        train_data_image.append((item[0],))
        train_data_label.append(item[1])

    result = {
        "image": train_data_image,
        "label": train_data_label
    }

    return result


# 获取test_data
def get_test_data():
    """
    使用test()来获取测试数据

    Args:
    Return:
        result -- 包含测试数据(image)和标签(label)的python字典
    """
    test_data_creator = test()
    test_data_image = []
    test_data_label = []

    for item in test_data_creator():
        test_data_image.append((item[0],))
        test_data_label.append(item[1])

    result = {
        "image": test_data_image,
        "label": test_data_label
    }

    return result


# 训练集准确度
def train_accuracy(probs_train, train_data):
    """
    根据训练数据集来计算训练准确度train_accuracy

    Args:
        probs_train -- 训练数据集的预测结果，调用paddle.infer()来获取
        train_data -- 训练数据集

    Return:
        train_accuracy -- 训练准确度train_accuracy
    """
    train_right = 0
    train_total = len(train_data['label'])
    for i in range(len(probs_train)):
        if float(probs_train[i][0]) > 0.5 and train_data['label'][i] == 1:
            train_right += 1
        elif float(probs_train[i][0]) < 0.5 and train_data['label'][i] == 0:
            train_right += 1
    train_accuracy = (float(train_right) / float(train_total)) * 100

    return train_accuracy


# 测试集准确度
def test_accuracy(probs_test, test_data):
    """
    根据测试数据集来计算测试准确度test_accuracy

    Args:
        probs_test -- 测试数据集的预测结果，调用paddle.infer()来获取
        test_data -- 测试数据集

    Return:
        test_accuracy -- 测试准确度test_accuracy
    """
    test_right = 0
    test_total = len(test_data['label'])
    for i in range(len(probs_test)):
        if float(probs_test[i][0]) > 0.5 and test_data['label'][i] == 1:
            test_right += 1
        elif float(probs_test[i][0]) < 0.5 and test_data['label'][i] == 0:
            test_right += 1
    test_accuracy = (float(test_right) / float(test_total)) * 100

    return test_accuracy


def main():
    """
    预测结果并检验模型准确率
    Args:
    Return:
    """
    global PARAMETERS
    paddle.init(use_gpu=False, trainer_count=1)
    load_data()
    if not os.path.exists('params_pass_199.tar'):
        print("Params file doesn't exists.")
        return
    with open('params_pass_199.tar', 'r') as f:
        PARAMETERS = paddle.parameters.Parameters.from_tar(f)

    # 输入层，paddle.layer.data表示数据层
    # name=’image’：名称为image
    # type=paddle.data_type.dense_vector(DATADIM)：数据类型为DATADIM维稠密向量
    image = paddle.layer.data(
        name='image', type=paddle.data_type.dense_vector(DATADIM))

    # 输入层，paddle.layer.data表示数据层
    # name=’label’：名称为image
    # type=paddle.data_type.dense_vector(DATADIM)：数据类型为DATADIM维向量

    y_predict = paddle.layer.fc(
        input=image, size=1, act=paddle.activation.Sigmoid())

    # 获取测试数据和训练数据，用来验证模型准确度
    train_data = get_train_data()
    test_data = get_test_data()

    # 根据train_data和test_data预测结果
    probs_train = paddle.infer(
        output_layer=y_predict, parameters=PARAMETERS, input=train_data['image']
    )
    probs_test = paddle.infer(
        output_layer=y_predict, parameters=PARAMETERS, input=test_data['image']
    )

    # 计算train_accuracy和test_accuracy
    print("train_accuracy: {} %".format(train_accuracy(probs_train, train_data)))
    print("test_accuracy: {} %".format(test_accuracy(probs_test, test_data)))


if __name__ == "__main__":
    main()
