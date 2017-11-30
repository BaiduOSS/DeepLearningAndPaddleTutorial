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

用学习到的模型进行预测
与train-with-paddle.py不同，这里不需要重新训练模型，只需要加载训练生成的parameters.tar
文件来获取模型参数，对这组参数也就是训练完的模型进行检测。
1.载入数据和预处理：load_data()
2.初始化
3.配置网络结构
4.获取训练和测试数据
5.从parameters.tar文件直接获取模型参数
6.根据模型参数和测试数据来预测结果
"""

import os

import numpy as np
import paddle.v2 as paddle

from lr_utils import load_dataset

TEST_SET = None
PARAMETERS = None
DATADIM = None
CLASSES = None


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
    global TEST_SET, DATADIM, CLASSES

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    # 定义纬度
    DATADIM = num_px * num_px * 3

    # 展开数据
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1)

    # 归一化数据
    test_set_x = test_set_x_flatten / 255.

    TEST_SET = np.hstack((test_set_x, test_set_y.T))

    CLASSES = classes


# 读取训练数据或测试数据，服务于train()和test()
def read_data(data_set):
    """
        一个reader
        Args:
            data_set -- 要获取的数据集
        Return:
            reader -- 用于获取训练数据集及其标签的生成器generator
    """

    def reader():
        """
        一个reader
        Args:
        Return:
            data[:-1], data[-1:] -- 使用yield返回生成器(generator)，
                    data[:-1]表示前n-1个元素，也就是训练数据，data[-1:]表示最后一个元素，也就是对应的标签
        """
        for data in data_set:
            yield data[:-1], data[-1:]

    return reader


# 测试数据集
def test():
    """
    定义一个reader来获取测试数据集及其标签

    Args:
    Return:
        read_data -- 用于获取测试数据集及其标签的reader
    """
    global TEST_SET

    return read_data(TEST_SET)


# 获取data，服务于get_train_data()和get_test_data()
def get_data(data_creator):
    """
    使用参数data_creator来获取测试数据

    Args:
        data_creator -- 数据来源,可以是train()或者test()
    Return:
        result -- 包含测试数据(image)和标签(label)的python字典
    """
    data_creator = data_creator
    data_image = []
    data_label = []

    for item in data_creator():
        data_image.append((item[0],))
        data_label.append(item[1])

    result = {
        "image": data_image,
        "label": data_label
    }

    return result


# 获取test_data
def get_test_data():
    """
    使用test()来获取测试数据

    Args:
    Return:
        get_data(test()) -- 包含测试数据(image)和标签(label)的python字典
    """
    return get_data(test())


# 二分类结果
def get_binary_result(probs):
    """
    将预测结果转化为二分类结果

    Args:
        probs -- 预测结果
    Return:
        binary_result -- 二分类结果
    """
    binary_result = []
    for i in range(len(probs)):
        if float(probs[i][0]) > 0.5:
            binary_result.append(1)
        elif float(probs[i][0]) < 0.5:
            binary_result.append(0)
    return binary_result


def main():
    """
    预测结果并检验模型准确率
    Args:
    Return:
    """
    global PARAMETERS
    paddle.init(use_gpu=False, trainer_count=1)
    load_data()
    if not os.path.exists('params_pass_1900.tar'):
        print("Params file doesn't exists.")
        return
    with open('params_pass_1900.tar', 'r') as f:
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

    # 获取测试数据
    test_data = get_test_data()

    # 根据test_data预测结果
    probs = paddle.infer(
        output_layer=y_predict, parameters=PARAMETERS, input=test_data['image']
    )

    # 将结果转化为二分类结果
    binary_result = get_binary_result(probs)

    index = 12
    print ("y = " + str(binary_result[index]) + ", you predicted that it is a \"" +
           CLASSES[binary_result[index]].decode("utf-8") + "\" picture.")


if __name__ == "__main__":
    main()
