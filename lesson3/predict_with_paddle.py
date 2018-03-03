#!/usr/bin/env python
#  -*- coding:utf-8 -*-

"""
Authors: weixing(wx_crome@163.com)
Date:    2017/11/12 17:23:06

用学习到的模型进行预测
与train-with-paddle.py不同，这里不需要重新训练模型，只需要加载训练生成的parameters.tar
文件来获取模型参数，对这组参数也就是训练完的模型进行检测。
1.载入数据和预处理：load_data()
2.从parameters.tar文件直接获取模型参数
3.初始化
4.配置网络结构
5.获取测试数据
6.根据测试数据获得预测结果
7.将预测结果转化为二分类结果
8.预测图片是否为猫
"""

import numpy as np
import paddle.v2 as paddle

from utils import load_data_sets

TEST_SET = None
PARAMETERS = None
DATA_DIM = None
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
    global TEST_SET, DATA_DIM, CLASSES

    train_x_ori, train_y, test_x_ori, test_y, classes = \
        load_data_sets()
    m_test = test_x_ori.shape[0]
    num_px = train_x_ori.shape[1]

    # 定义纬度
    DATA_DIM = num_px * num_px * 3

    # 展开数据
    test_x_flatten = test_x_ori.reshape(m_test, -1)

    # 归一化数据
    test_x = test_x_flatten / 255.

    TEST_SET = np.hstack((test_x, test_y.T))

    CLASSES = classes


def read_data(data_set):
    """
        读取训练数据或测试数据，服务于train()和test()
        Args:
            data_set: 要获取的数据集
        Return:
            reader: 用于获取训练数据集及其标签的生成器generator
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


def test():
    """
    定义一个reader来获取测试数据集及其标签

    Args:
    Return:
        read_data: 用于获取测试数据集及其标签的reader
    """
    global TEST_SET

    return read_data(TEST_SET)


def get_data(data_creator):
    """
    获取data，服务于get_train_data()和get_test_data()

    Args:
        data_creator: 数据来源,可以是train()或者test()
    Return:
        result: 包含测试数据(image)和标签(label)的python字典
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


def get_binary_result(probs):
    """
    将预测结果转化为二分类结果
    Args:
        probs: 预测结果
    Return:
        binary_result: 二分类结果
    """
    binary_result = []
    for i in range(len(probs)):
        if float(probs[i][0]) > 0.5:
            binary_result.append(1)
        elif float(probs[i][0]) < 0.5:
            binary_result.append(0)
    return binary_result


def network_config():
    """
    配置网络结构和设置参数
    Args:
    Return:
        y_predict: 输出层，Sigmoid作为激活函数
    """
    # 输入层，paddle.layer.data表示数据层
    # name=’image’：名称为image
    # type=paddle.data_type.dense_vector(DATA_DIM)：数据类型为DATA_DIM维稠密向量
    image = paddle.layer.data(
        name='image', type=paddle.data_type.dense_vector(DATA_DIM))

    # 输出层，paddle.layer.fc表示全连接层，input=image: 该层输入数据为image
    # size=1：神经元个数，act=paddle.activation.Sigmoid()：激活函数为Sigmoid()
    y_predict = paddle.layer.fc(
        input=image, size=1, act=paddle.activation.Sigmoid())

    return y_predict


def main():
    """
    main entry 预测结果并检验模型准确率
    Args:
    Return:
    """
    global PARAMETERS

    # 载入数据
    load_data()

    # 载入参数
    with open('params_pass_1900.tar', 'r') as param_f:
        PARAMETERS = paddle.parameters.Parameters.from_tar(param_f)

    # 初始化
    paddle.init(use_gpu=False, trainer_count=1)

    # 配置网络结构
    y_predict = network_config()

    # 获取测试数据
    test_data = get_data(test())

    # 根据test_data预测结果
    probs = paddle.infer(
        output_layer=y_predict, parameters=PARAMETERS, input=test_data['image']
    )

    # 将结果转化为二分类结果
    binary_result = get_binary_result(probs)

    # 预测图片是否为猫
    index = 15
    print ("y = " + str(binary_result[index]) +
           ", you predicted that it is a \"" +
           CLASSES[binary_result[index]].decode("utf-8") + "\" picture.")


if __name__ == "__main__":
    main()
