#!/usr/bin/env python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: yin xiaoting(y_tink@163.com)
Date:    2017/11/16

使用paddlepaddle来做线性回归，拟合房屋价格与房屋面积的线性关系，具体步骤如下：
1.载入数据和预处理：load_data()
2.定义train()和test()用于读取训练数据和测试数据，分别返回一个reader
3.初始化
4.配置网络结构和设置参数：
    - 定义成本函数cost
    - 创建parameters
    - 定义优化器optimizer
5.定义event_handler
6.定义trainer
7.开始训练
8.打印参数和结果print_parameters()
9.展示学习曲线plot_costs()
"""
import numpy as np
import paddle.v2 as paddle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CODEMASTER_TRAIN_DATA = None
X_RAW = None
CODEMASTER_TEST_DATA = None


# 载入数据
def load_data(filename, feature_num=2, ratio=0.8):
    """
    载入数据并进行数据预处理

    Args:
        filename -- 数据存储文件，从该文件读取数据
        feature_num -- 数据特征数量
        ratio -- 训练集占总数据集比例
    Return:
    """
    # 如果测试数据集和训练数据集都不为空，就不再载入数据load_data
    global CODEMASTER_TRAIN_DATA, CODEMASTER_TEST_DATA, X_RAW
    if CODEMASTER_TRAIN_DATA is not None and CODEMASTER_TEST_DATA is not None:
        return
    # data = np.loadtxt()表示将数据载入后以矩阵或向量的形式存储在data中
    # delimiter=',' 表示以','为分隔符
    data = np.loadtxt(filename, delimiter=',')
    X_RAW = data.T[0].copy()
    # axis=0 表示按列计算
    # data.shape[0]表示data中一共多少列
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
        axis=0) / data.shape[0]

    # 归一化，data[:, i] 表示第i列的元素
    for i in xrange(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # offset用于划分训练数据集和测试数据集，例如0.8表示训练集占80%
    offset = int(data.shape[0] * ratio)
    CODEMASTER_TRAIN_DATA = data[:offset].copy()
    CODEMASTER_TEST_DATA = data[offset:].copy()


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


# 获取训练数据集
def train():
    """
    定义一个reader来获取训练数据集及其标签

    Args:
    Return:
        read_data -- 用于获取训练数据集及其标签的reader
    """
    global CODEMASTER_TRAIN_DATA
    load_data('data.txt')

    return read_data(CODEMASTER_TRAIN_DATA)


# 获取测试数据集
def test():
    """
    定义一个reader来获取测试数据集及其标签

    Args:
    Return:
        read_data -- 用于获取测试数据集及其标签的reader
    """
    global CODEMASTER_TEST_DATA
    load_data('data.txt')

    return read_data(CODEMASTER_TEST_DATA)


#配置网络结构
def netconfig():
    """
    配置网络结构
    Args:
    Return:
        image -- 输入层，DATADIM维稠密向量
        y_predict -- 输出层，Linear作为激活函数
        y_label -- 标签数据，1维稠密向量
        cost -- 损失函数
        parameters -- 模型参数
        optimizer -- 优化器
        feeding -- 数据映射，python字典
    """
    # 输入层，paddle.layer.data表示数据层,name=’x’：名称为x,
    # type=paddle.data_type.dense_vector(1)：数据类型为1维稠密向量
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(1))

    # 输出层，paddle.layer.fc表示全连接层，input=x: 该层输入数据为x
    # size=1：神经元个数，act=paddle.activation.Linear()：激活函数为Linear()
    y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())

    # 标签数据，paddle.layer.data表示数据层，name=’y’：名称为y
    # type=paddle.data_type.dense_vector(1)：数据类型为1维稠密向量
    y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))

    # 定义成本函数为均方差损失函数square_error_cost
    cost = paddle.layer.square_error_cost(input=y_predict, label=y)

    # 利用cost创建parameters
    parameters = paddle.parameters.create(cost)

    # 创建optimizer，并初始化momentum
    optimizer = paddle.optimizer.Momentum(momentum=0)

    # 数据层和数组索引映射，用于trainer训练时喂数据
    feeding = {'x': 0, 'y': 1}

    data = [x, y_predict, y, cost, parameters, optimizer, feeding]

    return data


# 展示模型训练曲线
def plot_costs(costs):
    """
    利用costs展示模型的训练曲线

    Args:
        costs -- 记录了训练过程的cost变化的list，每一百次迭代记录一次
    Return:
    """
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("House Price Distributions of Beijing Beiyuan Area")
    plt.show()
    plt.savefig('costs.png')


# 输出参数结果
def print_parameters(parameters):
    """
        打印训练结果的参数以及测试结果
        Args:
            parameters -- 训练结果的参数
        Return:
        """
    print("Result Parameters as below:")
    a = parameters.get('___fc_layer_0__.w0')[0]
    b = parameters.get('___fc_layer_0__.wbias')[0]
    print(a, b)

    x0 = X_RAW[0]
    y0 = a * CODEMASTER_TRAIN_DATA[0][0] + b

    x1 = X_RAW[1]
    y1 = a * CODEMASTER_TRAIN_DATA[1][0] + b

    a = (y0 - y1) / (x0 - x1)
    b = (y1 - a * x1)

    print 'a = ', a
    print 'b = ', b


def main():
    """
    初始化，定义神经网络结构，训练
    Args:
    Return:
    """
    # 初始化，设置是否使用gpu，trainer数量
    paddle.init(use_gpu=False, trainer_count=1)

    # 配置网络结构和设置参数
    x, y_predict, y, cost, parameters, optimizer, feeding = netconfig()

    # 记录成本cost
    costs = []

    # 构造trainer,配置三个参数cost、parameters、update_equation，它们分别表示成本函数、参数和更新公式。
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # 处理事件
    def event_handler(event):
        """
        事件处理器，可以根据训练过程的信息作相应操作

        Args:
            event -- 事件对象，包含event.pass_id, event.batch_id, event.cost等信息
        Return:
        """
        if isinstance(event, paddle.event.EndIteration):
            if event.pass_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)
                costs.append(event.cost)

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                reader=paddle.batch(test(), batch_size=2),
                feeding=feeding)
            print "Test %d, Cost %f" % (event.pass_id, result.cost)

    """        
    # 模型训练
    paddle.reader.shuffle(train(), buf_size=500)：表示trainer从train()这个reader中读取了buf_size=500
    大小的数据并打乱顺序
    paddle.batch(reader(), batch_size=256)：表示从打乱的数据中再取出batch_size=256大小的数据进行一次迭代训练
    feeding：用到了之前定义的feeding索引，将数据层x和y输入trainer
    event_handler：事件管理机制，可以自定义event_handler，根据事件信息作相应的操作
    num_passes：定义训练的迭代次数
    """
    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(train(), buf_size=500),
            batch_size=256),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=300)

    # 打印参数结果
    print_parameters(parameters)

    # 展示学习曲线
    plot_costs(costs)

if __name__ == '__main__':
    main()

