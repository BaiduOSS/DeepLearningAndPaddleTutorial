#!/usr/bin/env python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: Jiahui Liu(2505774110@qq.com)
Date:    2017/11/17 17:27:06

使用PaddlePaddle框架来实现浅层神经网络识别花型图案，关键步骤如下（除第三步外与逻辑回归代码一致,另外本实验无测试集）：
1.载入数据和预处理：load_data()
2.初始化
3.配置网络结构（增加一层隐藏层并设置7个节点）
4.定义成本函数cost
5.定义优化器optimizer
6.定义reader()用于读取训练数据
7.预测并测试准确率train_accuracy
"""


import sys
import numpy as np
import paddle.v2 as paddle
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


# 载入数据(cat/non-cat)
def load_data():
    """
    载入数据，数据项包括：
        train_set_x：原始训练数据集
        train_set_y：原始训练数据标签

    Args:
    Return:
    """
    global TRAINING_SET, DATADIM

    train_set_x, train_set_y= load_planar_dataset()

    # 定义纬度
    DATADIM = 2

    TRAINING_SET = np.hstack((train_set_x.T, train_set_y.T))



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


def main():
    """
    定义神经网络结构，训练、预测、检验准确率并打印学习曲线
    Args:
    Return:
    """
    global DATADIM
    # 初始化，设置是否使用gpu，trainer数量
    paddle.init(use_gpu=False, trainer_count=1)

    # 载入数据
    load_data()

    # 输入层，paddle.layer.data表示数据层,name=’image’：名称为image,
    # type=paddle.data_type.dense_vector(DATADIM)：数据类型为DATADIM维稠密向量
    image = paddle.layer.data(
        name='image', type=paddle.data_type.dense_vector(DATADIM))
    h1 = paddle.layer.fc(
        input=image, size=4, act=paddle.activation.Tanh())
    # 输出层，paddle.layer.fc表示全连接层，input=image: 该层输入数据为image
    # size=1：神经元个数，act=paddle.activation.Sigmoid()：激活函数为Sigmoid()
    y_predict = paddle.layer.fc(
        input=h1, size=1, act=paddle.activation.Sigmoid())

    # 数据层，paddle.layer.data表示数据层，name=’label’：名称为label
    # type=paddle.data_type.dense_vector(1)：数据类型为1维稠密向量
    y_label = paddle.layer.data(
        name='label', type=paddle.data_type.dense_vector(1))

    # 定义成本函数为交叉熵损失函数multi_binary_label_cross_entropy_cost
    cost = paddle.layer.multi_binary_label_cross_entropy_cost(input=y_predict, label=y_label)

    # 利用cost创建parameters
    parameters = paddle.parameters.create(cost)

    # 创建optimizer，并初始化momentum和learning_rate
    optimizer = paddle.optimizer.Momentum(momentum=0, learning_rate=0.0075)

    # 数据层和数组索引映射，用于trainer训练时喂数据
    feeding = {
        'image': 0,
        'label': 1}

    # 记录成本cost
    costs = []

    # 事件处理
    def event_handler(event):
        """
        事件处理器，可以根据训练过程的信息作相应操作

        Args:
            event -- 事件对象，包含event.pass_id, event.batch_id, event.cost等信息
        Return:
        """
        if isinstance(event, paddle.event.EndIteration):
            if event.pass_id % 1000 == 0:
                print("Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id, event.cost))
                costs.append(event.cost)
                # with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                #     parameters.to_tar(f)

    # 构造trainer,配置三个参数cost、parameters、update_equation，它们分别表示成本函数、参数和更新公式。
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    """
    模型训练
    paddle.reader.shuffle(train(), buf_size=5000)：表示trainer从train()这个reader中读取了buf_size=5000
    大小的数据并打乱顺序
    paddle.batch(reader(), batch_size=256)：表示从打乱的数据中再取出batch_size=256大小的数据进行一次迭代训练
    feeding：用到了之前定义的feeding索引，将数据层image和label输入trainer
    event_handler：事件管理机制，可以自定义event_handler，根据事件信息作相应的操作
    num_passes：定义训练的迭代次数
    """
    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(train(), buf_size=5000),
            batch_size=256),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=10000)

    # 获取训练数据，用来验证模型准确度
    train_data = get_train_data()


    # 根据train_data预测结果，output_layer表示输出层，parameters表示模型参数，input表示输入的测试数据
    probs_train = paddle.infer(
        output_layer=y_predict, parameters=parameters, input=train_data['image']
    )
    # 计算train_accuracy
    print("train_accuracy: {} %".format(train_accuracy(probs_train, train_data)))



if __name__ == '__main__':
    main()
