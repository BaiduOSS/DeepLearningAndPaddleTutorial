<<<<<<< HEAD
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
使用paddle框架实现深层神经网络识别猫的问题，关键步骤如下：
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
8.预测infer()并输出准确率train_accuracy和test_accuracy
9.展示学习曲线plot_costs()
"""

import matplotlib
import numpy as np
import paddle.v2 as paddle

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dnn_utils

TRAINING_SET = None
TEST_SET = None
DATADIM = None


# 载入数据(cat/non-cat)
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

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = dnn_utils.load_dataset()
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
    global TRAINING_SET

    return read_data(TRAINING_SET)


# 获取测试数据集
def test():
    """
    定义一个reader来获取测试数据集及其标签
    Args:
    Return:
        read_data -- 用于获取测试数据集及其标签的reader
    """
    global TEST_SET

    return read_data(TEST_SET)


# 配置网络结构和设置参数
def netconfig():
    """
    配置网络结构和设置参数
    Args:
    Return:
        image -- 输入层，DATADIM维稠密向量
        y_predict -- 输出层，Sigmoid作为激活函数
        y_label -- 标签数据，1维稠密向量
        cost -- 损失函数
        parameters -- 模型参数
        optimizer -- 优化器
        feeding -- 数据映射，python字典
    """
    # 输入层，paddle.layer.data表示数据层,name=’image’：名称为image,
    # type=paddle.data_type.dense_vector(DATADIM)：数据类型为DATADIM维稠密向量
    image = paddle.layer.data(
        name='image', type=paddle.data_type.dense_vector(DATADIM))

    # 隐藏层1，paddle.layer.fc表示全连接层，input=image: 该层输入数据为image
    # size=20：神经元个数，act=paddle.activation.Relu()：激活函数为Relu()
    h1 = paddle.layer.fc(
        input=image, size=20, act=paddle.activation.Relu())

    # 隐藏层2，paddle.layer.fc表示全连接层，input=h1: 该层输入数据为h1
    # size=7：神经元个数，act=paddle.activation.Relu()：激活函数为Relu()
    h2 = paddle.layer.fc(
        input=h1, size=7, act=paddle.activation.Relu())

    # 隐藏层3，paddle.layer.fc表示全连接层，input=h2: 该层输入数据为h2
    # size=5：神经元个数，act=paddle.activation.Relu()：激活函数为Relu()
    h3 = paddle.layer.fc(
        input=h2, size=5, act=paddle.activation.Relu())

    # 输出层，paddle.layer.fc表示全连接层，input=h3: 该层输入数据为h3
    # size=1：神经元个数，act=paddle.activation.Sigmoid()：激活函数为Sigmoid()
    y_predict = paddle.layer.fc(
        input=h3, size=1, act=paddle.activation.Sigmoid())

    # 标签数据，paddle.layer.data表示数据层，name=’label’：名称为label
    # type=paddle.data_type.dense_vector(1)：数据类型为1维稠密向量
    y_label = paddle.layer.data(
        name='label', type=paddle.data_type.dense_vector(1))

    # 定义成本函数为交叉熵损失函数multi_binary_label_cross_entropy_cost
    cost = paddle.layer.multi_binary_label_cross_entropy_cost(input=y_predict, label=y_label)

    # 利用cost创建parameters
    parameters = paddle.parameters.create(cost)

    # 创建optimizer，并初始化momentum和learning_rate
    optimizer = paddle.optimizer.Momentum(momentum=0, learning_rate=0.000075)

    # 数据层和数组索引映射，用于trainer训练时喂数据
    feeding = {
        'image': 0,
        'label': 1}

    data = [image, y_predict, y_label, cost, parameters, optimizer, feeding]

    return data


# 获取data
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


# 计算准确度，服务于train_accuracy()和test_accuracy()
def calc_accuracy(probs, data):
    """
    根据数据集来计算准确度accuracy
    Args:
        probs -- 数据集的预测结果，调用paddle.infer()来获取
        data -- 数据集
    Return:
        calc_accuracy -- 训练准确度
    """
    right = 0
    total = len(data['label'])
    for i in range(len(probs)):
        if float(probs[i][0]) > 0.5 and data['label'][i] == 1:
            right += 1
        elif float(probs[i][0]) < 0.5 and data['label'][i] == 0:
            right += 1
    accuracy = (float(right) / float(total)) * 100
    return accuracy


# 预测
def infer(y_predict, parameters):
    """
    预测并输出模型准确率
    Args:
        y_predict -- 输出层，DATADIM维稠密向量
        parameters -- 训练完成的模型参数
    Return:
    """
    # 获取测试数据和训练数据，用来验证模型准确度
    train_data = get_data(train())
    test_data = get_data(test())

    # 根据train_data和test_data预测结果，output_layer表示输出层，parameters表示模型参数，input表示输入的测试数据
    probs_train = paddle.infer(
        output_layer=y_predict, parameters=parameters, input=train_data['image']
    )
    probs_test = paddle.infer(
        output_layer=y_predict, parameters=parameters, input=test_data['image']
    )

    # 计算train_accuracy和test_accuracy
    print("train_accuracy: {} %".format(calc_accuracy(probs_train, train_data)))
    print("test_accuracy: {} %".format(calc_accuracy(probs_test, test_data)))


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
    plt.title("Learning rate = 0.000075")
    plt.show()
    plt.savefig('costs.png')


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

    # 配置网络结构和设置参数
    image, y_predict, y_label, cost, parameters, optimizer, feeding = netconfig()

    # 记录成本cost
    costs = []

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
                print("Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id, event.cost))
                costs.append(event.cost)
                with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                    parameters.to_tar(f)

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
        num_passes=3000)

    # 预测
    infer(y_predict, parameters)

    # 展示学习曲线
    plot_costs(costs)


if __name__ == '__main__':
    main()
=======
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

使用paddle框架实现深层神经网络识别猫的问题，关键步骤如下：
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
8.预测infer()并输出准确率train_accuracy和test_accuracy
9.展示学习曲线plot_costs()
"""

import matplotlib
import numpy as np
import paddle.v2 as paddle

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dnn_utils

TRAINING_SET = None
TEST_SET = None
DATADIM = None


# 载入数据(cat/non-cat)
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

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = dnn_utils.load_dataset()
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
    global TRAINING_SET

    return read_data(TRAINING_SET)


# 获取测试数据集
def test():
    """
    定义一个reader来获取测试数据集及其标签

    Args:
    Return:
        read_data -- 用于获取测试数据集及其标签的reader
    """
    global TEST_SET

    return read_data(TEST_SET)


# 配置网络结构和设置参数
def netconfig():
    """
    配置网络结构和设置参数
    Args:
    Return:
        image -- 输入层，DATADIM维稠密向量
        y_predict -- 输出层，Sigmoid作为激活函数
        y_label -- 标签数据，1维稠密向量
        cost -- 损失函数
        parameters -- 模型参数
        optimizer -- 优化器
        feeding -- 数据映射，python字典
    """
    # 输入层，paddle.layer.data表示数据层,name=’image’：名称为image,
    # type=paddle.data_type.dense_vector(DATADIM)：数据类型为DATADIM维稠密向量
    image = paddle.layer.data(
        name='image', type=paddle.data_type.dense_vector(DATADIM))

    # 隐藏层1，paddle.layer.fc表示全连接层，input=image: 该层输入数据为image
    # size=20：神经元个数，act=paddle.activation.Relu()：激活函数为Relu()
    h1 = paddle.layer.fc(
        input=image, size=20, act=paddle.activation.Relu())

    # 隐藏层2，paddle.layer.fc表示全连接层，input=h1: 该层输入数据为h1
    # size=7：神经元个数，act=paddle.activation.Relu()：激活函数为Relu()
    h2 = paddle.layer.fc(
        input=h1, size=7, act=paddle.activation.Relu())

    # 隐藏层3，paddle.layer.fc表示全连接层，input=h2: 该层输入数据为h2
    # size=5：神经元个数，act=paddle.activation.Relu()：激活函数为Relu()
    h3 = paddle.layer.fc(
        input=h2, size=5, act=paddle.activation.Relu())

    # 输出层，paddle.layer.fc表示全连接层，input=h3: 该层输入数据为h3
    # size=1：神经元个数，act=paddle.activation.Sigmoid()：激活函数为Sigmoid()
    y_predict = paddle.layer.fc(
        input=h3, size=1, act=paddle.activation.Sigmoid())

    # 标签数据，paddle.layer.data表示数据层，name=’label’：名称为label
    # type=paddle.data_type.dense_vector(1)：数据类型为1维稠密向量
    y_label = paddle.layer.data(
        name='label', type=paddle.data_type.dense_vector(1))

    # 定义成本函数为交叉熵损失函数multi_binary_label_cross_entropy_cost
    cost = paddle.layer.multi_binary_label_cross_entropy_cost(input=y_predict, label=y_label)

    # 利用cost创建parameters
    parameters = paddle.parameters.create(cost)

    # 创建optimizer，并初始化momentum和learning_rate
    optimizer = paddle.optimizer.Momentum(momentum=0, learning_rate=0.000075)

    # 数据层和数组索引映射，用于trainer训练时喂数据
    feeding = {
        'image': 0,
        'label': 1}

    data = [image, y_predict, y_label, cost, parameters, optimizer, feeding]

    return data


# 获取data
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


# 计算准确度，服务于train_accuracy()和test_accuracy()
def calc_accuracy(probs, data):
    """
    根据数据集来计算准确度accuracy

    Args:
        probs -- 数据集的预测结果，调用paddle.infer()来获取
        data -- 数据集

    Return:
        calc_accuracy -- 训练准确度
    """
    right = 0
    total = len(data['label'])
    for i in range(len(probs)):
        if float(probs[i][0]) > 0.5 and data['label'][i] == 1:
            right += 1
        elif float(probs[i][0]) < 0.5 and data['label'][i] == 0:
            right += 1
    accuracy = (float(right) / float(total)) * 100
    return accuracy


# 预测
def infer(y_predict, parameters):
    """
    预测并输出模型准确率

    Args:
        y_predict -- 输出层，DATADIM维稠密向量
        parameters -- 训练完成的模型参数

    Return:
    """
    # 获取测试数据和训练数据，用来验证模型准确度
    train_data = get_data(train())
    test_data = get_data(test())

    # 根据train_data和test_data预测结果，output_layer表示输出层，parameters表示模型参数，input表示输入的测试数据
    probs_train = paddle.infer(
        output_layer=y_predict, parameters=parameters, input=train_data['image']
    )
    probs_test = paddle.infer(
        output_layer=y_predict, parameters=parameters, input=test_data['image']
    )

    # 计算train_accuracy和test_accuracy
    print("train_accuracy: {} %".format(calc_accuracy(probs_train, train_data)))
    print("test_accuracy: {} %".format(calc_accuracy(probs_test, test_data)))


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
    plt.title("Learning rate = 0.000075")
    plt.show()
    plt.savefig('costs.png')


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

    # 配置网络结构和设置参数
    image, y_predict, y_label, cost, parameters, optimizer, feeding = netconfig()

    # 记录成本cost
    costs = []

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
                print("Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id, event.cost))
                costs.append(event.cost)
                with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                    parameters.to_tar(f)

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
        num_passes=3000)

    # 预测
    infer(y_predict, parameters)

    # 展示学习曲线
    plot_costs(costs)


if __name__ == '__main__':
    main()
>>>>>>> a3348d4a8549ae5508a50dcea4d14fb83ee201a5
