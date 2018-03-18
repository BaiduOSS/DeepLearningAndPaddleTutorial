#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Authors: xiake(kedou1993@163.com)
    Date:    2017/11/29

    使用paddle框架实现逻辑数字识别案例，关键步骤如下：
    1.定义分类器网络结构
    2.初始化
    3.配置网络结构
    4.定义成本函数cost
    5.定义优化器optimizer
    6.定义事件处理函数
    7.进行训练
    8.利用训练好的模型进行预测
"""
import os

import matplotlib
import numpy as np
import paddle.v2 as paddle
from paddle.v2.plot import Ploter
from PIL import Image

matplotlib.use('Agg')

WITH_GPU = os.getenv('WITH_GPU', '0') != '0'

STEP = 0

# 绘图相关标注
TRAIN_TITLE_COST = "Train cost"
TEST_TITLE_COST = "Test cost"

TRAIN_TITLE_ERROR = "Train error rate"
TEST_TITLE_ERROR = "Test error rate"


def softmax_regression(img):
    """
    定义softmax分类器：
        只通过一层简单的以softmax为激活函数的全连接层，可以得到分类的结果
    Args:
        img -- 输入的原始图像数据
    Return:
        predict_image -- 分类的结果
    """
    predict = paddle.layer.fc(
        input=img, size=10, act=paddle.activation.Softmax())
    return predict


def multilayer_perceptron(img):
    """
    定义多层感知机分类器：
        含有两个隐藏层（即全连接层）的多层感知器
        其中两个隐藏层的激活函数均采用ReLU，输出层的激活函数用Softmax
    Args:
        img -- 输入的原始图像数据
    Return:
        predict_image -- 分类的结果
    """
    # 第一个全连接层
    hidden1 = paddle.layer.fc(input=img, size=128,
                              act=paddle.activation.Relu())
    # 第二个全连接层
    hidden2 = paddle.layer.fc(
        input=hidden1, size=64, act=paddle.activation.Relu())
    # 第三个全连接层，需要注意输出尺寸为10,，对应0-9这10个数字
    predict = paddle.layer.fc(
        input=hidden2, size=10, act=paddle.activation.Softmax())
    return predict


def convolutional_neural_network(img):
    """
    定义卷积神经网络分类器：
        输入的二维图像，经过两个卷积-池化层，使用以softmax为激活函数的全连接层作为输出层
    Args:
        img -- 输入的原始图像数据
    Return:
        predict_image -- 分类的结果
    """
    # 第一个卷积-池化层
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())

    # 第二个卷积-池化层
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # 全连接层
    predict = paddle.layer.fc(
        input=conv_pool_2, size=10, act=paddle.activation.Softmax())
    return predict


def network_config():
    """
    配置网络结构
    Args:
    Return:
        images -- 输入层
        label -- 标签数据
        predict_image -- 输出层
        cost -- 损失函数
        parameters -- 模型参数
        optimizer -- 优化器
    """

    # 输入层:
    # paddle.layer.data表示数据层,
    # name=’pixel’：名称为pixel,对应输入图片特征
    # type=paddle.data_type.dense_vector(784)：数据类型为784维(输入图片的尺寸为28*28)稠密向量
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))

    # 数据层:
    # paddle.layer.data表示数据层,
    # name=’label’：名称为label,对应输入图片的类别标签
    # type=paddle.data_type.dense_vector(10)：数据类型为10维(对应0-9这10个数字)稠密向量

    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))

    # 选择分类器：
    # 在此之前已经定义了3种不同的分类器，在下面的代码中,
    # 我们可以通过保留某种方法的调用语句、注释掉其余两种，以选择特定的分类器

    # predict_image = softmax_regression(images)
    # predict_image = multilayer_perceptron(images)
    predict = convolutional_neural_network(images)

    # 定义成本函数，addle.layer.classification_cost()函数内部采用的是交叉熵损失函数
    cost = paddle.layer.classification_cost(input=predict, label=label)

    # 利用cost创建参数parameters
    parameters = paddle.parameters.create(cost)

    # 创建优化器optimizer，下面列举了2种常用的优化器，不同类型优化器选一即可
    # 创建Momentum优化器，并设置学习率(learning_rate)、动量(momentum)和正则化项(regularization)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.01 / 128.0,
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

    # 创建Adam优化器，并设置参数beta1、beta2、epsilon
    # optimizer = paddle.optimizer.Adam(beta1=0.9, beta2=0.99, epsilon=1e-06)

    config_data = [predict, cost, parameters, optimizer]

    return config_data


def plot_init():
    """
    绘图初始化函数：
        初始化绘图相关变量
    Args:
    Return:
        cost_ploter -- 用于绘制cost曲线的变量
        error_ploter -- 用于绘制error_rate曲线的变量
    """
    # 绘制cost曲线所做的初始化设置
    cost_ploter = Ploter(TRAIN_TITLE_COST, TEST_TITLE_COST)

    # 绘制error_rate曲线所做的初始化设置
    error_ploter = Ploter(TRAIN_TITLE_ERROR, TEST_TITLE_ERROR)

    ploter = [cost_ploter, error_ploter]

    return ploter


def load_image(image_file):
    """
    定义读取输入图片的函数：
        读取指定路径下的图片，将其处理成分类网络输入数据对应形式的数据，如数据维度等
    Args:
        file -- 输入图片的文件路径
    Return:
        im_data -- 分类网络输入数据对应形式的数据
    """
    im_data = Image.open(image_file).convert('L')
    im_data = im_data.resize((28, 28), Image.ANTIALIAS)
    im_data = np.array(im_data).astype(np.float32).flatten()
    im_data = im_data / 255.0
    return im_data


def infer(predict, parameters, image_file):
    """
    定义判断输入图片类别的函数：
        读取并处理指定路径下的图片，然后调用训练得到的模型进行类别预测
    Args:
        predict_image -- 输出层
        parameters -- 模型参数
        image_file -- 输入图片的文件路径
    Return:
    """
    # 读取并预处理要预测的图片
    test_data = []
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    test_data.append((load_image(cur_dir + image_file),))

    # 利用训练好的分类模型，对输入的图片类别进行预测
    probs = paddle.infer(
        output_layer=predict, parameters=parameters, input=test_data)
    lab = np.argsort(-probs)
    print "Label of image/infer_3.png is: %d" % lab[0][0]


def main():
    """
    主函数：
        定义神经网络结构，训练模型并打印学习曲线、预测测试数据类别
    Args:
    Return:
    """
    # 初始化，设置是否使用gpu，trainer数量
    paddle.init(use_gpu=WITH_GPU, trainer_count=1)

    # 定义神经网络结构
    predict, cost, parameters, optimizer = network_config()

    # 构造trainer,配置三个参数cost、parameters、update_equation，它们分别表示成本函数、参数和更新公式
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # 初始化绘图变量
    cost_ploter, error_ploter = plot_init()

    # lists用于存储训练的中间结果，包括cost和error_rate信息，初始化为空
    lists = []

    def event_handler_plot(event):
        """
        定义event_handler_plot事件处理函数：
            事件处理器，可以根据训练过程的信息做相应操作：包括绘图和输出训练结果信息
        Args:
            event -- 事件对象，包含event.pass_id, event.batch_id, event.cost等信息
        Return:
        """
        global STEP
        if isinstance(event, paddle.event.EndIteration):
            # 每训练100次（即100个batch），添加一个绘图点
            if STEP % 100 == 0:
                cost_ploter.append(TRAIN_TITLE_COST, STEP, event.cost)
                # 绘制cost图像，保存图像为‘train_test_cost.png’
                cost_ploter.plot('./train_test_cost')
                error_ploter.append(
                    TRAIN_TITLE_ERROR, STEP,
                    event.metrics['classification_error_evaluator'])
                # 绘制error_rate图像，保存图像为‘train_test_error_rate.png’
                error_ploter.plot('./train_test_error_rate')
            STEP += 1
            # 每训练100个batch，输出一次训练结果信息
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        if isinstance(event, paddle.event.EndPass):
            # 保存参数至文件
            with open('params_pass_%d.tar' % event.pass_id, 'w') as param_f:
                trainer.save_parameter_to_tar(param_f)
            # 利用测试数据进行测试
            result = trainer.test(reader=paddle.batch(
                paddle.dataset.mnist.test(), batch_size=128))
            print "Test with Pass %d, Cost %f, %s\n" % (
                event.pass_id, result.cost, result.metrics)
            # 添加测试数据的cost和error_rate绘图数据
            cost_ploter.append(TEST_TITLE_COST, STEP, result.cost)
            error_ploter.append(
                TEST_TITLE_ERROR, STEP,
                result.metrics['classification_error_evaluator'])
            # 存储测试数据的cost和error_rate数据
            lists.append((
                event.pass_id, result.cost,
                result.metrics['classification_error_evaluator']))

    # 训练模型：
    # paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=8192)：
    # 表示trainer从paddle.dataset.mnist.train()这个reader中读取了buf_size=8192大小的数据并打乱顺序
    # paddle.batch(reader(), batch_size=128)：
    # 表示从打乱的数据中再取出batch_size=128大小的数据进行一次迭代训练
    # event_handler：事件处理函数，可以自定义event_handler，根据事件信息做相应的操作，
    # 下方代码中选择的是event_handler_plot函数
    # num_passes：定义训练的迭代次数

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=128),
        event_handler=event_handler_plot,
        num_passes=10)

    # 在多次迭代中，找到在测试数据上表现最好的一组参数，并输出相应信息
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
    print 'The classification accuracy is %.2f%%' % \
          (100 - float(best[2]) * 100)

    # 预测输入图片的类型
    infer(predict, parameters, '/image/infer_3.png')


if __name__ == '__main__':
    main()
