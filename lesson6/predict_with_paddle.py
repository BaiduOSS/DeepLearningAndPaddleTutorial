#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Authors: xiake(kedou1993@163.com)
Date:    2017/11/29

使用PaddlePaddle框架实现数字识别案例的预测任务，关键步骤如下：
1.定义分类器网络结构
2.读取训练好的模型参数
3.预测结果
"""

import os

import numpy as np
import paddle.v2 as paddle
from PIL import Image

WITH_GPU = os.getenv('WITH_GPU', '0') != '0'


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
        predict_image -- 输出层
    """

    # 输入层:
    # paddle.layer.data表示数据层,
    # name=’pixel’：名称为pixel,对应输入图片特征
    # type=paddle.data_type.dense_vector(784)：
    # 数据类型为784维(输入图片的尺寸为28*28)稠密向量

    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))

    # 选择分类器：
    # 在此之前已经定义了3种不同的分类器，在下面的代码中,
    # 我们可以通过保留某种方法的调用语句、注释掉其余两种，以选择特定的分类器

    # predict_image = softmax_regression(images)
    # predict_image = multilayer_perceptron(images)
    predict = convolutional_neural_network(images)

    return predict


def load_image(image_file):
    """
    定义读取输入图片的函数：
        读取指定路径下的图片，将其处理成分类网络输入数据对应形式的数据，如数据维度等
    Args:
        file -- 输入图片的文件路径
    Return:
        img_data -- 分类网络输入数据对应形式的数据
    """
    img_data = Image.open(image_file).convert('L')
    img_data = img_data.resize((28, 28), Image.ANTIALIAS)
    img_data = np.array(img_data).astype(np.float32).flatten()
    img_data = img_data / 255.0
    return img_data


def main():
    """
    定义网络结构、读取模型参数并预测结果
    Args:
    Return:
    """
    paddle.init(use_gpu=WITH_GPU)

    # 定义神经网络结构
    predict = network_config()

    # 读取模型参数
    if not os.path.exists('params_pass_9.tar'):
        print "Params file doesn't exists."
        return
    with open('params_pass_9.tar', 'r') as param_f:
        parameters = paddle.parameters.Parameters.from_tar(param_f)

    # 读取并预处理要预测的图片
    test_data = []
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    test_data.append((load_image(cur_dir + '/image/infer_3.png'),))

    # 利用训练好的分类模型，对输入的图片类别进行预测
    probs = paddle.infer(
        output_layer=predict, parameters=parameters, input=test_data)
    lab = np.argsort(-probs)
    print "Label of image/infer_3.png is: %d" % lab[0][0]


if __name__ == '__main__':
    main()
