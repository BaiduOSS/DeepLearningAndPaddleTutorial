# -*- coding:utf-8 -*-
import sys
import numpy as np

import paddle.v2 as paddle
import h5py
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset



TRAINING_SET = None
TEST_SET = None
datadim = None

# 载入数据(cat/non-cat)
def load_data():
    global TRAINING_SET, TEST_SET, datadim

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    # 定义纬度
    datadim = num_px * num_px * 3
    # 数据展开,注意此处为了方便处理，没有加上.T的转置操作
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1)
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1)

    # 归一化
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    # print(train_set_x.shape)
    # print(train_set_y.shape)

    TRAINING_SET = np.hstack((train_set_x, train_set_y.T))
    TEST_SET = np.hstack((test_set_x, test_set_y.T))


    # print(TRAINING_SET.shape)
    # print(TEST_SET.shape)
    # print(TEST_SET_X.shape)
    # print(TEST_SET_Y.shape)

# 训练数据集
def train():
    global TRAINING_SET
    def reader():
        for data in TRAINING_SET:
            yield data[:-1], data[-1:]
    return reader
# 测试数据集
def test():
    global TEST_SET
    def reader():
        for data in TEST_SET:
            yield data[:-1], data[-1:]
    return reader

# 获取test_data
def get_test_data():
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

# 获取train_data
def get_train_data():
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
    train_right = 0
    train_total = len(train_data['label'])
    for i in range(len(probs_train)):
        if float(probs_train[i][0]) > 0.5 and train_data['label'][i] == 1:
            train_right += 1
        elif float(probs_train[i][0]) < 0.5 and train_data['label'][i] == 0:
            train_right += 1
    return (float(train_right) / float(train_total)) * 100

# 测试集准确度
def test_accuracy(probs_test, test_data):
    test_right = 0
    test_total = len(test_data['label'])
    for i in range(len(probs_test)):
        if float(probs_test[i][0]) > 0.5 and test_data['label'][i] == 1:
            test_right += 1
        elif float(probs_test[i][0]) < 0.5 and test_data['label'][i] == 0:
            test_right += 1
    return (float(test_right) / float(test_total)) * 100



def main():
    global datadim
    # init
    paddle.init(use_gpu=False, trainer_count=1)
    load_data()

    image = paddle.layer.data(
        name='image', type=paddle.data_type.dense_vector(datadim))

    # 输入层，paddle.layer.data表示数据层
    # name=’label’：名称为image
    # type=paddle.data_type.dense_vector(datadim)：数据类型为datadim维向量

    y_predict = paddle.layer.fc(
        input=image, size=1, act=paddle.activation.Sigmoid())

    # 输出层，paddle.layer.fc表示全连接层
    # input=image: 该层输入数据为image
    # size=1：神经元个数
    # act=paddle.activation.Sigmoid()：激活函数为Sigmoid()
    y_label = paddle.layer.data(
        name='label', type=paddle.data_type.dense_vector(1))

    cost = paddle.layer.multi_binary_label_cross_entropy_cost(input=y_predict, label=y_label)

    # 创建parameters
    parameters = paddle.parameters.create(cost)

    #创建optimizer
    optimizer = paddle.optimizer.Momentum(momentum=0, learning_rate=0.01)

    feeding = {
        'image': 0,
        'label': 1}

    costs = []
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print("Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id, event.cost))
            if event.pass_id % 100 == 0:
                costs.append(event.cost)
            # with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
            #     parameters.to_tar(f)


# 创建trainer
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(train(), buf_size=50000),
            batch_size=2),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=1000)

    # 获取测试数据和训练数据，用来验证模型准确度
    train_data = get_train_data()
    test_data = get_test_data()

    # 根据train_data和test_data预测结果
    probs_train = paddle.infer(
        output_layer=y_predict, parameters=parameters, input=train_data['image']
    )
    probs_test = paddle.infer(
        output_layer=y_predict, parameters=parameters, input=test_data['image']
    )

    # 计算train_accuracy和test_accuracy
    print("train_accuracy: {} %".format(train_accuracy(probs_train, train_data)))
    print("test_accuracy: {} %".format(test_accuracy(probs_test, test_data)))

    # Plot learning curve (with costs)
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = 0.01")
    plt.show()
    plt.savefig("costs.jpg")

if __name__ == '__main__':
    main()
