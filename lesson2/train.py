#-*- coding:utf-8 -*-

import numpy as np
import paddle.v2 as paddle

CODEMASTER_TRAIN_DATA = None
X_RAW = None
CODEMASTER_TEST_DATA = None


def load_data(filename, feature_num=2, ratio=0.8):
    #如果测试数据集和训练数据集都不为空，就不再载入数据load_data
    global CODEMASTER_TRAIN_DATA, CODEMASTER_TEST_DATA,X_RAW
    if CODEMASTER_TRAIN_DATA is not None and CODEMASTER_TEST_DATA is not None:
        return
    #data = np.loadtxt()表示将数据载入后以矩阵或向量的形式存储在data中
    #delimiter=',' 表示以','为分隔符
    data = np.loadtxt(filename, delimiter=',')
    X_RAW = data.T[0].copy()
    #axis=0 表示按列计算
    #data.shape[0]表示data中一共多少列
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
        axis=0) / data.shape[0]
    #归一化，data[:, i] 表示第i列的元素
    for i in xrange(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    #offset用于划分训练数据集和测试数据集，例如0.8表示训练集占80%
    offset = int(data.shape[0] * ratio)
    CODEMASTER_TRAIN_DATA = data[:offset].copy()
    CODEMASTER_TEST_DATA = data[offset:].copy()

def train():
    global CODEMASTER_TRAIN_DATA
    load_data("data.txt")

    #yield作用同return，但是返回的是生成器(generator)，生成器只能调用一次，实时计算
    def reader():
        for d in CODEMASTER_TRAIN_DATA:
            yield d[:-1], d[-1:]

    return reader


def test():
    global CODEMASTER_TEST_DATA
    load_data("data.txt")

    def reader():
        for d in CODEMASTER_TEST_DATA:
            yield d[:-1], d[-1:]

    return reader


def main():
    # init
    paddle.init(use_gpu=False, trainer_count=1)

    # network config
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(1))
    y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
    y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
    cost = paddle.layer.mse_cost(input=y_predict, label=y)

    # create parameters
    parameters = paddle.parameters.create(cost)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(momentum=0)

    # stochastic gradient descent
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # mapping data
    feeding = {'x': 0, 'y': 1}

    # event_handler to print training and testing info
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                reader=paddle.batch(test(), batch_size=2),
                feeding=feeding)
            print "Test %d, Cost %f" % (event.pass_id, result.cost)

    # training
    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(train(), buf_size=500),
            batch_size=2),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=300)

    # print result parameter
    print("Result Parameters as below:")
    a = parameters.get('___fc_layer_0__.w0')[0]
    b = parameters.get('___fc_layer_0__.wbias')[0]
    print(a,b)

    x0 = X_RAW[0]
    y0 = a * CODEMASTER_TRAIN_DATA[0][0] + b

    x1 = X_RAW[1]
    y1 = a * CODEMASTER_TRAIN_DATA[1][0] + b

    a = (y0 - y1) / (x0 - x1)
    b = (y1 - a * x1)

    print 'a = ', a
    print 'b = ', b
   


if __name__ == '__main__':
    main()
