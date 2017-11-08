# -*- coding:utf-8 -*-
import sys
import paddle.v2 as paddle
# import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import paddle.v2.dataset.uci_housing as uci_housing


def main():
    paddle.init(use_gpu = FALSE,train_count = 1)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    # 定义纬度
    datadim = num_px * num_px * 3

    # 数据展开,注意此处为了方便处理，没有加上.T的转置操作
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1)
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1)
    #归一化
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    # 定义训练数据集
    def train():
        def reader():
            for i in range(len(train_set_x)):
                yield train_set_x[i], train_set_y[0][i]

        return reader

    # 定义测试数据集
    def test():
        def reader():
            for i in range(len(test_set_x)):
                yield test_set_x[i], test_set_y[0][i]

        return reader

    image = paddle.layer.data(
        name='image', type=paddle.data_type.dense_vector(datadim)

    j = 4
    def multi_nn(img):
        h1 = paddle.layer.fc(input=img, size =j , act = paddle.activation.Tanh())
        y_out = paddle.layer.fc(input = h1, size = 1, act = paddle.activation.Sigmoid())
        return y_out
    yout = multi_nn(image)
    lbl = paddle.layer.data(
        name='label', type=paddle.data_type.dense_vector(1))

    cost = paddle.layer.multi_binary_label_cross_entropy_cost(input=yout, label=lbl)

    # 创建parameters
    parameters = paddle.parameters.create(cost)

    # 创建optimizer
    optimizer = paddle.optimizer.Momentum(momentum=0, learning_rate=0.001)

    feeding = {
        'image': 0,
        'label': 1}

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print("\nPass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id, event.cost))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            # 保存parameters
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                parameters.to_tar(f)

            result = trainer.test(
                reader=paddle.batch(
                    test(), batch_size=30
                ),
                feeding=feeding
            )
            print("\nTest with Pass %d" % event.pass_id)

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(train(), buf_size=50000),
            batch_size=60),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=200)

    probs_test = paddle.infer(
        output_layer=out, parameters=parameters, input=test_set_x
    )

    probs_train = paddle.infer(
        output_layer=out, parameters=parameters, input=train_set_x
    )

    test_right = 0
    train_right = 0
    test_total_length = len(test_set_x)
    train_total_length = len(train_set_x)

    #计算准确率
    for i in range(len(probs_train)):
        if float(probs_train[i][0]) > 0.5 and train_set_y[0][i] == 1:
            train_right += 1
        elif float(probs_train[i][0]) < 0.5 and train_set_y[0][i] == 0:
            train_right += 1
    print("train_accuracy: {} %".format((float(train_right) / float(train_total_length)) * 100))
    print(train_right, train_total_length)
    for i in range(len(probs_test)):
        if float(probs_test[i][0]) > 0.5 and test_set_y[0][i] == 1:
            test_right += 1
        elif float(probs_test[i][0]) < 0.5 and test_set_y[0][i] == 0:
            test_right += 1

    print("test_accuracy: {} %".format((float(test_right) / float(test_total_length)) * 100))
    print(test_right, test_total_length)

if __name__=='__main__':
    main()