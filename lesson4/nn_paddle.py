# -*- coding:utf-8-*-
import PIL
import sys
import numpy as np
import paddle.v2 as paddle
# import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import paddle.v2.dataset.uci_housing as uci_housing


TRAINING_SET = None
TEST_SET = None
datadim = None
def main():
    global TRAINING_SET, TEST_SET, datadim

    paddle.init(use_gpu = False,trainer_count=1)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]


    datadim = num_px * num_px * 3


    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1)
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1)


    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    TRAINING_SET = np.hstack((train_set_x, train_set_y.T))
    TEST_SET = np.hstack((test_set_x, test_set_y.T))





    def train():
        global TRAINING_SET
        def reader():
            for data in TRAINING_SET:
                yield data[:-1], data[-1:]
        return reader


    def test():
        global TEST_SET
        def reader():
            for data in TEST_SET:
                yield data[:-1], data[-1:]
        return reader

    image = paddle.layer.data(
        name='image', type=paddle.data_type.dense_vector(datadim))


    def multi_nn(img,j):
        h1 = paddle.layer.fc(input=img, size =j , act = paddle.activation.Tanh())
        y_out = paddle.layer.fc(input = h1, size = 1, act = paddle.activation.Sigmoid())
        return y_out
    j = 4
    yout = multi_nn(image,j)


    # y_out = paddle.layer.fc(input=image, size=1, act=paddle.activation.Sigmoid())
    lbl = paddle.layer.data(
        name='label', type=paddle.data_type.dense_vector(1))

    cost = paddle.layer.multi_binary_label_cross_entropy_cost(input=yout, label=lbl)

    #     parameters
    parameters = paddle.parameters.create(cost)

    #     optimizer
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
            #     parameters
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
            batch_size=30),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=1000)

    test_data_creator = test()
    test_data_image = []
    test_data_label = []

    for item in test_data_creator():
        test_data_image.append((item[0], ))
        test_data_label.append(item[1])

    train_data_creator = train()
    train_data_image = []
    train_data_label = []

    for item in train_data_creator():
        train_data_image.append((item[0],))
        train_data_label.append(item[1])


    probs_test = paddle.infer(
        output_layer=yout, parameters=parameters, input=test_data_image
    )

    probs_train = paddle.infer(
        output_layer=yout, parameters=parameters, input=train_data_image
    )

    test_right = 0
    train_right = 0
    test_total = len(test_data_image)
    train_total = len(train_data_image)

    for i in range(len(probs_train)):
        if float(probs_train[i][0]) > 0.5 and train_data_label[i] == 1:
            train_right += 1
        elif float(probs_train[i][0]) < 0.5 and train_data_label[i] == 0:
            train_right += 1
    print("train_accuracy: {} %".format((float(train_right) / float(train_total)) * 100))
    print(train_right, train_total)

    for i in range(len(probs_test)):
        if float(probs_test[i][0]) > 0.5 and test_data_label[i] == 1:
            test_right += 1
        elif float(probs_test[i][0]) < 0.5 and test_data_label[i] == 0:
            test_right += 1

    print("test_accuracy: {} %".format((float(test_right) / float(test_total)) * 100))
    print(test_right, test_total)

if __name__=='__main__':
    main()