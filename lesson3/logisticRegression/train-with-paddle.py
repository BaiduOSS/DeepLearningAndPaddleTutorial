import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import paddle.v2 as paddle

TRAINING_DATA = None
TEST_DATA = None
TEST_DATA_X = None
TEST_DATA_Y = None

def load_data():
    global TRAINING_DATA, TEST_DATA_X, TEST_DATA_Y, TEST_DATA
    
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y,classes = load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3]).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3]).T
    train_set_x = train_set_x_flatten/255.0
    test_set_x = test_set_x_flatten/255.0

    TRAINING_DATA = np.hstack([train_set_x.T,train_set_y.T])

    TEST_DATA = np.hstack([test_set_x.T,test_set_y.T])
    TEST_DATA_X = test_set_x.T
    TEST_DATA_Y = test_set_y.T
    print(TRAINING_DATA.shape)
    print(TEST_DATA)

def train(): 
    global TRAINING_DATA
    load_data()
    def reader():
        for d in TRAINING_DATA:
            yield d[:-1],d[-1:]
    return reader

def test():
    global TEST_DATA
    load_data()
    def reader():
        for d in TEST_DATA:
            yield d[:-1],d[-1:]
    return reader


def main():
    # init
    paddle.init(use_gpu=False, trainer_count=1)

    # network config
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(12288))
    y_predict = paddle.layer.fc(input = x, size =1,act = paddle.activation.Sigmoid())
    y_label = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))

    cost = paddle.layer.multi_binary_label_cross_entropy_cost(input=y_predict, label=y_label)
    parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Momentum(momentum=0, learning_rate=0.005)

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

     # mapping data
    feeding = {'x': 0, 'y': 1}
    

    # event_handler to print training and testing info
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            print "Pass %d, Batch %d, Cost %f" % (
                event.pass_id, event.batch_id, event.cost)


    # training
    reader = paddle.reader.shuffle(train(), buf_size=50000)
    batch_reader = paddle.batch(reader,batch_size=256)
    trainer.train(
        batch_reader,
        feeding=feeding,
        event_handler=event_handler,
        num_passes=2000)

    # infer test
    test_data_creator = test()
    test_data = []
    test_label = []

    for item in test_data_creator():
        test_data.append((item[0],))
        test_label.append(item[1])

    probs = paddle.infer(
        output_layer=y_predict, parameters=parameters, input=test_data)

    right_number =0
    total_number = len(test_data)
    for i in xrange(len(probs)):
        #print "label=" + str(test_label[i]) + ", predict=" + str(probs[i])
        if float(probs[i][0]) >= 0.5 and test_label[i] ==1 :
            right_number += 1
        elif float(probs[i][0]) < 0.5 and test_label[i] ==0:
            right_number += 1

    print("right_number is {0} in {1} samples".format(right_number,total_number))



if __name__ == '__main__':
    main()
