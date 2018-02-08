# -*- coding:utf-8 -*-
import dnn_function
import dnn_utils


def main():
    X_train, Y_train, X_test, Y_test, classes = dnn_utils.load_dataset()

    # 获取数据相关信息
    train_num = X_train.shape[0]
    test_num = X_test.shape[0]
    # 本例中num_px=64
    px_num = X_train.shape[1]

    # 转换数据形状
    data_dim = px_num * px_num * 3
    X_train = X_train.reshape(train_num, data_dim).T
    X_test = X_test.reshape(test_num, data_dim).T

    X_train = X_train / 255.
    X_test = X_test / 255.

    layer = [12288, 20, 7, 5, 1]
    parameters = dnn_function.deep_neural_network(X_train, Y_train, layer, 2500)
    print('Train Accuracy:', dnn_function.predict_result(parameters, X_train, Y_train), '%')
    print('Test Accuracy:', dnn_function.predict_result(parameters, X_test, Y_test), '%')

if __name__ == '__main__':
    main()
