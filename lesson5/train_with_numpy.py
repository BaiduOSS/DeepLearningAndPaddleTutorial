#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    Authors: Jiahui Liu(2505774110@qq.com)
    Date:    2017/11/17 17:27:06
"""

import utils


def main():
    """
    main entry
    """
    train_X, train_Y, test_X, test_Y, classes = utils.load_data_sets()

    # 获取数据相关信息
    train_num = train_X.shape[0]
    test_num = test_X.shape[0]

    # 本例中num_px=64
    px_num = train_X.shape[1]

    # 转换数据形状
    data_dim = px_num * px_num * 3
    train_X = train_X.reshape(train_num, data_dim).T
    test_X = test_X.reshape(test_num, data_dim).T

    train_X = train_X / 255.
    test_X = test_X / 255.

    layer = [12288, 20, 7, 5, 1]
    parameters = utils.deep_neural_network(train_X, train_Y, layer, 2500)
    print 'Train Accuracy:', utils.predict_image(
        parameters, train_X, train_Y), '%'
    print 'Test Accuracy:', utils.predict_image(
        parameters, test_X, test_Y), '%'


if __name__ == '__main__':
    main()
