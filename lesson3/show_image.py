#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Authors: weixing(wx_crome@163.com)
    Date:    2017/11/12 17:23:06
"""


import matplotlib.pyplot as plt

import utils


def main():
    """
    show some images in the train dataset
    """
    train_x_ori, train_y, test_x_ori, test_y, classes = \
        utils.load_data_sets()
    print "1. load data from the dataset"
    print "there is " + str(train_y.shape[1]) + " train data and " \
          + str(test_y.shape[1]) + " test data"
    print "train set categories as " + str(classes)

    print "2. show train sets label, 0 means not cat, 1 means cat"
    print train_y

    print "3. show image(2) with label 1, it should be a cat image"
    index = 2
    plt.imshow(train_x_ori[index])
    plt.pause(30)

    print "4. show image(1) with label 0, it should not be a cat image"
    index = 1
    plt.imshow(train_x_ori[index])
    plt.pause(30)

    print "5. show test sets label, 0 means not cat, 1 means cat"
    print test_y

    print "6. show image(1) with label 1, it should be a cat image"
    index = 1
    plt.imshow(test_x_ori[index])
    plt.pause(30)


if __name__ == '__main__':
    main()
