#!/usr/bin/env python
#  -*- coding:utf-8 -*-

"""
    Authors: weixing(wx_crome@163.com)
    Date:    2017/11/12 17:23:06

    用于载入数据，目标数据源为两个.h5文件，分别为：
    train_images.h5：训练数据集（猫图片）
    test_images.h5：测试数据集（猫图片）
"""
import h5py
import numpy as np


def load_data_sets():
    """
    用于从两个.h5文件中分别加载训练数据和测试数据

    Args:
    Return:
        train_x_ori: 原始训练数据集
        train_y: 原始训练数据标签
        test_x_ori: 原始测试数据集
        test_y: 原始测试数据标签
        classes(cat/non-cat): 分类list
    """
    train_data = h5py.File('datasets/train_images.h5', "r")
    # train set features
    train_x_ori = np.array(train_data["train_set_x"][:])
    # train set labels
    train_y_ori = np.array(train_data["train_set_y"][:])

    test_data = h5py.File('datasets/test_images.h5', "r")
    # test set features
    test_x_ori = np.array(test_data["test_set_x"][:])
    # test set labels
    test_y_ori = np.array(test_data["test_set_y"][:])
    # the list of classes
    classes = np.array(test_data["list_classes"][:])

    train_y_ori = train_y_ori.reshape((1, train_y_ori.shape[0]))
    test_y_ori = test_y_ori.reshape((1, test_y_ori.shape[0]))

    result = [train_x_ori, train_y_ori, test_x_ori,
              test_y_ori, classes]
    return result
