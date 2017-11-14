#!/usr/bin/env python
#  -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
This module provide configure file management service in i18n environment.

Authors: weixing(wx_crome@163.com)
Date:    2017/11/12 17:23:06

用于载入数据，目标数据源为两个.h5文件，分别为：
train_catvnoncat.h5：训练数据集（猫图片）
test_catvnoncat.h5：测试数据集（猫图片）
"""
import numpy as np
import h5py
    
    
def load_dataset():
    """
    用于从两个.h5文件中分别加载训练数据和测试数据

    Args:
    Return:
        train_set_x_orig -- 原始训练数据集
        train_set_y -- 原始训练数据标签
        test_set_x_orig -- 原始测试数据集
        test_set_y -- 原始测试数据标签
        classes(cat/non-cat) -- 分类list
    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    dataset = [train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes]
    return dataset

