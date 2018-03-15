#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Numpy中实现广播机制说明脚本
    Created on 2017-11-16
    author: denglelai
"""
import numpy as np


def main():
    """
    show broadcast operation in numpy
    """
    array_a = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    array_b = np.array([
        [1, 2, 3],
        [1, 2, 3]
    ])

    # 维度一样的array，对位计算
    # array([[2, 4, 6],
    #       [5, 7, 9]])

    print "相同维度array, 进行对位运算, 结果为：\n" + str(array_a + array_b)

    array_c = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])

    array_d = np.array([2, 2, 2])
    # 广播机制让计算的表达式保持简洁
    # array_d和array_c的每一行分别进行运算
    # array([[ 3,  4,  5],
    #       [ 6,  7,  8],
    #       [ 9, 10, 11],
    #       [12, 13, 14]])
    print "广播机制下, c和d进行每一行分别计算, 结果为：\n" + str(array_c + array_d)


if __name__ == '__main__':
    main()
