#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Numpy中实现矩阵转置基础操作脚本
    Created on 2017-11-16
    author: denglelai
"""
import numpy as np


def main():
    """
    show matrix's transpose
    """

    # 初始化矩阵x, x为
    # [[ 0  1  2  3]
    # [ 4  5  6  7]
    # [ 8  9 10 11]]

    matrix_x = np.array(np.arange(12).reshape((3, 4)))
    print "原始矩阵：\n" + str(matrix_x)

    # 利用函数transpose()实现转置，转置后为矩阵t：
    # [[ 0  4  8]
    # [ 1  5  9]
    # [ 2  6 10]
    # [ 3  7 11]]
    matrix_y = matrix_x.transpose()
    print "转置后矩阵：\n" + str(matrix_y)


if __name__ == '__main__':
    main()
