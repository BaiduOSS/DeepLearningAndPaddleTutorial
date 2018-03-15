#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Numpy中实现基本数学计算脚本
    Created on 2017-11-16
    author: denglelai
"""
import numpy as np


def main():
    """
    run basic operations of numpy
    """

    # 绝对值，1
    a_variable = np.abs(-1)
    print "-1的绝对值为:" + str(a_variable)

    # sin函数，1.0
    a_variable = np.sin(np.pi / 2)
    print "pi/2的正弦值为:" + str(a_variable)

    # tanh逆函数，0.500001071578
    a_variable = np.arctanh(0.462118)
    print "tanh(0.462118)值为:" + str(a_variable)

    # e为底的指数函数，20.0855369232
    a_variable = np.exp(3)
    print "e的3次方值为:" + str(a_variable)

    # 2的3次方，8
    a_variable = np.power(2, 3)
    print "2的3次方值为:" + str(a_variable)

    # 点积，1*3+2*4=11
    a_variable = np.dot([1, 2], [3, 4])
    print "向量[1. 2]与向量[3. 4]点乘值为:" + str(a_variable)

    # 开方，5
    a_variable = np.sqrt(25)
    print "25的2次方根值为:" + str(a_variable)

    # 求和，10
    a_variable = np.sum([1, 2, 3, 4])
    print "对[1, 2, 3, 4]中元素求和结果为:" + str(a_variable)

    # 平均值，5.5
    a_variable = np.mean([4, 5, 6, 7])
    print "对[1, 2, 3, 4]中元素求平均结果为:" + str(a_variable)

    # 标准差，0.968245836552
    a_variable = np.std([1, 2, 3, 2, 1, 3, 2, 0])
    print "对[1, 2, 3, 2, 1, 3, 2, 0]中元素求标准差结果为:" + str(a_variable)


if __name__ == '__main__':
    main()
