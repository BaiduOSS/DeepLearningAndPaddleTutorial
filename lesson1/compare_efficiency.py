#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Numpy中实现向量化与非向量化计算效率比较脚本
    Created on 2017-11-16
    author: denglelai
"""
import time
import numpy as np


def main():
    """
    show two version of vector operation
    """

    # 初始化两个1000000维的随机向量v1,v2用于矩阵相乘计算
    vector_1 = np.random.rand(1000000)
    vector_2 = np.random.rand(1000000)
    result = 0

    # 矩阵相乘-非向量化版本
    tic = time.time()
    for i in range(1000000):
        result = result + vector_1[i] * vector_2[i]
    toc = time.time()
    print "非向量化-计算结果：" + str(result)
    print "非向量化-计算时间：" + str((toc - tic) * 1000) + "ms" + "\n"

    # 矩阵相乘-向量化版本
    tic = time.time()
    result = np.dot(vector_1, vector_2)
    toc = time.time()
    print "向量化-计算结果：" + str(result)
    print "向量化-计算时间：" + str((toc - tic) * 1000)+"ms"


if __name__ == '__main__':
    main()
