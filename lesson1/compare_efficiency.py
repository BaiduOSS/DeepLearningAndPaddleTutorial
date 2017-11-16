#!/usr/bin/env python2.7
# -*- encoding:utf-8 -*-
"""
Numpy中实现向量化与非向量化计算效率比较脚本
Created on 2017-11-16
@author: denglelai@baidu.com
@copyright: www.baidu.com
"""
import numpy as np
import time


#初始化两个1000000维的随机向量v1,v2用于矩阵相乘计算
v1 = np.random.rand(1000000)
v2 = np.random.rand(1000000)
v = 0

#矩阵相乘-非向量化版本
tic = time.time()
for i in range(1000000):
    v = v + v1[i] * v2[i]
toc = time.time()
print "非向量化-计算结果：" + str(v) + "ms"
print "非向量化-计算时间：" + str((toc - tic) * 1000) + "ms" + "\n"

#矩阵相乘-向量化版本
tic = time.time()
v = np.dot(v1, v2)
toc = time.time()
print "向量化-计算结果：" + str(v) + "ms"
print "向量化-计算时间：" + str((toc - tic) * 1000)+"ms"