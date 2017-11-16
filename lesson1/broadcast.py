#!/usr/bin/env python2.7
# -*- encoding:utf-8 -*-
"""
Numpy中实现广播机制说明脚本
Created on 2017-11-16
@author: denglelai@baidu.com
@copyright: www.baidu.com
"""
import numpy as np


a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

b = np.array([
    [1, 2, 3],
    [1, 2, 3]
])

'''
维度一样的array，对位计算
array([[2, 4, 6],
       [5, 7, 9]])
'''
print "相同维度array, 进行对位运算, 结果为：\n" + str(a + b)

c = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])
d = np.array([2, 2, 2])

'''
广播机制让计算的表达式保持简洁
d和c的每一行分别进行运算
array([[ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11],
       [12, 13, 14]])
'''
print "广播机制下, c和d进行每一行分别计算, 结果为：\n" + str(c + d)