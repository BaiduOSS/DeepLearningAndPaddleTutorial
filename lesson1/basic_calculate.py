#!/usr/bin/env python2.7
# -*- encoding:utf-8 -*-
"""
Numpy中实现基本数学计算脚本
Created on 2017-11-16
@author: denglelai@baidu.com
@copyright: www.baidu.com
"""
import numpy as np


# 绝对值，1
a = np.abs(-1)
print "-1的绝对值为:" + str(a)

# sin函数，1.0
b = np.sin(np.pi / 2)
print "pi/2的正弦值为:" + str(b)

# tanh逆函数，0.50000107157840523
c = np.arctanh(0.462118)
print "tanh(0.462118)值为:" + str(c)

# e为底的指数函数，20.085536923187668
d = np.exp(3)
print "e的3次方值为:" + str(d)

# 2的3次方，8
f = np.power(2, 3)
print "2的3次方值为:" + str(f)

# 点积，1*3+2*4=11
g = np.dot([1, 2], [3, 4])
print "向量[1. 2]与向量[3. 4]点乘值为:" + str(g)

# 开方，5
h = np.sqrt(25)
print "25的2次方根值为:" + str(h)

# 求和，10
l = np.sum([1, 2, 3, 4])
print "对[1, 2, 3, 4]中元素求和结果为:" + str(l)

# 平均值，5.5
m = np.mean([4, 5, 6, 7])
print "对[1, 2, 3, 4]中元素求平均结果为:" + str(m)

# 标准差，0.96824583655185426
p = np.std([1, 2, 3, 2, 1, 3, 2, 0])
print "对[1, 2, 3, 2, 1, 3, 2, 0]中元素求标准差结果为:" + str(p)
