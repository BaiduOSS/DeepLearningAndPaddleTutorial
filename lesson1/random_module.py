#!/usr/bin/env python2.7
# -*- encoding:utf-8 -*-
"""
Numpy中random模块说明脚本
Created on 2017-11-16
@author: denglelai@baidu.com
@copyright: www.baidu.com
"""
import numpy as np

# 设置随机数种子
np.random.seed(42)

# 产生一个1x3，[0,1)之间的浮点型随机数
# array([[ 0.37454012,  0.95071431,  0.73199394]])
# 后面的例子就不在注释中给出具体结果了
print "产生一个1x3，[0,1)之间的浮点型随机数Array:\n" + str(np.random.rand(1, 3))

# 产生一个[0,1)之间的浮点型随机数
print "产生一个[0,1)之间的浮点型随机数:\n" + str(np.random.random())

# 从a中有放回的随机采样7个
a = np.array([1, 2, 3, 4, 5, 6, 7])
print "从a中有放回的随机采样7个:\n" + str(np.random.choice(a, 7))

# 从a中无放回的随机采样7个
print "从a中无放回的随机采样7个:\n" + str(np.random.choice(a, 7, replace=False))

# 对a进行乱序并返回一个新的array
print "对a进行乱序并返回一个新的array:\n" + str(np.random.permutation(a))

# 生成一个长度为9的随机bytes序列并作为str返回
# '\x96\x9d\xd1?\xe6\x18\xbb\x9a\xec'
print "生成一个长度为9的随机bytes序列并作为str返回:\n" + str([np.random.bytes(9)])