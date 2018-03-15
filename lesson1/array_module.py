#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Array 基础操作脚本
    Created on 2017-11-16
    author: denglelai
"""

import numpy as np


def main():
    """
    show np.array related operations
    """
    # a是python中的list类型
    a = [1, 2, 3, 4]
    # 数组化之后b的类型变为 array
    b = np.array(a)
    # a的类型 <type 'list'>
    print "原始的类型为：" + str(type(a))
    # b的类型 <type 'numpy.ndarray'>
    print "数组化之后的类型为：" + str(type(b))
    # shape参数表示array的大小，这里是(4,)
    print "Array的大小为：" + str(b.shape)
    # 调用argmax()函数可以求得array中的最大值的索引，这里是3
    print "Array中最大元素索引为：" + str(b.argmax())
    # 调用max()函数可以求得array中的最大值，这里是4
    print "Array中最大元素值为：" + str(b.max())
    # 调用mean()函数可以求得array中的平均值，这里是2.5
    print "Array中元素平均值为：" + str(b.mean())


if __name__ == '__main__':
    main()
