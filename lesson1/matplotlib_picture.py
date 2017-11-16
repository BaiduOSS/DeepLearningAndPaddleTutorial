#!/usr/bin/env python2.7
# -*- encoding:utf-8 -*-
"""
Matplotlib模块图像处理示例脚本
Created on 2017-11-16
@author: denglelai@baidu.com
@copyright: www.baidu.com
"""
import matplotlib.pyplot as plt
import numpy as np

def rgb2gray(rgb):
    """
    图像转化为灰度图实现
    args:
        rgb: 彩色图像
    reurn:
        np.dot: 灰度图像
    """
    return np.dot(rgb[...,
                  : 3], [0.299, 0.587, 0.114])

# 读取一张小白狗的照片并显示
plt.figure('A Little White Dog')
little_dog_img = plt.imread('./image/little_white_dog.jpg')
plt.imshow(little_dog_img)

# Z是小白狗的照片，img0就是Z，img1是Z做了个简单的变换
Z = plt.imread('./image/little_white_dog.jpg')
Z = rgb2gray(Z)
img0 = Z
img1 = 1 - Z

# cmap指定为'gray'用来显示灰度图
fig = plt.figure('Auto Normalized Visualization')
ax0 = fig.add_subplot(121)
ax0.imshow(img0, cmap='gray')
ax1 = fig.add_subplot(122)
ax1.imshow(img1, cmap='gray')
plt.show()