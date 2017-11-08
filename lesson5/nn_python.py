# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#初始化参数
def inin_para(row,col,m):
    pa = np.random.randn(row,col)*m
    return np.mat(pa)

#计算z1（隐藏层加权偏移后的中间量）
def z1_out(w1, b1, x):
    z1 = w1 * x + b1
    return z1

#计算h1（z1激活后的值）
def h1_out(z1):
    z = np.tanh(z1)
    return z


#计算z2（输出层的中间量）
def z2_out(w2,b2,h1):
    z2 = w2 * h1 + b2
    return z2

#计算输出y
def y_out(z2):
    z = 1 / (1 + np.exp(-z2))
    return z

#整合上述函数，直接输出y
def calculate(x, w1, w2, b1, b2):
    z1 = z1_out(w1, b1, x)
    h1 = h1_out(z1)
    z2 = z2_out(w2,b2,h1)
    y_ou = y_out(z2)
    return y_ou

#BP
#计算dy
def dy_out(y_ou, y):
    dy_ou = y_ou - y
    return dy_ou

#计算dz2
def dz2_out(dy_out, y_ou):
    dz = 1 - y_ou
    dz = np.multiply(y_ou, dz)
    dz = np.multiply(dy_out,dz)
    return dz

#计算dw2
def dw2_out(dz22, h1, n):
    dw22 = dz22 * (h1.T) / n
    return dw22

#计算db2
def db2_out(dz22, n):
    db22 = np.sum(dz22, axis=1)/n
    return db22

#计算dh1
def dh1_out(dz2, w2):
    dh11 = w2.T * dz2
    return dh11

#计算dz1
def dz1_out(dh11, h1):
    hh = np.multiply(h1,h1)
    hh = 1 - hh
    dz1 = np.multiply(dh11, hh)
    return dz1

#计算dw1
def dw1_out(dz11, x, n):
    dw11 = (dz11 * x.T) / n
    return dw11

#计算db1
def db1_out(dz11, n):
    db11 = np.sum(dz11, axis=1) / n
    return db11

#更新参数
def update(p, dp, u):
    return p - (u * dp)

#计算正确率
def accuracy(y,yout):
    a = 1 - np.mean(np.abs(1*(yout>0.5)-y))
    return a

#载入数据
# 下载数据集(cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(m_train,-1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test,-1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

x = np.mat(train_set_x)
y = np.mat(train_set_y)

#设置相关量
j = 4 #每个隐藏层节点苏
u = 0.09 #学习率
n = 400 #数据个数

#初始化参数
w1 = inin_para(j,x.shape[0],0.01)
w2 = inin_para(y.shape[0],j,0.01)
b1 = inin_para(j,1,0.01)
b2 = inin_para(y.shape[0],1,0.01)



#训练
for i in range(2000):

    yout = calculate(x, w1, w2, b1, b2) #计算y
    dyout = dy_out(yout, y) #计算dy
    dz2 = dz2_out(dyout, yout) #计算dz2

    z1 = z1_out(w1, b1, x)
    h1 = h1_out(z1)
    dw2 = dw2_out(dz2, h1, n) #计算dw2
    w2 = update(w2, dw2, u) #更新w2

    db2 = db2_out(dz2, n)
    b2 = update(b2, db2, u) #更新b2

    dh1 = dh1_out(dz2, w2)
    dz1 = dz1_out(dh1, h1)

    dw1 = dw1_out(dz1, x, n)
    w1 = update(w1, dw1, u) #更新w1

    db1 = db1_out(dz1, n)
    b1 = update(b1, db1, u) #更新b1

print('Train_accuracy:',accuracy(y,yout))

#test
yout = calculate(test_set_x,w1,w2,b1,b2)
print('Test_accuracy:',accuracy(test_set_y,yout))

