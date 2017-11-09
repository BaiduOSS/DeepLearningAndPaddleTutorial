# -*- coding:utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import h5py
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#初始化某个参数
def parameters(row,col,b):
    #p = np.zeros((row,col),dtype=float) + b
    p = np.random.randn(row,col) * b
    return np.mat(p)

#初始化全部参数
def all_the_para(i,j,W,B,x,y):
    pw = 0.5
    pb = 0
    for k in range(i):
        if ((k == 0) & (k != i - 1)):
            w = parameters(j, x.shape[0], pw)
            W.append(w)
            b = parameters(j, 1, pb)
            B.append(b)
        if ((k > 0) & (k < i - 1)):
            w = parameters(j, j, pw)
            W.append(w)
            b = parameters(j, 1, pb)
            B.append(b)
        if (k == i - 1):
            w = parameters(y.shape[0], j, pw)
            W.append(w)
            b = parameters(y.shape[0], 1, pb)
            B.append(b)

def z_out(x,w,b):
    zz = w * x + b
    return zz


def a_out(z):
    aa = np.maximum(0,z)
    return aa

def y_out(z):
    yy = 1 / (1 + np.exp(-z))
    return yy

def calculate(A,W,B,Z,i):
    for k in range(i):
        z = z_out(A[k], W[k], B[k])
        Z.append(z)
        if(k < i-1):
            a = a_out(z)
            A.append(a)
        if(k == i-1):
            yout = y_out(z)
            A.append(yout)


def dy_out(y,yout):
    dy = yout - y
    return dy


def dz_out(dy, yout):
    dz = 1 * (yout > 0)
    dz = np.multiply(dy,dz)
    return dz


def dzy_out(dy, yout):
    dzy = np.multiply(yout,1-yout)
    dzy = np.multiply(dy,dzy)
    return dzy

def da_out(dz,w):
    da = w.T * dz
    return da


def db_out(dz,n):
    db = np.sum(dz,axis=1) / n
    return db


def dw_out(dz,a,n):
    dw = (dz * a.T) / n
    return dw

def update(p,dp,u):
    p_update = p - (u * dp)
    return p_update


def bp(i, dA, A, dZ, W, B, u):
    for k in range(i):
        if(k == 0): #若k为0，此时指向输出层，则计算输出层的dz
            dz = dzy_out(dA[k], A[i-k])
            dZ.append(dz)
        if(k > 0 ): #若k大于0，此时指向隐藏层，则计算隐藏层的dz
            dz = dz_out(dA[k], A[i-k])
            dZ.append(dz)
        if( k != i - 1): #若k不等于i-1,即此时仍指向隐藏层或输出层，计算da
            da = da_out(dZ[k], W[i - k - 1])
            dA.append(da)
        #梯度下降
        dw = dw_out(dZ[k], A[i - k - 1], n)
        #dW.append(dw)
        db = db_out(dZ[k], n)
        #dB.append(db)

        # 更新参数
        w_up = update(W[i - k - 1], dw, u)
        W[i - k - 1] = w_up
        b_up = update(B[i - k - 1], db, u)
        B[i - k - 1] = b_up


def accuracy(y,yout):
    ac = 1 - np.mean(np.abs(y-(1*(yout>0.5))))
    return ac

# 下载数据集(cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(m_train,-1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test,-1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

W = []
B = []
n = 209 #有209个数据
u = 0.89 #学习率
i = 5 #有5层
j = 7 #隐藏层有7个节点
all_the_para(i, j, W, B, train_set_x, train_set_y) #初始化参数



for item in range(2000):
    A = [] #存放输入X和隐藏层节点值，以及输出值
    Z = [] #存放加权偏移后的中间量
    A.append(train_set_x) #载入输入值X
    calculate(A, W, B, Z, i) #正向计算
    dA = [] #存放隐藏层节点导数
    dZ = [] #存放中间量导数
    dA.append(dy_out(train_set_y, A[i])) #存放dy
    bp(i, dA, A, dZ, W, B, u) #更新参数w,b
a = accuracy(train_set_y, A[i]) #计算训练数据准确率
train_accuracy = []
train_accuracy.append(a)

#计算测试数据准确率
A = []
A.append(test_set_x)
calculate(A, W, B, Z, i)
a = accuracy(test_set_y, A[i])
test_accuracy = []
test_accuracy.append(a)

print('train accuracy:', train_accuracy)
print('test accuracy:', test_accuracy)


