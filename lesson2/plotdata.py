#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_data(data,a,b):
   x = data[:,0]
   y = data[:,1]
   y_predict = x*a + b;
   plt.scatter(x,y,marker='.',c='r',label='True')
   plt.title('House Price Distributions of Beijing Beiyuan Area in 2016/12')
   plt.xlabel('House Area ')
   plt.ylabel('House Price ')
   plt.xlim(0,250)
   plt.ylim(0,2500)
   predict = plt.plot(x,y_predict,label='Predict')
   plt.legend(loc='upper left')
   plt.savefig('result.png')
   plt.show()

data = np.loadtxt('data.txt', delimiter=',')
X_RAW = data.T[0].copy()
plot_data(data,7.1,-61.1)
