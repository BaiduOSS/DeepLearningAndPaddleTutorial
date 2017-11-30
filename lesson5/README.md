# Lesson5: Deep Neural Network(DNN)

本实验内容主要为分别利用python和paddlepaddle框架来实现Deep Neural Network，用于解决识别猫问题。

>**注意**: 实验假设读者已经对Numpy以及Paddlepaddle知识有基本的了解，并且掌握了深度学习的基本概念和深层神经网络的实现原理。


包含内容

* [datasets](datasets)：数据集，包含两个h5py文件
	* train_catvnoncat.h5：训练数据集
	* test_catvnoncat.h5：测试数据集

* [dnn_utils.py](lr_utils.py)：工具类，包含load_data()函数，用于载入数据

* [dnn_app_utils_v2.py](dnn_app_utils_v2.py):工具类，包含DNN的Python版本实现需要的函数

* [train_with_numpy.py](train_with_numpy.py)：DNN的Python版本实现

* [train_with_paddle.py](train_with_paddle.py)：DNN的Paddlepaddle版本实现

* [costs.png](costs.png)：成本变化曲线图
