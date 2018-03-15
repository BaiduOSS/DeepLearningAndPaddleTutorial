# Lesson3: 逻辑回归（Logistic Regression）

本实验内容主要为分别利用python和paddlepaddle框架来实现Logistic Regression，用于解决识别猫问题。

>**注意**: 实验假设读者已经对Numpy以及Paddlepaddle知识有基本的了解，并且掌握了深度学习的基本概念和Logistic Regression的实现原理。


包含内容

* [datasets](datasets)：数据集，包含两个h5py文件
	* train_images.h5：训练数据集
	* test_images.h5：测试数据集

* [utils.py](utils.py)：工具类，包含load_data()函数，用于载入数据

* [show_image.py](show_image.py): 显示数据集情况

* [train_with_numpy.py](train_with_numpy.py)：Logistic Regression的Python版本实现

* [train_with_paddle.py](train_with_paddle.py)：Logistic Regression的Paddlepaddle版本实现

* [predict_with_paddle.py](predict_with_paddle.py)：对训练完成的模型进行预测和检验，使用Paddlepaddle实现（在教材中将该部分合并至train_with_paddle.py中，但实际上应单独作为文件）

* [costs.png](costs.png)：成本变化曲线图
 
