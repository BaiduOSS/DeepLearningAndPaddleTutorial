# lesson1:Logistic Regression

***

本实验内容主要为分别利用python和paddlepaddle框架来实现Logistic Regression，用于解决识别猫问题。

>**注意**: 实验假设读者已经对Numpy以及Paddlepaddle知识有基本的了解，并且掌握了深度学习的基本概念和Logistic Regression的实现原理。


包含内容

* [datasets](datasets)：数据集，包含train_catvnoncat.h5和test_catvnoncat.h5两个h5py文件，分别为训练数据和测试数据来源

* [lr_util.py](lr_util.py)：工具类，包含load_data()函数，用于载入数据

* [train.py](train.py)：Logistic Regression的Python版本实现

* [train_with_paddle.py](train_with_paddle.py)：Logistic Regression的Paddlepaddle版本实现

* [predict.py](predict.py)：对训练完成的模型进行预测和检验，使用Paddlepaddle实现

* [costs.png](costs.png)：成本变化曲线图
 


