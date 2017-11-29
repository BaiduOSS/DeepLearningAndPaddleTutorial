# Lesson6: 数字识别（Recognize Digits） 
本实验内容主要为利用PaddlePaddle框架来实现神经网络分类器，用于解决手写数字识别问题。

>**注意**: 实验假设读者已经对Numpy以及PaddlePaddle知识有基本的了解，并且掌握了深度学习的基本概念、对卷积神经网络有基本的了解。

数据集：实验采用MNIST数据集

文件清单：

  * [image](image)：用于存放测试图片的文件夹
    * image/infer_3.png：测试图片——用训练得到的模型预测其类别。该图片的真实标签为‘3’

  * [train_with_paddle.py](train_with_paddle.py)：解决数字识别问题的PaddlePaddle代码实现
  
  * [predict_with_paddle.py](predict_with_paddle.py)：根据训练好的模型预测手写数字体类别的PaddlePaddle代码实现

  * [train_test_cost.png](train_test_cost.png)：训练过程中成本变化曲线图

  * [train_test_error_rate.png](train_test_error_rate.png):训练过程中错误率变化曲线

