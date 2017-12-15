# Lesson10: 算法优化（Optimization） 
本实验内容分为两部分:
* 第一部分主要为以第六章手写数字识别实验为基础，调整部分代码以说明算法优化的作用。对调整的代码以注释的形式进行了说明。
* 第二部分以预测两类数据的分界线为问题，通过对比加入正则化操作前后的实验结果，来说明正则化操作对模型训练的作用。

>**注意**: 实验假设读者已经对Numpy以及PaddlePaddle知识有基本的了解，并且掌握了深度学习的基本概念、对卷积神经网络有基本的了解。

第一部分
---
数据集：实验采用MNIST数据集

文件清单：

  * [image](image)：用于存放测试图片的文件夹
    * image/infer_3.png：测试图片——用训练得到的模型预测其类别。该图片的真实标签为‘3’
  
  * [train_with_paddle_default.py](train_with_paddle_default.py)：与第六章代码相比，差异在于学习率和动量设置的值不同，目的在于降低模型的总体表现。其他代码文件将在设置的基础上，加入其他模块（例如drop out），从而可以通过模型性能的提升来验证该模块的作用。同时，读者也可以在此文件基础上单独修改学习率、动量值、优化器类型等，以验证以上参数的影响
  
  * [train_with_paddle_dropout.py](train_with_paddle_dropout.py)：与train_with_paddle_default.py相比，差异在于在卷积神经网络分类器中加入了drop out设置

  * [train_with_paddle_bn.py](train_with_paddle_bn.py)：与train_with_paddle_default.py相比，差异在于在卷积神经网络分类器中加入了batch normalization设置

第二部分
---
文件清单：

  * [datasets](regularization/datasets)：用于存放数据集
    * datasets/data.mat：数据集

  * [regularization_with_numpy.py](regularization/regularization_with_numpy.py)：对比3种模型（基础模型、加入L2正则化的模型、加入dropout的模型）的表现，体现正则化操作的作用

  * [reg_utils.py](regularization/reg_utils.py)：一些神经网络基本函数，如正向转播、反向传播等

  * [data.png](regularization/data.png)：输入数据的散点图

  * [cost_with_dropout.png](regularization/cost_with_dropout.png)：使用dropout的模型的cost曲线

  * [cost_with_L2-regularization.png](regularization/cost_with_L2-regularization.png)：使用L2正则化操作的模型的cost曲线

  * [cost_without_regularization.png](regularization/cost_without_regularization.png)：不使用正则化操作的基础模型的cost曲线

  * [decision_boundary_with_dropout.png](regularization/decision_boundary_with_dropout.png)：使用dropout的模型的分界线预测结果

  * [decision_boundary_with_L2-regularization.png](regularization/decision_boundary_with_L2-regularization.png)：使用L2正则化操作的模型的分界线预测结果

  * [decision_boundary_without_regularization.png](regularization/decision_boundary_without_regularization.png)：不使用正则化操作的基础模型的分界线预测结果



