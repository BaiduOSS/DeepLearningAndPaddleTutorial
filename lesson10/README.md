# Lesson10: 算法优化（Optimization） 
本实验内容主要为以第六章手写数字识别实验为基础，调整部分代码以说明算法优化的作用。对调整的代码以注释的形式进行了说明。

>**注意**: 实验假设读者已经对Numpy以及PaddlePaddle知识有基本的了解，并且掌握了深度学习的基本概念、对卷积神经网络有基本的了解。

数据集：实验采用MNIST数据集

文件清单：

  * [image](image)：用于存放测试图片的文件夹
    * image/infer_3.png：测试图片——用训练得到的模型预测其类别。该图片的真实标签为‘3’
  
  * [train_with_paddle_default.py](train_with_paddle_default.py)：与第六章代码相比，差异在于学习率和动量设置的值不同，目的在于降低模型的总体表现。其他代码文件将在设置的基础上，加入其他模块（例如drop out），从而可以通过模型性能的提升来验证该模块的作用。同时，读者也可以在此文件基础上单独修改学习率、动量值、优化器类型等，以验证以上参数的影响
  
  * [train_with_paddle_dropout.py](train_with_paddle_dropout.py)：与train_with_paddle_default.py相比，差异在于在卷积神经网络分类器中加入了drop out设置
  
  * [train_with_paddle_bn.py](train_with_paddle_bn.py)：与train_with_paddle_default.py相比，差异在于在卷积神经网络分类器中加入了batch normalization设置

