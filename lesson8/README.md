# Lesson8: 个性化推荐在PaddlePaddle Cloud上的实现（Recommend System with PaddlePaddle Cloud） 
本实验内容主要为在PaddlePaddle Cloud平台上实现特征融合推荐模型，进而实现分布式个性化电影推荐系统。

>**注意**: 实验假设读者已经对Numpy以及PaddlePaddle知识有基本的了解，并且掌握了深度学习的基本概念、对PaddlePaddle Cloud有基本的了解。

数据集：实验采用MovieLens-1M数据集

文件清单：

  * [prepare_data.py](prepare_data.py)：在PaddlePaddle Cloud平台上完成训练数据的预处理

  * [train_with_cloud.py](train_with_cloud.py)：在PaddlePaddle Cloud平台上分布式训练推荐模型
