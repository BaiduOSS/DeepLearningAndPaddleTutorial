# Lesson9: CTR预估

本实验代码与教材第九章配套，为介绍CTR预估在PaddlePaddle的实现
> 注意： 代码中相关知识的详细说明参考教材第九章的相关描述

包含内容

* [avazu_data_processer.py](./avazu_data_processer.py)： 对AVAZU数据集进行预处理
* [infer.py](./infer.py)： 用训练好的CTR模型进行预测
* [network_conf.py](./network_conf.py)： 建立网络参数的设置
* [reader.py](./reader.py)： 读取数据集脚本
* [train.py](./train.py): 实现CTR模型的训练
* [utils.py](./utils.py)： 在模型训练和预测过程中打印相关信息
