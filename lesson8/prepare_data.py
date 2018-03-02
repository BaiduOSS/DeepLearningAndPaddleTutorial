#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Authors: fuqiang(fqjeremybuaa@163.com)
Date:    2017/11/29

在paddlePaddle cloud平台上完成训练数据的预处理，关键步骤如下：
1.获取文件路径和训练器参数
2.根据训练文件路径对movielens数据集进行拆分
"""

import os

import paddle.v2.dataset as dataset

# USERNAME是PaddlePaddle Cloud平台登陆的用户名，直接替换相应字段即可,paddle@example.com
USERNAME = "xxx@example.com"

# 获取PaddlePaddle Cloud当前数据中心的环境变量值,PADDLE_CLOUD_CURRENT_DATACENTER
DC = os.getenv("PADDLE_CLOUD_CURRENT_DATACENTER")

# 设定在当前数据中心下缓存数据集的路径
dataset.common.DATA_HOME = "/pfs/%s/home/%s" % (DC, USERNAME)
TRAIN_FILES_PATH = os.path.join(dataset.common.DATA_HOME, "movielens")

# 获取训练器的相关参数
TRAINER_ID = int(os.getenv("PADDLE_INIT_TRAINER_ID"))
TRAINER_INSTANCES = int(os.getenv("PADDLE_INIT_NUM_GRADIENT_SERVERS"))


def main():
    """
    根据训练文件路径对movielens数据集进行拆分，并输出操作日志
    Args:
    Return:
    """
    # 判断训练是否在PaddlePaddle Cloud上执行
    if TRAINER_ID == -1 or TRAINER_INSTANCES == -1:
        print "no cloud environ found, must run on cloud"
        exit(1)

    print "\nBegin to convert data into " + dataset.common.DATA_HOME

    # 拆分数据
    dataset.common.convert(TRAIN_FILES_PATH,
                           dataset.movielens.train(), 1000, "train")
    print "\nConvert process is finished"
    print "\nPlease run 'paddlecloud file ls " + dataset.common.DATA_HOME \
          + "/movielens' to check if datas exist there"


if __name__ == '__main__':
    main()
