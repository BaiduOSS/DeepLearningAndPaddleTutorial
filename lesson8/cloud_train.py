#!/usr/bin/env python
# -*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Authors: fuqiang(fqjeremybuaa@163.com)
Date:    2017/11/19


在paddlePaddle cloud平台上分布式训练推荐模型，关键步骤如下：
1.从RecordIO文件路径创建数据reader读取文件
2.构造用户融合特征模型
3.构造电影融合特征模型
4.定义特征相似性度量inference和成本函数cost
5.定义模型训练器trainer
6.定义事件迭代训练模型
7.保存模型参数
"""

import os
import copy

import paddle.v2 as paddle

# USERNAME是PaddlePaddle Cloud平台登陆的用户名，直接替换相应字段即可
USERNAME = "paddle@example.com"

# 获取PaddlePaddle Cloud当前数据中心的环境变量值
DC = os.getenv("PADDLE_CLOUD_CURRENT_DATACENTER")

# 设定在当前数据中心下缓存数据集的路径
DATA_HOME = "/pfs/%s/home/%s" % (DC, USERNAME)
TRAIN_FILES_PATH = os.path.join(DATA_HOME, "movielens/train-*")


# 创建数据reader读取文件
def recordio(paths, buf_size=100):
    """
    从给定的被","分开的RecordIO文件路径创建数据reader
	Args:
		path -- recordio文件的路径
		buf_size -- buf大小设为100
    Return:
		dec.buffered(reader, buf_size) -- recordio文件的数据reader
    """

    import recordio as rec
    import paddle.v2.reader.decorator as dec
    import cPickle as pickle

	# 文件读取模块
    def reader():
		"""
		定义一个reader
		Args:
		Return:
		"""
        f = rec.reader(paths)
        while True:
            r = f.read()
            if r is None:
                break
            yield pickle.loads(r)
        f.close()

    return dec.buffered(reader, buf_size)

# 构造用户融合特征模型
def get_usr_combined_features():
    """
    构造用户融合特征模型，融合特征包括：
		user_id：用户编号
		gender_id：性别类别编号
		age_id：年龄分类编号
		job_id：职业类别编号
	以上特征信息从数据集中读取后分别变换成对应词向量，再输入到全连接层
    所有的用户特征再输入到一个全连接层中，将所有特征融合为一个200维的特征    
    Args:
    Return:
		usr_combined_features -- 用户融合特征模型
    """
	# 读取用户编号信息（user_id）
    uid = paddle.layer.data(
        name='user_id',
        type=paddle.data_type.integer_value(
            paddle.dataset.movielens.max_user_id() + 1))
    
	# 将用户编号变换为对应词向量
    usr_emb = paddle.layer.embedding(input=uid, size=32)
	
	# 将用户编号对应词向量输入到全连接层
    usr_fc = paddle.layer.fc(input=usr_emb, size=32)

	# 读取用户性别类别编号信息（gender_id）并做处理（同上）
    usr_gender_id = paddle.layer.data(
        name='gender_id', type=paddle.data_type.integer_value(2))
    usr_gender_emb = paddle.layer.embedding(input=usr_gender_id, size=16)
    usr_gender_fc = paddle.layer.fc(input=usr_gender_emb, size=16)

	# 读取用户年龄类别编号信息（age_id）并做处理（同上）
    usr_age_id = paddle.layer.data(
        name='age_id',
        type=paddle.data_type.integer_value(
            len(paddle.dataset.movielens.age_table)))
    usr_age_emb = paddle.layer.embedding(input=usr_age_id, size=16)
    usr_age_fc = paddle.layer.fc(input=usr_age_emb, size=16)

	# 读取用户职业类别编号信息（job_id）并做处理（同上）
    usr_job_id = paddle.layer.data(
        name='job_id',
        type=paddle.data_type.integer_value(
            paddle.dataset.movielens.max_job_id() + 1))
    usr_job_emb = paddle.layer.embedding(input=usr_job_id, size=16)
    usr_job_fc = paddle.layer.fc(input=usr_job_emb, size=16)

	# 所有的用户特征再输入到一个全连接层中，完成特征融合
    usr_combined_features = paddle.layer.fc(
        input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc],
        size=200,
        act=paddle.activation.Tanh())
    return usr_combined_features


# 构造电影融合特征模型
def get_mov_combined_features():
    """
    构造电影融合特征模型，融合特征包括：
		movie_id：电影编号
		category_id：电影类别编号
		movie_title：电影名
	以上特征信息经过相应处理后再输入到一个全连接层中，
	将所有特征融合为一个200维的特征    
    Args:
    Return:
		mov_combined_features -- 电影融合特征模型
    """

    movie_title_dict = paddle.dataset.movielens.get_movie_title_dict()
	
    # 读取电影编号信息（movie_id）
    mov_id = paddle.layer.data(
        name='movie_id',
        type=paddle.data_type.integer_value(
            paddle.dataset.movielens.max_movie_id() + 1))
			
	# 将电影编号变换为对应词向量
    mov_emb = paddle.layer.embedding(input=mov_id, size=32)
	
	# 将电影编号对应词向量输入到全连接层
    mov_fc = paddle.layer.fc(input=mov_emb, size=32)
	
	# 读取电影类别编号信息（category_id）
    mov_categories = paddle.layer.data(
        name='category_id',
        type=paddle.data_type.sparse_binary_vector(
            len(paddle.dataset.movielens.movie_categories())))
			
	# 将电影编号信息输入到全连接层
    mov_categories_hidden = paddle.layer.fc(input=mov_categories, size=32)

    # 读取电影名信息（movie_title）
    mov_title_id = paddle.layer.data(
        name='movie_title',
        type=paddle.data_type.integer_value_sequence(len(movie_title_dict)))
		
	# 将电影名变换为对应词向量
    mov_title_emb = paddle.layer.embedding(input=mov_title_id, size=32)
	
	# 将电影名对应词向量输入到卷积网络生成电影名时序特征
    mov_title_conv = paddle.networks.sequence_conv_pool(
        input=mov_title_emb, hidden_size=32, context_len=3)

	# 所有的电影特征再输入到一个全连接层中，完成特征融合
    mov_combined_features = paddle.layer.fc(
        input=[mov_fc, mov_categories_hidden, mov_title_conv],
        size=200,
        act=paddle.activation.Tanh())
		
    return mov_combined_features


def main():
    """
    定义神经网络结构，训练网络    
    Args:
    Return:
    """
	# 初始化
    paddle.init()
	
	# 构造用户融合特征，电影融合特征
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()
	
	# 计算用户融合特征和电影融合特征的余弦相似度
    inference = paddle.layer.cos_sim(
        a=usr_combined_features, b=mov_combined_features, size=1, scale=5)
		
	# 定义成本函数为均方误差函数
    cost = paddle.layer.square_cost(
        input=inference,
        label=paddle.layer.data(
            name='score', type=paddle.data_type.dense_vector(1)))

	# 利用cost创建parameters
    parameters = paddle.parameters.create(cost)
    """
    	定义模型训练器，配置三个参数
    	cost:成本函数
    	parameters:参数
    	update_equation:更新公式（模型采用Adam方法优化更新，并初始化学习率）
    	"""

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=paddle.optimizer.Adam(learning_rate=1e-4))
		
	# 数据层和数组索引映射，用于trainer训练时读取数据
    feeding = {
        'user_id': 0,
        'gender_id': 1,
        'age_id': 2,
        'job_id': 3,
        'movie_id': 4,
        'category_id': 5,
        'movie_title': 6,
        'score': 7
    }

	# 事件处理模块
    def event_handler(event):
        """
        事件处理器，可以根据训练过程的信息作相应操作
        Args:
            event -- 事件对象，包含event.pass_id, event.batch_id, event.cost等信息
        Return:
        """
        if isinstance(event, paddle.event.EndIteration):
			# 每100个batch输出一条记录，分别是当前的迭代次数编号，batch编号和对应损失值
            if event.batch_id % 100 == 0:
                print "Pass %d Batch %d Cost %.2f" % (
                    event.pass_id, event.batch_id, event.cost)

        """
        模型训练
        paddle.batch(reader(), batch_size=256)：表示从打乱的数据中再取出batch_size=256大小的数据进行一次迭代训练
        paddle.reader.shuffle(train(), buf_size=8192)：表示trainer从recordio(TRAIN_FILES_PATH)这个reader中读取了buf_size=8192
        大小的数据并打乱顺序
        event_handler：事件管理机制，可以自定义event_handler，根据事件信息作相应的操作
        feeding：用到了之前定义的feeding索引，将数据层信息输入trainer
        num_passes：定义训练的迭代次数
        """

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(recordio(TRAIN_FILES_PATH), buf_size=8192),
            batch_size=256),
        event_handler=event_handler,
        feeding=feeding,
        num_passes=1)

	# 定义用户编号值和电影编号值
    user_id = 234
    movie_id = 345

	# 根据已定义的用户、电影编号值从movielens数据集中读取数据信息
    user = paddle.dataset.movielens.user_info()[user_id]
    movie = paddle.dataset.movielens.movie_info()[movie_id]

	# 存储用户特征和电影特征
    feature = user.value() + movie.value()

	# 复制feeding值，并删除序列中的得分项
    infer_dict = copy.copy(feeding)
    del infer_dict['score']

	# 预测指定用户对指定电影的喜好得分值
    prediction = paddle.infer(
        output_layer=inference,
        parameters=parameters,
        input=[feature],
        feeding=infer_dict)
    score = (prediction[0][0] + 5.0) / 2
    print "[Predict] User %d Rating Movie %d With Score %.2f" % (user_id, movie_id, score)


if __name__ == '__main__':
    main()
