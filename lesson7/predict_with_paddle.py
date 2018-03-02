#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Authors: fuqiang(fqjeremybuaa@163.com)
Date:    2017/11/29

使用paddle框架实现个性化电影推荐系统的结果预测，
无需重新训练模型，只需加载模型文件。关键步骤如下：
1.初始化
2.配置网络结构
  - 构造用户融合特征模型
  - 构造电影融合特征模型
  - 定义特征相似性度量inference
  - 定义feeding
3.从parameters文件直接获取模型参数
4.根据模型参数和测试数据来预测结果
"""

import copy
import os

import paddle.v2 as paddle

PARAMETERS = None


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


# 配置网络结构
def netconfig():
    """
    配置网络结构
    Args:
    Return:
        inference -- 相似度
        feeding -- 数据映射，python字典
    """

    # 构造用户融合特征，电影融合特征
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()

    # 计算用户融合特征和电影融合特征的余弦相似度
    inference = paddle.layer.cos_sim(
        a=usr_combined_features, b=mov_combined_features, size=1, scale=5)

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

    data = [inference, feeding]

    return data


def main():
    """
    读取模型参数并预测结果
    Args:
    Return:
    """
    global PARAMETERS
    paddle.init(use_gpu=False)

    # 配置网络结构
    inference, feeding = netconfig()

    # 判断参数文件是否存在
    if not os.path.exists('params_pass_9.tar'):
        print "Params file doesn't exists."
        return

    # 从文件中读取参数
    with open('params_pass_9.tar', 'r') as param_f:
        PARAMETERS = paddle.parameters.Parameters.from_tar(param_f)

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
        parameters=PARAMETERS,
        input=[feature],
        feeding=infer_dict)
    score = (prediction[0][0] + 5.0) / 2
    print "[Predict] User %d Rating Movie %d With Score %.2f" % (
        user_id, movie_id, score)


if __name__ == '__main__':
    main()
