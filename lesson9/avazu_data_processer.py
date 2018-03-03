#!/usr/bin/env python2.7
# -*- encoding:utf-8 -*-
"""
Avazu数据集预处理脚本
Created on 2017-12-2

"""
import sys
import csv
import cPickle
import argparse
import os
import numpy as np

from utils import logger, TaskMode

parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")
parser.add_argument(
    '--data_path', type=str, required=True, help="path of the Avazu dataset")
parser.add_argument(
    '--output_dir', type=str, required=True, help="directory to output")
parser.add_argument(
    '--num_lines_to_detect',
    type=int,
    default=500000,
    help="number of records to detect dataset's meta info")
parser.add_argument(
    '--test_set_size',
    type=int,
    default=10000,
    help="size of the validation dataset(default: 10000)")
parser.add_argument(
    '--train_size',
    type=int,
    default=100000,
    help="size of the trainset (default: 100000)")
args = parser.parse_args()
'''
数据集内容如下:
    0. id: ad identifier
    1. click: 0/1 for non-click/click
    2. hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
    3. C1 -- anonymized categorical variable
    4. banner_pos
    5. site_id
    6. site_domain
    7. site_category
    8. app_id
    9. app_domain
    10. app_category
    11. device_id
    12. device_ip
    13. device_model
    14. device_type
    15. device_conn_type
    16. C14-C21 -- anonymized categorical variables
我们将利用下列域作为分类特征:
    - C1
    - banner_pos
    - site_category
    - app_category
    - device_type
    - device_conn_type
一些特征将作为ID特征:
    - id
    - site_id
    - app_id
    - device_id
`hour` 将被作为连续特征，并被转化为24维度的one-hot特征.
脚本输出以下三个文件:
1. train.txt
2. test.txt
3. infer.txt
'''

feature_dims = {}

categorial_features = ('C1 banner_pos site_category app_category ' +
                       'device_type device_conn_type').split()

id_features = 'id site_id app_id device_id _device_id_cross_site_id'.split()


def get_all_field_names(mode=0):
    '''
    @mode: int
        0 为训练模式，1为测试模式
    @return: list of str
    '''
    return categorial_features + ['hour'] + id_features + ['click'] \
        if mode == 0 else []


class CategoryFeatureGenerator(object):
    '''
    生成类别特征
    '''

    def __init__(self):
        self.dic = {'unk': 0}
        self.counter = 1

    def register(self, key):
        '''
        注册记录
        '''
        if key not in self.dic:
            self.dic[key] = self.counter
            self.counter += 1

    def size(self):
        return len(self.dic)

    def gen(self, key):
        '''
        为每一条记录转化为one-hot特征
        '''
        if key not in self.dic:
            res = self.dic['unk']
        else:
            res = self.dic[key]
        return [res]

    def __repr__(self):
        return '<CategoryFeatureGenerator %d>' % len(self.dic)


class IDfeatureGenerator(object):
    def __init__(self, max_dim, cross_fea0=None, cross_fea1=None):
        '''
        @max_dim: int
            id元素空间的维度
        '''
        self.max_dim = max_dim
        self.cross_fea0 = cross_fea0
        self.cross_fea1 = cross_fea1

    def gen(self, key):
        '''
       为每条记录生成one-hot特征
        '''
        return [hash(key) % self.max_dim]

    def gen_cross_fea(self, fea1, fea2):
        key = str(fea1) + str(fea2)
        return self.gen(key)

    def size(self):
        return self.max_dim


class ContinuousFeatureGenerator(object):
    def __init__(self, n_intervals):
        self.min = sys.maxint
        self.max = sys.minint
        self.n_intervals = n_intervals

    def register(self, val):
        self.min = min(self.minint, val)
        self.max = max(self.maxint, val)

    def gen(self, val):
        self.len_part = (self.max - self.min) / self.n_intervals
        return (val - self.min) / self.len_part


# 初始化所有特征类
fields = {}
for key in categorial_features:
    fields[key] = CategoryFeatureGenerator()
for key in id_features:
    # 针对交叉特征
    if 'cross' in key:
        feas = key[1:].split('_cross_')
        fields[key] = IDfeatureGenerator(10000000, *feas)
    # 对普通的ID特征
    else:
        fields[key] = IDfeatureGenerator(10000)

# 作为PaddlePaddle的feed_index使用
field_index = dict((key, id)
                   for id, key in enumerate(['dnn_input', 'lr_input', 'click']))


def detect_dataset(path, topn, id_fea_space=10000):
    '''
   稀疏化大数据集合的前n条记录
    '''
    # 生成分类数据对象
    logger.warning('detecting dataset')

    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_id, row in enumerate(reader):
            if row_id > topn:
                break

            for key in categorial_features:
                fields[key].register(row[key])

    for key, item in fields.items():
        feature_dims[key] = item.size()

    feature_dims['hour'] = 24
    feature_dims['click'] = 1

    feature_dims['dnn_input'] = np.sum(
        feature_dims[key] for key in categorial_features + ['hour']) + 1
    feature_dims['lr_input'] = np.sum(feature_dims[key]
                                      for key in id_features) + 1
    return feature_dims


def load_data_meta(meta_path):
    '''
    载入大数据集合信息
    '''
    feature_dims, fields = cPickle.load(open(meta_path, 'rb'))
    return feature_dims, fields


def concat_sparse_vectors(inputs, dims):
    '''
    连接多个稀疏特征
    @inputs: list
        稀疏向量列表
    @dims: list of int
        每个稀疏向量维度
    '''
    res = []
    assert len(inputs) == len(dims)
    start = 0
    for no, vec in enumerate(inputs):
        for v in vec:
            res.append(v + start)
        start += dims[no]
    return res


class AvazuDataset(object):
    '''
    载入Avazu数据集合作为训练集
    '''

    def __init__(self,
                 train_path,
                 n_records_as_test=-1,
                 fields=None,
                 feature_dims=None):
        self.train_path = train_path
        self.n_records_as_test = n_records_as_test
        self.fields = fields
        # default is train mode.
        self.mode = TaskMode.create_train()

        self.categorial_dims = [
            feature_dims[key] for key in categorial_features + ['hour']
        ]
        self.id_dims = [feature_dims[key] for key in id_features]

    def train(self):
        '''
        载入训练集
        '''
        logger.info("load trainset from %s" % self.train_path)
        self.mode = TaskMode.create_train()
        with open(self.train_path) as f:
            reader = csv.DictReader(f)

            for row_id, row in enumerate(reader):
                # 跳过前n条记录
                if self.n_records_as_test > 0 and row_id < self.n_records_as_test:
                    continue

                rcd = self._parse_record(row)
                if rcd:
                    yield rcd

    def test(self):
        '''
        载入测试集
        '''
        logger.info("load testset from %s" % self.train_path)
        self.mode = TaskMode.create_test()
        with open(self.train_path) as f:
            reader = csv.DictReader(f)

            for row_id, row in enumerate(reader):
                # skip top n lines
                if self.n_records_as_test > 0 and row_id > self.n_records_as_test:
                    break

                rcd = self._parse_record(row)
                if rcd:
                    yield rcd

    def infer(self):
        '''
        载入预测集
        '''
        logger.info("load inferset from %s" % self.train_path)
        self.mode = TaskMode.create_infer()
        with open(self.train_path) as f:
            reader = csv.DictReader(f)

            for row_id, row in enumerate(reader):
                rcd = self._parse_record(row)
                if rcd:
                    yield rcd

    def _parse_record(self, row):
        '''
       稀疏化CSV文件的一列获得记录
        '''
        record = []
        for key in categorial_features:
            record.append(self.fields[key].gen(row[key]))
        record.append([int(row['hour'][-2:])])
        dense_input = concat_sparse_vectors(record, self.categorial_dims)

        record = []
        for key in id_features:
            if 'cross' not in key:
                record.append(self.fields[key].gen(row[key]))
            else:
                fea0 = self.fields[key].cross_fea0
                fea1 = self.fields[key].cross_fea1
                record.append(
                    self.fields[key].gen_cross_fea(row[fea0], row[fea1]))

        sparse_input = concat_sparse_vectors(record, self.id_dims)

        record = [dense_input, sparse_input]

        if not self.mode.is_infer():
            record.append(list((int(row['click']), )))
        return record


def ids2dense(vec, dim):
    return vec


def ids2sparse(vec):
    return ["%d:1" % x for x in vec]


detect_dataset(args.data_path, args.num_lines_to_detect)
dataset = AvazuDataset(
    args.data_path,
    args.test_set_size,
    fields=fields,
    feature_dims=feature_dims)

output_trainset_path = os.path.join(args.output_dir, 'train.txt')
output_testset_path = os.path.join(args.output_dir, 'test.txt')
output_infer_path = os.path.join(args.output_dir, 'infer.txt')
output_meta_path = os.path.join(args.output_dir, 'data.meta.txt')

with open(output_trainset_path, 'w') as f:
    for id, record in enumerate(dataset.train()):
        if id and id % 10000 == 0:
            logger.info("load %d records" % id)
        if id > args.train_size:
            break
        dnn_input, lr_input, click = record
        dnn_input = ids2dense(dnn_input, feature_dims['dnn_input'])
        lr_input = ids2sparse(lr_input)
        line = "%s\t%s\t%d\n" % (' '.join(map(str, dnn_input)),
                                 ' '.join(map(str, lr_input)), click[0])
        f.write(line)
    logger.info('write to %s' % output_trainset_path)

with open(output_testset_path, 'w') as f:
    for id, record in enumerate(dataset.test()):
        dnn_input, lr_input, click = record
        dnn_input = ids2dense(dnn_input, feature_dims['dnn_input'])
        lr_input = ids2sparse(lr_input)
        line = "%s\t%s\t%d\n" % (' '.join(map(str, dnn_input)),
                                 ' '.join(map(str, lr_input)), click[0])
        f.write(line)
    logger.info('write to %s' % output_testset_path)

with open(output_infer_path, 'w') as f:
    for id, record in enumerate(dataset.infer()):
        dnn_input, lr_input = record
        dnn_input = ids2dense(dnn_input, feature_dims['dnn_input'])
        lr_input = ids2sparse(lr_input)
        line = "%s\t%s\n" % (' '.join(map(str, dnn_input)),
                             ' '.join(map(str, lr_input)), )
        f.write(line)
        if id > args.test_set_size:
            break
    logger.info('write to %s' % output_infer_path)

with open(output_meta_path, 'w') as f:
    lines = [
        "dnn_input_dim: %d" % feature_dims['dnn_input'],
        "lr_input_dim: %d" % feature_dims['lr_input']
    ]
    f.write('\n'.join(lines))
    logger.info('write data meta into %s' % output_meta_path)
