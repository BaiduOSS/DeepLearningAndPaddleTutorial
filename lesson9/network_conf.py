#!/usr/bin/env python2.7
# -*- encoding:utf-8 -*-
"""
����������ýű�
Created on 2017-12-2
@author: PaddlePaddle CTR Model
@copyright: www.baidu.com
"""
import paddle.v2 as paddle
from paddle.v2 import layer
from paddle.v2 import data_type as dtype
from utils import logger, ModelType


class CTRmodel(object):
    '''
    ���� wide && deep learning ģ�ͽ�����CTRģ��
    '''

    def __init__(self,
                 dnn_layer_dims,
                 dnn_input_dim,
                 lr_input_dim,
                 model_type=ModelType.create_classification(),
                 is_infer=False):
        '''
        @dnn_layer_dims: list of integer
            DNNÿһ���ά��
        @dnn_input_dim: int
            DNN�����Ĵ�С
        @lr_input_dim: int
            LR������С
        @is_infer: bool
            �Ƿ���Ԥ��ģ��
        '''
        self.dnn_layer_dims = dnn_layer_dims
        self.dnn_input_dim = dnn_input_dim
        self.lr_input_dim = lr_input_dim
        self.model_type = model_type
        self.is_infer = is_infer

        self._declare_input_layers()

        self.dnn = self._build_dnn_submodel_(self.dnn_layer_dims)
        self.lr = self._build_lr_submodel_()

        # ģ��Ԥ��
        if self.model_type.is_classification():
            self.model = self._build_classification_model(self.dnn, self.lr)
        if self.model_type.is_regression():
            self.model = self._build_regression_model(self.dnn, self.lr)

    def _declare_input_layers(self):
        self.dnn_merged_input = layer.data(
            name='dnn_input',
            type=paddle.data_type.sparse_binary_vector(self.dnn_input_dim))

        self.lr_merged_input = layer.data(
            name='lr_input',
            type=paddle.data_type.sparse_float_vector(self.lr_input_dim))

        if not self.is_infer:
            self.click = paddle.layer.data(
                name='click', type=dtype.dense_vector(1))

    def _build_dnn_submodel_(self, dnn_layer_dims):
        '''
        ����DNN��ģ��
        '''
        dnn_embedding = layer.fc(
            input=self.dnn_merged_input, size=dnn_layer_dims[0])
        _input_layer = dnn_embedding
        for i, dim in enumerate(dnn_layer_dims[1:]):
            fc = layer.fc(
                input=_input_layer,
                size=dim,
                act=paddle.activation.Relu(),
                name='dnn-fc-%d' % i)
            _input_layer = fc
        return _input_layer

    def _build_lr_submodel_(self):
        '''
        ����LR��ģ��
        '''
        fc = layer.fc(
            input=self.lr_merged_input, size=1, act=paddle.activation.Relu())
        return fc

    def _build_classification_model(self, dnn, lr):
        merge_layer = layer.concat(input=[dnn, lr])
        self.output = layer.fc(
            input=merge_layer,
            size=1,
            # ����sigmoid����Ԥ��CTR����
            act=paddle.activation.Sigmoid())

        if not self.is_infer:
            self.train_cost = paddle.layer.multi_binary_label_cross_entropy_cost(
                input=self.output, label=self.click)
        return self.output

    def _build_regression_model(self, dnn, lr):
        merge_layer = layer.concat(input=[dnn, lr])
        self.output = layer.fc(
            input=merge_layer, size=1, act=paddle.activation.Sigmoid())
        if not self.is_infer:
            self.train_cost = paddle.layer.square_error_cost(
                input=self.output, label=self.click)
        return self.output