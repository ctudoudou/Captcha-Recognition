#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/23 下午10:10
# @Author  : tudoudou
# @File    : model.py
# @Software: PyCharm


import keras
from keras import layers
from keras.models import model_from_json, load_model, Model
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten
from keras.layers import Input, MaxPooling2D, AveragePooling2D, Dense
from keras.layers import Dropout, UpSampling2D
import numpy as np


def conv_block(input_tensor, bn_axis, filters, phase, name, strides=(1, 1)):
    """
    Conv2D 塊，雙路雙卷積計算
    :param input_tensor:(tensor) 輸入張量
    :param filters:(tuple) 卷積核打包
    :param strides:(int) 卷積步長
    :param BN_axis:(int) 規範化卷積軸
    :return: model
    """
    filters1, filters2, filters3 = filters  # 解包卷積核數量
    Conv_base_name = 'Conv_' + name + '_' + str(phase) + '_phase_'
    BN_base_name = 'BN_' + name + '_' + str(phase) + '_phase_'
    x = Conv2D(
        filters=filters1, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2a'
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2a')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(
        filters=filters2, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2b'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2b')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(
        filters=filters3, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2c'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2c')(x)
    x = Activation(activation='relu')(x)

    y = Conv2D(filters3, (1, 1), strides=strides, name=Conv_base_name + '1a')(input_tensor)
    y = BatchNormalization(axis=bn_axis, name=BN_base_name + '1b')(y)

    x = layers.add([x, y])
    a = Activation('relu')(x)

    return a


def identity_block(input_tensor, bn_axis, filters, phase, name, strides=(1, 1)):
    """
    Conv2D 塊，雙路單卷積計算
    :param input_tensor:(tensor) 輸入張量
    :param filters:(tuple) 卷積核打包
    :param strides:(int) 卷積步長
    :param BN_axis:(int) 規範化卷積軸
    :return: model
    """
    filters1, filters2, filters3 = filters  # 解包卷積核數量
    Conv_base_name = 'Conv_' + name + '_' + str(phase) + '_phase_'
    BN_base_name = 'BN_' + name + '_' + str(phase) + '_phase_'
    x = Conv2D(
        filters=filters1, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2a'
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2a')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(
        filters=filters2, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2b'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2b')(x)
    x = Activation(activation='relu')(x)

    x = Conv2D(
        filters=filters3, kernel_size=(1, 1), strides=strides, name=Conv_base_name + '2c'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=BN_base_name + '2c')(x)
    x = Activation(activation='relu')(x)

    x = layers.add([x, input_tensor])
    a = Activation('relu')(x)

    return a


def my_resnet():
    inputs = Input(shape=(1, 80, 85))

    x = Conv2D(
        filters=4, kernel_size=(2, 4), padding='same', name='Conv1', data_format='channels_first')(inputs)
    x = BatchNormalization(axis=1, name='BN_Conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format='channels_first')(x)

    x = conv_block(input_tensor=x, bn_axis=1, filters=(4, 4, 64), phase=2, name='a')
    x = identity_block(input_tensor=x, bn_axis=1, filters=(4, 4, 64), phase=2, name='b')
    x = identity_block(input_tensor=x, bn_axis=1, filters=(4, 4, 64), phase=2, name='c')

    # x = conv_block(input_tensor=x, bn_axis=1, filters=(8, 8, 512), phase=3, name='a')
    # x = identity_block(input_tensor=x, bn_axis=1, filters=(8, 8, 512), phase=3, name='b')
    # x = identity_block(input_tensor=x, bn_axis=1, filters=(8, 8, 512), phase=3, name='c')

    x = AveragePooling2D((2, 2), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    # x = [Dense(11, activation='softmax', name='sotfmax11_%d' % (i + 1))(x) for i in range(4)]
    x = Dense(1130, activation='softmax', name='sotfmax1130')(x)
    #     x = GlobalMaxPooling2D()(x)

    model = Model(inputs, x, name='My_Resnet')
    return model


def create_model():
    """返回一個已創建好的 resnet model"""
    model = my_resnet()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def tr_model(modelname, X, y, eval_X, eval_y, batch_size, epochs):
    model = load_model(modelname)
    model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True)
    el = model.evaluate(eval_X, eval_y, batch_size=64)
    print(el)
    model.save(modelname)


def use_model(modelname, X, y, ):
    model = load_model(modelname)
    temp = model.predict(X)
    result = arr2list(temp)
    y = arr2list(y)
    num = 0
    for i in range(len(result)):
        if result[i] == y[i]:
            num += 1
        else:
            # print(result[i])
            pass

    print(num / len(result))


def arr2list(arr):
    result = []
    for i in range(len(arr[0])):
        te = ''
        for j in range(4):
            if np.argmax(arr[j][i]) == 10:
                pass
            else:
                te += str(np.argmax(arr[j][i]))
        result.append(te)

    return result


def predict(filename, modelname='test1.h5'):
    import cv2
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (50, 40))
    img = np.resize(img, (1, 1, 40, 50))
    model = load_model(modelname)
    for a in range(20):
        temp = model.predict([img])
        arr = []
        result = ''
        for i in temp:
            arr.append(np.argmax(i))
        for i in arr:
            i = str(i)
            if i != '10':
                result += i
        print(result)


if __name__ == '__main__':
    create_model()
