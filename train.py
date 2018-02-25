#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/23 下午10:27
# @Author  : tudoudou
# @File    : train.py
# @Software: PyCharm

from dataset import g_get_data
from model import create_model
import numpy as np


def train():
    model = create_model()
    # a = g_get_data()
    # b, c = next(a)
    # print(c)
    # b = np.random.randint(1, 100, [100, 1, 85, 80])
    # c = np.random.randint(1, 100, [100, 1130])
    # print(b)
    # print(c)
    # model.fit(b, c)
    model.fit_generator(g_get_data(), 256, 10)


if __name__ == '__main__':
    train()
