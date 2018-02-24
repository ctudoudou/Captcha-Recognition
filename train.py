#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/23 下午10:27
# @Author  : tudoudou
# @File    : train.py
# @Software: PyCharm

from dataset import g_get_data
from model import create_model


def train():
    model = create_model()
    model.fit_generator(g_get_data(), 256, 10)


if __name__ == '__main__':
    train()
