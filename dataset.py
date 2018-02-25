#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/23 下午10:06
# @Author  : tudoudou
# @File    : dataset.py
# @Software: PyCharm

import cv2
import numpy
from PIL import Image
from get_captcha import ImageCaptcha, HAN


def get_data(img):
    ca_img, code = img.get_image()
    img = cv2.cvtColor(numpy.asarray(ca_img), cv2.COLOR_RGB2BGR)
    ca_img = cv2.fastNlMeansDenoisingColored(img, None, 255, 255, 255, 4)
    ca_img = Image.fromarray(ca_img)

    ca_img_l = ca_img.crop((0, 0, 85, 80)).convert('L')
    ca_img_r = ca_img.crop((85, 0, 170, 80)).convert('L')
    # ca_img.show();ca_img_l.show();ca_img_r.show()
    ca_lab_l, ca_lab_r = code[0], code[1]
    l, r = numpy.zeros([1, 1130]), numpy.zeros([1, 1130])
    l[0][HAN.index(ca_lab_l)] = 1
    r[0][HAN.index(ca_lab_l)] = 1
    ca_img_l=numpy.array(ca_img_l)
    # return numpy.array(ca_img_l), numpy.array(ca_img_r), l, r
    return numpy.array([[ca_img_l]]), l


def g_get_data():
    img = ImageCaptcha(2, 170, 80, './MSYHMONO.ttf')
    while True:
        yield get_data(img)


if __name__ == '__main__':
    o = g_get_data()
    while True:
        print(next(o))
