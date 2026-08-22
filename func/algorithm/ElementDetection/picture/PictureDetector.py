# -*- coding: utf-8 -*-
# @Time    : 2020/9/29 10:57
# @Author  : chenguodong
# @File    : ImageDetection.py
# @Software: PyCharm

from .yolov5onnx import YOLOv5ONNX


class PictureDetector(object):

    def __init__(self, model_path, device='cpu'):
        self.yolov5 = YOLOv5ONNX(model_path)

    def detect(self, img):
        pic_boxes = self.yolov5.infer(img)
        return pic_boxes
