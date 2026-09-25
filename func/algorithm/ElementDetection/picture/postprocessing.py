import cv2
import numpy as np


def area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def iou(bbox1, bbox2):
    '''
    计算两个方框之间的交并比
    :param bbox: 方框
    :return: 返回交并比
    '''
    # 计算重叠区域面积
    overlapWidth = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    overlapHeight = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    overlapArea = max(overlapHeight, 0) * max(overlapWidth, 0)

    # 计算重叠区域的IoU
    IoU = float(overlapArea) / (min(area(bbox1), area(bbox2)))

    return IoU


def merge_bbox(bbox):
    '''合并重叠方框'''
    flag = [0] * len(bbox)
    for i in range(len(bbox)):
        bbox_i = bbox[i]
        if flag[i] == 1 and i != 0:
            continue

        for j in range(len(bbox)):
            if i == j or flag[j] == 1:
                continue

            bbox_j = bbox[j]

            if iou(bbox_i, bbox_j) > 0.9:
                flag[j] = 1

                temp_xmin = min(bbox_i[0], bbox_j[0])
                temp_ymin = min(bbox_i[1], bbox_j[1])
                temp_xmax = max(bbox_i[2], bbox_j[2])
                temp_ymax = max(bbox_i[3], bbox_j[3])

                bbox_i[0] = temp_xmin
                bbox_i[1] = temp_ymin
                bbox_i[2] = temp_xmax
                bbox_i[3] = temp_ymax

        bbox[i] = bbox_i

    mergeList = []  # 存储合并后的结果
    for idx, i in enumerate(flag):
        if i == 0:
            mergeList.append(bbox[idx])

    return mergeList


def verticalContourProjection(img):
    h, w = img.shape

    upContHist = [0] * w
    downContHist = [0] * w

    for c in range(w):
        # 从上往下遍历每一列，记录第一个出现的白色像素位置
        for r1 in range(h):
            if img[r1, c] == 255:
                upContHist[c] = r1
                break
        # 从下往上遍历每一列，记录第一个出现的白色像素位置
        for r2 in range(h - 1, -1, -1):
            if img[r2, c] == 255:
                downContHist[c] = r2
                break

    return upContHist, downContHist


def horizontalContourProjection(img):
    '''
    左右轮廓投影
    :param img: 待投影的二值图像
    :return: 返回投影后的左右轮廓
    '''
    h, w = img.shape

    leftContHist = [0] * h
    rightContHist = [0] * h

    for r in range(h):
        # 从左往右遍历每一行，记录第一个出现的白色像素位置
        for c1 in range(w):
            if img[r, c1] == 255:
                leftContHist[r] = c1
                break

        # 从右往左遍历每一行，记录第一个出现的白色像素位置
        for c2 in range(w - 1, -1, -1):
            if img[r, c2] == 255:
                rightContHist[r] = c2
                break

    return leftContHist, rightContHist


def upDownContourProjection(img, src_img=None):
    '''
    上下轮廓投影
    :param img: 待投影区域
    :param src_img: 原始图像
    :return: 返回图像的上下边界
    '''
    # 去躁
    img = cv2.medianBlur(img, 3)

    # 投影获取上下轮廓
    upContHist, downContHist = verticalContourProjection(img)

    # 计算列表中非零元素的最大最小值
    y1 = min(filter(lambda x: x >= 0, upContHist))
    y2 = max(filter(lambda x: x >= 0, downContHist))

    return y1, y2


def leftRightContourProjection(img, src_img=None):
    '''
    左右轮廓投影
    :param img: 待投影区域
    :param src_img: 原始图像
    :return: 返回图像的上下边界
    '''
    # 投影并获取左右轮廓
    leftHist, rightHist = horizontalContourProjection(img)

    # 计算列表中最大最小值
    x1 = min(filter(lambda x: x >= 0, leftHist))
    x2 = max(filter(lambda x: x >= 0, rightHist))

    return x1, x2


def modify_bbox(bbox, img):
    '''修正方框位置'''

    # 获取图片区域
    for idx, p in enumerate(bbox):
        x1, y1, x2, y2 = p
        roi = img[y1:y2, x1:x2]
        # 预处理
        g_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 灰度化
        g_roi = cv2.medianBlur(g_roi, 3)
        _, th_roi = cv2.threshold(g_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 二值化
        _y1, _y2 = upDownContourProjection(th_roi)
        _x1, _x2 = leftRightContourProjection(th_roi)
        bbox[idx] = [x1 + _x1, y1 + _y1, x1 + _x2, y1 + _y2]

    return bbox


def postprocessing(bbox, image):
    '''方框检测后处理'''
    # 重叠区域筛选
    bbox = merge_bbox(bbox)

    # 位置修正
    bbox = modify_bbox(bbox, img=image)

    return bbox
