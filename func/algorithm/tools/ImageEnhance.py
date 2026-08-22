# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 9:01
# @Author  : wy09
# @File    : ImageEnhance.py
# @Software: PyCharm


import cv2 as cv
import numpy as np
import traceback


def gamma_enhance(img, gamma=3):
    img = np.float32(img)
    img = np.power(img, gamma)
    i_min = np.min(img)
    i_max = np.max(img)
    img = (img - i_min) / (i_max - i_min) * 255
    img = np.uint8(np.clip(img, 0, 255))
    return img


def ahe(img, n=15, alpha=0.2):
    gray_img = np.float32(img)
    g_mean = np.mean(gray_img)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (n, n))
    mean_img = cv.filter2D(gray_img, -1, kernel, borderType=cv.BORDER_REFLECT) / (n * n)
    std_img = np.sqrt(cv.filter2D((gray_img - mean_img) ** 2, -1, kernel, borderType=cv.BORDER_REFLECT)) / (n * n)
    G = alpha * (g_mean / std_img)
    G = np.clip(G, 0.5, 10)
    new_img = mean_img + G * (gray_img - mean_img)
    y, x = np.where(std_img < 0.1)
    new_img[y, x] = gray_img[y, x]
    new_img = np.clip(new_img, 0, 255)
    new_img = np.uint8(new_img)
    return new_img


def usm(img):
    blur_img = cv.GaussianBlur(img, (3, 3), 0)
    lap_img = cv.Laplacian(blur_img, -1, ksize=1, scale=1.0, delta=0, borderType=cv.BORDER_DEFAULT)
    new_img = cv.addWeighted(blur_img, 1.0, lap_img, -1.0, 0)
    return new_img


def get_color_table(th, alpha=0.05):
    table = []
    for i in range(256):
        t = int(1 / (1 + np.e ** (-alpha*(i - th))) * 255)
        if t < 0:
            t = 0
        if t > 255:
            t = 255
        table.append(t)
    return np.asarray(table, dtype=np.uint8)


def auto_enhance(gray_img, d=-1):
    try:
        en_img = auto_enhance3(gray_img, d)
        new_img = reduction_noise(en_img)
        return new_img
    except Exception:
        traceback.print_exc()


def auto_enhance2(gray_img):
    try:
        # temp1 = cv.bilateralFilter(gray_img, 40, 30, 30)
        # temp2 = usm(temp1)
        binary_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 12)
        new_img = cv.cvtColor(binary_img, cv.COLOR_GRAY2BGR)
        return new_img
    except Exception:
        traceback.print_exc()


def adaptive_enhance(gray_img):
    try:
        new_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 12)
        new_img = cv.cvtColor(new_img, cv.COLOR_GRAY2BGR)
        return new_img
    except Exception:
        traceback.print_exc()


def auto_enhance3(gray_img, d=-1):
    h, w = gray_img.shape[:2]
    r = int(min(h, w)/32)
    if 0 < d < 40:
        gray_img = cv.bilateralFilter(gray_img, d, 20, 20)
    if r%2==0:
        r += 1
    blur_img = cv.GaussianBlur(gray_img, (r, r), 0)
    gray_img_float = np.float32(gray_img)
    blur_img_float = np.float32(blur_img)
    gray_img_float = (gray_img_float/blur_img_float)*255
    new_gray_img = np.uint8(np.clip(gray_img_float, 0, 255))
    temp2 = usm(new_gray_img)
    # temp2 = new_gray_img
    th, binary_img = cv.threshold(temp2, 127, 255, cv.THRESH_OTSU)
    table = get_color_table(th)
    new_img = cv.LUT(temp2, table)
    return new_img


def enhance_image(image):
    imfloat = np.float32(image) / 255
    imblur = cv.blur(imfloat, (51, 51), cv.BORDER_CONSTANT)
    rim = (imfloat / imblur) * 255
    rim = np.clip(rim, 0, 255)
    rim = np.uint8(rim)
    return rim


def reduction_noise(gray_img):
    blur_img = cv.GaussianBlur(gray_img, (15, 15), 0)
    # th, binary_img = cv.threshold(blur_img, 127, 255, cv.THRESH_OTSU)
    th = 127
    binary_img = cv.adaptiveThreshold(blur_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 12)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    morph_img = cv.erode(binary_img, kernel)
    mask = np.ones_like(gray_img, dtype=np.uint8) * 255
    y, x = np.where(morph_img < th)
    mask[y, x] = gray_img[y, x]
    return mask
