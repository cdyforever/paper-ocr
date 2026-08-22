import cv2
import numpy as np
from .utils import get_mH


def enhance_image( image ):
    imfloat = np.float32( image ) / 255
    imblur = cv2.blur( imfloat, (51, 51), cv2.BORDER_CONSTANT)
    rim = (imfloat / imblur) * 255
    rim = np.clip(rim, 0, 255)
    rim = np.uint8( rim )
    return rim


def get_binary_and_mH(srcim):
    rim = enhance_image(srcim)
    grayim = cv2.cvtColor(rim, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(grayim, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    kernelvertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    vbin = cv2.dilate(bin, kernelvertical)
    #cv2.imshow("vbin", vbin)
    #cv2.waitKey(0)
    _, label, stats, centroids = cv2.connectedComponentsWithStats(vbin, connectivity=8)
    sizelist = []
    boxlist = []
    for j in range(1, stats.shape[0]):
        rectmsg = stats[j]
        x, y, w, h, area = rectmsg
        sizelist.append((w, h))
    mediaH = get_mH(sizelist, minW=16)
    # print("mediaH is : ", mediaH)
    return bin, mediaH