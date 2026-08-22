import numpy as np
import cv2
from .utils import filter_boxes, box_score_fast, rotate_to_horizonal, get_paper_angle, get_rects, dst_rrect


def top_move(zonestd, cx, cy, w, h, zh):
    toplist = []
    for i in range(h // 4, (zh - h) // 2):
        topline = zonestd[cy - h // 2 - i, cx - w // 2: cx + w // 2]
        topsum = np.sum(topline)
        toplist.append(topsum)
        if topsum < 1:
            break
    topmove = toplist.index(min(toplist)) + h // 4 if len(toplist) > 0 else 0
    return topmove


def bottom_move(zonestd, cx, cy, w, h, zh):
    bottomlist = []
    for i in range(h // 4, (zh - h) // 2):
        bottomline = zonestd[cy + h // 2 + i, cx - w // 2: cx + w // 2]
        bottomsum = np.sum(bottomline)
        bottomlist.append(bottomsum)
        if bottomsum < 1:
            break
    bottommove = bottomlist.index(min(bottomlist)) + h // 4 if len(bottomlist) > 0 else 0
    return bottommove


def left_move(zonestd, cx, cy, w, h, zw):
    leftlist = []
    for i in range(h // 2, (zw - w) // 2):
        leftline = zonestd[cy - h // 2: cy + h // 2, cx - w // 2 - i]
        leftsum = np.sum(leftline)
        leftlist.append(leftsum)
        if leftsum < 1:
            break
    leftmove = leftlist.index(min(leftlist)) + h // 2 if len(leftlist) > 0 else 0
    return leftmove


def right_move(zonestd, cx, cy, w, h, zw):
    rightlist = []
    right_move = 0
    for i in range(h // 2, (zw - w) // 2 - 2):
        rightline = zonestd[cy - h // 2: cy + h // 2, cx + w // 2 + i - 2: cx + w // 2 + i + 2]
        rightsum = np.sum(rightline)
        rightlist.append(rightsum)
        if rightsum < 1:
            break
    rightmove = rightlist.index(min(rightlist)) + h // 2 if len(rightlist) > 0 else 0
    return rightmove


def move_and_cal(zone, oldrect):
    zh, zw = zone.shape[:2]
    zonestd = zone / 255
    cx, cy = zw // 2, zh // 2
    _, (w, h), _ = oldrect
    w = int(w)
    h = int(h)
    topmove = top_move(zonestd, cx, cy, w, h, zh)
    bottommove = bottom_move(zonestd, cx, cy, w, h, zh)
    leftmove = left_move(zonestd, cx, cy, w, h, zw)
    rightmove = right_move(zonestd, cx, cy, w, h, zw)
    cyplus = (bottommove - topmove) // 2
    heightplus = bottommove + topmove + 4
    widthplus = leftmove + rightmove
    return heightplus, widthplus, cyplus


def get_rect_and_mat(srcimg, oldrect, newrect, i):
    (cx, cy), (cw, ch), angle = newrect
    tmprect = ((cx + 300, cy + 300), (cw, ch), angle)
    point = cv2.boxPoints(tmprect)
    x, y, w, h = cv2.boundingRect(point)
    srcmat = srcimg[y: y + h, x: x + w]
    zone = rotate_to_horizonal(srcmat, tmprect, (255, 255, 255))
    heightplus, widthplus, cyplus = move_and_cal(zone, oldrect)
    (ocx, ocy), (ow, oh), oangle = oldrect
    refinerect = ((ocx, ocy + cyplus), (ow + widthplus, oh + heightplus), oangle)
    return refinerect


def after_process(bin, rrects, mH):
    height, width = bin.shape[:2]
    tmpbin = np.zeros((height + 600, width + 600), dtype=np.uint8)
    tmpbin[300: -300, 300: -300] = bin
    refineboxes = []
    rotaterects = []
    paper_angle = get_paper_angle(rrects)
    for i in range(len(rrects)):
        oldrect = rrects[i]
        cx, cy = oldrect[0]
        cw, ch = oldrect[1]
        # get origin rect bigger zone.....
        oldrect = ((cx, cy), (cw, ch), paper_angle)
        newrect = ((cx, cy), (cw + ch * 3, ch * 5), paper_angle)
        refinerect = get_rect_and_mat(tmpbin, oldrect, newrect, i)
        pts = cv2.boxPoints(refinerect)
        refinebox = cv2.boundingRect(pts)
        refineboxes.append(refinebox)
    refineboxes = filter_boxes(refineboxes)
    for box in refineboxes:
        rx, ry, rw, rh = box
        rotaterect = ((rx + rw / 2, ry + rh / 2), (float(rw), float(rh)), 0)
        rotaterects.append(rotaterect)
    return refineboxes, rotaterects


#############################################################################################
def binarize(pred, thresh):
    return pred > thresh


def decode_from_result(pred, dest_width, dest_height):
    thresh = 0.3
    min_size = 1
    bitmap = binarize(pred, thresh)
    Height, Width = bitmap.shape
    bitmap = (bitmap * 255).astype(np.uint8)
    ret, contours, hierarchy = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    num_contours = len(contours)
    contour_rect_list = []
    for index in range(num_contours):
        contour = contours[index].squeeze(1)
        rrect = get_rects(contour)
        w, h = rrect[1]
        if h < min_size:
            continue
        if cv2.contourArea(contour) < 8:
            continue
        score = box_score_fast(pred, contour)
        if score < thresh:
            continue
        contour_rect_list.append((rrect, contour))
    # first get medianHeight of the contours and maxWidth of the contours......
    # split eposion for distort line......
    final_contour_rrect_list = []
    # circle the contour rect list........ split those could split contours......
    for i in range(len(contour_rect_list)):
        rrect, contour = contour_rect_list[i]
        # get if split .... and then split use matrix A
        rrect = dst_rrect(rrect, dest_height, dest_width, Height, Width)
        final_contour_rrect_list.append(rrect)
    return final_contour_rrect_list, bitmap