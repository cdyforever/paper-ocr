import numpy as np
import cv2
from .utils import get_rects, box_score_fast, get_medianHeight_maxWidth, dst_rrect, split_contour, if_split
from skimage.morphology import skeletonize


######## Get rotate mat ########
def rotate_to_horizonal(src, rotaterect, fill_color):
    angle = rotaterect[2]
    oriw, orih = rotaterect[1]
    row, col = src.shape[0:2]
    maxBorder = int( max(col, row) * 1.414 )
    dx = int( (maxBorder - col) / 2 )
    dy = int( (maxBorder - row) / 2 )
    dst = cv2.copyMakeBorder(src, dy, dy, dx, dx, cv2.BORDER_CONSTANT, fill_color)
    drow, dcol = dst.shape[0:2]
    center = (float(dcol) / 2 , float(drow) / 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    dst = cv2.warpAffine(dst, mat, (dcol, drow))
    x = int( (dcol - oriw) / 2 ) + 1
    y = int( (drow - orih) / 2 ) + 1
    dst = dst[y:y + int(orih) - 2, x:x + int(oriw) - 2]
    return dst


def top_move(zonestd, cx, cy, w, h, zh):
    toplist = []
    for i in range(1, (zh - h) // 2):
        topline = zonestd[cy - h // 2 - i, cx - w // 2: cx + w // 2]
        topsum = np.sum(topline)
        toplist.append(topsum)
        if topsum < 2:
            break
    topmove = toplist.index(min(toplist))
    return topmove


def bottom_move(zonestd, cx, cy, w, h, zh):
    bottomlist = []
    for i in range(1, (zh - h) // 2):
        bottomline = zonestd[cy + h // 2 + i, cx - w // 2: cx + w // 2]
        bottomsum = np.sum(bottomline)
        bottomlist.append(bottomsum)
        if bottomsum < 2:
            break
    bottommove = bottomlist.index(min(bottomlist))
    return bottommove


def left_move(zonestd, cx, cy, w, h, zw):
    leftlist = []
    for i in range(h // 2, (zw - w) // 2):
        leftline = zonestd[cy - h // 2: cy + h // 2, cx - w // 2 - i]
        leftsum = np.sum(leftline)
        leftlist.append(leftsum)
        if leftsum < 2:
            break
    leftmove = leftlist.index(min(leftlist)) + h // 2
    return leftmove


def right_move(zonestd, cx, cy, w, h, zw, single=True):
    rightlist = []
    right_move = 0
    if single:
        for i in range(h // 2, (zw - w) // 2 - 2):
            rightline = zonestd[cy - h // 2: cy + h // 2, cx + w // 2 + i - 2: cx + w // 2 + i + 2]
            rightsum = np.sum(rightline)
            rightlist.append(rightsum)
            if rightsum < 2:
                break
        rightmove = rightlist.index(min(rightlist)) + h // 2
    else:
        for i in range((w - zw) // 2 + 2, (zw - w) // 2 - 2):
            rightline = zonestd[cy - h // 2: cy + h // 2, cx + w // 2 + i - 2: cx + w // 2 + i + 2]
            rightsum = np.sum(rightline)
            rightlist.append(rightsum)
            if rightsum < 2:
                break
        rightmove = rightlist.index(min(rightlist)) + (w - zw) // 2 + 2
    return rightmove


def move_and_cal(zone, oldrect):
    zh, zw = zone.shape[:2]
    zonestd = zone / 255
    cx, cy = zw // 2, zh // 2
    _, (w, h), _ = oldrect
    w = int(w)
    h = int(h)
    cyplus = 0
    heightplus = 0
    topmove = top_move(zonestd, cx, cy, w, h, zh)
    bottommove = bottom_move(zonestd, cx, cy, w, h, zh)
    leftmove = left_move(zonestd, cx, cy, w, h, zw)
    rightmove = right_move(zonestd, cx, cy, w, h, zw)
    if abs(bottommove - topmove) < 8:
        heightplus = max(bottommove, topmove) * 2 + 1
    else:
        cyplus = (bottommove - topmove) // 2
        heightplus = bottommove + topmove + 2
    widthplus = leftmove + rightmove
    return heightplus, widthplus, cyplus


def get_rect_and_mat(srcimg, oldrect, newrect):
    (cx, cy), (cw, ch), angle = newrect
    tmprect = ((cx + 300, cy + 300), (cw, ch), angle)
    point = cv2.boxPoints(tmprect)
    x, y, w, h = cv2.boundingRect(point)
    srcmat = srcimg[y: y + h, x: x + w]
    zone = rotate_to_horizonal(srcmat, tmprect, (255, 255, 255))
    heightplus, widthplus, cyplus = move_and_cal(zone, oldrect)
    (ocx, ocy), (ow, oh), oangle = oldrect
    refinerect = ((ocx, ocy + cyplus), (ow + widthplus, oh + heightplus), oangle)
    return refinerect, zone


def move_and_cal_v2(zone, oldrect, idx, offset):
    zh, zw = zone.shape[:2]
    zonestd = zone / 255
    cx, cy = zw // 2, zh // 2
    _, (w, h), _ = oldrect
    w = int(w)
    h = int(h)
    cyplus = 0
    heightplus = 0
    topmove = top_move(zonestd, cx, cy, w, h, zh)
    bottommove = bottom_move(zonestd, cx, cy, w, h, zh)
    if idx == 0:
        leftmove = left_move(zonestd, cx, cy, w, h, zw)
    else:
        leftmove = -offset
    rightmove = right_move(zonestd, cx, cy, w, h, zw, False)
    if abs(bottommove - topmove) < 8:
        heightplus = max(bottommove, topmove) * 2 + 1
    else:
        cyplus = (bottommove - topmove) // 2
        heightplus = bottommove + topmove + 2
    widthplus = leftmove + rightmove
    cxplus = (rightmove - leftmove) // 2
    newoffset = rightmove
    return heightplus, widthplus, cxplus, cyplus, newoffset


def get_rect_and_mat_v2(srcimg, oldrect, newrect, offset, idx):
    (cx, cy), (cw, ch), angle = newrect
    tmprect = ((cx + 300, cy + 300), (cw, ch), angle)
    point = cv2.boxPoints(tmprect)
    x, y, w, h = cv2.boundingRect(point)
    srcmat = srcimg[y: y + h, x: x + w]
    zone = rotate_to_horizonal(srcmat, tmprect, (255, 255, 255))
    if idx == 0:
        heightplus, widthplus, cxplus, cyplus, newoffset = move_and_cal_v2(zone, oldrect, 0, offset)
    else:
        heightplus, widthplus, cxplus, cyplus, newoffset = move_and_cal_v2(zone, oldrect, -1, offset)
    (ocx, ocy), (ow, oh), oangle = oldrect
    refinerect = ((ocx + cxplus, ocy + cyplus), (ow + widthplus, oh + heightplus), oangle)
    return refinerect, newoffset



def after_process(bin, rrects, split_rrects):
    height, width = bin.shape[:2]
    tmpbin = np.zeros((height + 600, width + 600), dtype=np.uint8)
    tmpbin[300: -300, 300: -300] = bin
    refinerects = []
    for oldrect in rrects:
        (cx, cy), (w, h), angle = oldrect
        newrect = ((cx, cy), (w + h * 4, h * 4), angle)
        refinerect, _ = get_rect_and_mat(tmpbin, oldrect, newrect)
        refinerects.append(refinerect)
    for rects in split_rrects:
        offset = 0
        for i in range(len(rects)):
            oldrect = rects[i]
            (cx, cy), (w, h), angle = oldrect
            newrect = ((cx, cy), (w + h * 3, h * 4), angle)
            refinerect, offset = get_rect_and_mat_v2(tmpbin, oldrect, newrect, offset, i)
            refinerects.append(refinerect)
    return refinerects


def binarize(pred, thresh):
    return pred > thresh


def decode_from_result(pred, dest_width, dest_height):
    thresh = 0.2
    min_size = 2
    bitmap = binarize(pred, thresh)
    Height, Width = bitmap.shape
    bitmap = (bitmap * 255).astype(np.uint8)
    ret, contours, hierarchy = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    num_contours = len(contours)
    contour_rect_list = []
    height_list = []
    for index in range(num_contours):
        contour = contours[index].squeeze(1)
        rect = get_rects(contour)
        _width, _height = rect[1]
        if _height < min_size:
            continue
        score = box_score_fast(pred, contour)
        if score < thresh:
            continue
        height_list.append((_height, _width))
        contour_rect_list.append((rect, contour))
    # first get medianHeight of the contours and maxWidth of the contours......
    medianHeight, maxWidth = get_medianHeight_maxWidth(height_list)
    print(medianHeight, maxWidth)
    # split eposion for distort line......
    eposion = medianHeight / 3
    final_contour_rect_list = []
    final_contour_split_list = []
    # circle the contour rect list........ split those could split contours......
    for i in range(len(contour_rect_list)):
        rect, contour = contour_rect_list[i]
        # get if split .... and then split use matrix A
        bool_split = if_split(rect, contour, maxWidth, medianHeight)
        if bool_split:
            instance_center_mask = np.zeros((Height, Width), dtype=np.uint8)
            cv2.drawContours(instance_center_mask, [contour], -1, 1, -1)
            skeleton = skeletonize(instance_center_mask)
            skeleton = np.transpose(skeleton)
            skeleton_xy = np.argwhere(skeleton > 0)
            A = cv2.approxPolyDP(skeleton_xy, eposion, False)
            contours = split_contour(contour, A, medianHeight)
            split_list = []
            for _con in contours:
                _rect = get_rects(_con)
                if _rect[1][0] > medianHeight and _rect[1][1] > 1:
                    split_list.append(_rect)
            if len(split_list) > 1:
                new_split_list = []
                for split in split_list:
                    new_split_list.append(dst_rrect(split, dest_height, dest_width, Height, Width))
                final_contour_split_list.append(new_split_list)
            else:
                rrect = dst_rrect(rect, dest_height, dest_width, Height, Width)
                final_contour_rect_list.append(rrect)
        else:
            rrect = dst_rrect(rect, dest_height, dest_width, Height, Width)
            final_contour_rect_list.append(rrect)
    return final_contour_rect_list, final_contour_split_list, bitmap