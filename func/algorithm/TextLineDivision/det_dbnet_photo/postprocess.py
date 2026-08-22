
import numpy as np
import cv2
from .utils import polygon_intersection, refine_rrect


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
    for i in range(h // 4, (zh - h) // 2):
        topline = zonestd[cy - h // 2 - i, cx - w // 2: cx + w // 2]
        topsum = np.sum(topline)
        toplist.append(topsum)
        if topsum < 1:
            break
    topmove = toplist.index(min(toplist)) + h // 4
    return topmove


def bottom_move(zonestd, cx, cy, w, h, zh):
    bottomlist = []
    for i in range(h // 4, (zh - h) // 2):
        bottomline = zonestd[cy + h // 2 + i, cx - w // 2: cx + w // 2]
        bottomsum = np.sum(bottomline)
        bottomlist.append(bottomsum)
        if bottomsum < 1:
            break
    bottommove = bottomlist.index(min(bottomlist)) + h // 4
    return bottommove


def left_move(zonestd, cx, cy, w, h, zw):
    leftlist = []
    for i in range(h // 2, (zw - w) // 2):
        leftline = zonestd[cy - h // 2: cy + h // 2, cx - w // 2 - i]
        leftsum = np.sum(leftline)
        leftlist.append(leftsum)
        if leftsum < 1:
            break
    leftmove = leftlist.index(min(leftlist)) + h // 4 if len(leftlist) > 0 else h // 4
    return leftmove


def right_move(zonestd, cx, cy, w, h, zw):
    rightlist = []
    for i in range(h // 2, (zw - w) // 2):
        rightline = zonestd[cy - h // 2: cy + h // 2, cx + w // 2 + i]
        rightsum = np.sum(rightline)
        rightlist.append(rightsum)
        if rightsum < 1:
            break
    rightmove = rightlist.index(min(rightlist)) + h // 4 if len(rightlist) > 0 else h // 4
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


def filter_zone(zone, oldrect):
    zh, zw = zone.shape[:2]
    _, (w, h), _ = oldrect
    w = int(w)
    h = int(h)
    ifzone = zone[(zh - h) // 2 : (zh + h) // 2 ,(zw - w) // 2: (zw + w) // 2]
    contournum = np.sum(ifzone) / 255
    if contournum < 24:
        return False
    else:
        return True


def get_rects_and_mat(srcimg, oldrect, newrect):
    (cx, cy), (cw, ch), angle = newrect
    tmprect = ((cx + 300, cy + 300), (cw, ch), angle)
    point = cv2.boxPoints(tmprect)
    x, y, w, h = cv2.boundingRect(point)
    srcmat = srcimg[y: y + h, x: x + w]
    zone = rotate_to_horizonal(srcmat, tmprect, (255, 255, 255))
    ifDrop = filter_zone(zone, oldrect)
    if not ifDrop:
        return None, None
    heightplus, widthplus, cyplus = move_and_cal(zone, oldrect)
    (ocx, ocy), (ow, oh), oangle = oldrect
    #refinerect = ((ocx, ocy + cyplus), (ow + oh * 2, oh + heightplus), oangle)
    refinerect = ((ocx, ocy + cyplus), (ow + widthplus + 8, oh + heightplus + 4), oangle)
    return refinerect, zone


def right_points(box, w, h):
    box[:, 0] = np.clip(box[:, 0], 0, w-1)
    box[:, 1] = np.clip(box[:, 1], 0, h-1)
    return box


def connect_rects(rect1, rect2):
    pts1 = cv2.boxPoints(rect1)
    pts2 = cv2.boxPoints(rect2)
    pts = np.concatenate((pts1, pts2))
    dstrect = cv2.minAreaRect(pts)
    dstrect = refine_rrect(dstrect)
    return dstrect


def filter_rects(rect_list):
    resultcontours = []
    for i in range(len(rect_list)):
        if i >= len(rect_list):
            break
        srcrect = rect_list[i]
        j = i + 1
        while j < len(rect_list):
            dstrect = rect_list[j]
            flag = polygon_intersection(srcrect, dstrect)
            if flag > 0.5:
                rect_list.pop(j)
                srcrect = connect_rects(srcrect, dstrect)
                j = i + 1  # from first position......
            else:
                j += 1
        resultcontours.append(srcrect)
    return resultcontours


def after_process(bin, rrects):
    height, width = bin.shape[:2]
    idx = 1
    tmpbin = np.zeros((height + 600, width + 600), dtype=np.uint8)
    tmpbin[300: -300, 300: -300] = bin
    refinerects = []
    for oldrect in rrects:
        idx += 1
        (cx, cy), (w, h), angle = oldrect
        newrect = ((cx, cy), (w + h * 3, h * 4), angle)
        # pts = cv2.boxPoints(newrect)
        # pts = right_points(pts, width, height)
        refinerect, zone = get_rects_and_mat(tmpbin, oldrect, newrect)
        if refinerect is not None:
            refinerects.append(refinerect)
    refinerects = filter_rects(refinerects) 
    return refinerects


def from_rects_to_mat(srcimg, rects):
    height, width = srcimg.shape[:2]
    mats = []
    for rotaterect in rects:
        point = cv2.boxPoints(rotaterect)
        x, y, w, h = cv2.boundingRect(point)
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, width - x)
        h = min(h, height - y)
        srcmat = srcimg[y: y + h, x: x + w]
        dst = rotate_to_horizonal(srcmat, rotaterect, (255, 255, 255))
        mats.append(dst)
    return mats
