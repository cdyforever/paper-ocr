import cv2
import numpy as np
from shapely.geometry import Polygon


def get_mH(srcsizelist, minW = 32):
    mH = -1
    heights = []
    for w, h in srcsizelist:
        if 8 < h < 128 and w > minW:
            heights.append(h)
    heights.sort()
    size = len(heights)
    if size < 1:
        mH = 16
    elif size < 2:
        mH = heights[0]
    else:
        mH = (heights[int(size / 2)] + heights[int(size/2) - 1]) / 2
    return mH


def get_resizeratio(mH):
    basemH = 8
    resizeratio = basemH / mH
    return resizeratio


def refine_rrect(rrect):
    newrrect = rrect
    if rrect[2] < -45:
        a, b, c = rrect
        c += 90
        d, e = b
        newrrect = (a, (e, d), c)
    if rrect[2] == 90:
        a, b, c = rrect
        d, e = b
        newrrect = (a, (e, d), 0)
    return newrrect


def draw_rotate_rect(src_im, rotate_rect, color=(0, 0, 255), thickness=2):
    #src_im = input_im.copy()
    pts = cv2.boxPoints(rotate_rect)
    pts = np.int0(pts)
    pt1 = (pts[0][0], pts[0][1])
    pt2 = (pts[1][0], pts[1][1])
    pt3 = (pts[2][0], pts[2][1])
    pt4 = (pts[3][0], pts[3][1])
    cv2.line(src_im, pt1, pt2, color, thickness)
    cv2.line(src_im, pt1, pt4, color, thickness)
    cv2.line(src_im, pt3, pt2, color, thickness)
    cv2.line(src_im, pt3, pt4, color, thickness)
    return src_im


def final_contours(contourlist, conposlist):
    contour_list = []
    split_point = []
    for i in range(len(conposlist)):
        split_point.append(conposlist[i][1])
    if len(split_point) < 2:
        return [np.array(contourlist)]
    startsplitpoint = split_point.pop(0)
    contour_single = []
    for pts in contourlist:
        if pts[0] > startsplitpoint[0]:
            contour_single.append(startsplitpoint)
            newcontour = np.array(contour_single)
            contour_list.append( newcontour )
            if len(split_point) == 0:
                break
            contour_single = []
            startsplitpoint = split_point.pop(0)
        else:
            contour_single.append((pts))
    return contour_list


def split_contour(contour, A, medianHeight):
    contourlist = contour.tolist()
    contourlist = sorted(contourlist, key=lambda x: x[0])
    A = np.squeeze(A, axis=1)
    connum = A.shape[0] - 1
    A = A.tolist()
    conposlist = []
    for i in range(connum):
        conpos = [A[i], A[i + 1]]
        conposlist.append(conpos)
    if connum > 1:
        if conposlist[0][1][0] - conposlist[0][0][0] < medianHeight * 3:
            savestart = conposlist[0][0]
            conposlist.pop(0)
            conposlist[0][0] = savestart
        if conposlist[-1][1][0] - conposlist[-1][0][0] < medianHeight * 3:
            saveend = conposlist[-1][1]
            conposlist.pop(-1)
            conposlist[-1][1] = saveend
    contours = final_contours(contourlist, conposlist)
    return contours


def if_split(rect, contour, maxWidth, medianHeight):
    if rect[1][0] < maxWidth * 3 / 5:
        return False
    area1 = cv2.contourArea(contour)
    contourheight = area1 / rect[1][0]
    if contourheight > medianHeight * 1.5:
        return False
    area2 = rect[1][1] * rect[1][0]
    if area2 / area1 < 2:
        return False
    contourlist = contour.tolist()
    contourlist = sorted(contourlist, key=lambda x: x[0])
    contourdict = dict()
    for x, y in contourlist:
        if x not in contourdict.keys():
            contourdict[x] = [y]
        else:
            contourdict[x].append(y)
    split = True
    for k, v in contourdict.items():
        v = sorted(v)
        conheight = v[-1] - v[0]
        if conheight > medianHeight * 1.5:
            split = False
            break
    return split


def polygon_intersection(rect1, rect2):
    g = cv2.boxPoints(rect1)
    p = cv2.boxPoints(rect2)
    g = np.asarray(g)
    p = np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    if inter == 0:
        return 0
    else:
        return max(inter/g.area, inter/p.area)


def get_resize_width_height(h, w, ratio):
    re_size = max(h, w) * ratio
    if re_size < 320:
        re_size = 320
    else:
        re_size = (int(re_size) // 32) * 32
    if h > w:
        scale_h = re_size / h
        tar_h = re_size
        tar_w = int(w * scale_h)
        tar_w = tar_w - tar_w % 32
        tar_w = max(32, tar_w)
        # scale_w = tar_w / w
    else:
        scale_w = re_size / w
        tar_w = re_size
        tar_h = int(h * scale_w)
        tar_h = tar_h - tar_h % 32
        tar_h = max(32, tar_h)
        # scale_h = tar_h / h
    # print(tar_w, tar_h)

    return tar_w, tar_h


def if_distort(rect, contour, maxWidth, medianHeight):
    if rect[1][0] < maxWidth * 1 / 5 or rect[1][0] < medianHeight * 4:
        return False
    area1 = cv2.contourArea(contour)
    area2 = rect[1][1] * rect[1][0]
    if area1 / area2 > 0.5:
        return False
    else:
        return True

    
def getTextLineCenterTrace(contour):
    contourlist = contour.tolist()
    # sort contour list......
    contourlist = sorted(contourlist, key=lambda x: x[0])
    allwidth = contourlist[-1][0] - contourlist[0][0]
    contourdict = dict()
    heightlist = []
    for x, y in contourlist:
        if x not in contourdict.keys():
            contourdict[x] = [y]
        else:
            contourdict[x].append(y)
    tracelist = []
    for k, v in contourdict.items():
        v = sorted(v)
        height = abs(v[0] - v[-1])
        heightlist.append(height)
    maxheight = max(heightlist)
    return contourdict, allwidth, maxheight
