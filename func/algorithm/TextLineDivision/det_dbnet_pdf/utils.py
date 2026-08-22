import cv2
import numpy as np


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


def get_medianHeight_maxWidth(sizelist):
    heights = []
    widths = []
    for h, w in sizelist:
        if 4 < h < 64:
            heights.append(h)
            widths.append(w)
    heights.sort()
    widths.sort()
    size = len(heights)
    if size < 1:
        mH = 16
        maxW = 64
    elif size < 2:
        mH = heights[0]
        maxW = widths[0]
    else:
        half = size // 2
        mH = (heights[half] + heights[~half]) / 2
        maxW = widths[-1]
    return mH, maxW


def get_resizeratio(mH):
    basemH = 12
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


def get_rects(contour):
    rect = cv2.minAreaRect(contour)
    rect = refine_rrect(rect)
    return rect


def get_paper_angle(rrects):
    assert len(rrects) > 0
    anglelist = []
    for rrect in rrects:
        angle = rrect[2]
        w, h = rrect[1]
        anglelist.append((angle,  w))
    angle_list = sorted(anglelist, key=lambda x: x[1], reverse=True)
    LEN = len(angle_list)
    save_angle_list = []
    for i in range(LEN):
        if len(save_angle_list) > 3:
            break
        save_angle_list.append(angle_list[i][0])
    save_angle_npy = np.array(save_angle_list)
    return np.mean(save_angle_npy)


def dst_rect(origin_rect, dest_height, dest_width, Height, Width):
    x, y, w, h = origin_rect
    wratio = dest_width / Width
    hratio = dest_height / Height
    newx = wratio * x
    newy = hratio * y
    neww = wratio * w
    newh = hratio * h
    dstrect = (int(newx), int(newy), int(neww), int(newh))
    return dstrect


def dst_rrect(origin_rect, dest_height, dest_width, Height, Width):
    newcx = origin_rect[0][0] * dest_width / Width
    newcy = origin_rect[0][1] * dest_height / Height
    neww = (origin_rect[1][0]) * dest_width / Width
    newh = (origin_rect[1][1]) * dest_height / Height
    dstrect = ((newcx, newcy), (neww, newh), origin_rect[2])
    return dstrect


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def draw_rect(src_im, rect, color=(0, 0, 255), thickness=2):
    x, y, w, h = rect
    cv2.rectangle(src_im, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return src_im


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
        if pts[0] >= startsplitpoint[0]:
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


################## Box utils ###################
def xywh2xyxy(rect):
    x, y, w, h = rect
    xyxy = (x, y, x + w, y + h)
    return xyxy


def xyxy2xywh(rect):
    x1, y1, x2, y2 = rect
    xywh = (x1, y1, x2-x1, y2-y1)
    return xywh


def bbox_iou(rect1, rect2):
    x1, y1, x2, y2 = xywh2xyxy(rect1)
    x3, y3, x4, y4 = xywh2xyxy(rect2)
    xx1 = max(x1, x3)
    yy1 = max(y1, y3)
    h1 = y2 - y1
    xx2 = min(x2, x4)
    yy2 = min(y2, y4)
    h2 = y4 - y3
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = min(area1, area2)
    iou = inter / union
    hiou = h / min(h1, h2)
    if iou > 0.5:
        return True
    elif iou > 0 and hiou > 0.7:
        return True
    else:
        return False


def join_rects(rects):
    xyxylist = []
    for rect in rects:
        xyxylist.append(xywh2xyxy(rect))
    xyxylistnpy = np.array(xyxylist)
    o = np.min(xyxylistnpy, axis=0)
    m = np.max(xyxylistnpy, axis=0)
    r = (o[0], o[1], m[2], m[3])
    rr = xyxy2xywh(r)
    return rr


def filter_boxes(rects):
    bagresults = []
    while len(rects) > 0:
        orirect = rects.pop()
        j = 0
        while j < len(rects):
            drect = rects[j]
            if bbox_iou(orirect, drect):
                orirect = join_rects((orirect, drect))
                rects.pop(j)
                j = 0
            j = j + 1
        bagresults.append(orirect)
    return bagresults


# rotate rect to horizon
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


def from_rects_to_mat(srcimg, rects):
    mats = []
    for rect in rects:
        x, y, w, h = rect
        srcmat = srcimg[y: y + h, x: x + w, :]
        mats.append(srcmat)
    return mats