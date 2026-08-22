import cv2
import numpy as np
import pyclipper
from .utils import split_contour, if_distort, polygon_intersection, getTextLineCenterTrace
from skimage.morphology import skeletonize


class SegDetectorRepresenter:
    def __init__(self, box_thresh=0, max_candidates=1000):
        self.min_size = 3
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates


    def __call__(self, preds, oriimg, height, width):
        pred = preds[0, :, :]
        pred = cv2.resize(pred, (width, height), cv2.INTER_CUBIC)
        bitmap = self.binarize(pred)
        frects, distort_rect_mat = self.boxes_from_bitmap(pred, bitmap, oriimg)
        return frects, distort_rect_mat


    def binarize(self, pred, thresh=0.2):
        return pred > thresh


    def boxes_from_bitmap(self, pred, bitmap, oriimg, Debug=False):
        bitmap = (bitmap * 255).astype(np.uint8)
        ret, contours, hierarchy = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        num_contours = len(contours)
        contour_rect_list = []
        height_list = []
        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            rect = self.get_rects(contour)
            _width, _height = rect[1]
            if _height < self.min_size:
                continue
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue
            height_list.append((_height, _width))
            contour_rect_list.append((rect, contour))
        medianHeight, maxWidth = self.get_medianHeight_maxWidth(height_list)
        other_rect = []
        distort_rect_mat = []
        for i in range(len(contour_rect_list)):
            rect, contour = contour_rect_list[i]
            bool_split = if_distort(rect, contour, maxWidth, medianHeight)
            if bool_split:
                contourdict, allwidth, maxheight = getTextLineCenterTrace(contour)
                saveheight = max(maxheight, medianHeight)
                textline = np.ones((int(saveheight) * 2, int(allwidth) + 2, 3), dtype=np.uint8) * 255
                j = 0
                for k, v in contourdict.items():
                    v = sorted(v)
                    cv = (v[0] + v[-1]) // 2
                    try:
                        textline[:, j, :] = oriimg[cv - int(saveheight): cv + int(saveheight), k, :]
                    except:
                        pass
                    j += 1
                distort_rect_mat.append((rect, textline))
            else:
                other_rect.append((rect, contour))
        final_rect_list = other_rect
        frects = []
        for i in range(len(final_rect_list)):
            frect, fcon = final_rect_list[i]
            neww =  frect[1][0] + medianHeight
            frects.append((frect[0], (neww, frect[1][1]), frect[2]))
        return frects, distort_rect_mat


    def get_boxes_height(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        bounding_box = self.refine_rrect(bounding_box)
        height = bounding_box[1][1]
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, height


    def get_rects(self, contour):
        rect = cv2.minAreaRect(contour)
        rect = self.refine_rrect(rect)
        return rect


    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


    def refine_rrect(self, rrect):
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


    def get_medianHeight_maxWidth(self, sizelist):
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