#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from PIL import Image
from skimage import measure

from .image import letterbox_image, exp, minAreaLine, draw_lines, line_to_line, sqrt
from .config import tableNetPath, GPU, SIZE
# from .darknet import load_net, predict_image, array_to_image

# if not GPU:
#     tableNet = cv2.dnn.readNetFromDarknet(tableNetPath.replace('.weights', '.cfg'), tableNetPath)
# else:
#     from .darknet import load_net, predict_image, array_to_image
#     tableNet = load_net(tableNetPath.replace('table.weights', 'table-darknet.cfg').encode(), tableNetPath.encode(), 0)


class TableLineDetector(object):

    # def __init__(self, table_param_dir, device='cpu'):
    def __init__(self, table_param_dir):
        table_weight_path = os.path.join(table_param_dir, 'table.weights')

        # self.device = device
        # if self.device != 'cuda':
        #     table_cfg_path = os.path.join(table_param_dir, 'table.cfg')
        #     self.table_net = cv2.dnn.readNetFromDarknet(table_cfg_path, table_weight_path)
        # else:
        #     table_cfg_path = os.path.join(table_param_dir, 'table-darknet.cfg')
        #     self.table_net = load_net(table_cfg_path.encode(), table_weight_path.encode(), 0)
        table_cfg_path = os.path.join(table_param_dir, 'table.cfg')
        self.table_net = cv2.dnn.readNetFromDarknet(table_cfg_path, table_weight_path)

    def dnn_cpu_predict(self, img):
        img_resize, fx, fy, dx, dy = letterbox_image(img, SIZE)
        img_w, img_h = SIZE
        img_resize = np.array(img_resize)
        image = cv2.dnn.blobFromImage(img_resize, 1, size=(img_w, img_h), swapRB=False)
        image = np.array(image) / 255
        self.table_net.setInput(image)
        out = self.table_net.forward()
        out = exp(out[0])
        out = out[:, dy:, dx:]
        return out, fx, fy

    # def darknet_gpu_predict(self, img):
    #     img_resize, fx, fy, dx, dy = letterbox_image(img, SIZE)
    #     img_w, img_h = SIZE
    #     im = array_to_image(img_resize)
    #     out = predict_image(self.table_net, im)
    #     values = []
    #     for i in range(2 * img_w * img_h):
    #         values.append(out[i])
    #     out = np.array(values).reshape((2, img_h, img_w))
    #     out = out[:, dy:, dx:]
    #     return out, fx, fy

    def get_table_rowcols(self, img, prob, row=100, col=100):
        # if self.device != 'cuda':
        #     out, fx, fy = self.dnn_cpu_predict(img)
        # else:
        #     out, fx, fy = self.darknet_gpu_predict(img)
        out, fx, fy = self.dnn_cpu_predict(img)

        rows = out[0]
        cols = out[1]

        labels = measure.label(rows > prob, connectivity=2)
        regions = measure.regionprops(labels)
        rows_lines = [minAreaLine(line.coords) for line in regions if line.bbox[3] - line.bbox[1] > row]

        labels = measure.label(cols > prob, connectivity=2)
        regions = measure.regionprops(labels)
        cols_lines = [minAreaLine(line.coords) for line in regions if line.bbox[2] - line.bbox[0] > col]

        tmp = np.zeros(SIZE[::-1], dtype='uint8')
        tmp = draw_lines(tmp, cols_lines + rows_lines, color=255, lineW=1)
        labels = measure.label(tmp > 0, connectivity=2)
        regions = measure.regionprops(labels)

        for region in regions:
            ymin, xmin, ymax, xmax = region.bbox
            label = region.label
            if ymax - ymin < 20 or xmax - xmin < 20:
                labels[labels == label] = 0
        labels = measure.label(labels > 0, connectivity=2)

        ind_y, ind_x = np.where(labels > 0)
        if len(ind_x) > 0:
            xmin, xmax = ind_x.min(), ind_x.max()
            ymin, ymax = ind_y.min(), ind_y.max()
            rows_lines = [p for p in rows_lines if
                          xmin <= p[0] <= xmax and xmin <= p[2] <= xmax and ymin <= p[1] <= ymax and ymin <= p[3] <= ymax]
            cols_lines = [p for p in cols_lines if
                          xmin <= p[0] <= xmax and xmin <= p[2] <= xmax and ymin <= p[1] <= ymax and ymin <= p[3] <= ymax]
            rows_lines = [[box[0] / fx, box[1] / fy, box[2] / fx, box[3] / fy] for box in rows_lines]
            cols_lines = [[box[0] / fx, box[1] / fy, box[2] / fx, box[3] / fy] for box in cols_lines]
            return rows_lines, cols_lines
        else:
            return None, None

    @staticmethod
    def adjust_lines(RowsLines, ColsLines, alph=50):
        ##调整line
        nrow = len(RowsLines)
        ncol = len(ColsLines)
        new_rows_lines = []
        new_cols_lines = []
        for i in range(nrow):
            x1, y1, x2, y2 = RowsLines[i]
            cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
            for j in range(nrow):
                if i != j:
                    x3, y3, x4, y4 = RowsLines[j]
                    cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                    if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                        continue
                    else:
                        r = sqrt((x1, y1), (x3, y3))
                        if r < alph:
                            new_rows_lines.append([x1, y1, x3, y3])
                        r = sqrt((x1, y1), (x4, y4))
                        if r < alph:
                            new_rows_lines.append([x1, y1, x4, y4])

                        r = sqrt((x2, y2), (x3, y3))
                        if r < alph:
                            new_rows_lines.append([x2, y2, x3, y3])
                        r = sqrt((x2, y2), (x4, y4))
                        if r < alph:
                            new_rows_lines.append([x2, y2, x4, y4])

        for i in range(ncol):
            x1, y1, x2, y2 = ColsLines[i]
            cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
            for j in range(ncol):
                if i != j:
                    x3, y3, x4, y4 = ColsLines[j]
                    cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                    if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                        continue
                    else:
                        r = sqrt((x1, y1), (x3, y3))
                        if r < alph:
                            new_cols_lines.append([x1, y1, x3, y3])
                        r = sqrt((x1, y1), (x4, y4))
                        if r < alph:
                            new_cols_lines.append([x1, y1, x4, y4])

                        r = sqrt((x2, y2), (x3, y3))
                        if r < alph:
                            new_cols_lines.append([x2, y2, x3, y3])
                        r = sqrt((x2, y2), (x4, y4))
                        if r < alph:
                            new_cols_lines.append([x2, y2, x4, y4])

        return new_rows_lines, new_cols_lines

    def get_table_ceilboxes2(self, img, prob, row=100, col=100, alph=50):
        """
        获取单元格
        """
        w, h = SIZE
        rows_lines, cols_lines = self.get_table_rowcols(img, prob, row, col)

        if rows_lines is not None:
            new_rows_lines, new_cols_lines = TableLineDetector.adjust_lines(rows_lines, cols_lines, alph=alph)
            rows_lines = new_rows_lines + rows_lines
            cols_lines = cols_lines + new_cols_lines

            nrow = len(rows_lines)
            ncol = len(cols_lines)

            for i in range(nrow):
                for j in range(ncol):
                    rows_lines[i] = line_to_line(rows_lines[i], cols_lines[j], 32)
                    cols_lines[j] = line_to_line(cols_lines[j], rows_lines[i], 32)
            return cols_lines, rows_lines
        else:
            return None, None

    def table_line_detect(self, cv_img, box=None):
        if box is None:
            h, w = cv_img.shape[:2]
            box = [0, 0, w, h]
        sub_img = cv_img[box[1]:box[3], box[0]:box[2]]

        img = Image.fromarray(cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB))
        cols_lines, rows_lines = self.get_table_ceilboxes2(img, prob=0.5, row=10, col=10, alph=10)
        return cols_lines, rows_lines


# if __name__ == '__main__':
#     import os
#     import time
#
#     # config = parser()
#     # p= config.jpgPath
#     p = "D:/data/img1105/14.png"
#     box = None
#     if os.path.exists(p):
#         start = time.time()
#         # img =Image.open(p).convert('RGB')
#         cv_img = cv2.imread(p)
#         # if box is None:
#         #     h, w = cv_img.shape[:2]
#         #     box = [0, 0, w, h]
#         # sub_img = cv_img[box[1]:box[3], box[0]:box[2]]
#         #
#         # img = Image.fromarray(cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB))
#         # ColsLines, RowsLines = get_table_ceilboxes2(img,prob=0.5,row=10,col=10,alph=10)
#         ColsLines, RowsLines = table_line_detect(cv_img)
#         print("table detection cast time:", time.time() - start)
#         print("ColsLines:", len(ColsLines), ColsLines)
#         print("RowsLines:", len(RowsLines), RowsLines)
