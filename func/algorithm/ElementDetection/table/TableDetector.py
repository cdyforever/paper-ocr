# -*- coding: utf-8 -*-
# @Time    : 2020/9/29 10:56
# @Author  : wy09
# @File    : TableDetection.py
# @Software: PyCharm


import numpy as np
from .table import TableLineDetector


class TableDetector(object):

    # def __init__(self, model_dir, device='cpu'):
    #     self.table_line_detector = TableLineDetector(model_dir, device)
    def __init__(self, model_dir):
        self.table_line_detector = TableLineDetector(model_dir)

    def detect(self, img, box=None):
        cols_lines, rows_lines = self.table_line_detector.table_line_detect(img, box)
        if cols_lines is None or rows_lines is None:
            return []
        if len(cols_lines) < 3 or len(rows_lines) < 3:
            return []
        table_lines_set = TableDetector.calculate_table_lines_set(cols_lines, rows_lines)
        if len(table_lines_set) == 0:
            return []
        else:
            return table_lines_set

    @staticmethod
    def calculate_table_lines_set(col_lines, row_lines):
        col_mat = np.asarray(col_lines)
        row_mat = np.asarray(row_lines)
        col_mat = col_mat[np.argsort(col_mat[:, 0])]
        row_mat = row_mat[np.argsort(row_mat[:, 1])]
        mat = np.zeros([len(col_mat), len(row_mat)])
        for i in range(len(col_mat)):
            for j in range(len(row_mat)):
                t = TableDetector.intersect(col_mat[i], row_mat[j])
                if t:
                    mat[i, j] = 1
        line_segment_list = []
        for i in range(len(col_mat)):
            row_set = set(np.where(mat[i] == 1)[0])
            col_set = set()
            for r_ind in row_set:
                t = mat[:, r_ind]
                temp_col_set = set(np.where(t == 1)[0])
                col_set = col_set.union(temp_col_set)
                mat[:, r_ind] = 0
            mat[i] = 0
            if len(row_set) > 0 and len(col_set) > 0:
                line_segment_list.append((row_set, col_set))
            if (mat == 0).all():
                break
        # 合并部分没有检测到的表格线
        while 1:
            n = len(line_segment_list)
            flag = False
            t1, t2 = 0, 0
            for i in range(n):
                j = i + 1
                while j < n:
                    set1 = line_segment_list[i][1] & line_segment_list[j][1]
                    set2 = line_segment_list[i][0] & line_segment_list[j][0]
                    if len(set1) > 0 or len(set2) > 0:
                        t1 = i
                        t2 = j
                        flag = True
                        break
                    j += 1
            if flag:
                s1 = line_segment_list[t1][0] | line_segment_list[t2][0]
                s2 = line_segment_list[t1][1] | line_segment_list[t2][1]
                line_segment_list.pop(t1)
                line_segment_list.pop(t2 - 1)
                line_segment_list.append((s1, s2))
            else:
                break
        # 表格判断条件一：横与行必须又三条或以上的线段
        temp_line_segment_list = []
        for i in range(len(line_segment_list)):
            tem = line_segment_list[i]
            if len(tem[0]) > 2 and len(tem[1]) > 2:
                temp_line_segment_list.append(line_segment_list[i])
        line_segment_list = temp_line_segment_list
        # 确定表格区域
        # 表格判断条件二：左右两端线段近似相等于H，上下两端的线段近似相等于W
        table_lines_set = []
        for e in line_segment_list:
            row_ind = sorted(list(e[0]))
            row_lines = row_mat[row_ind]
            col_ind = sorted(list(e[1]))
            col_lines = col_mat[col_ind]
            x_min = min(np.min(col_lines[:, 0]), np.min(col_lines[:, 2]), np.min(row_lines[:, 0]),
                        np.min(row_lines[:, 2]))
            x_max = max(np.max(col_lines[:, 0]), np.max(col_lines[:, 2]), np.max(row_lines[:, 0]),
                        np.max(row_lines[:, 2]))
            y_min = min(np.min(col_lines[:, 1]), np.min(col_lines[:, 3]), np.min(row_lines[:, 1]),
                        np.min(row_lines[:, 3]))
            y_max = max(np.max(col_lines[:, 1]), np.max(col_lines[:, 3]), np.max(row_lines[:, 1]),
                        np.max(row_lines[:, 3]))
            w = x_max - x_min
            h = y_max - y_min
            t_w = (row_lines[0][2] - row_lines[0][0])
            d_w = (row_lines[-1][2] - row_lines[-1][0])
            l_h = (col_lines[0][3] - col_lines[0][1])
            r_h = (col_lines[-1][3] - col_lines[-1][1])
            if 0.9 < t_w / w < 1.1 and 0.9 < d_w / w < 1.1 and 0.9 < l_h / h < 1.1 and 0.9 < r_h / h < 1.1:
                table_lines_set.append((row_lines, col_lines, [x_min, y_min, x_max, y_max]))
        return table_lines_set

    @staticmethod
    def intersect(col_line, row_line, th=-5):
        cw = (col_line[0] + col_line[2]) / 2
        rh = (row_line[1] + row_line[3]) / 2
        min_w = min(row_line[0], row_line[2])
        max_w = max(row_line[0], row_line[2])
        min_h = min(col_line[1], col_line[3])
        max_h = max(col_line[1], col_line[3])
        if cw - min_w > th and max_w - cw > th:
            if rh - min_h > th and max_h - rh > th:
                return True
            else:
                return False
        else:
            return False
