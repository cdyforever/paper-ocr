"""
    Integrating distributed text into a completed subject

    Author: Chen Yu
    Date  : 7/13/2021
"""

from sklearn.cluster import DBSCAN
import numpy as np

import json


INT_MAX = 1e+7
INT_MIN = 0
ERROR_RATIO = 0.03
MAX_GAP_RATIO = 1.1

SUBJECT_NUM = '1234567890'

# hyper-parameter of DBSCAN
DBSCAN_MIN_SAMPLES = 2


class OCRResultInfo(object):

    def __init__(self, page_ocr_results):
        self.page_ocr_results = page_ocr_results
        self.rows_x_scope = []       # 每行box的x轴跨度
        self.rows_y_scope = []       # 每行box的y轴跨度
        self.rows_h = []             # 每行box的高度
        self.rows_begin_clusters = []

        self.avg_h = 0               # 平均行高
        self.x_error = 0             # x轴box位置可容忍误差
        self.page_x_scope = [0, 0]   # 整页x轴跨度

        self._init(page_ocr_results)

    def _init(self, page_ocr_results):
        rows_x_scope, rows_y_scope = [], []
        for row_ocr_results in page_ocr_results:
            # get margin x coordinates of each row, [left, right]
            row_x_scope, row_y_scope = OCRResultInfo._get_row_scope(row_ocr_results)
            rows_x_scope.append(row_x_scope)
            rows_y_scope.append(row_y_scope)
        self.rows_x_scope = np.asarray(rows_x_scope)
        self.rows_y_scope = np.asarray(rows_y_scope)
        self.rows_box = np.concatenate([self.rows_x_scope, self.rows_y_scope], axis=1)[:, [0, 2, 1, 3]]
        self.rows_h = np.asarray([y[1] - y[0] for y in self.rows_y_scope], dtype=np.int32)
        self.avg_h = OCRResultInfo._get_avg_h(self.rows_h)
        self.x_error, self.page_x_scope = OCRResultInfo._get_effective_error(self.rows_x_scope)
        self.rows_begin_clusters = OCRResultInfo._begin_point_clustering(self.rows_x_scope[:, 0], self.x_error)

    @staticmethod
    def _get_row_scope(row_results):
        x_scope = np.asarray([INT_MAX, INT_MIN], dtype=np.int32)
        y_scope = np.asarray([INT_MAX, INT_MIN], dtype=np.int32)
        for row_elem in row_results:
            pos = np.asarray(row_elem['pos'], dtype=np.int32)
            x_scope[0] = np.min([pos[0], x_scope[0]])
            x_scope[1] = np.max([pos[2], x_scope[1]])
            y_scope[0] = np.min([pos[1], y_scope[0]])
            y_scope[1] = np.max([pos[3], y_scope[1]])
        return x_scope, y_scope

    @staticmethod
    def _get_avg_h(ocr_h):
        q1 = np.percentile(ocr_h, 30)
        q3 = np.percentile(ocr_h, 70)
        iqr = q3 - q1
        effective_h = []
        for h in ocr_h:
            if (q3 + .25 * iqr) > h > (q1 - .25 * iqr):
                effective_h.append(h)
        avg_h = np.nanmean(effective_h).astype(np.int32)
        return avg_h

    @staticmethod
    def _get_effective_error(ocr_scope):
        ratio = ERROR_RATIO
        ocr_scope_np = np.asarray(ocr_scope)
        right_scope = np.max(ocr_scope_np[:, 1])
        left_scope = np.min(ocr_scope_np[:, 0])
        return np.int32(ratio * (right_scope - left_scope)), [left_scope, right_scope]

    @staticmethod
    def _begin_point_clustering(rows_x_left_scope, x_error):
        x = rows_x_left_scope.reshape(-1, 1)
        pred = DBSCAN(eps=x_error, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(x)
        return pred

    def __getitem__(self, item):
        return self.page_ocr_results[item], self.rows_x_scope[item], self.rows_y_scope[item]

    def __len__(self):
        return len(self.page_ocr_results)


class SubjectMaker(object):

    def __init__(self):
        pass

    def _is_option_region(self):
        pass

    def _is_last_row_in_subject(self):
        pass

    def __call__(self, page_ocr_results):
        ocr_result_info = OCRResultInfo(page_ocr_results)
        subject_cluster_idx = _get_subject_cluster(ocr_result_info)
        subjects_idx = _get_subject_idx_by_cluster_idx(ocr_result_info, subject_cluster_idx)
        subjects_box, subjects_contained_row_idx = _concatenate_rows2subjects(ocr_result_info, subjects_idx)
        content = _convert2json(ocr_result_info, subjects_box, subjects_contained_row_idx)
        return content


def _get_subject_cluster(ocr_result_info):
    clusters = ocr_result_info.rows_begin_clusters
    cls_idx = np.unique(clusters[clusters >= 0])
    max_cnt, subject_cluster = 0, 0
    for cls in cls_idx:
        cnt = 0
        results_idx = np.where(clusters == cls)[0]
        for result_idx in results_idx:
            result = ocr_result_info.page_ocr_results[result_idx][0]
            if 'ench' in result.keys():
                if _is_subject_begin(result['ench'].replace(' ', '')):
                    cnt += 1
        if cnt > max_cnt:
            max_cnt, subject_cluster = cnt, cls
    return subject_cluster


def _is_subject_begin(text):
    content = text.replace(' ', '')
    subject_num = SUBJECT_NUM
    if len(content) > 0:
        return content[0] in subject_num
    else:
        return False


def _get_subject_idx_by_cluster_idx(ocr_result_info, cluster_idx):
    subjects_idx = [row_idx for row_idx in np.where(ocr_result_info.rows_begin_clusters == cluster_idx)[0] if
                    _is_subject_begin(ocr_result_info.page_ocr_results[row_idx][0].get('ench', '#'))]
    return subjects_idx


def _concatenate_serial_rows(rows_box, head_idx, rear_idx, avg_h):
    row_max_gap = avg_h * MAX_GAP_RATIO
    target_box = np.asarray(rows_box[head_idx])
    target_contained_row_idx = [head_idx]
    cur_y_end = rows_box[head_idx][3]
    cur_idx = head_idx
    while cur_idx + 1 < rear_idx:
        next_idx = cur_idx + 1
        next_y_start = rows_box[next_idx][1]
        if next_y_start - cur_y_end > row_max_gap:
            break
        else:
            cur_idx = next_idx
            cur_y_end = rows_box[next_idx][3]
            target_contained_row_idx.append(next_idx)
    target_box[3] = cur_y_end
    return target_box, target_contained_row_idx


def _concatenate_rows2subjects(ocr_result_info, subjects_idx):
    rows_box = ocr_result_info.rows_box
    avg_h = ocr_result_info.avg_h
    subjects_box = []
    subjects_contained_row_idx = []
    for num, subject_idx in enumerate(subjects_idx):
        # update end pos of each subject box
        head_idx = subject_idx
        if num + 1 >= len(subjects_idx):
            rear_idx = len(rows_box)
        else:
            rear_idx = subjects_idx[num + 1]
        subject_box, subject_contained_row_idx = _concatenate_serial_rows(rows_box, head_idx, rear_idx, avg_h)
        subjects_box.append(subject_box)
        subjects_contained_row_idx.append(subject_contained_row_idx)
    return subjects_box, subjects_contained_row_idx


def _form_subject_json(num, content, box):
    subject_msg = {
        'No': num,
        'content': content,
        'box': {
            'start_x': str(box[0]),
            'start_y': str(box[1]),
            'end_x': str(box[2]),
            'end_y': str(box[3]),
        }
    }
    return subject_msg


def _convert2json(ocr_result_info, subjects_box, subjects_contained_row_idx):

    page_ocr_results = ocr_result_info.page_ocr_results
    subjects_msg = []
    for num, subject_contained_row_idx in enumerate(subjects_contained_row_idx):
        content = ''
        box = subjects_box[num]
        for row_idx in subject_contained_row_idx:
            for text in page_ocr_results[row_idx]:
                if 'ench' in text.keys():
                    content += text['ench']
                elif 'formula' in text.keys():
                    content += text['formula']
                content += ' '
        subjects_msg.append(_form_subject_json(num, content, box))
    return subjects_msg
