import numpy as np
import cv2

from .ench.ench_recognize import EnchRecognizeModel
from .formula_recognize.formula_rec_module import FormulaRecognizeModel
from .formula_recognize.mathpixapi import MathPixAPI
from .formula_detect.formula_detect_pdf import detectFormula as detectFormula_PDF
from .formula_detect.formula_detect_photo import detectFormula as detectFormula_Photo
from .formula_detect.formula_detect_infer import PDFFormulaDetectModel
from .ResStructure import TextLineRecognizeResult


class TextRecognizer(object):

    def __init__(self,
                 pdf_formula_detect_model_path,
                 photo_formula_detect_model_dir,
                 formula_recognize_model_dir,
                 ench_recognize_model_dir
                 ):
        self.mathpix_api = MathPixAPI()
        self.pdf_formula_detect_model = PDFFormulaDetectModel(pdf_formula_detect_model_path)
        # self.photo_formula_detect_model = PhotoFormulaDetectModel(photo_formula_detect_model_dir)
        self.formula_recognize_model = FormulaRecognizeModel(formula_recognize_model_dir)
        self.chn_eng_recognize_model = EnchRecognizeModel(ench_recognize_model_dir)
        self.photo_formula_detect_model = self.chn_eng_recognize_model

    def predict_pdf(self, result_rotate_list, median_height, ench_formula_list, enable_mathpix=False):
        page_result = \
            self._pdf_text_line_recognize(median_height, ench_formula_list, result_rotate_list,
                                          enable_mathpix=enable_mathpix)
        return page_result

    def predict_photo(self, result_rotate_list, median_height, ench_formula_list, enable_mathpix=False):
        page_result = \
            self._photo_text_line_recognize(median_height, ench_formula_list, result_rotate_list,
                                            enable_mathpix=enable_mathpix)
        return page_result

    def predict_table(self, table_img_list, table_lines_set):
        table_result = \
            self._table_text_recognition(table_img_list, table_lines_set)
        return table_result

    def _table_text_recognition(self, table_imgs, table_lines):
        table_result = []
        for ind, table_line in enumerate(table_lines):
            b = table_line[2]
            table_img = table_imgs[ind]
            cell_info = self._cell_detect(table_line[0], table_line[1])
            for cell in cell_info:
                box = cell["box"]
                if box[3] - box[1] <= 0 or box[2] - box[0] <= 0:
                    continue
                x1, y1, x2, y2 = box[0] - b[0], box[1] - b[1], box[2] - b[0], box[3] - b[1]
                cell_img = table_img[int(y1):int(y2), int(x1):int(x2)]
                new_img = self._vertical2horizontal(cell_img)
                res = self.chn_eng_recognize_model.predict(new_img)
                cell["res"] = [{"ench": res, "pos": box}]
            table_result.append({"table": cell_info, "pos": b})
        return table_result

    def _pdf_text_line_recognize(self, median_height, ench_formula_list, result_rotate_list, enable_mathpix=False):
        page_result = []
        for idx, ench_formula in enumerate(ench_formula_list):
            box = cv2.boxPoints(result_rotate_list[idx])
            box = np.int0(box)
            x, y, w, h = cv2.boundingRect(box)
            pos = [x, y, x + w, y + h]
            segments_pos = detectFormula_PDF(ench_formula, self.pdf_formula_detect_model, median_height)
            text_line_result = self._recognize_text_line(ench_formula, segments_pos, median_height, pos,
                                                         enable_mathpix=enable_mathpix)
            if len(text_line_result) > 0:
                page_result.append(text_line_result)
        return page_result

    def _photo_text_line_recognize(self, median_height, ench_formula_list, result_rotate_list, enable_mathpix=False):
        page_result = []
        for idx, ench_formula in enumerate(ench_formula_list):
            box = cv2.boxPoints(result_rotate_list[idx])
            box = np.int0(box)
            x, y, w, h = cv2.boundingRect(box)
            pos = [x, y, x + w, y + h]
            segments_pos = detectFormula_Photo(ench_formula, self.photo_formula_detect_model, median_height)
            text_line_result = self._recognize_text_line(ench_formula, segments_pos, median_height, pos,
                                                         enable_mathpix=enable_mathpix)
            if len(text_line_result) > 0:
                page_result.append(text_line_result)
        return page_result

    def _recognize_text_line(self, ench_formula, segments_pos, median_height, text_line_pos,
                             enable_mathpix=False):
        if enable_mathpix is True:
            is_success, text_line_result = self._recognize_complete_text_by_mathpix(ench_formula, text_line_pos)
            if is_success is False:
                text_line_result = \
                    self._recognize_segmented_text(ench_formula, text_line_pos, segments_pos, median_height,
                                                   enable_mathpix=enable_mathpix)
        else:
            text_line_result = \
                self._recognize_segmented_text(ench_formula, text_line_pos, segments_pos, median_height,
                                               enable_mathpix=enable_mathpix)
        return text_line_result

    def _recognize_segmented_text(self, ench_formula, text_line_pos, segments_pos, median_height,
                                  enable_mathpix=False):
        text_line_result = TextLineRecognizeResult(text_line_pos, is_completed_by_mathpix=False)
        for seg_idx, (pos, text_type) in enumerate(segments_pos):
            x1, y1, x2, y2 = pos
            zone = ench_formula[y1: y2, x1:  x2]
            text_label = 'native'
            if text_type == 'formula':
                if enable_mathpix is True:
                    res = self.mathpix_api.get_result(zone)
                    is_success = False if res is None else True
                    if is_success:
                        # print('seg %d : mathpix' % seg_idx)
                        # print(res)
                        text_label = 'mathpix'
                        seg_results_split, labels = TextRecognizer._split_ench_formula(res)
                        for seg_result, seg_type in zip(seg_results_split, labels):
                            text_line_result.push_back(text_str=seg_result,
                                                       text_type=seg_type,
                                                       text_label='mathpix',
                                                       text_pos=text_line_pos)
                    else:
                        res = self.formula_recognize_model.predict(zone, median_height)
                else:
                    res = self.formula_recognize_model.predict(zone, median_height)
            elif text_type == 'ench':
                res = self.chn_eng_recognize_model.predict(zone)
            else:
                res = u''
            # add result when recognized not by mathpix
            if text_label != 'mathpix':
                # print('seg %d : native' % seg_idx)
                # print(res)
                text_line_result.push_back(text_str=res,
                                           text_type=text_type,
                                           text_label=text_label,
                                           text_pos=pos)
        return text_line_result

    def _recognize_complete_text_by_mathpix(self, ench_formula, text_line_pos):
        res = self.mathpix_api.get_result(ench_formula)
        # print('complete : mathpix')
        # print(res)
        is_success = False if res is None else True
        text_line_result = TextLineRecognizeResult(text_line_pos, is_completed_by_mathpix=True)
        if is_success:
            seg_results_split, labels = TextRecognizer._split_ench_formula(res)
            for seg_result, seg_type in zip(seg_results_split, labels):
                text_line_result.push_back(text_str=seg_result,
                                           text_type=seg_type,
                                           text_label='mathpix',
                                           text_pos=text_line_pos)
        return is_success, text_line_result

    @staticmethod
    def _split_ench_formula(str_):
        index = 0
        pos = []
        while True:
            index = str_.find("$", index)
            if index == -1:
                break
            pos.append(index)
            index += 1
        split_str = []
        str_label = []
        if len(pos) == 0 or len(pos) == 1:
            return [str_], ['ench']
        if len(pos) % 2 == 0:
            arr = []
            i = 0
            while i < len(pos):
                arr.append([pos[i], pos[i + 1]])
                i += 2
            flag = 0
            for i in range(len(arr)):
                start = arr[i][0]
                end = arr[i][1]
                if 0 < start != flag:
                    str_label.append('ench')
                    split_str.append(str_[flag:start])
                str_label.append('formula')
                split_str.append(str_[start + 1:end])
                flag = end + 1
            if arr[-1][1] < len(str_) - 1:
                str_label.append('ench')
                split_str.append(str_[flag:len(str_)])
            return split_str, str_label
        else:
            return [str_], ['ench']

    @staticmethod
    def _cell_detect(row_lines, col_lines, padding=5):
        min_x = min(np.min(row_lines[:, [0, 2]]), np.min(col_lines[:, [0, 2]]))
        min_y = min(np.min(row_lines[:, [1, 3]]), np.min(col_lines[:, [1, 3]]))
        row_lines[:, [0, 2]] -= min_x
        row_lines[:, [1, 3]] -= min_y
        col_lines[:, [0, 2]] -= min_x
        col_lines[:, [1, 3]] -= min_y
        row_lines = np.int32(row_lines)
        col_lines = np.int32(col_lines)
        max_x = max(np.max(row_lines[:, [0, 2]]), np.max(col_lines[:, [0, 2]]))
        max_y = max(np.max(row_lines[:, [1, 3]]), np.max(col_lines[:, [1, 3]]))
        white_img = np.ones([max_y + 2 * padding, max_x + 2 * padding], dtype=np.uint8) * 255
        for e in row_lines:
            cv2.line(white_img, (e[0] + padding, e[1] + padding), (e[2] + padding, e[3] + padding), (0, 0, 0),
                     thickness=3)
        for e in col_lines:
            cv2.line(white_img, (e[0] + padding, e[1] + padding), (e[2] + padding, e[3] + padding), (0, 0, 0),
                     thickness=3)
        n, labels = cv2.connectedComponents(white_img)
        area = (max_y + 2 * padding) * (max_x + 2 * padding)
        box_list = []
        for i in range(n):
            if i == 0:
                continue
            y, x = np.where(labels == i)
            point = np.asarray([y, x])
            rect = cv2.minAreaRect(point.T)
            box = cv2.boxPoints(rect)
            x1 = np.min(box[:, 1])
            y1 = np.min(box[:, 0])
            x2 = np.max(box[:, 1])
            y2 = np.max(box[:, 0])
            if rect[1][0] * rect[1][1] > 0.95 * area:
                continue
            else:
                # 避免与单元格边界重合
                box_list.append([x1 + padding, y1 + padding, x2 - padding, y2 - padding])
        box_info = []
        for e in box_list:
            tx1 = 0
            min_tx1 = 1e6
            tx2 = 0
            min_tx2 = 1e6
            ty1 = 0
            min_ty1 = 1e6
            ty2 = 0
            min_ty2 = 1e6
            for ind, row in enumerate(row_lines):
                t1 = abs((row[1] + row[3]) / 2 - e[1])
                t2 = abs((row[1] + row[3]) / 2 - e[3])
                if t1 < min_tx1:
                    tx1 = ind
                    min_tx1 = t1
                if t2 < min_tx2:
                    tx2 = ind
                    min_tx2 = t2
            for ind, col in enumerate(col_lines):
                t1 = abs((col[0] + col[2]) / 2 - e[0])
                t2 = abs((col[0] + col[2]) / 2 - e[2])
                if t1 < min_ty1:
                    ty1 = ind
                    min_ty1 = t1
                if t2 < min_ty2:
                    ty2 = ind
                    min_ty2 = t2
            e[0] += min_x
            e[1] += min_y
            e[2] += min_x
            e[3] += min_y
            for i in range(len(e)):
                e[i] = int(e[i])
            box_info.append({"cols": list(range(ty1, ty2)), "rows": list(range(tx1, tx2)), "box": e})
        return box_info

    @staticmethod
    def _vertical2horizontal(img):
        ih, iw = img.shape[:2]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 13)
        y, x = np.where(binary_img < 10)
        if len(x) < 10:
            return img
        x1 = np.min(x)
        x2 = np.max(x)
        y1 = np.min(y)
        y2 = np.max(y)
        x1 = max(x1 - 1, 0)
        x2 = min(x2 + 1, iw)
        y1 = max(y1 - 1, 0)
        y2 = min(y2 + 1, ih)
        h = y2 - y1
        w = x2 - x1
        sub_img = img[y1:y2, x1:x2]
        if h > w:
            mat1 = binary_img[y1:y2, x1:x2]
            mat1 = (255 - mat1) / 255
            col_sum = np.sum(mat1, axis=1)
            s0 = -1
            l = len(col_sum)
            i = 0
            segment_list = []
            max_h = -1
            while i < l:
                if col_sum[i] > 1:
                    s0 = i
                    while i < l and col_sum[i] > 1:
                        i += 1
                    segment_list.append((s0, i))
                    max_h = max(max_h, i - s0)
                i += 1
            # 合并上下结构的字 （待）
            # 旋转括号等小型字符 （待）
            p_lr = int(0.1 * w)
            joint_img = None
            for e in segment_list:
                img_t = sub_img[e[0]:e[1], :]
                h_t = e[1] - e[0]
                p_h = max_h - h_t
                p_top = int(p_h / 2)
                p_down = p_h - p_top
                img_t = cv2.copyMakeBorder(img_t, p_top, p_down, p_lr, p_lr, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                if joint_img is None:
                    joint_img = img_t
                else:
                    joint_img = np.hstack((joint_img, img_t))
            return joint_img
        else:
            return sub_img
