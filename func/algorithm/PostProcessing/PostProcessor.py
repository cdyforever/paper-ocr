# -*- coding: utf-8 -*-
# @Time    : 2020/11/16 11:36
# @Author  : wy09
# @File    : PostProcessingProcess.py
# @Software: PyCharm


import numpy as np
import json


# TODO: post process BUG?
class PostProcessor(object):

    def __call__(self,
                 table_result,
                 page_result,
                 pic_boxes,
                 img_region
                 ):
        merge_line_recognition_result = self.merge_line(page_result)
        package_result = self.result_package(pic_boxes, table_result,
                                             merge_line_recognition_result,
                                             (img_region[0], img_region[1]))
        return package_result

    @staticmethod
    def merge_line(ench_formula_recognition_result, overlap_rate=0.4):
        ench_formula_recognition_result = sorted(ench_formula_recognition_result, key=lambda e: e[0]["pos"][1])
        new_result = []
        for i in range(len(ench_formula_recognition_result)):
            block_res = ench_formula_recognition_result[i]
            cur_box = block_res[0]["pos"]
            new_result.append({"pos": cur_box, "line": block_res})
        return new_result

    @staticmethod
    def result_package(pic_box_list, tables_result, rec_result, relative_loc=(0, 0)):
        res = []
        l1 = len(rec_result)
        l2 = len(pic_box_list)
        l3 = len(tables_result)
        i1, i2, i3 = 0, 0, 0
        switch = -1
        while i1 != l1 or i2 != l2 or i3 != l3:
            if switch == -1 or switch == 1:
                if i1 == l1:
                    h1 = 1e6
                else:
                    e1 = rec_result[i1]
                    box = e1["pos"]
                    h1 = int(box[1] + relative_loc[1])
            if switch == -1 or switch == 2:
                if i2 == l2:
                    h2 = 1e6
                else:
                    e2 = pic_box_list[i2]
                    h2 = e2[1]
            if switch == -1 or switch == 3:
                if i3 == l3:
                    h3 = 1e6
                else:
                    e3 = tables_result[i3]
                    h3 = e3["pos"][1]
            idx = np.argmin([h1, h2, h3])
            if idx == 0:
                for block in e1["line"]:
                    xt1, yt1, xt2, yt2 = block["pos"]
                    xt1 += relative_loc[0]
                    yt1 += relative_loc[1]
                    xt2 += relative_loc[0]
                    yt2 += relative_loc[1]
                    block["pos"] = [xt1, yt1, xt2, yt2]
                res.append(e1["line"])
                switch = 1
                i1 += 1
            elif idx == 1:
                x1, y1, x2, y2 = e2
                x1 += relative_loc[0]
                y1 += relative_loc[1]
                x2 += relative_loc[0]
                y2 += relative_loc[1]
                res.append([{"pic": None, "pos": [x1, y1, x2, y2]}])
                switch = 2
                i2 += 1
            else:
                x1, y1, x2, y2 = e3["pos"]
                x1 += relative_loc[0]
                y1 += relative_loc[1]
                x2 += relative_loc[0]
                y2 += relative_loc[1]
                e3["pos"] = [x1, y1, x2, y2]
                res.append([e3])
                switch = 3
                i3 += 1
        return res

    @staticmethod
    def merge_box(box1, box2):
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return [x1, y1, x2, y2]

    @staticmethod
    def section_overlap(box1, box2):
        section1 = (box1[1], box1[3])
        section2 = (box2[1], box2[3])
        if section2[0] > section1[0] and section2[1] < section1[1]:
            return 1
        if section2[0] < section1[0] and section2[1] > section1[1]:
            return 1
        if section2[0] > section1[0] and section2[0] < section1[1]:
            overlap_len = section1[1] - section2[0]
            min_len = min(section1[1] - section1[0], section2[1] - section2[0])
            return overlap_len / min_len
        if section2[1] > section1[0] and section2[1] < section1[1]:
            overlap_len = section2[1] - section1[0]
            min_len = min(section1[1] - section1[0], section2[1] - section2[0])
            return overlap_len / min_len
        return 0

    @staticmethod
    def neaten_exception_result(content):
        state = content["state"]
        package_result = {}
        if state >= 1:
            pass
        if state >= 2:
            pass
        if state >= 3:
            package_result["pic_box_list"] = content["pic_box_list"]
            package_result["table_lines_set"] = content["table_lines_set"]
        if state >= 4:
            package_result["line_ench_formula_pos_list"] = content["line_ench_formula_pos_list"]
        if state >= 5:
            package_result["table_recognition_result"] = content["table_recognition_result"]
            package_result["ench_formula_recognition_result"] = content["ench_formula_recognition_result"]
        return package_result

    @staticmethod
    def record_cast_time(content):
        detail_cast_time = {}
        layout_cast_time = content.get("layout_cast_time")
        if layout_cast_time is not None:
            detail_cast_time["layout_cast_time"] = round(layout_cast_time, 3)
        table_pic_detect_cast_time = content.get("table_pic_detect_cast_time")
        if table_pic_detect_cast_time is not None:
            detail_cast_time["table_pic_detect_cast_time"] = round(table_pic_detect_cast_time, 3)
        english_chinese_split_cast_time = content.get("english_chinese_split_cast_time")
        if english_chinese_split_cast_time is not None:
            detail_cast_time["english_chinese_split_cast_time"] = round(english_chinese_split_cast_time, 3)
        text_recognition_cast_time = content.get("text_recognition_cast_time")
        if text_recognition_cast_time is not None:
            detail_cast_time["text_recognition_cast_time"] = round(text_recognition_cast_time, 3)
        post_process_cast_time = content.get("post_process_cast_time")
        if post_process_cast_time is not None:
            detail_cast_time["post_process_cast_time"] = round(post_process_cast_time, 3)
        return detail_cast_time
