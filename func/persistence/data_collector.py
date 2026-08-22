"""
    Data collector
    This module is used to saving OCR results of each stage to analyse and improve algorithms

    Author: Chen Yu
    Date  : 7/29/2021
"""

from .db_interface.mysql import MySQLInterface
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PWD, MYSQL_DATABASE, \
    PIC_COLLECT_DIR

import os
import uuid
import cv2
import time


class DataCollector(object):

    def __init__(self, host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, pwd=MYSQL_PWD, db=MYSQL_DATABASE,
                 pic_save_dir=PIC_COLLECT_DIR
                 ):
        self.mysql = MySQLInterface(host, port, user, pwd, db)
        self.pic_save_dir = pic_save_dir
        os.makedirs(self.pic_save_dir, exist_ok=True)

    def _collect_segment_recognition_result(self, img_name, page_result):
        segment_record = page_result2segment_record(img_name, page_result)
        self.mysql.write_segment_record(segment_record)

    def _collect_bbox_detection_result(self, img_name, page_result):
        bbox_records = page_result2bbox_record(img_name, page_result)
        self.mysql.write_bbox_record(bbox_records)

    def _collect_ocr_result(self, img_name, combined_msg):
        ocr_record = {
            'time': get_cur_time(),
            'img_name': img_name,
            'result': combined_msg
        }
        self.mysql.write_ocr_record(ocr_record)

    def collect_ocr_result(self, img, combined_msg, page_result):
        img_id = produce_image_id()
        img_save_name = img_id + '.jpg'
        cv2.imwrite(os.path.join(self.pic_save_dir, img_save_name), img)
        self._collect_bbox_detection_result(img_save_name, page_result=page_result)
        self._collect_segment_recognition_result(img_save_name, page_result=page_result)
        self._collect_ocr_result(img_save_name, combined_msg)
        return img_save_name


def page_result2bbox_record(img_name, page_result):
    bbox_record = []
    for line_idx, text_line_result in enumerate(page_result):
        record = dict()
        record['text_type'] = 'line'
        record['bbox'] = text_line_result.text_line_pos
        record['line_idx'] = line_idx
        record['img_name'] = img_name
        record['time'] = get_cur_time()
        record['subbox'] = []
        if text_line_result.is_by_mathpix_completed is False:
            for text_str, text_type, text_label, text_pos in text_line_result:
                record['subbox'].append((text_type, text_pos))
        bbox_record.append(record)
    return bbox_record


def page_result2segment_record(img_name, page_result):
    segment_record = []
    for line_idx, text_line_result in enumerate(page_result):
        for text_str, text_type, text_label, text_pos in text_line_result:
            record = dict()
            if text_line_result.is_by_mathpix_completed:
                record['text_type'] = 'line'
            else:
                record['text_type'] = text_type
            record['text_str'] = text_str
            record['bbox'] = text_pos
            record['label'] = text_label
            record['line_idx'] = line_idx
            record['img_name'] = img_name
            record['time'] = get_cur_time()
            segment_record.append(record)
    return segment_record


def produce_image_id():
    return str(uuid.uuid1().hex)


def produce_segment_id():
    return str(uuid.uuid1().hex)


def get_cur_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
