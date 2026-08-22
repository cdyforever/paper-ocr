"""
    MySQL interface

    Author: Chen Yu
    Date  : 7/29/2021
"""

import base64
import pymysql

SEG_INSERT_SQL = """INSERT INTO segment (img_name, line_idx, bbox, label, text_type, text_str, submit_time) \
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP);"""
BBOX_INSERT_SQL = """INSERT INTO bbox (img_name, line_idx, bbox, text_type, subbox, submit_time) \
                     VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP);"""
RES_INSERT_SQL = """INSERT INTO ocr_result (img_name, result, submit_time) \
                    VALUES (%s, %s, CURRENT_TIMESTAMP);"""


class MySQLInterface(object):

    def __init__(self, host, port, user, pwd, db):
        self.conn = pymysql.connect(host=host, port=port, user=user, password=pwd,
                                    autocommit=True)
        cursor = self.conn.cursor()
        create_db_sql = "CREATE DATABASE IF NOT EXISTS %s" % db
        cursor.execute(create_db_sql)
        cursor.close()

        self.conn.select_db(db)
        self.cursor = self.conn.cursor()
        self._init()

    def _init(self):
        create_segment_sql = "CREATE TABLE IF NOT EXISTS segment(" \
                             "img_name VARCHAR (255), line_idx INT (50), bbox VARCHAR (255), " \
                             "label VARCHAR (255), text_type VARCHAR (255), text_str MEDIUMTEXT, " \
                             "submit_time DATETIME) CHARACTER SET UTF8"
        create_bbox_sql = "CREATE TABLE IF NOT EXISTS bbox(" \
                          "img_name VARCHAR (255), line_idx INT (50), bbox VARCHAR (255), " \
                          "text_type VARCHAR (255), subbox MEDIUMTEXT, " \
                          "submit_time DATETIME) CHARACTER SET UTF8"
        create_res_sql = "CREATE TABLE IF NOT EXISTS ocr_result(" \
                         "img_name VARCHAR (255), result MEDIUMTEXT, submit_time DATETIME) CHARACTER SET UTF8"
        self.cursor.execute(create_segment_sql)
        self.cursor.execute(create_bbox_sql)
        self.cursor.execute(create_res_sql)

    def write_segment_record(self, segment_record):
        submit_data = []
        for record in segment_record:
            text_str = base64.b64encode(str(record['text_str']).encode())
            bbox = str(record['bbox'])
            data = (record['img_name'], record['line_idx'], bbox,
                    record['label'], record['text_type'], text_str)
            submit_data.append(data)
        self.cursor.executemany(SEG_INSERT_SQL, submit_data)
        self.conn.commit()
        # TODO: try

    def write_bbox_record(self, bbox_records):
        submit_data = []
        for record in bbox_records:
            subbox = base64.b64encode(str(record['subbox']).encode())
            bbox = str(record['bbox'])
            data = (record['img_name'], record['line_idx'], bbox,
                    record['text_type'], subbox)
            submit_data.append(data)
        self.cursor.executemany(BBOX_INSERT_SQL, submit_data)
        self.conn.commit()

    def write_ocr_record(self, ocr_record):
        res = base64.b64encode(str(ocr_record['result']).encode())
        data = (ocr_record['img_name'], res)
        self.cursor.execute(RES_INSERT_SQL, data)
        self.conn.commit()
