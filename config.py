"""
    Global configuration for Canpoint OCR server

    Author: Chen Yu
    Date  : 7/29/2021
"""

import os
import sys


# -------------- ALGORITHM CONFIG -------------- #
# the sys path is needed for YOLO5 module
root_path = os.getcwd()
sys.path.insert(0, root_path + "/func/algorithm/ElementDetection/picture")
WEIGHTS_DIR = os.path.join(os.getcwd(), 'func/algorithm')
# element detection
TABLE_DETECT_MODEL_DIR = os.path.join(WEIGHTS_DIR, 'weights/element_detect_model/table_model')
PICTURE_DETECT_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'weights/element_detect_model/image_detect.onnx')
# text line division
PDF_TEXT_LINE_DIVIDE_MODEL_PATH = os.path.join(WEIGHTS_DIR,
                                               'weights/text_line_divide_model/dbnet_pdfv2.onnx')
PHOTO_TEXT_LINE_DIVIDE_MODEL_PATH = os.path.join(WEIGHTS_DIR,
                                                 'weights/text_line_divide_model/dbnet_photo.onnx')
# text recognition
PDF_FORMUlA_DETECT_MODEL_PATH = \
    os.path.join(WEIGHTS_DIR, 'weights/text_recognize_model/pdf_formula_detect/latex-detect-v2.onnx')
PHOTO_FORMULA_DETECT_MODEL_DIR = os.path.join(WEIGHTS_DIR,
                                              'weights/text_recognize_model/photo_formula_detect')
FORMULA_RECOGNIZE_MODEL_DIR = os.path.join(WEIGHTS_DIR, 'weights/text_recognize_model/formula_recognize')
ENCH_RECOGNIZE_MODEL_DIR = os.path.join(WEIGHTS_DIR, 'weights/text_recognize_model/ench_recognize')

MATHPIX_AUTH = 'Bearer _xTyBIGWYJzJeGgwlzOC1UIpnuQBaoBRuooBFOsMj8bBtNXf5XYs9yTu8F1bob95mqUyKoWC3PPvESQI0_9SRQ'
ENABLE_MATHPIX = False
DEVICE = 'cpu'
CALL_LIMIT = 20


# -------------- DATA PRESISTENCE CONFIG -------------- #
# mysql
MYSQL_HOST = "123.60.217.149"
MYSQL_PORT = 9920
MYSQL_USER = "root"
MYSQL_PWD = ""
MYSQL_DATABASE = "canpoint_ocr"
PIC_COLLECT_DIR = os.path.join(os.getcwd(), 'data/pictures')


# -------------- SERVER -------------- #
WEIGHTS_DOWNLOAD_URL = 'http://123.60.217.149/weights.zip'
PORT = 9911
