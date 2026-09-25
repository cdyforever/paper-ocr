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

# -------------- PURE ONNX OCR (no PaddlePaddle) -------------- #
# Converted from PaddleOCR PP-OCRv4 via paddle2onnx.
# See convert_paddleocr_to_onnx.py
PURE_ONNX_OCR = True

ONNX_DET_MODEL_PATH = os.path.join(root_path,
                                   'weights/text_line_detect/dbnet_det.onnx')
ONNX_REC_MODEL_PATH = os.path.join(root_path,
                                   'weights/text_recognition/crnn_rec.onnx')
ONNX_CLS_MODEL_PATH = os.path.join(root_path,
                                   'weights/text_recognition/cls.onnx')
ONNX_REC_DICT_PATH = os.path.join(root_path,
                                  'weights/text_recognition/ppocr_keys_v1.txt')

# -------------- PURE ONNX FORMULA DETECTION (MFD) -------------- #
# YOLOv8 from MinerU / PDF-Extract-Kit, exported to ONNX.
# Classes: 0=embedding (inline formula), 1=isolated (display formula)
FORMULA_DETECT_ONNX_PATH = os.path.join(
    root_path, 'weights/formula_detect/mfd_yolov8_fp16.onnx')
USE_FORMULA_DETECT_ONNX = True

# -------------- PURE ONNX FORMULA RECOGNITION (MFR) -------------- #
# UniMERNet-tiny (VisionEncoderDecoder: VariableUnimerNet + custom mBART
# with MBartSqueezeAttention), exported as separate encoder/decoder ONNX.
# See export_unimernet_onnx.py
FORMULA_RECOGNIZE_ONNX_DIR = os.path.join(
    root_path, 'weights/formula_recognize/onnx_fp16')
FORMULA_RECOGNIZE_TOKENIZER_DIR = os.path.join(
    root_path, 'weights/formula_recognize/unimernet_tiny')
USE_FORMULA_RECOGNIZE_ONNX = True

# -------------- PAGE-LEVEL DECOUPLED PIPELINE -------------- #
# DBNet 是为「横向文本行」训练的, 公式 (分式/根式/上下标/积分号) 不是它的
# 目标形态。实测: 公式区域平均只有 43.9% 的面积落在 DBNet 行框内,
# 31 个公式里 17 个 (55%) 完全丢失; 行内 MFD 因此只能找到 6/31。
#
# 因此把顺序反过来: 页面级 MFD 先检测公式 -> 涂白 -> DBNet 只切纯文本行。
# 见 func/algorithm/onnx_page_pipeline.py
USE_PAGE_LEVEL_DECOUPLING = True
# 公式区域涂白时向外扩的像素数 (防止笔画残留被 DBNet 当成文本)
PAGE_MASK_MARGIN = 6
# 文本行落在公式内的面积占比超过该值则丢弃 ('filter' 模式使用)
PAGE_DROP_LINE_FORMULA_RATIO = 0.5

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
