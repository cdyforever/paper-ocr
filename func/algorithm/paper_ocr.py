"""
    Paper OCR
    This module integrate all parts of OCR algorithm, providing a calling method for user

    Author: Chen Yu
    Date  : 7/29/2021
"""

from .ElementDetection import TableElementDetector, PictureElementDetector
from .TextLineDivision import PDFTextLineDetector, PhotoTextLineDetector
from .TextRecognition import TextRecognizer
from .PostProcessing import PostProcessor, SubjectMaker
from config import DEVICE, \
    TABLE_DETECT_MODEL_DIR, PICTURE_DETECT_MODEL_PATH, \
    PDF_TEXT_LINE_DIVIDE_MODEL_PATH, PHOTO_TEXT_LINE_DIVIDE_MODEL_PATH, \
    PDF_FORMUlA_DETECT_MODEL_PATH, PHOTO_FORMULA_DETECT_MODEL_DIR, \
    FORMULA_RECOGNIZE_MODEL_DIR, ENCH_RECOGNIZE_MODEL_DIR, \
    ENABLE_MATHPIX

import time


class PaperOCR(object):

    def __init__(self):
        self.table_detector = TableElementDetector(TABLE_DETECT_MODEL_DIR)
        self.picture_detector = PictureElementDetector(PICTURE_DETECT_MODEL_PATH, DEVICE)
        # TODO: ONNX runtime device option
        self.pdf_text_line_detector = PDFTextLineDetector(PDF_TEXT_LINE_DIVIDE_MODEL_PATH)
        self.photo_text_line_detector = PhotoTextLineDetector(PHOTO_TEXT_LINE_DIVIDE_MODEL_PATH)
        self.text_recognizer = TextRecognizer(
            pdf_formula_detect_model_path=PDF_FORMUlA_DETECT_MODEL_PATH,
            photo_formula_detect_model_dir=PHOTO_FORMULA_DETECT_MODEL_DIR,
            formula_recognize_model_dir=FORMULA_RECOGNIZE_MODEL_DIR,
            ench_recognize_model_dir=ENCH_RECOGNIZE_MODEL_DIR
        )
        self.post_processor = PostProcessor()
        self.subject_maker = SubjectMaker()

    def predict_subject(self, img, img_type, enable_mathpix=ENABLE_MATHPIX):
        combined_msg, page_result, cost_time = self.predict(img, img_type, enable_mathpix)
        combined_msg = self.subject_maker(combined_msg)
        return combined_msg, page_result, cost_time

    def predict(self, img, img_type, enable_mathpix=ENABLE_MATHPIX):
        start_time = time.time()
        ih, iw = img.shape[:2]
        img_region = [0, 0, iw, ih]
        # fill in the regions of tables and pictures with white
        # in the image after detection finished
        table_result, img = self._table_text_recognize(img)
        pic_imgs, pic_boxes, img = self.picture_detector.predict(img)

        if img_type == 'pdf':
            page_result = self._pdf_text_recognize(img, enable_mathpix=enable_mathpix)
        elif img_type == 'photo':
            page_result = self._photo_text_recognize(img, enable_mathpix=enable_mathpix)
        else:
            # TODO: warning unknown type
            page_result = self._photo_text_recognize(img, enable_mathpix=enable_mathpix)
        # TODO: reconstruct post process
        page_result_extracted = self._extract_page_result(page_result)
        combined_msg = self.post_processor(table_result, page_result_extracted, pic_boxes, img_region)
        cost_time = time.time() - start_time

        return combined_msg, page_result, cost_time

    def _pdf_text_recognize(self, img, enable_mathpix):
        result_rotate_list, median_height, ench_formula_list = self.pdf_text_line_detector.detect(img)
        ench_formula_recognition_result = \
            self.text_recognizer.predict_pdf(result_rotate_list, median_height, ench_formula_list,
                                             enable_mathpix=enable_mathpix)
        return ench_formula_recognition_result

    def _photo_text_recognize(self, img, enable_mathpix):
        result_rotate_list, median_height, ench_formula_list = self.photo_text_line_detector.detect(img)
        ench_formula_recognition_result = \
            self.text_recognizer.predict_photo(result_rotate_list, median_height, ench_formula_list,
                                               enable_mathpix=enable_mathpix)
        return ench_formula_recognition_result

    def _table_text_recognize(self, img):
        table_imgs, table_lines, img_updated = self.table_detector.predict(img)
        table_recognition_result = self.text_recognizer.predict_table(table_imgs, table_lines)
        return table_recognition_result, img_updated

    def _extract_page_result(self, page_result):
        """ temporary method for transform result to adapting old post process """
        page_result_extracted = []
        for text_line_result in page_result:
            line_results = []
            text_line_pos = text_line_result.text_line_pos
            for text_str, text_type, text_label, text_pos in text_line_result:
                msg = dict()
                msg[text_type] = text_str
                msg['pos'] = text_line_pos
                line_results.append(msg)
            page_result_extracted.append(line_results)
        return page_result_extracted
