
from .det_dbnet_pdf.utils import draw_rotate_rect

import cv2
import os


def _draw_image_formula_text(img_url, im, latex_lists, text_lists):
    save_im = im.copy()
    for latex in latex_lists:
        l, t, r, b = latex
        cv2.rectangle(save_im, (l, t), (r, b), (0, 0, 255), 3)
    for text in text_lists:
        l, t, r, b = text
        cv2.rectangle(save_im, (l, t), (r, b), (255, 0, 20), 3)
    cv2.imwrite("./DIR_FormulaTextDetect/" + os.path.basename(img_url), save_im)


def _draw_image_textline(img_url, im, result_rotate_list):
    save_im = im.copy()
    for i in range(len(result_rotate_list)):
        rotate_rect = result_rotate_list[i]
        save_im = draw_rotate_rect(save_im, rotate_rect)
    cv2.imwrite("./DIR_TextLineDetect/" + os.path.basename(img_url), save_im)


def _save_textline(img_url, rotate_mats):
    imname = os.path.basename(img_url)
    imname = os.path.splitext(imname)[0]
    for i in range(len(rotate_mats)):
        mat = rotate_mats[i]
        imgname = imname + "_" + str(i)
        cv2.imwrite("./DIR_TextLine/" + imgname + ".jpg", mat)
