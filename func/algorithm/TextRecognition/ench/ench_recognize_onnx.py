# -*- coding: utf-8 -*-
"""
EnchRecognizeModel — 纯 ONNX 实现（替代 Paddle Fluid 版本）
============================================================

与原 `ench/ench_recognize.py` 的 `EnchRecognizeModel` **接口完全兼容**：

    model = EnchRecognizeModel(model_dir)
    text = model.predict(img)      # -> str
    chars = model.ocr(img)         # -> list[str]

但底层使用 onnxruntime 加载 PP-OCRv4 识别模型，**不再依赖 paddle.fluid**。

模型文件（由 convert_paddleocr_to_onnx.py 生成）：
    weights/text_recognition/crnn_rec.onnx
    weights/text_recognition/ppocr_keys_v1.txt

用法：
    from func.algorithm.TextRecognition.ench.ench_recognize_onnx import EnchRecognizeModel
    model = EnchRecognizeModel('path/to/model_dir')   # 或直接传 onnx 文件路径
"""

import os
import math
import numpy as np
import cv2
import onnxruntime as ort


# 默认模型位置（相对于项目根目录）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', '..'))
_DEFAULT_ONNX = os.path.join(_PROJECT_ROOT, 'weights', 'text_recognition', 'crnn_rec.onnx')
_DEFAULT_DICT = os.path.join(_PROJECT_ROOT, 'weights', 'text_recognition', 'ppocr_keys_v1.txt')


def read_charset(model_dir):
    """
    读取字符字典，保持与原 Paddle 版本一致的布局。

    原版布局: [dict...] + ['卍', '']
    PP-OCRv4 的 CTC 布局: ['blank'] + [dict...] + [' ']

    这里返回 PP-OCRv4 布局（索引 0 为 blank），因为 ONNX 模型输出的
    类别数是 len(dict) + 2。
    """
    dict_path = os.path.join(model_dir, 'ppocr_keys_v1.txt')
    if not os.path.exists(dict_path):
        dict_path = _DEFAULT_DICT

    with open(dict_path, encoding='utf-8') as fp:
        alphabet = [line.rstrip('\n') for line in fp]

    # PP-OCRv4: blank 在索引 0，空格在末尾
    alphabet = ['blank'] + alphabet + [' ']
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    return alphabet, inv_alph_dict


class EnchRecognizeModel(object):
    """
    纯 ONNX 版本的英文/中文识别模型。

    与 Paddle Fluid 版本接口保持一致，可直接替换。
    """

    def __init__(self, model_dir, use_gpu=False, rec_image_height=48):
        """
        Args:
            model_dir: 模型目录（含 ppocr_keys_v1.txt）或直接是 .onnx 文件路径
            use_gpu: 是否使用 GPU
            rec_image_height: 识别输入高度（PP-OCRv4 为 48）
        """
        # 解析模型路径
        if model_dir and model_dir.endswith('.onnx'):
            self.model_path = model_dir
            self.model_dir = os.path.dirname(model_dir)
        else:
            self.model_dir = model_dir
            candidate = os.path.join(model_dir, 'crnn_rec.onnx')
            self.model_path = candidate if os.path.exists(candidate) else _DEFAULT_ONNX

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f'ONNX 识别模型不存在: {self.model_path}\n'
                f'请先运行 convert_paddleocr_to_onnx.py 生成模型')

        self.rec_image_height = rec_image_height

        # 字典
        self.alphabeta, self.alphadict = read_charset(self.model_dir)

        # 执行提供者
        available = ort.get_available_providers()
        if use_gpu and 'CUDAExecutionProvider' in available:
            providers = [('CUDAExecutionProvider', {'device_id': 0}),
                         'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # -- 预处理 -------------------------------------------------------------
    def resize_norm_img(self, img, max_wh_ratio=None):
        """
        归一化文本行图像，输出 CHW float32。

        与原版不同：PP-OCRv4 需要固定高度 48 并按宽高比缩放。
        """
        img_c, img_h = 3, self.rec_image_height
        h, w = img.shape[:2]
        ratio = w / float(h)

        if max_wh_ratio is not None and max_wh_ratio > ratio:
            ratio = max_wh_ratio

        img_w = int(img_h * ratio)
        img_w = max(img_w, 1)

        resized_w = img_w if math.ceil(img_h * (w / float(h))) > img_w \
            else int(math.ceil(img_h * (w / float(h))))
        resized_w = max(resized_w, 1)

        resized_image = cv2.resize(img, (resized_w, img_h))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    # -- 推理 ---------------------------------------------------------------
    def __call__(self, img):
        """
        运行识别模型。

        Returns:
            (text_list, char_index_array)
            保持与原版返回结构一致：rec_idx 为解码后的字符序列，
            pred_idx 为逐帧 argmax 索引。
        """
        norm_img = self.resize_norm_img(img)
        norm_img_batch = norm_img[np.newaxis, :].copy()

        preds = self.session.run(
            [self.output_name], {self.input_name: norm_img_batch})[0]

        # preds: (1, T, C)
        probs = preds[0]
        pred_idx = np.argmax(probs, axis=1)

        # CTC 贪婪解码
        selection = np.ones(len(pred_idx), dtype=bool)
        selection[1:] = pred_idx[1:] != pred_idx[:-1]
        selection &= pred_idx != 0  # blank

        char_list = []
        for i in np.where(selection)[0]:
            idx = int(pred_idx[i])
            if 0 <= idx < len(self.alphabeta):
                char_list.append(self.alphabeta[idx])

        return char_list, pred_idx

    def predict(self, img):
        """
        识别文本，返回字符串（与原版接口一致）。
        """
        char_list, _ = self.__call__(img)
        return u''.join(char_list)

    def ocr(self, img):
        """
        返回字符列表（与原版接口一致）。
        """
        char_list, _ = self.__call__(img)
        return char_list

    def predict_with_score(self, img):
        """
        扩展接口：返回 (text, mean_confidence)。
        """
        norm_img = self.resize_norm_img(img)
        norm_img_batch = norm_img[np.newaxis, :].copy()

        preds = self.session.run(
            [self.output_name], {self.input_name: norm_img_batch})[0]
        probs = preds[0]
        pred_idx = np.argmax(probs, axis=1)
        pred_prob = probs.max(axis=1)

        selection = np.ones(len(pred_idx), dtype=bool)
        selection[1:] = pred_idx[1:] != pred_idx[:-1]
        selection &= pred_idx != 0

        chars, confs = [], []
        for i in np.where(selection)[0]:
            idx = int(pred_idx[i])
            if 0 <= idx < len(self.alphabeta):
                chars.append(self.alphabeta[idx])
                confs.append(float(pred_prob[i]))

        text = u''.join(chars)
        score = float(np.mean(confs)) if confs else 0.0
        return text, score


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    model = EnchRecognizeModel(_DEFAULT_ONNX)
    print(f'模型: {model.model_path}')
    print(f'字典大小: {len(model.alphabeta)}')

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            text, score = model.predict_with_score(img)
            print(f'识别结果: {text!r}  score={score:.4f}')
