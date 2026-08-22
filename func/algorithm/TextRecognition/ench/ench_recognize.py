# -*- coding: utf-8 -*-
import cv2
import numpy as np
import paddle.fluid as fluid
# config the paddle
from paddle.fluid.core import AnalysisConfig
# construct the model and get input and output......
from paddle.fluid.core import create_paddle_predictor

MKLDNN = False         # intel MKLDNN
GPU = False            # gpu cuda
USE_ZERO_COPY = False  # only gpu use......


def read_charset(model_dir):
    charset_fp = model_dir+"/ppocr_keys_v1.txt"
    alphabet = []
    with open(charset_fp, encoding='utf-8') as fp:
        for line in fp:
            alphabet.append(line.rstrip('\n'))
    alphabet.extend(['卍', ''])
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    return alphabet, inv_alph_dict

def read_charsetV2(charset_fp):
    alphabet = ['卍']
    with open(charset_fp, encoding='utf-8') as fp:
        for line in fp:
            alphabet.append(line.rstrip('\n'))
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    return alphabet, inv_alph_dict

def config_config(config):
    # 8000 is 8g
    if GPU:
        config.enable_use_gpu(8000, 0)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(6)
        if MKLDNN:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
    config.disable_glog_info()
    if USE_ZERO_COPY:
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
    else:
        config.switch_use_feed_fetch_ops(True)
    return config


class EnchRecognizeModel(object):
    def __init__(self, model_dir):
        self.alphabeta, self.alphadict = read_charset(model_dir)
        model_file_path = model_dir + "/model"
        params_file_path = model_dir + "/params"
        config = AnalysisConfig(model_file_path, params_file_path)
        config = config_config(config)
        predictor = create_paddle_predictor(config)
        input_names = predictor.get_input_names()
        output_names = predictor.get_output_names()
        output_tensors = []
        input_tensor = None
        for name in input_names:
            input_tensor = predictor.get_input_tensor(name)
        for output_name in output_names:
            output_tensor = predictor.get_output_tensor(output_name)
            output_tensors.append(output_tensor)
        self.predictor, self.input_tensor, self.output_tensors = predictor, input_tensor, output_tensors


    def resize_norm_img(self, img):
        h, w = img.shape[:2]
        scale = 32 / img.shape[0]
        new_width = int(scale * img.shape[1])
        resized_image = cv2.resize(img, (new_width, 32))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        # padding_im = np.zeros((3, 32, 320), dtype=np.float32)
        # padding_im[:, :, 0:new_width] = resized_image
        return resized_image


    def __call__(self, img):
        resized_image = self.resize_norm_img(img)
        norm_img_batch = resized_image[np.newaxis, :]
        norm_img_batch = norm_img_batch.copy()
        if USE_ZERO_COPY:
            self.input_tensor.copy_from_cpu(norm_img_batch)
            self.predictor.zero_copy_run()
        else:
            norm_img_batch = fluid.core.PaddleTensor(norm_img_batch)
            self.predictor.run([norm_img_batch])

        # rec_idx return the output argmax value
        rec_idx_batch = self.output_tensors[0].copy_to_cpu()
        rec_idx = np.squeeze(rec_idx_batch, axis=1)
        # lod get the output situation such as [0-15] is img 1  [16-19] is img 2......
        # rec_idx_lod = self.output_tensors[0].lod()[0]
        # predict_lod = self.output_tensors[1].lod()[0]
        predict_batch = self.output_tensors[1].copy_to_cpu()
        pred_idx = np.argmax(predict_batch, axis=1)
        return rec_idx, pred_idx

    def predict(self, img):
        rec_idx, pred_idx= self.__call__(img)
        char_list = []
        ori_list = []
        for i in range(len(rec_idx)):
            char_list.append(self.alphabeta[rec_idx[i]])
        for i in range(len(pred_idx)):
            ori_list.append(self.alphabeta[pred_idx[i]])
        c_l = u''.join(char_list)
        return c_l

    def ocr(self, img):
        _, pred_idx= self.__call__(img)
        ori_list = []
        for i in range(len(pred_idx)):
            ori_list.append(self.alphabeta[pred_idx[i]])
        return ori_list