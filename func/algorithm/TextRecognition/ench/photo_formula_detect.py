from __future__ import print_function
import numpy as np
import MNN
import cv2
import os

def read_charset(charset_fp):
    alphabet = ['卍']
    with open(charset_fp, encoding='utf-8') as fp:
        for line in fp:
            alphabet.append(line.rstrip('\n'))
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    return alphabet, inv_alph_dict


class PhotoFormulaDetectModel:
    def __init__(self, model_dir):
        model_path = os.path.join(model_dir, 'cnocr.mnn')
        label_cn_path = os.path.join(model_dir, 'label_cn.txt')

        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()
        self.alphabeta, self.alphadict = read_charset(label_cn_path)

    def ctc_label(self, p):
        ret = []
        p1 = [0] + p
        for i, _ in enumerate(p):
            c1 = p1[i]
            c2 = p1[i + 1]
            if (c2 == 0 or c2 != c1) and c1 != 0 and len(ret) > 0:
                ret[-1][-1] = i
            if c2 == 0 or c2 == c1:
                continue
            ret.append([c2, i, -1])
        if len(ret) == 0:
            return [], []
        if ret[-1][-1] < 0:
            ret[-1][-1] = len(p)
        label_ids = [ele[0] for ele in ret]
        start_end_idx = [(ele[1], ele[2]) for ele in ret]
        return label_ids, start_end_idx

    def predict(self, img):
        h, w = img.shape[:2]
        scale = 32 / img.shape[0]
        new_width = int(scale * img.shape[1])
        img = cv2.resize(img, (new_width, 32))
        img = np.array(([img]), dtype=np.float32)
        img /= 255.0
        tmp_input = MNN.Tensor( (1, 1, 32, new_width), MNN.Halide_Type_Float, img, MNN.Tensor_DimensionType_Caffe)
        input_tensor = self.interpreter.getSessionInput(self.session)
        self.interpreter.resizeTensor(input_tensor, (1, 1, 32, new_width))
        self.interpreter.resizeSession(self.session)
        input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)
        output_tensor = self.interpreter.getSessionOutput(self.session)
        out = output_tensor.getData()
        ids = np.argmax(out, axis=2)
        ids = np.squeeze(ids, axis=1)
        return ids

    def decode(self, pred_text):
        char_list = []
        embding = []
        for i in range(len(pred_text)):
            if pred_text[i] != 0 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or
                                           (i > 1 and pred_text[i] == pred_text[i - 2])):
                char_list.append(self.alphabeta[pred_text[i]])
                embding.append(pred_text[i])
        c_l = u''.join(char_list)
        # print(c_l)
        return c_l, embding

if __name__ == "__main__":
    im = cv2.imread("../../DIR_TextLine/8.jpg", 0)
    model = PhotoFormulaDetectModel(version="0114")
    pred = model.predict(im)
    a, b = model.decode(pred)