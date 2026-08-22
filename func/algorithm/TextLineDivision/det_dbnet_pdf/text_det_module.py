import onnxruntime as rt
import numpy as np
import cv2
from .preprocess import get_binary_and_mH
from .postprocess_pdf import after_process, decode_from_result
from .utils import get_resize_width_height, get_resizeratio, from_rects_to_mat

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


class DBNET:

    def __init__(self, MODEL_PATH):
        self.sess = rt.InferenceSession(MODEL_PATH)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def process(self, img, ratio):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        #make sure min resize is 32
        tar_w, tar_h = get_resize_width_height(h, w, ratio)
        img = cv2.resize(img, (tar_w, tar_h), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(img, axis=0)
        out = self.sess.run(["out"], {"input": transformed_image.astype(np.float32)})
        pred = out[0][0][0]
        #rect_list, rect_split_list, bitmap = postprocess.decode_from_result(pred, w, h)
        rect_list, bitmap = decode_from_result(pred, w, h)
        #return rect_list, rect_split_list, pred, bitmap
        return rect_list, pred, bitmap


class PDFTextLineDetectModel(object):

    def __init__(self, model_path):
        self.net = DBNET(MODEL_PATH=model_path)

    def detect(self, srcim):
        binary, mH = get_binary_and_mH(srcim)
        ratio = get_resizeratio(mH)
        rects, _, _, = self.net.process(srcim, ratio)
        rectlist, rotatelist = after_process(binary, rects, mH)
        refine_mats = from_rects_to_mat(srcim, rectlist)
        return rotatelist, mH, refine_mats
