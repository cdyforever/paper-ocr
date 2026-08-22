import onnxruntime as rt
import numpy as np
import cv2
from .decode import SegDetectorRepresenter
from .preprocess import get_binary_and_mH
from .postprocess import after_process, from_rects_to_mat
from .utils import get_resize_width_height, get_resizeratio
from func.algorithm.tools.ImageEnhance import enhance_image


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def Singleton(cls):
    _instance = {}
    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


class SingletonType(type):
    def __init__(cls, *args, **kwargs):
        super(SingletonType, cls).__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        cls.__init__(obj, *args, **kwargs)
        return obj


def draw_rotate_rect(src_im, rotate_rect, color=(0, 0, 255), thickness=2):
    #src_im = input_im.copy()
    pts = cv2.boxPoints(rotate_rect)
    pts = np.int0(pts)
    pt1 = (pts[0][0], pts[0][1])
    pt2 = (pts[1][0], pts[1][1])
    pt3 = (pts[2][0], pts[2][1])
    pt4 = (pts[3][0], pts[3][1])
    cv2.line(src_im, pt1, pt2, color, thickness)
    cv2.line(src_im, pt1, pt4, color, thickness)
    cv2.line(src_im, pt3, pt2, color, thickness)
    cv2.line(src_im, pt3, pt4, color, thickness)
    return src_im


class DBNET(metaclass=SingletonType):
    def __init__(self, MODEL_PATH):
        self.sess = rt.InferenceSession(MODEL_PATH)
        self.decode_handel = SegDetectorRepresenter()

    def process(self, img, img_enhance, ratio):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        #make sure min resize is 32
        tar_w, tar_h = get_resize_width_height(h, w, ratio)
        # print("tar w tar h ", tar_h, tar_w)
        img = cv2.resize(img, (tar_w, tar_h), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img /= 255.0
        img -= mean
        img /= std
        img = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(img, axis=0)
        out = self.sess.run(["out"], {"input": transformed_image.astype(np.float32)})
        frects, distort_rect_mat = self.decode_handel(out[0][0], img_enhance, h, w)
        return frects, distort_rect_mat


class PhotoTextLineDetectModel(object):

    def __init__(self, model_path):
        self.net = DBNET(MODEL_PATH=model_path)

    def detect(self, srcim):
        img_enhance = enhance_image(srcim)
        binary, mH = get_binary_and_mH(srcim)
        ratio = get_resizeratio(mH)
        rects, distort_rect_mat = self.net.process(srcim, img_enhance, ratio)
        # TODO: photo text line detect BUG 7.23
        # TODO: how to work in this part
        resultlist = after_process(binary, rects)
        refine_mats = from_rects_to_mat(srcim, resultlist)
        for rect, mat in distort_rect_mat:
            resultlist.append(rect)
            refine_mats.append(mat)
        return resultlist, mH, refine_mats
