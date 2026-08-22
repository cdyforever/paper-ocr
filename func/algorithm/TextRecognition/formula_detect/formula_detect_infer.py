import os
os.environ["ORT_LOGGING_LEVEL_ERROR"] = "2"
import cv2
import onnxruntime
import numpy as np


class PDFFormulaDetectModel:
    def __init__(self, model_path):
        self.sess = onnxruntime.InferenceSession(model_path)

    def sigmoid(self, out):
        out = 1/(1+np.exp(-out))
        return out

    def infer(self, img):
        img = img.astype(np.float32)
        img -= 127.5
        img /= 127.5
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, :]
        out = self.sess.run(["out"], {"input": img})[0]
        out = np.squeeze(out)
        out = self.sigmoid(out)
        return out


def draw_predict(srcimg, res, basename):
    h, w, _ = srcimg.shape
    for i in range(len(res)):
        item = res[i]
        if item > 0.25:
            pos = i * 4
            cv2.line(srcimg, (pos, 0), (pos, h), (0, 0, 255))
    cv2.imwrite(basename, srcimg)
    return


def if_connect(ret, l, r):
    ll, lr = l
    rl, rr = r
    lret = np.max(ret[ll: lr + 1])
    rret = np.max(ret[rl: rr + 1])
    value = lret if lr - ll > rr - rl else rret
    if min(lr - ll, rr - rl) < 2:
        ret[ll: lr + 1] = value
        ret[rl: rr + 1] = value
        return True, value
    elif lret == rret:
        return True, value
    else:
        return False, lret


def analyze_result(res):
    ret = res > 0.25
    ret = ret.astype(np.uint8)
    all_zone = []
    text_zone = []
    latex_zone = []
    startidx = 0
    for i in range(ret.shape[0] - 1):
        if ret[i + 1] != ret[i]:
            all_zone.append((startidx, i))
            startidx = i + 1
    if startidx < ret.shape[0]:
        all_zone.append((startidx, ret.shape[0] - 1))
    i = 0
    while i < len(all_zone) - 1:
        ifcon, value = if_connect(ret, all_zone[i], all_zone[i + 1])
        if ifcon:
            all_zone[i] = (all_zone[i][0], all_zone[i + 1][1])
            all_zone.pop(i + 1)
        else:
            zone = all_zone.pop(i)
            if value == 1:
                text_zone.append(zone)
            else:
                latex_zone.append(zone)
    final_zone = all_zone[0]
    final_value = np.max(ret[final_zone[0] : final_zone[1] + 1])
    if final_value == 0:
        latex_zone.append(final_zone)
    else:
        text_zone.append(final_zone)
    return text_zone, latex_zone