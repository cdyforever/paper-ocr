import cv2
import onnxruntime
import numpy as np

# vocab
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',\
            'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',\
            'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',\
            'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '[', ']', '(', ')',\
            '^', '_', '-', '/', '|', '.', '>', '<', ':', ',', ';', '=', '+', '*', '\\\\', '\\left\\{',\
            '\\right\\}', '[', ']', '(', ')', '\\left|', '\\right|', '\\sqrt', '\\frac',\
            '\\pm', '\\mp', '\\therefore', '\\because', '\\Delta', '\\log', '\\lg', '\\odot', '\\times', '\\div',\
            '\\oplus', '\\ln', '\\leqslant', '\\geqslant', '\\neq', '\\leq', '\\geq', '\\approx', '\\perp', '\\cong',\
            '\\subset', '\\subseteq', '\\otimes', '\\widehat', '\\lim', '\\quad', '\\in', '\\boldsymbol', '\\varnothing',\
            '\\exists', '\\forall', '\\prime', '\\hat', '\\sum', '\\%', '\\min', '\\max', '\\sin', '\\cos', '\\tan',\
            '\\cot', '\\circ', '\\cdot', '\\angle', '\\alpha', '\\beta', '\\gamma', '\\mathbf', '\\theta', '\\varphi',\
            '\\rho', '\\lambda', '\\nsubseteq', '\\mu', '\\pi', '\\supseteq', '\\neg', '\\wedge', '\\vee', '\\Sigma',\
            '&', '\\infty', '\\square', '\\sim', '\\omega', '\\Omega', '\\overline', '\\overrightarrow', '\\Leftrightarrow',\
            '\\Rightarrow', '\\rightarrow', '\\dot', '\\bigoplus', '\\mathrm', '\\triangle', '\\cup', '\\cap', '\\not',\
            '\\dots', '\\Lambda', '\\begin{array}', '\\end{array}']


class FormulaRecognizeModel:
    def __init__(self, basic_dir="./data/model"):
        so = onnxruntime.SessionOptions()
        so.enable_cpu_mem_arena = False
        self.sess1 = onnxruntime.InferenceSession(basic_dir+"/mlp-encoder.onnx", so)
        self.sess2 = onnxruntime.InferenceSession(basic_dir+"/mlp-init-decoder.onnx", so)
        self.sess3 = onnxruntime.InferenceSession(basic_dir+"/mlp-decoder.onnx", so)

    def infer(self, img, median_height):
        ratio = 32.0 / median_height
        if median_height < 32:
            newwidth = int(ratio * img.shape[1])
            newheight = int(ratio * img.shape[0])
            img = cv2.resize(img, (newwidth, newheight), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        img = 1 - img / 255.
        img = img[np.newaxis, :]
        transformed_image = np.expand_dims(img, axis=0)
        out = self.sess1.run(["out"], {"input": transformed_image.astype(np.float32)})[0]
        inith, initc, inito = self.sess2.run(["inith", "initc", "inito"], {"input": out})
        tgt = np.ones((1, 1), dtype=np.int64) * 176
        retlist = []
        for i in range(200):
            ret = self.sess3.run(["oh", "oc", "oo", "logit"], {"ih": inith, "ic": initc, "io":inito, "fea":out, "tgt":tgt})
            inith, initc, inito, logit = ret
            tgt = np.argmax(logit, axis=1)
            if tgt == 0:
                break
            idx = tgt[0]
            retlist.append(idx - 1)
            tgt = tgt[np.newaxis, :]
        return retlist

    def predict(self, img, median_height=None):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retlist = self.infer(img, median_height)
        reco_result = " ".join([vocab[item] for item in retlist])
        # print("ori reco is : ", reco_result)
        if "\\left\\{ \\begin{array}" in reco_result:
            reco_result = reco_result.replace("\\end{array}", "\\end{array}\\right.")
        if "\\left|" in reco_result:
            reco_result = reco_result.replace("\\left|", "|")
        if "\\right|" in reco_result:
            reco_result = reco_result.replace("\\right|", "|")
        if "\\left(" in reco_result:
            reco_result = reco_result.replace("\\left(", "(")
        if "\\right)" in reco_result:
            reco_result = reco_result.replace("\\right)", ")")
        # print("new reco is", reco_result)
        return reco_result
