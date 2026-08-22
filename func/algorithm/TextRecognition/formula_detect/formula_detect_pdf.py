import cv2
import time
import numpy as np
from .formula_detect_infer import analyze_result, draw_predict

#计算水平IOU
def h_iou(box1, box2):
    if box2[0] < box1[1]:
        return True
    else:
        return False

#对于分割的单个文字区域，进行合并
def concat_one(box1, box2, mSize):
    if abs(box1[1] - box2[0]) < 4 and abs(box2[1] - box1[0]) < mSize * 1.2:
        return True
    else:
        return False

#计算分割的BOX之间的IOU
def cal_iou(box1, box2):
    a,b,c,d = box1
    e,f,g,h = box2
    interw = min(b,f) - max(a,e)
    interh = min(d,h) - max(c,g)
    if interh <= 0 or interw <= 0:
        return 0
    else:
        area1 = (b-a)*(d-c)
        area2 = (f-e)*(h-g)
        uarea = max(area1, area2)
        return (interh * interw)/uarea

#计算某个BOX和全部BOXES的IOU
def cal_ious(box, boxes):
    for item in boxes:
        iou = cal_iou(box, item)
        if iou < 1 and iou > 0:
            return iou
    return 0

#根据若干个BOX获取包含他们的区域的边界
def get_text_positon(result_text):
    _np = np.array(result_text)
    _left = np.min(_np[:, 0])
    _right = np.max(_np[:, 1])
    _top = np.min(_np[:, 2])
    _bottom = np.max(_np[:, 3])
    position = [_left, _top, _right, _bottom]
    return position


#分析文本线，获取所有的字符区域，中值高度，确定为latex的字符区域序号
def anaylize_textline(inputim, medianH):
    print(inputim.shape)
    currentboximg = cv2.cvtColor(inputim, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(currentboximg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = thresh.shape
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #dilation = cv2.dilate(thresh, kernel)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in cnts:
        _rect = cv2.boundingRect(cnt)
        _x, _y, _w, _h = _rect
        boxes.append((_x, _x + _w, _y, _y + _h))
    result = sorted(boxes, key=lambda s: s[0])
    mSize = medianH
    if mSize < 8:
        print("Current text line has none char......")
        return [], [], None, None, thresh
    i = 0
    while i < len(result) - 1:
        if h_iou(result[i], result[i + 1]) or concat_one(result[i], result[i + 1], mSize):
            result[i] = (min(result[i][0], result[i + 1][0]), max(result[i][1], result[i + 1][1]),
                         min(result[i][2], result[i + 1][2]), max(result[i][3], result[i + 1][3]))
            result.remove(result[i + 1])
        else:
            i += 1
    i = 0
    while i < len(result) - 1:
        if result[i][3] - result[i][2] < 16 and result[i][1] - result[i][0] > mSize * 2:
            result.remove(result[i])
        else:
            i += 1
    j = 0
    islatex = False
    latexidx = []
    latexcenter = 0
    while j < len(result):
        item = result[j]
        if ((item[3] - item[2]) > mSize * 1.5 or abs((item[3] + item[2])/2 - latexcenter) > mSize*0.5) and islatex == True:
            result[j - 1] = (min(result[j][0], result[j - 1][0]), max(result[j][1], result[j - 1][1]),
                             min(result[j][2], result[j - 1][2]), max(result[j][3], result[j - 1][3]))
            result.remove(item)
        elif item[3] - item[2] > mSize * 1.5:
            latexidx.append(j)
            latexcenter = (item[3] + item[2]) / 2
            islatex = True
            j += 1
        else:
            islatex = False
            j += 1
    otheridx = [ idx for idx in range(len(result)) if idx not in latexidx]
    allcharresult = [result[i] for i in range(len(result)) if i in otheridx]
    chartop = 0
    charbottom = h
    latex_zone = []
    other_zone = []
    if len(allcharresult) > 0:
       all_range = get_text_positon(allcharresult)
       if all_range[3] - all_range[1] < mSize:
           chartop = max(int(all_range[1]/2 + all_range[3]/2 - mSize/2) - 8, 0)
           charbottom = min(int(chartop + mSize) + 8, h)
       else:
           chartop = max(int(all_range[1]) - 8, 0)
           charbottom = min(int(all_range[3]) + 8, h)
    if len(latexidx) == len(result):
        latex_zone = [(0, 0, w, h)]
        other_zone = []
    elif len(latexidx) == 0:
        latex_zone = []
        other_zone = [[0, w]]
    else:
        latexresult = [result[i] for i in range(len(result)) if i in latexidx]
        latex_zone = [(latexresult[i][0], 0, latexresult[i][1], h) for i in range(len(latexresult))]
        k = 0
        while k < len(latex_zone) - 1:
            if latex_zone[k + 1][0] - latex_zone[k][2] < mSize * 2:
                latex_zone[k] = (latex_zone[k][0], 0, latex_zone[k+1][2], h)
                latex_zone.remove(latex_zone[k+1])
            else:
                k += 1
        other_zone = [[0, latex_zone[0][0] - 2] , [latex_zone[-1][2] + 2 , w]]
        if 0 in latexidx:
            other_zone.pop(0)
        if len(result) - 1 in latexidx:
            other_zone.pop()
        for i in range(len(latex_zone) - 1):
            single_zone = [latex_zone[i][2] + 2, latex_zone[i+1][0] - 2]
            other_zone.insert( i + 1, single_zone)
    return latex_zone, other_zone, chartop, charbottom, thresh


def decode_result(text_zone_ori, latex_zone_ori, left, chartop, charbottom):
    text_zones = []
    latex_zones = []
    for text in text_zone_ori:
        l, r = text
        text_zones.append((l * 4 + left, chartop, r * 4 + left, charbottom))
    for latex in latex_zone_ori:
        l, r = latex
        latex_zones.append((l * 4 + left, chartop, r * 4 + left, charbottom))
    return text_zones, latex_zones


def connect_latex(latex_list):
    latex_list = sorted(latex_list, key=lambda x : x[0])
    i = 0
    while i < len(latex_list) - 1:
        if latex_list[i + 1][0] - latex_list[i][2] < 16:
            newzone = (latex_list[i][0], min(latex_list[i][1], latex_list[i + 1][1]), latex_list[i + 1][2], max(latex_list[i][3], latex_list[i + 1][3]))
            latex_list[i] = newzone
            latex_list.remove(latex_list[i + 1])
        else:
            i += 1
    return latex_list


def filter_result(thresh, latex_zones, text_zones):
    final_latex_zones = []
    final_text_zones = []
    for latex in latex_zones:
        l, t, r, b = latex
        threshzone = thresh[t:b, l:r]
        if np.sum(threshzone) / 255 < 8:
            continue
        final_latex_zones.append(latex)
    for text in text_zones:
        l, t, r, b = text
        threshzone = thresh[t:b, l:r]
        if np.sum(threshzone) /255 < 8:
            continue
        final_text_zones.append(text)
    return final_latex_zones, final_text_zones


def get_formula_text_dict(final_latex_list, final_text_list):
    latex_text_list = []
    for latex in final_latex_list:
        latex_text_list.append((latex[0], latex[2], "latex"))
    for text in final_text_list:
        latex_text_list.append((text[0], text[2], "text"))
    latex_text_list = sorted(latex_text_list, key=lambda x: x[0])
    return latex_text_list


def detectFormula(im, basemodel, medianH):
    height_, width_ = im.shape[:2]
    detect_zones = []
    latex_ori, other_ori, chartop, charbottom, thresh = anaylize_textline(im, medianH)
    if latex_ori == None and other_ori == None:
        return [],[]
    zone_dict = {"latex": [], "text": []}
    for zone in latex_ori:
        zone_dict["latex"].append(zone)
    for other_zone in other_ori:
        left, right = other_zone
        zone_img = im[chartop: charbottom, left: right]
        res = basemodel.infer(zone_img)
        text_zone_ori, latex_zone_ori = analyze_result(res)
        text_zone_end, latex_zone_end = decode_result(text_zone_ori, latex_zone_ori, left, chartop, charbottom)
        for zone in latex_zone_end:
            zone_dict["latex"].append(zone)
        for zone in text_zone_end:
            zone_dict["text"].append(zone)
    latex_zones = connect_latex(zone_dict["latex"])
    text_zones = zone_dict["text"]
    latex_zones, text_zones = filter_result(thresh, latex_zones, text_zones)
    latex_text_list = get_formula_text_dict(latex_zones, text_zones)
    for left, right, cls in latex_text_list:
        if cls == "text":
            text_zone = (left, chartop, right, charbottom)
            detect_zones.append((text_zone, "ench"))
        else:
            latex_zone = (left, 0, right, height_)
            detect_zones.append((latex_zone, "formula"))

    return detect_zones