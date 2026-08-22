import cv2
import numpy as np

not_chinese = u"0123456789∞ABCDEFGHIJKLMNOОPQRSTUVWXYZabcdefghijklmnπopqrstuvwxyzａ￥≌~～≈≠_⊥/｜|│±+.·:。,，—－―-–>*×÷≥≤<∠△▽□=()（）[]²°〇○‘’”′…{}卍%ø√√αβγη∵∴'"
not_care = u"，,.·。十一ABCDabcd0123456789:()（）[]”"
math_symbol = u"±－-×÷/><≥≤=≈∠√°△□%⊥≌≠"
n_english = u"0123456789ABCDEFGHIJKLMNOОPQRSTUVWXYZabcdefghijklmnπopqrstuvwxyzαβ"
question_head = u"0123456789ABCD"


nclass = 6624
not_chinese = not_chinese[:]
math_symbol = math_symbol[:]


def last_right_index(origin_pred, idx):
    for i in range(idx + 1, len(origin_pred)):
        if origin_pred[i] != '' and origin_pred[i] != "." and origin_pred[i] != "．":
            return i - idx
    return -1


def replace_for_split(origin_pred, pred_string):
    hasNonChinese = False
    for pred_char in pred_string:
        if not_chinese.find(pred_char) != -1:
            hasNonChinese = True
    if not hasNonChinese:
        return origin_pred
    idxp1 = pred_string.find('.')
    idxp2 = pred_string.find('．')
    idxp1 = -1 if idxp1 > 2 else idxp1
    idxp2 = -1 if idxp2 > 2 else idxp2
    idx = max(idxp1, idxp2)
    if idx != -1:
        pointchar = pred_string[idx]
        idxori = origin_pred.index(pointchar)
        dis = last_right_index(origin_pred, idxori)
        #if 0 < dis < 4:
        #    return origin_pred
        for i in range(idxori + 1):
            origin_pred[i] = '田'
    return origin_pred


def _decode(origin_pred, firstTime=False):
    chinese_at = [-1]
    not_chinese_at = []
    math_at = []
    hard_at = []
    all_length = len(origin_pred)
    pred_string = ""
    for i in range(all_length):
        pred_char = origin_pred[i]
        pred_string += pred_char
    if firstTime:
       origin_pred = replace_for_split(origin_pred, pred_string)
    chn_sentense = ""
    for i in range(len(origin_pred)):
        pred_char = origin_pred[i]
        if not_chinese.find(pred_char) == -1 and not_care.find(pred_char) == -1:
            chinese_at.append(i)
            chn_sentense += pred_char
        elif pred_char != '' and not_care.find(pred_char) == -1:
            not_chinese_at.append(i)
        # Fill math symbols...
        if math_symbol.find(pred_char) != -1:
            math_at.append(i)
        # Fill hard items...
    chinese_at.append(all_length)
    return chinese_at, math_at, not_chinese_at, all_length


#文字识别服务接口——调用中英文识别模型获取识别结果
def text_recognize(img_array, model, new_pt, firstTime=False):
    img_detect = img_array[new_pt[1]: new_pt[3], new_pt[0]: new_pt[2]]
    origin_pred = model.ocr(img_detect)
    ch_at, math_at, n_ch_at, length = _decode(origin_pred, firstTime)
    return origin_pred, ch_at, math_at, n_ch_at, length


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


#获取一个区域的最大高度
def get_max_height(char_text):
    _np = np.array(char_text)
    return np.max(_np[:,3] - _np[:, 2])


#获取所有字符的中值高度
def get_median_height(result, mediaH):
    hlist = [item[3] - item[2] for item in result if item[3] - item[2] > 8]
    hlist = sorted(hlist)
    if len(hlist) == 0:
        return mediaH
    elif hlist[len(hlist) * 3 // 4] < mediaH:
        return mediaH
    else:
        return hlist[len(hlist) * 3 // 4]


#分析文本线，获取所有的字符区域，中值高度，确定为latex的字符区域序号
def anaylize_textline(currentboximg, mediaH):
    if len(currentboximg.shape) == 3:
        currentboximg = cv2.cvtColor(currentboximg, cv2.COLOR_BGR2GRAY)
    h, w = currentboximg.shape
    #thresh = cv2.adaptiveThreshold(currentboximg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 55, 15)
    ret, thresh = cv2.threshold(currentboximg, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 8:
            continue
        _rect = cv2.boundingRect(cnt)
        _x, _y, _w, _h = _rect
        if (_x == 0 or _x + _w == w) and _w < mediaH / 3:
            continue
        boxes.append((_x, _x + _w, _y, _y + _h))
    result = sorted(boxes, key=lambda s: s[0])
    mediaH = get_median_height(result, mediaH)
    i = 0
    while i < len(result) - 1:
        if h_iou(result[i], result[i + 1]) or concat_one(result[i], result[i + 1], mediaH):
            result[i] = (min(result[i][0], result[i + 1][0]), max(result[i][1], result[i + 1][1]),
                         min(result[i][2], result[i + 1][2]), max(result[i][3], result[i + 1][3]))
            result.remove(result[i + 1])
        else:
            i += 1
    # Remove line and......
    i = 0
    while i < len(result) - 1:
        if result[i][3] - result[i][2] < 16 and result[i][1] - result[i][0] > mediaH * 2:
            result.remove(result[i])
        else:
            i += 1
    #Second, latex idx ......
    j = 0
    islatex = False
    latexidx = []
    latexcenter = 0
    while j < len(result):
        item = result[j]
        if ((item[3] - item[2]) > mediaH * 1.5 or abs((item[3] + item[2])/2 - latexcenter) > mediaH*0.5) and islatex == True:
            result[j - 1] = (min(result[j][0], result[j - 1][0]), max(result[j][1], result[j - 1][1]),
                             min(result[j][2], result[j - 1][2]), max(result[j][3], result[j - 1][3]))
            result.remove(item)
        elif item[3] - item[2] > mediaH * 1.5:
            latexidx.append(j)
            latexcenter = (item[3] + item[2]) / 2
            islatex = True
            j += 1
        else:
            islatex = False
            j += 1
    otheridx = [ idx for idx in range(len(result)) if idx not in latexidx]
    allcharresult = [result[i] for i in range(len(result)) if i in otheridx]
    if len(allcharresult) > 0:
        chartop = charbottom = 0
        all_range = get_text_positon(allcharresult)
        if all_range[3] - all_range[1] < mediaH:
            chartop = max(int(all_range[1]/2 + all_range[3]/2 - mediaH/2 - 4), 0)
            charbottom = min(int(chartop + mediaH + 8), h)
        else:
            chartop = max(int(all_range[1]) - 4, 0)
            charbottom = min(int(all_range[3]) + 4, h)
        return result, latexidx, chartop, charbottom
    else:
        return result, latexidx, 0, h


#寻找最邻近的位置（输入为中文字符位置，返回中文字符之间的非中文区域的位置）
def findNearNch(h, l, _list):
    containlist = []
    for item in _list:
        if item < h and item > l:
            containlist.append(item)
        if item > h:
            break
    if len(containlist) < 1:
        return 0, -1
    else:
        _l = l
        _r = h
        return _l, _r


def get_form_and_text_pos(formula_at, result_text, pos, length, startidx):
    formula_ids = []
    boxwidth = pos[2] - pos[0]
    for form in formula_at:
        formleft, formright = form
        formleftpos = float(formleft) / length * boxwidth + pos[0]
        formrightpos = float(formright) / length * boxwidth + pos[0]
        leftidx, rightidx = -1, -1
        leftdismin = 255
        rightdismin = 255
        for i in range(len(result_text)):
            l, r, _, _ = result_text[i]
            leftidx = i
            if formleftpos < l:
                break
        for i in range(len(result_text)):
            l, r, _, _ = result_text[i]
            rightidx = i
            if formrightpos < r:
                rightidx = i - 1
                break
        formlist = list(range(leftidx + startidx, rightidx + startidx + 1))
        formula_ids.append(formlist)
    return formula_ids


def get_all_formula_ids(formula_from_text_ids, latexidx):
    all_length = latexidx[-1]
    latexidx = latexidx[1:-1]
    finallatexidx = []
    for i in range(len(formula_from_text_ids)):
        form_zone = formula_from_text_ids[i]
        for form in form_zone:
            finallatexidx += form
    finallatexidx += latexidx
    finallatexidx = sorted(finallatexidx)
    finaltextidx = [i for i in range(all_length) if i not in finallatexidx]
    final_text_list = []
    final_latex_list = []
    final_text = []
    final_latex = []
    for i in range(len(finaltextidx)):
        item = finaltextidx[i]
        final_text.append(item)
        if i == len(finaltextidx) - 1:
            final_text_list.append(final_text)
        elif finaltextidx[i + 1] - finaltextidx[i] > 1:
            final_text_list.append(final_text)
            final_text = []
    for i in range(len(finallatexidx)):
        item = finallatexidx[i]
        final_latex.append(item)
        if i == len(finallatexidx) - 1:
            final_latex_list.append(final_latex)
        elif finallatexidx[i + 1] - finallatexidx[i] > 1:
            final_latex_list.append(final_latex)
            final_latex = []
    return final_latex_list, final_text_list


def split_preprocess(currentboximg, mH):
    currentboximggray = cv2.cvtColor(currentboximg, cv2.COLOR_BGR2GRAY)
    height, width = currentboximggray.shape
    ret, thresh = cv2.threshold(currentboximggray, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in cnts:
        _rect = cv2.boundingRect(cnt)
        _x, _y, _w, _h = _rect
        boxes.append((_x, _x + _w, _y, _y + _h))
    result = sorted(boxes, key=lambda s: s[0])
    lastleft = 0
    lastright = 0
    splitlist = []
    for i in range(len(result)):
        box = result[i]
        if box[0] - lastright > mH and i > 0:
            saveright = (lastright + box[0]) // 2
            splitlist.append((lastleft, saveright))
            lastleft = (lastright + box[0]) // 2
            lastright = max(box[1], lastright)
        else:
            lastright = max(box[1], lastright)
    splitlist.append((lastleft, width))
    savelist = []
    saveposlist = []
    for split in splitlist:
        if split[1] - split[0] < 5:
            continue
        else:
            newleft = split[0]
            newright = split[1]
            savelist.append(currentboximg[:, newleft: newright])
            saveposlist.append((newleft, newright))
    return savelist, saveposlist


def get_formula_text_dict(final_latex_list, final_text_list):
    latex_text_list = []
    for latex in final_latex_list:
        latex_text_list.append((latex[0], latex[-1], "latex"))
    for text in final_text_list:
        latex_text_list.append((text[0], text[-1], "text"))
    latex_text_list = sorted(latex_text_list, key=lambda x: x[0])
    return latex_text_list


#######公式检测接口，返回公式对应的坐标，以及非公式（文本）对应的坐标#######
def detectFormula(im, basemodel, medianH):
    height_, width_ = im.shape[:2]
    detect_zones = []
    savelist, saveposlist = split_preprocess(im, medianH)
    for splitim, splitpos in zip(savelist, saveposlist):
        splitleft, splitright = splitpos
        result, latexidx, charTop, charBottom = anaylize_textline(splitim, medianH)
        # 图片上没有元素
        allnum = len(result)
        if result == None:
            continue
        # elif allnum == 1 and len(latexidx) == 0:
        #    text_zones.append([splitleft, charTop, splitright, charBottom])
        #    continue
        latexidx.insert(0, -1)
        latexidx.append(allnum)
        results_text = []
        results_text_start_idx = []
        texts_positon = []
        for i in range(1, len(latexidx)):
            result_text = result[latexidx[i - 1] + 1: latexidx[i]]
            if len(result_text) > 0:
                leftside = 0 if latexidx[i - 1] == -1 else result[latexidx[i - 1]][1]
                rightside = splitright - splitleft - 1 if latexidx[i] == allnum else result[latexidx[i]][0]
                results_text.append(result_text)
                results_text_start_idx.append(latexidx[i - 1] + 1)
                t_pos = [leftside, charTop, rightside, charBottom]
                texts_positon.append(t_pos)
        formula_from_text_ids = []
        for i in range(len(texts_positon)):
            pos = texts_positon[i]
            result_text = results_text[i]
            start_idx = results_text_start_idx[i]
            pred, ch_at, _, n_ch_at, length = text_recognize(splitim, basemodel, pos, True)
            formula_at = []
            # if len(ch_at) == 2:
            #    formula_at.append([0, length])
            # else:
            for j in range(1, len(ch_at)):
                if ch_at[j] - ch_at[j - 1] > 2:
                    start, end = findNearNch(ch_at[j], ch_at[j - 1], n_ch_at)
                    if end == start + 1 and pred[end] != pred[start] or end - start > 1:
                        formula_at.append([start, end])
            formula_ids = get_form_and_text_pos(formula_at, result_text, pos, length, start_idx)
            formula_from_text_ids.append(formula_ids)
        final_latex_list, final_text_list = get_all_formula_ids(formula_from_text_ids, latexidx)
        latex_text_list = get_formula_text_dict(final_latex_list, final_text_list)
        for left, right, cls in latex_text_list:
            _left = (result[left - 1][1] + result[left][0]) // 2 if left > 0 else 0
            _right = (result[right + 1][0] + result[right][
                1]) // 2 if right < allnum - 1 else splitright - splitleft - 1
            if cls == "text":
                text_zone = (_left + splitleft, charTop, _right + splitleft, charBottom)
                detect_zones.append((text_zone, "ench"))
            else:
                latex_zone = (_left + splitleft, 0, _right + splitleft, height_)
                detect_zones.append((latex_zone, "formula"))
    return detect_zones
