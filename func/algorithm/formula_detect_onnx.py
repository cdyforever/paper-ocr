"""
Pure ONNX Formula Detection (MFD)
=================================

基于 MinerU / PDF-Extract-Kit 的 YOLOv8 公式检测模型，
用纯 ONNX Runtime 推理，**不依赖 ultralytics / torch / PaddlePaddle**。

模型输出类别:
    0: embedding  (行内公式)
    1: isolated   (独立公式)

用法:
    from func.algorithm.formula_detect_onnx import FormulaDetectorONNX

    det = FormulaDetectorONNX()
    formulas = det(image)          # -> [{'box': [x1,y1,x2,y2], 'type': 'embedding', 'score': 0.89}, ...]
    vis = det.draw(image, formulas)
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MFD_MODEL = _PROJECT_ROOT / 'weights' / 'formula_detect' / 'mfd_yolov8_fp16.onnx'

# YOLOv8 类别名（来自模型 names）
CLASS_NAMES = {0: 'embedding', 1: 'isolated'}


class FormulaDetectorONNX:
    """
    纯 ONNX 公式检测器 (YOLOv8 MFD)。

    检测文档中的数学公式区域，区分行内公式与独立公式。
    """

    def __init__(self,
                 model_path: str = str(DEFAULT_MFD_MODEL),
                 use_gpu: bool = False,
                 imgsz: int = 1024,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 split_tall_boxes: bool = True,
                 tall_box_ratio: float = 2.5,
                 dedup_iou: float = 0.9,
                 split_child_iou: float = 0.6,
                 verbose: bool = True):
        """
        Args:
            split_tall_boxes: 是否对过高检测框做切分 (修复跨行合并)
            tall_box_ratio: 框高 > 中位高 * 该值 时触发切分
            dedup_iou: 同位置跨类框的 IoU 阈值, 超过则去重
            split_child_iou: 切分出的子框若与某个独立检测框 IoU >= 该值,
                             说明该子框已被单独检出, 丢弃以免重复识别
        """
        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.split_tall_boxes = split_tall_boxes
        self.tall_box_ratio = tall_box_ratio
        self.dedup_iou = dedup_iou
        self.split_child_iou = split_child_iou
        self.verbose = verbose

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f'MFD ONNX 模型不存在: {self.model_path}\n'
                f'请先运行 test_and_export_mfd.py 生成模型')

        available = ort.get_available_providers()
        if use_gpu and 'CUDAExecutionProvider' in available:
            providers = [('CUDAExecutionProvider', {'device_id': 0}),
                         'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        t0 = time.time()
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # 从模型读取输入尺寸
        shape = self.session.get_inputs()[0].shape
        if isinstance(shape[2], int) and isinstance(shape[3], int):
            self.imgsz = shape[2]

        if self.verbose:
            print(f'[FormulaDetectorONNX] 加载完成 {time.time()-t0:.2f}s  '
                  f'imgsz={self.imgsz}  providers={providers}')

    # -- 预处理 -------------------------------------------------------------
    def _letterbox(self, img: np.ndarray, new_shape: int = 1024
                   ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """等比缩放 + 灰边填充 (YOLO letterbox)。"""
        shape = img.shape[:2]  # h, w
        r = min(new_shape / shape[0], new_shape / shape[1])

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape - new_unpad[0]
        dh = new_shape - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (left, top)

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """BGR -> letterbox -> RGB -> CHW -> [0,1] -> NCHW float32."""
        lb, ratio, pad = self._letterbox(img, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(blob), ratio, pad

    # -- 后处理 -------------------------------------------------------------
    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray,
             iou_thres: float) -> List[int]:
        """标准 NMS。boxes: [N,4] xyxy, scores: [N]"""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        return keep

    def _postprocess(self, output: np.ndarray, ratio: float,
                     pad: Tuple[int, int],
                     orig_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Args:
            output: [1, 4+nc, N]  YOLOv8 输出
            ratio: letterbox 缩放比
            pad: (left, top)
            orig_shape: (h, w)
        """
        pred = output[0]                    # [4+nc, N]
        pred = pred.transpose(1, 0)         # [N, 4+nc]

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]          # [N, nc]

        if class_scores.size == 0:
            return []

        cls_ids = class_scores.argmax(axis=1)
        confs = class_scores[np.arange(len(cls_ids)), cls_ids]

        mask = confs >= self.conf_thres
        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        # xywh(中心) -> xyxy
        cx, cy, w, h = (boxes_xywh[:, 0], boxes_xywh[:, 1],
                        boxes_xywh[:, 2], boxes_xywh[:, 3])
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # 按类别分别做 NMS
        keep_all = []
        for c in np.unique(cls_ids):
            idx = np.where(cls_ids == c)[0]
            keep = self._nms(boxes[idx], confs[idx], self.iou_thres)
            keep_all.extend(idx[keep])
        keep_all = np.array(sorted(keep_all), dtype=int)
        if len(keep_all) == 0:
            return []

        boxes = boxes[keep_all]
        confs = confs[keep_all]
        cls_ids = cls_ids[keep_all]

        # 还原到原图坐标
        left, top = pad
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio

        oh, ow = orig_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

        results = []
        for b, c, s in zip(boxes, cls_ids, confs):
            results.append({
                'box': [float(v) for v in b],
                'type': CLASS_NAMES.get(int(c), f'class_{int(c)}'),
                'class_id': int(c),
                'score': float(s),
            })

        # 按阅读顺序排序 (先上后下, 再左到右)
        results.sort(key=lambda r: (r['box'][1], r['box'][0]))
        return results

    # -- 后处理: 去重与切分 --------------------------------------------------
    @staticmethod
    def _iou(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
        inter = iw * ih
        ua = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
        return inter / ua if ua > 0 else 0.0

    def _dedup_cross_class(self, results: List[Dict[str, Any]]
                           ) -> List[Dict[str, Any]]:
        """
        去除跨类重叠框: 同一位置同时被标为 embedding 与 isolated 时,
        保留置信度更高的那个。
        """
        if self.dedup_iou <= 0 or len(results) < 2:
            return results

        # 按置信度降序, 优先保留高置信
        ordered = sorted(results, key=lambda r: -r['score'])
        kept: List[Dict[str, Any]] = []
        for r in ordered:
            dup = False
            for k in kept:
                if (r['type'] != k['type'] and
                        self._iou(r['box'], k['box']) >= self.dedup_iou):
                    dup = True
                    break
            if not dup:
                kept.append(r)

        kept.sort(key=lambda r: (r['box'][1], r['box'][0]))
        return kept

    @staticmethod
    def _line_bbox(ln):
        """返回一行的 (top, bottom, left, right)"""
        top = min(c[1] for c in ln)
        bot = max(c[1] + c[3] for c in ln)
        left = min(c[0] for c in ln)
        right = max(c[0] + c[2] for c in ln)
        return top, bot, left, right

    @classmethod
    def _merge_fraction_groups(cls, lines, med_h):
        """
        合并被分数线拆开的分子/分母。

        策略: 分数线的典型形态是「一行的 x 范围显著窄于同行其他内容」
        且「与上一行间距很小」。这里不依赖单独的分数线检测, 而是判断
        相邻两行是否构成分式:
          - 垂直间距 < 1.8 * med_h (紧邻)
          - 较窄的那一行宽度 < 较宽行的 0.8 倍 (分子/分母比整行窄)
          - 两行 x 范围有重叠
        满足则合并。

        Args:
            lines: 聚类后的行列表
            med_h: 字符高度中位数

        Returns:
            合并后的行列表
        """
        if len(lines) < 2:
            return lines

        def bbox(ln):
            return cls._line_bbox(ln)

        # 按 top 排序
        ordered = sorted(lines, key=lambda ln: bbox(ln)[0])

        merged = [list(ordered[0])]
        for ln in ordered[1:]:
            prev = merged[-1]
            p_top, p_bot, p_left, p_right = bbox(prev)
            c_top, c_bot, c_left, c_right = bbox(ln)

            gap = c_top - p_bot
            p_w = p_right - p_left
            c_w = c_right - c_left
            x_overlap = min(p_right, c_right) - max(p_left, c_left)

            # 判定是否为分式的分子/分母对
            narrow_ratio = (min(p_w, c_w) / max(p_w, c_w)) if max(p_w, c_w) > 0 else 1.0
            is_fraction_pair = (
                0 <= gap < med_h * 1.8          # 紧邻 (允许少量重叠)
                and narrow_ratio < 0.8          # 一方明显更窄
                and x_overlap > 0               # x 有重叠
            )

            if is_fraction_pair:
                prev.extend(ln)
            else:
                merged.append(list(ln))
        return merged

    @classmethod
    def _split_box_by_components(cls, img: np.ndarray, box, gap_ratio: float = 0.45
                                 ) -> List[List[float]]:
        """
        把过高的检测框按「连通域 + 行聚类」切成多个子框。

        相比逐行投影, 该法能保留分式 (分子/分母) 的完整性:
        先做连通域行聚类, 再把被拆开的分子/分母合并回来。

        Args:
            img: 原图 (BGR)
            box: [x1, y1, x2, y2]
            gap_ratio: 行间距 > gap_ratio * 中位字符高 则分行

        Returns:
            子框列表; 无法切分时返回 [box]
        """
        H, W = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return [box]

        roi = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        n, _labels, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
        comps = []
        for i in range(1, n):
            cx, cy, cw, ch, area = stats[i]
            if area < 12:                        # 噪点
                continue
            if ch > roi.shape[0] * 0.9:          # 整幅边框
                continue
            comps.append((int(cx), int(cy), int(cw), int(ch),
                          int(area), float(cents[i][1])))

        if not comps:
            return [box]

        heights = sorted(c[3] for c in comps)
        med_h = max(1, heights[len(heights) // 2])

        # 按 y 中心排序后做行聚类
        comps.sort(key=lambda c: c[5])
        lines = [[comps[0]]]
        for c in comps[1:]:
            cur = lines[-1]
            cur_top = min(x[1] for x in cur)
            cur_bot = max(x[1] + x[3] for x in cur)
            c_top, c_bot = c[1], c[1] + c[3]
            overlap = min(cur_bot, c_bot) - max(cur_top, c_top)
            gap = max(c_top - cur_bot, cur_top - c_bot)
            if overlap > 0 or gap < med_h * gap_ratio:
                cur.append(c)
            else:
                lines.append([c])

        # 合并被分数线拆开的分子/分母
        lines = cls._merge_fraction_groups(lines, med_h)

        # 输出子框 (按 y 排序)
        boxes = []
        for ln in lines:
            top, bot, _l, _r = cls._line_bbox(ln)
            if bot - top < 15:
                continue
            pad = 5
            boxes.append([float(x1), float(max(0, y1 + top - pad)),
                          float(x2), float(min(H, y1 + bot + pad))])
        boxes.sort(key=lambda b: b[1])

        # 切出多个子框才有意义
        return boxes if len(boxes) > 1 else [box]

    def _split_tall_boxes(self, img: np.ndarray,
                          results: List[Dict[str, Any]]
                          ) -> Tuple[List[Dict[str, Any]], List[Dict]]:
        """
        切分过高的检测框 (修复跨行合并)。

        注意: 一个「合并框」往往和某个**已经正确检出的单题框**重叠
        (实测: 757px 合并框被切成 4 份, 其中 1 份与另一条 score 更高的
        独立检测框 IoU=0.85)。因此切分后必须剔除已被更好框覆盖的子框,
        否则会重复识别同一个公式。

        Returns:
            (切分后的结果列表, 切分记录)
        """
        if not self.split_tall_boxes or not results:
            return results, []

        heights = sorted(r['box'][3] - r['box'][1] for r in results)
        med_h = heights[len(heights) // 2]
        limit = med_h * self.tall_box_ratio

        # 先收集「不需要切分」的框, 它们是判断子框是否冗余的参照
        small = [r for r in results if (r['box'][3] - r['box'][1]) <= limit]
        tall = [r for r in results if (r['box'][3] - r['box'][1]) > limit]

        out: List[Dict[str, Any]] = list(small)
        records = []
        for r in tall:
            h = r['box'][3] - r['box'][1]
            subs = self._split_box_by_components(img, r['box'])

            if len(subs) <= 1:
                out.append(r)                     # 切不动, 原样保留
                continue

            # 剔除已被 small 中某个框覆盖的子框 (避免与独立检测重复)
            kept_subs = []
            dropped_subs = []
            for s in subs:
                dup = None
                for k in small:
                    if self._iou(s, k['box']) >= self.split_child_iou:
                        dup = k
                        break
                if dup is None:
                    kept_subs.append(s)
                else:
                    dropped_subs.append({'child': s, 'covered_by': dup['box'],
                                         'covered_by_score': dup['score']})

            if not kept_subs:
                # 整个合并框都被独立检测覆盖 -> 丢弃, 避免重复识别
                records.append({'original': r['box'], 'height': h,
                                'limit': limit, 'n_split': 0,
                                'children': [], 'dropped_children': dropped_subs,
                                'action': 'discard_tall_box'})
                continue

            records.append({'original': r['box'], 'height': h,
                            'limit': limit, 'n_split': len(kept_subs),
                            'children': kept_subs,
                            'dropped_children': dropped_subs,
                            'action': 'split'})
            for s in kept_subs:
                out.append({
                    'box': s,
                    'type': r['type'],
                    'class_id': r['class_id'],
                    'score': r['score'],
                    'split_from': r['box'],
                })

        out.sort(key=lambda r: (r['box'][1], r['box'][0]))
        return out, records

    # -- 主接口 -------------------------------------------------------------
    def __call__(self, img, return_debug: bool = False):
        """
        检测公式区域。

        Args:
            img: BGR 图像 (numpy) 或图片路径
            return_debug: 为 True 时返回 (results, debug), debug 含
                          原始框 / 切分记录 / 中间张量

        Returns:
            results: [{'box': [x1,y1,x2,y2], 'type': ..., 'score': ...}, ...]
            (return_debug=True 时) (results, debug)
        """
        if isinstance(img, (str, Path)):
            img = cv2.imread(str(img))
            if img is None:
                raise FileNotFoundError(f'无法读取图片: {img}')

        orig_shape = img.shape[:2]
        blob, ratio, pad = self._preprocess(img)

        t0 = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        self.last_time = time.time() - t0

        raw = self._postprocess(outputs[0], ratio, pad, orig_shape)
        deduped = self._dedup_cross_class(raw)
        final, split_records = self._split_tall_boxes(img, deduped)

        if not return_debug:
            return final

        debug = {
            'input_shape': list(orig_shape),
            'imgsz': self.imgsz,
            'letterbox_ratio': ratio,
            'letterbox_pad': list(pad),
            'raw_output_shape': list(outputs[0].shape),
            'raw_boxes': raw,
            'after_dedup': deduped,
            'split_records': split_records,
            'final_boxes': final,
        }
        return final, debug

    def detect(self, img, return_debug: bool = False):
        return self(img, return_debug=return_debug)

    # -- 可视化 -------------------------------------------------------------
    def draw(self, img: np.ndarray,
             results: List[Dict[str, Any]]) -> np.ndarray:
        """绘制检测框（使用 PIL 渲染标签以支持中文）。"""
        vis = img.copy()
        colors = {
            'embedding': (0, 165, 255),   # 橙 (BGR)
            'isolated': (0, 200, 0),      # 绿
        }
        for item in results:
            x1, y1, x2, y2 = [int(v) for v in item['box']]
            c = colors.get(item['type'], (255, 0, 0))
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)

        # PIL 渲染标签
        try:
            from PIL import Image, ImageDraw, ImageFont
            font = None
            for cand in (r'C:\Windows\Fonts\msyh.ttc',
                         r'C:\Windows\Fonts\simhei.ttf'):
                if os.path.exists(cand):
                    font = ImageFont.truetype(cand, 18)
                    break
            if font:
                pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                d = ImageDraw.Draw(pil)
                for item in results:
                    x1, y1, x2, y2 = [int(v) for v in item['box']]
                    label = f"{item['type']} {item['score']:.2f}"
                    ty = y1 - 22 if y1 - 22 > 0 else y2 + 2
                    bbox = d.textbbox((x1, ty), label, font=font)
                    d.rectangle([bbox[0] - 2, bbox[1] - 1,
                                 bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255))
                    rgb = (255, 120, 0) if item['type'] == 'embedding' else (0, 150, 0)
                    d.text((x1, ty), label, fill=rgb, font=font)
                vis = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                return vis
        except Exception:
            pass

        for item in results:
            x1, y1, x2, y2 = [int(v) for v in item['box']]
            c = colors.get(item['type'], (255, 0, 0))
            cv2.putText(vis, f"{item['type']} {item['score']:.2f}",
                        (x1, max(y1 - 5, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        return vis


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    det = FormulaDetectorONNX()
    target = sys.argv[1] if len(sys.argv) > 1 else 'weights/onnx_ocr/formula_test_input.jpg'
    res = det(target)
    print(f'\n检测到 {len(res)} 个公式区域:')
    for r in res:
        print(f"  {r['type']:10} score={r['score']:.3f}  box={[int(v) for v in r['box']]}")
