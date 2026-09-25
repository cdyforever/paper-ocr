"""
Pure ONNX Page-Level Pipeline (文本 / 公式 解耦)
================================================

背景
----
原流程是「DBNet 先切文本行 -> 在每一行内部再做公式检测」。
实测发现这个顺序对公式密集的页面是结构性失效的:

    * 公式区域平均只有 43.9% 的面积落在 DBNet 行框内
    * 31 个公式区域里 17 个 (55%) 可用率 < 50%, 属于完全丢失
    * 行内 MFD 只能在 31 个公式中找到 6 个 (81% 丢失)

原因不是 DBNet 模型差, 而是它的训练目标是「横向文本行」,
公式 (分式/根式/上下标/积分号) 不是它的目标形态, 因此经常
只框住公式的一小部分或干脆断开。

解决思路
--------
把顺序反过来 —— **页面级公式检测先行, 再做文本检测**:

    1. MFD (YOLOv8) 在整页上检测公式区域          <- 公式不依赖 DBNet
    2. 把这些区域涂白, 得到「纯文本图」
    3. DBNet 只在纯文本图上切行                    <- 文本不受公式干扰
    4. 文本行 -> PP-OCRv4 识别;  公式区域 -> UniMERNet 识别
    5. 按阅读顺序合并

好处:
    * 公式区域 100% 完整交给 MFR (实测覆盖度 59.3% -> 100%)
    * DBNet 不再被公式的碎片干扰, 文本行更干净
    * 公式检测/识别与 DBNet 完全解耦, 各自可用

用法:
    from func.algorithm.onnx_page_pipeline import OnnxPagePipeline

    pipe = OnnxPagePipeline()
    items, debug = pipe(image, return_debug=True)
    print(pipe.to_markdown(items))
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 两个模块都是「纯 ONNX」，直接按文件路径导入以免触发 legacy 包的
# paddle 依赖（func/algorithm/__init__.py -> paper_ocr -> paddle）。
import importlib.util as _ilu
import sys as _sys


def _load_sibling(name: str):
    key = f'_onnx_page_{name}'
    if key in _sys.modules:
        return _sys.modules[key]
    path = Path(__file__).resolve().parent / f'{name}.py'
    spec = _ilu.spec_from_file_location(key, path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


class OnnxPagePipeline:
    """
    页面级纯 ONNX 流水线: 公式检测 -> 涂白 -> 文本检测/识别 -> 公式识别 -> 合并。

    与旧流程的区别: 公式检测在**页面级**执行, 不依赖 DBNet 的行切分结果。
    """

    def __init__(self,
                 use_gpu: bool = False,
                 verbose: bool = True,
                 # 文本检测
                 det_limit_side_len: int = 960,
                 det_db_thresh: float = 0.3,
                 det_db_box_thresh: float = 0.6,
                 det_db_unclip_ratio: float = 1.5,
                 # 公式检测
                 mfd_conf_thres: float = 0.25,
                 mfd_split_tall_boxes: bool = True,
                 # 涂白
                 mask_margin: int = 6,
                 mask_mode: str = 'mask',
                 drop_line_formula_ratio: float = 0.5,
                 # 公式识别
                 max_new_tokens: int = 512):
        """
        Args:
            mask_margin: 涂白时向公式框外扩的像素数 (防止笔画残留)
            mask_mode:
                'mask'   -- 公式区域涂白后再跑 DBNet (推荐)
                'filter' -- DBNet 照常跑, 事后丢弃大部分落在公式内的行
                'none'   -- 不做任何处理 (对照组)
            drop_line_formula_ratio: mask_mode='filter' 时,
                行框落在公式内的面积占比超过该值就丢弃
        """
        self.verbose = verbose
        self.mask_margin = int(mask_margin)
        self.mask_mode = mask_mode
        self.drop_line_formula_ratio = float(drop_line_formula_ratio)

        eng_mod = _load_sibling('onnx_ocr_engine')
        mfd_mod = _load_sibling('formula_detect_onnx')
        mfr_mod = _load_sibling('formula_recognize_onnx')

        t0 = time.time()
        self.ocr = eng_mod.OnnxOCREngine(
            use_gpu=use_gpu, verbose=False,
            det_limit_side_len=det_limit_side_len,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio)
        self.mfd = mfd_mod.FormulaDetectorONNX(
            use_gpu=use_gpu, verbose=False,
            conf_thres=mfd_conf_thres,
            split_tall_boxes=mfd_split_tall_boxes)
        self.mfr = mfr_mod.FormulaRecognizerONNX(
            use_gpu=use_gpu, verbose=False,
            max_new_tokens=max_new_tokens)

        if self.verbose:
            print(f'[OnnxPagePipeline] 三个模型加载完成 {time.time()-t0:.2f}s '
                  f'(mask_mode={mask_mode})')

    # -- 几何工具 -----------------------------------------------------------
    @staticmethod
    def _quad_to_xyxy(box) -> List[float]:
        b = np.array(box, dtype=np.float32).reshape(-1, 2)
        return [float(b[:, 0].min()), float(b[:, 1].min()),
                float(b[:, 0].max()), float(b[:, 1].max())]

    @staticmethod
    def _inter_area(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _mask_regions(self, img: np.ndarray, boxes: List[List[float]],
                      margin: Optional[int] = None) -> np.ndarray:
        """把指定矩形区域涂白, 得到「纯文本图」。"""
        m = self.mask_margin if margin is None else margin
        H, W = img.shape[:2]
        out = img.copy()
        for b in boxes:
            x1 = max(0, int(b[0]) - m)
            y1 = max(0, int(b[1]) - m)
            x2 = min(W, int(b[2]) + m)
            y2 = min(H, int(b[3]) + m)
            if x2 > x1 and y2 > y1:
                out[y1:y2, x1:x2] = 255
        return out

    def _formula_ratio(self, box: List[float],
                       formulas: List[Dict[str, Any]]) -> float:
        """行框面积中被公式区域覆盖的比例。"""
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area <= 0:
            return 0.0
        cov = sum(self._inter_area(box, f['box']) for f in formulas)
        return min(1.0, cov / area)

    @staticmethod
    def _cluster_lines(items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """把相邻的元素聚成「视觉行」: 垂直重叠超过较矮者一半即视为同一行。"""
        if not items:
            return []
        items = sorted(items, key=lambda r: (r['box'][1], r['box'][0]))
        lines: List[List[Dict[str, Any]]] = []
        for it in items:
            b = it['box']
            h = b[3] - b[1]
            placed = False
            for ln in lines:
                ref = ln[0]['box']
                ref_h = ref[3] - ref[1]
                ov = min(b[3], ref[3]) - max(b[1], ref[1])
                if ov > 0.5 * min(h, ref_h):
                    ln.append(it)
                    placed = True
                    break
            if not placed:
                lines.append([it])
        for ln in lines:
            ln.sort(key=lambda r: r['box'][0])
        lines.sort(key=lambda ln: min(r['box'][1] for r in ln))
        return lines

    @classmethod
    def _reading_order(cls, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按阅读顺序排序 (先按行聚合, 行内从左到右)。"""
        return [it for ln in cls._cluster_lines(items) for it in ln]

    # -- 主流程 -------------------------------------------------------------
    def __call__(self,
                 img,
                 return_debug: bool = False,
                 recognize_formula: bool = True,
                 recognize_text: bool = True):
        """
        Args:
            img: BGR 图像或路径
            return_debug: 返回 (items, debug)
            recognize_formula: 是否跑 MFR (关闭可只测检测)
            recognize_text: 是否跑文本识别 (关闭可只测检测)

        Returns:
            items: 按阅读顺序排列的列表, 每项:
                {'type': 'text',   'box': [x1,y1,x2,y2], 'text': str, 'score': float}
                {'type': 'formula','box': [x1,y1,x2,y2], 'latex': str,
                 'formula_type': 'embedding'|'isolated', 'score': float}
        """
        if isinstance(img, (str, Path)):
            img = cv2.imread(str(img))
            if img is None:
                raise FileNotFoundError(f'无法读取图片: {img}')

        H, W = img.shape[:2]
        debug: Dict[str, Any] = {'orig_shape': [H, W], 'mask_mode': self.mask_mode}

        # ---- 1) 页面级公式检测 (不依赖 DBNet) ----
        t0 = time.time()
        formulas, mfd_debug = self.mfd(img, return_debug=True)
        debug['mfd_time'] = time.time() - t0
        debug['formula_boxes'] = [dict(f) for f in formulas]
        debug['mfd_raw_boxes'] = mfd_debug['raw_boxes']
        debug['mfd_after_dedup'] = mfd_debug['after_dedup']
        debug['mfd_split_records'] = mfd_debug['split_records']

        # ---- 2) 准备文本图 ----
        fboxes = [f['box'] for f in formulas]
        if self.mask_mode == 'mask':
            text_img = self._mask_regions(img, fboxes)
        else:
            text_img = img

        # ---- 3) 文本检测 + 识别 ----
        t0 = time.time()
        if recognize_text:
            text_results, det_debug = self.ocr(text_img, return_debug=True)
        else:
            boxes, scores, det_debug = self.ocr.detect(text_img, return_debug=True)
            text_results = [{'box': b.astype(int).tolist(), 'text': '', 'score': 0.0}
                            for b in boxes]
        debug['ocr_time'] = time.time() - t0
        debug['det_debug'] = det_debug

        # 记录 DBNet 在原图 (未涂白) 上的表现, 用于对照
        if self.mask_mode == 'mask':
            raw_boxes, _ = self.ocr.detect(img)
            debug['det_boxes_original'] = [self._quad_to_xyxy(b) for b in raw_boxes]
        else:
            debug['det_boxes_original'] = [self._quad_to_xyxy(r['box'])
                                           for r in text_results]

        # ---- 4) 过滤: 丢弃仍然大面积落在公式内的行 ----
        # 仅在 'filter' 模式生效; 'none' 是对照组, 不做任何处理
        kept_text: List[Dict[str, Any]] = []
        dropped_text: List[Dict[str, Any]] = []
        for r in text_results:
            box = self._quad_to_xyxy(r['box'])
            ratio = self._formula_ratio(box, formulas)
            if ratio >= self.drop_line_formula_ratio and self.mask_mode == 'filter':
                dropped_text.append({'box': box, 'text': r['text'],
                                     'score': r['score'], 'formula_ratio': ratio})
                continue
            kept_text.append({'type': 'text', 'box': box, 'text': r['text'],
                              'score': r['score'], 'formula_ratio': ratio})
        debug['dropped_text_lines'] = dropped_text
        debug['text_lines'] = kept_text

        # ---- 5) 公式识别 (在原图上裁剪, 保证笔画完整) ----
        t0 = time.time()
        formula_items: List[Dict[str, Any]] = []
        for f in formulas:
            x1, y1, x2, y2 = [int(v) for v in f['box']]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            item = {'type': 'formula', 'box': [float(v) for v in f['box']],
                    'formula_type': f['type'], 'score': float(f['score']),
                    'latex': ''}
            if recognize_formula and x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                try:
                    item['latex'] = self.mfr(crop)
                except Exception as exc:      # 单条失败不影响整页
                    item['latex'] = ''
                    item['error'] = str(exc)
            formula_items.append(item)
        debug['mfr_time'] = time.time() - t0
        debug['formula_items'] = formula_items

        # ---- 6) 合并 + 阅读顺序 ----
        items = self._reading_order(kept_text + formula_items)
        debug['n_text'] = len(kept_text)
        debug['n_formula'] = len(formula_items)
        debug['n_total'] = len(items)

        if return_debug:
            return items, debug
        return items

    # -- 输出辅助 -----------------------------------------------------------
    @classmethod
    def to_markdown(cls, items: List[Dict[str, Any]],
                    line_gap: float = 1.6) -> str:
        """
        把结果拼成 Markdown: 行内公式用 `$...$`, 独立公式用 `$$...$$`。

        Args:
            line_gap: 两行间距超过「中位行高 * 该值」时插入空行分段
        """
        items = [it for it in items if it.get('box')]
        if not items:
            return ''
        lines = cls._cluster_lines(items)

        heights = [it['box'][3] - it['box'][1] for it in items]
        med_h = float(np.median(heights)) if heights else 20.0

        out: List[str] = []
        prev_bottom = None
        for ln in lines:
            top = min(it['box'][1] for it in ln)
            bottom = max(it['box'][3] for it in ln)
            if (prev_bottom is not None and
                    top - prev_bottom > med_h * line_gap):
                out.append('')
            prev_bottom = bottom

            # 整行只有一个独立公式 -> 块级公式
            if (len(ln) == 1 and ln[0]['type'] == 'formula'
                    and ln[0].get('formula_type') == 'isolated'
                    and ln[0].get('latex', '').strip()):
                out.append(f"$$\n{ln[0]['latex'].strip()}\n$$")
                out.append('')
                continue

            parts: List[str] = []
            for it in ln:
                if it['type'] == 'text':
                    txt = it.get('text', '').strip()
                    if txt:
                        parts.append(txt)
                else:
                    tex = it.get('latex', '').strip()
                    if not tex:
                        continue          # 识别失败/未开启 -> 不留空 $ $
                    if it.get('formula_type') == 'isolated':
                        parts.append(f'$${tex}$$')
                    else:
                        parts.append(f'${tex}$')
            if parts:
                out.append(' '.join(parts))

        # 折叠连续空行
        cleaned: List[str] = []
        for ln in out:
            if ln == '' and cleaned and cleaned[-1] == '':
                continue
            cleaned.append(ln)
        return '\n'.join(cleaned).strip()

    def draw(self, img: np.ndarray, items: List[Dict[str, Any]]) -> np.ndarray:
        """文本框绿色 / 行内公式橙色 / 独立公式蓝色。"""
        vis = img.copy()
        for it in items:
            x1, y1, x2, y2 = [int(v) for v in it['box']]
            if it['type'] == 'text':
                c, tag = (0, 200, 0), it.get('text', '')[:28]
            elif it.get('formula_type') == 'isolated':
                c, tag = (255, 120, 0), 'formula: ' + it.get('latex', '')[:24]
            else:
                c, tag = (0, 165, 255), it.get('latex', '')[:24]
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 3)
            if tag:
                cv2.putText(vis, tag, (x1, max(y1 - 6, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        return vis

    def close(self):
        pass


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    pipe = OnnxPagePipeline()
    target = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
    items, dbg = pipe(target, return_debug=True)
    print(f"\n文本 {dbg['n_text']} 行 / 公式 {dbg['n_formula']} 个")
    print('-' * 70)
    for it in items:
        if it['type'] == 'text':
            print(f"[text   ] {it['text']}")
        else:
            print(f"[formula] {it['formula_type']:9} {it['latex']}")
