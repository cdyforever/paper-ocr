"""
Pure ONNX OCR Engine
====================

A complete OCR pipeline (text detection + text recognition) that runs
**only** on ONNX Runtime. No PaddlePaddle / PaddleOCR framework needed at
inference time.

Models (converted from PaddleOCR PP-OCRv4):
  - Detection : weights/text_line_detect/dbnet_det.onnx      (DBNet)
  - Recognition: weights/text_recognition/crnn_rec.onnx       (PP-OCRv4 Rec)
  - Dictionary: weights/text_recognition/ppocr_keys_v1.txt

Usage:
    from func.algorithm.onnx_ocr_engine import OnnxOCREngine

    engine = OnnxOCREngine()
    results = engine(image)
    for r in results:
        print(r['text'], r['score'])
"""

import math
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort

try:
    import pyclipper
    from shapely.geometry import Polygon
    _HAS_CLIPPER = True
except ImportError:  # pragma: no cover
    _HAS_CLIPPER = False


# ----------------------------------------------------------------------------
# Default paths (resolved relative to the project root)
# ----------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../paper-ocr
DEFAULT_DET_MODEL = _PROJECT_ROOT / 'weights' / 'text_line_detect' / 'dbnet_det.onnx'
DEFAULT_REC_MODEL = _PROJECT_ROOT / 'weights' / 'text_recognition' / 'crnn_rec.onnx'
DEFAULT_DICT = _PROJECT_ROOT / 'weights' / 'text_recognition' / 'ppocr_keys_v1.txt'


# ============================================================================
# Detection post-processing (DBNet, faithful port of PaddleOCR DBPostProcess)
# ============================================================================
class DBPostProcess:
    """DBNet post-processing: bitmap -> quadrilateral boxes."""

    def __init__(self,
                 thresh: float = 0.3,
                 box_thresh: float = 0.6,
                 max_candidates: int = 1000,
                 unclip_ratio: float = 1.5,
                 min_size: int = 3):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
        """Mean score of the bitmap inside the box (fast, bbox based)."""
        h, w = bitmap.shape[:2]
        box = box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0])

    @staticmethod
    def _get_mini_boxes(contour: np.ndarray) -> Tuple[List, float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def _unclip(self, box: np.ndarray) -> np.ndarray:
        if not _HAS_CLIPPER:
            # Fallback: simple expansion by unclip_ratio using the bbox
            return box
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return np.array(expanded[0]) if len(expanded) else box

    # -- main ---------------------------------------------------------------
    def boxes_from_bitmap(self,
                          pred: np.ndarray,
                          bitmap: np.ndarray,
                          dest_width: int,
                          dest_height: int) -> Tuple[np.ndarray, List[float]]:
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)

        boxes, scores = [], []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            points = np.array(points)
            score = self._box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self._unclip(points).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int32))
            scores.append(score)

        if not boxes:
            return np.zeros((0, 4, 2), dtype=np.int32), []
        return np.array(boxes, dtype=np.int32), scores

    def __call__(self, pred: np.ndarray, shape_list: List) -> List[Dict]:
        """
        Args:
            pred: probability map of shape (1, 1, H, W)
            shape_list: list of [src_h, src_w, ratio_h, ratio_w] per image

        Returns:
            list of dicts with 'points' (N,4,2) and 'scores' (list of float)
        """
        segmentation = pred > self.thresh
        results = []

        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            mask = segmentation[batch_index, 0]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index, 0], mask, src_w, src_h)
            results.append({'points': boxes, 'scores': scores})

        return results


# ============================================================================
# CTC label decoder (faithful port of PaddleOCR CTCLabelDecode)
# ============================================================================
class CTCLabelDecode:
    """Greedy CTC decoder using a character dictionary."""

    def __init__(self, dict_path: str, use_space_char: bool = True):
        self.character = self._load_dict(dict_path, use_space_char)
        self.blank_idx = 0

    @staticmethod
    def _load_dict(dict_path: str, use_space_char: bool) -> List[str]:
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f'Character dictionary not found: {dict_path}')
        with open(dict_path, 'r', encoding='utf-8') as f:
            chars = [line.rstrip('\n') for line in f]
        # PaddleOCR layout: ['blank'] + dict + [' ']
        chars = ['blank'] + chars
        if use_space_char:
            chars.append(' ')
        return chars

    def __call__(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        """
        Args:
            preds: array of shape (B, T, C) with softmax probabilities

        Returns:
            list of (text, mean_score)
        """
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        results = []
        for b in range(preds_idx.shape[0]):
            idx_seq = preds_idx[b]
            prob_seq = preds_prob[b]

            selection = np.ones(len(idx_seq), dtype=bool)
            selection[1:] = idx_seq[1:] != idx_seq[:-1]   # remove duplicates
            selection &= idx_seq != self.blank_idx          # remove blank

            chars, confs = [], []
            for i in np.where(selection)[0]:
                idx = int(idx_seq[i])
                if 0 <= idx < len(self.character):
                    chars.append(self.character[idx])
                    confs.append(float(prob_seq[i]))

            text = ''.join(chars)
            score = float(np.mean(confs)) if confs else 0.0
            results.append((text, score))
        return results


# ============================================================================
# Box utilities
# ============================================================================
def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop a (possibly rotated) quadrilateral region via perspective transform."""
    points = np.array(points, dtype=np.float32)
    img_crop_width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])))
    img_crop_width = max(img_crop_width, 1)
    img_crop_height = max(img_crop_height, 1)

    pts_std = np.float32([[0, 0],
                          [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)

    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def sorted_boxes(dt_boxes) -> List[np.ndarray]:
    """Sort boxes top-to-bottom, then left-to-right (PaddleOCR logic)."""
    boxes = [np.asarray(b) for b in dt_boxes]
    num_boxes = len(boxes)
    _boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break
    return _boxes


def sorted_boxes_with_scores(dt_boxes, scores):
    """
    Same ordering as :func:`sorted_boxes`, but keeps the parallel
    detection-score array aligned with the reordered boxes.
    """
    boxes = [np.asarray(b) for b in dt_boxes]
    n = len(boxes)
    if n == 0:
        return [], []

    idx = sorted(range(n), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
    for i in range(n - 1):
        for j in range(i, -1, -1):
            a, b = boxes[idx[j + 1]], boxes[idx[j]]
            if abs(a[0][1] - b[0][1]) < 10 and (a[0][0] < b[0][0]):
                idx[j], idx[j + 1] = idx[j + 1], idx[j]
            else:
                break

    return [boxes[i] for i in idx], [scores[i] for i in idx]


# ============================================================================
# Main engine
# ============================================================================
class OnnxOCREngine:
    """
    Pure ONNX OCR engine (detection + recognition).

    No PaddlePaddle dependency: only onnxruntime + opencv + numpy.
    """

    def __init__(self,
                 det_model_path: str = str(DEFAULT_DET_MODEL),
                 rec_model_path: str = str(DEFAULT_REC_MODEL),
                 dict_path: str = str(DEFAULT_DICT),
                 use_gpu: bool = False,
                 det_limit_side_len: int = 960,
                 det_db_thresh: float = 0.3,
                 det_db_box_thresh: float = 0.6,
                 det_db_unclip_ratio: float = 1.5,
                 rec_image_height: int = 48,
                 rec_batch_num: int = 6,
                 verbose: bool = True):
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.dict_path = dict_path
        self.use_gpu = use_gpu

        # Detection params
        self.det_limit_side_len = det_limit_side_len
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio

        # Recognition params
        self.rec_image_height = rec_image_height
        self.rec_batch_num = rec_batch_num

        self.verbose = verbose

        # Execution providers
        available = ort.get_available_providers()
        if use_gpu and 'CUDAExecutionProvider' in available:
            providers = [('CUDAExecutionProvider', {'device_id': 0}),
                         'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.providers = providers

        # Load models
        t0 = time.time()
        self._load_det_model()
        self._load_rec_model()
        self.decoder = CTCLabelDecode(dict_path)
        self.postprocess_op = DBPostProcess(
            thresh=det_db_thresh,
            box_thresh=det_db_box_thresh,
            unclip_ratio=det_db_unclip_ratio)

        if self.verbose:
            print(f'[OnnxOCREngine] models loaded in {time.time() - t0:.2f}s '
                  f'(providers={providers})')
            print(f'[OnnxOCREngine] dict size = {len(self.decoder.character)}')

    # -- model loading ------------------------------------------------------
    def _load_det_model(self):
        if not os.path.exists(self.det_model_path):
            raise FileNotFoundError(f'Detection model not found: {self.det_model_path}')
        self.det_session = ort.InferenceSession(self.det_model_path, providers=self.providers)
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_output_name = self.det_session.get_outputs()[0].name

    def _load_rec_model(self):
        if not os.path.exists(self.rec_model_path):
            raise FileNotFoundError(f'Recognition model not found: {self.rec_model_path}')
        self.rec_session = ort.InferenceSession(self.rec_model_path, providers=self.providers)
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_output_name = self.rec_session.get_outputs()[0].name

    # -- detection ----------------------------------------------------------
    @staticmethod
    def _resize_image_type0(img: np.ndarray, limit_side_len: int):
        """Resize keeping aspect ratio; sides rounded to multiples of 32."""
        h, w, _ = img.shape
        if max(h, w) > limit_side_len:
            ratio = float(limit_side_len) / (h if h > w else w)
        else:
            ratio = 1.0

        resize_h = max(int(round(h * ratio / 32) * 32), 32)
        resize_w = max(int(round(w * ratio / 32) * 32), 32)
        return cv2.resize(img, (resize_w, resize_h)), ratio

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        """scale 1/255, mean/std ImageNet, HWC -> CHW."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        scale = 1.0 / 255.0
        data = img.astype(np.float32) * scale
        data = (data - mean) / std
        return data.transpose(2, 0, 1)

    def detect(self, img: np.ndarray, return_debug: bool = False):
        """
        Run text detection.

        Args:
            img: BGR image
            return_debug: also return intermediate tensors (probability map,
                          binarized bitmap, resized input) for inspection.

        Returns:
            (boxes [N,4,2] int32, scores list)
            or (boxes, scores, debug) when return_debug=True
        """
        ori_h, ori_w = img.shape[:2]

        resized, ratio = self._resize_image_type0(img, self.det_limit_side_len)
        resized_h, resized_w = resized.shape[:2]

        norm = self._normalize(resized)
        inp = np.expand_dims(norm, axis=0)

        t0 = time.time()
        preds = self.det_session.run([self.det_output_name], {self.det_input_name: inp})[0]
        self.det_time = time.time() - t0

        # ratio_h, ratio_w = original / resized
        ratio_h = ori_h / resized_h
        ratio_w = ori_w / resized_w
        shape_list = [[ori_h, ori_w, ratio_h, ratio_w]]

        result = self.postprocess_op(preds, shape_list)[0]
        boxes = result['points']
        scores = result['scores']

        if not return_debug:
            return boxes, scores

        prob_map = preds[0, 0]                      # (H, W) probability map
        bitmap = (prob_map > self.postprocess_op.thresh).astype(np.uint8)
        debug = {
            'orig_shape': [ori_h, ori_w],
            'resized_shape': [resized_h, resized_w],
            'limit_side_len': self.det_limit_side_len,
            'scale_ratio': ratio,
            'det_time': self.det_time,
            'prob_map': prob_map,                   # float32 (H, W)
            'prob_min': float(prob_map.min()),
            'prob_max': float(prob_map.max()),
            'prob_mean': float(prob_map.mean()),
            'bitmap': bitmap,                       # uint8 (H, W) 0/1
            'bitmap_ratio': float(bitmap.mean()),
            'thresh': self.postprocess_op.thresh,
            'box_thresh': self.postprocess_op.box_thresh,
            'unclip_ratio': self.postprocess_op.unclip_ratio,
            'num_boxes': len(boxes),
            'boxes': boxes,
            'scores': scores,
        }
        return boxes, scores, debug

    # -- recognition --------------------------------------------------------
    def _resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        """Resize + normalize a single text-line crop for the rec model."""
        img_c, img_h = 3, self.rec_image_height
        img_w = int(img_h * max_wh_ratio)

        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = img_w if math.ceil(img_h * ratio) > img_w else int(math.ceil(img_h * ratio))
        resized_w = max(resized_w, 1)

        resized = cv2.resize(img, (resized_w, img_h))
        resized = resized.astype(np.float32)
        resized = resized.transpose((2, 0, 1)) / 255.0
        resized -= 0.5
        resized /= 0.5

        padding = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padding[:, :, 0:resized_w] = resized
        return padding

    def recognize(self, img_crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Run text recognition on a list of cropped text-line images.

        Returns:
            list of (text, score)
        """
        if not img_crops:
            return []

        img_h = self.rec_image_height
        max_wh_ratio = 0.0
        for crop in img_crops:
            h, w = crop.shape[:2]
            max_wh_ratio = max(max_wh_ratio, w / float(h))

        results: List[Tuple[str, float]] = []
        t0 = time.time()

        for start in range(0, len(img_crops), self.rec_batch_num):
            batch = img_crops[start:start + self.rec_batch_num]
            norm_imgs = [self._resize_norm_img(c, max_wh_ratio) for c in batch]
            inp = np.stack(norm_imgs, axis=0)

            preds = self.rec_session.run([self.rec_output_name], {self.rec_input_name: inp})[0]
            results.extend(self.decoder(preds))

        self.rec_time = time.time() - t0
        return results

    # -- full pipeline ------------------------------------------------------
    def __call__(self,
                 img: np.ndarray,
                 det: bool = True,
                 rec: bool = True,
                 cls: bool = False,
                 return_crops: bool = False,
                 return_debug: bool = False):
        """
        Run the full OCR pipeline.

        Args:
            img: BGR image (numpy array) or path to an image
            det: run detection
            rec: run recognition
            cls: unused (kept for API compatibility)
            return_crops: include cropped images in the result
            return_debug: also return detection intermediates
                          (prob map / bitmap / raw boxes)

        Returns:
            list of dicts: {'box', 'text', 'score', 'crop'(optional)}
            or (results, debug) when return_debug=True
        """
        if isinstance(img, (str, Path)):
            img = cv2.imread(str(img))
            if img is None:
                raise FileNotFoundError(f'Could not read image: {img}')

        if not det:
            # Recognize the whole image as a single text line
            crops = [img]
            decoded = self.recognize(crops) if rec else [('', 1.0)]
            out = []
            for (text, score) in decoded:
                item = {'box': None, 'text': text, 'score': score}
                if return_crops:
                    item['crop'] = img
                out.append(item)
            if return_debug:
                return out, {'det': None, 'note': 'det=False'}
            return out

        # 1) detection
        if return_debug:
            boxes, det_scores, det_debug = self.detect(img, return_debug=True)
        else:
            boxes, det_scores = self.detect(img)
            det_debug = None

        if len(boxes) == 0:
            if return_debug:
                return [], {'det': det_debug}
            return []

        # 2) sort boxes (keeping detection scores aligned)
        dt_boxes, det_scores = sorted_boxes_with_scores(boxes, det_scores)

        # 3) crop text lines
        crops = []
        valid_boxes = []
        valid_det_scores = []
        for box, dscore in zip(dt_boxes, det_scores):
            crop = get_rotate_crop_image(img, np.array(box, dtype=np.float32))
            if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            crops.append(crop)
            valid_boxes.append(box)
            valid_det_scores.append(dscore)

        if not crops:
            if return_debug:
                return [], {'det': det_debug}
            return []

        # 4) recognition
        if rec:
            decoded = self.recognize(crops)
        else:
            decoded = [('', 1.0)] * len(crops)

        # 5) assemble results
        out = []
        dropped = []
        for i, (box, (text, score)) in enumerate(zip(valid_boxes, decoded)):
            # Drop empty / very low confidence predictions (false positives
            # such as noise regions the detector fired on).
            if not text.strip() or score < 0.5:
                dropped.append({'box': box.astype(int).tolist(),
                                'text': text, 'score': float(score)})
                continue
            item = {
                'box': box.astype(int).tolist(),
                'text': text,
                'score': float(score),
                'det_score': float(valid_det_scores[i]),
            }
            if return_crops:
                item['crop'] = crops[i]
            out.append(item)

        if return_debug:
            det_debug['num_sorted'] = len(dt_boxes)
            det_debug['num_crops'] = len(crops)
            det_debug['num_final'] = len(out)
            det_debug['dropped'] = dropped
            return out, {'det': det_debug}
        return out

    # -- convenience --------------------------------------------------------
    def detect_only(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text boxes only."""
        boxes, scores = self.detect(img)
        if len(boxes) == 0:
            return []
        return [{'box': b.astype(int).tolist(), 'score': float(s)}
                for b, s in zip(sorted_boxes(boxes), scores)]

    def draw(self, img: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Draw boxes + recognized text onto a copy of the image.

        Uses PIL for label rendering so that CJK text displays correctly
        (cv2.putText cannot render non-ASCII glyphs).
        """
        vis = img.copy()
        for item in results:
            box = np.array(item['box'], dtype=np.int32)
            cv2.polylines(vis, [box], True, (0, 255, 0), 2)

        # Render labels with PIL so Chinese characters are shown properly.
        try:
            from PIL import Image, ImageDraw, ImageFont

            font = None
            for candidate in (r'C:\Windows\Fonts\msyh.ttc',
                              r'C:\Windows\Fonts\simhei.ttf',
                              r'C:\Windows\Fonts\simsun.ttc'):
                if os.path.exists(candidate):
                    font = ImageFont.truetype(candidate, 20)
                    break

            if font is not None:
                pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil)
                for item in results:
                    box = np.array(item['box'], dtype=np.int32)
                    text = item.get('text', '')
                    score = item.get('score', 0.0)
                    label = f'{text} ({score:.2f})'
                    x, y = int(box[0][0]), int(box[0][1]) - 24
                    if y < 0:
                        y = int(box[2][1]) + 4
                    # background for readability
                    bbox = draw.textbbox((x, y), label, font=font)
                    draw.rectangle([bbox[0] - 2, bbox[1] - 1,
                                    bbox[2] + 2, bbox[3] + 1],
                                   fill=(255, 255, 255))
                    draw.text((x, y), label, fill=(200, 0, 0), font=font)
                vis = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                return vis
        except Exception:
            pass  # fall through to cv2 rendering

        for item in results:
            box = np.array(item['box'], dtype=np.int32)
            text = item.get('text', '')
            score = item.get('score', 0.0)
            label = f'{text} ({score:.2f})'
            x, y = int(box[0][0]), max(int(box[0][1]) - 5, 15)
            cv2.putText(vis, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        return vis


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    engine = OnnxOCREngine()
    target = sys.argv[1] if len(sys.argv) > 1 else 'test_standalone.jpg'
    res = engine(target)
    print(f'\nDetected {len(res)} text lines:')
    for r in res:
        print(f"  {r['text']!r}  score={r['score']:.4f}")
