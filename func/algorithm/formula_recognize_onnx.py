"""
Pure ONNX Formula Recognition (MFR) — UniMERNet-tiny
====================================================

基于 UniMERNet-tiny 的公式识别（图片 -> LaTeX），
用纯 ONNX Runtime 推理，**不依赖 torch / transformers**。

模型 (由 export_unimernet_onnx.py 导出):
    unimernet_encoder.onnx : pixel_values -> encoder_hidden_states
    unimernet_decoder.onnx : (input_ids, encoder_hidden_states) -> logits

架构说明:
    - Encoder: VariableUnimerNet (Swin 风格分层 ViT, 含 StemLayer 卷积下采样)
    - Decoder: 自定义 mBART, 关键改动是 **MBartSqueezeAttention**
               (q/k 投影维度 = embed_dim/2, v 保持 embed_dim)
    - 自回归解码在 Python 侧循环完成

用法:
    from func.algorithm.formula_recognize_onnx import FormulaRecognizerONNX

    rec = FormulaRecognizerONNX()
    latex = rec(formula_crop_image)
"""

import os
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 默认使用 FP16 模型 (体积减半, 精度与 FP32 完全一致)
DEFAULT_ONNX_DIR = _PROJECT_ROOT / 'weights' / 'formula_recognize' / 'onnx_fp16'
# tokenizer.json 在 unimernet_tiny 目录
DEFAULT_TOKENIZER_DIR = _PROJECT_ROOT / 'weights' / 'formula_recognize' / 'unimernet_tiny'


class FormulaRecognizerONNX:
    """
    纯 ONNX 公式识别器 (UniMERNet-tiny)。

    将公式区域图片识别为 LaTeX 字符串。
    """

    def __init__(self,
                 onnx_dir: str = str(DEFAULT_ONNX_DIR),
                 tokenizer_dir: str = str(DEFAULT_TOKENIZER_DIR),
                 use_gpu: bool = False,
                 max_new_tokens: int = 512,
                 verbose: bool = True):
        self.onnx_dir = Path(onnx_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

        # 配置
        cfg_path = self.onnx_dir / 'unimernet_config.json'
        if not cfg_path.exists():
            raise FileNotFoundError(f'配置不存在: {cfg_path}')
        self.cfg = json.loads(cfg_path.read_text(encoding='utf-8'))

        self.bos_id = self.cfg['bos_token_id']
        self.pad_id = self.cfg['pad_token_id']
        self.eos_id = self.cfg['eos_token_id']
        self.vocab_size = self.cfg['vocab_size']
        self.image_size = tuple(self.cfg['image_size'])       # (H, W)
        self.image_mean = self.cfg['image_mean']
        self.image_std = self.cfg['image_std']

        # 执行提供者
        available = ort.get_available_providers()
        if use_gpu and 'CUDAExecutionProvider' in available:
            providers = [('CUDAExecutionProvider', {'device_id': 0}),
                         'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.providers = providers

        # 加载模型
        t0 = time.time()
        enc_path = self.onnx_dir / 'unimernet_encoder.onnx'
        dec_path = self.onnx_dir / 'unimernet_decoder.onnx'
        if not enc_path.exists():
            raise FileNotFoundError(f'encoder 模型不存在: {enc_path}')
        if not dec_path.exists():
            raise FileNotFoundError(f'decoder 模型不存在: {dec_path}')

        self.enc_sess = ort.InferenceSession(str(enc_path), providers=providers)
        self.dec_sess = ort.InferenceSession(str(dec_path), providers=providers)

        self.enc_input = self.enc_sess.get_inputs()[0].name
        self.enc_output = self.enc_sess.get_outputs()[0].name
        self.dec_inputs = [i.name for i in self.dec_sess.get_inputs()]
        self.dec_output = self.dec_sess.get_outputs()[0].name

        # tokenizer (用 tokenizers 库直接加载，避免依赖 transformers)
        self._load_tokenizer()

        if self.verbose:
            print(f'[FormulaRecognizerONNX] 加载完成 {time.time()-t0:.2f}s  '
                  f'vocab={self.vocab_size}  providers={providers}')

    # -- tokenizer ----------------------------------------------------------
    def _load_tokenizer(self):
        """用 tokenizers 库加载 BPE tokenizer。"""
        tok_path = self.tokenizer_dir / 'tokenizer.json'
        if not tok_path.exists():
            raise FileNotFoundError(f'tokenizer.json 不存在: {tok_path}')

        try:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(str(tok_path))
            self._decode_fn = self._decode_tokenizers
            return
        except ImportError:
            pass

        # 回退: 用 transformers (如果装了)
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_dir), trust_remote_code=True)
            self._decode_fn = self._decode_transformers
            return
        except ImportError:
            raise ImportError('需要 tokenizers 或 transformers 来解码')

    def _decode_tokenizers(self, ids: List[int]) -> str:
        """tokenizers 库解码。"""
        # 去掉特殊 token (bos/pad/eos)
        ids = [i for i in ids if i not in (self.bos_id, self.pad_id, self.eos_id)]
        if not ids:
            return ''
        try:
            return self._tokenizer.decode(ids, skip_special_tokens=True)
        except TypeError:
            return self._tokenizer.decode(ids)

    def _decode_transformers(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    # -- 预处理 -------------------------------------------------------------
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        公式图 -> [1, 3, 192, 672] float32

        流程: 灰度 -> resize(672,192) -> /255 -> 标准化 -> 3 通道堆叠
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        h, w = self.image_size
        resized = cv2.resize(gray, (w, h)).astype(np.float32) / 255.0
        resized = (resized - self.image_mean) / self.image_std
        blob = np.stack([resized] * 3, axis=0)[None]   # [1,3,H,W]
        return np.ascontiguousarray(blob.astype(np.float32))

    # -- 推理 ---------------------------------------------------------------
    def encode(self, img: np.ndarray) -> np.ndarray:
        """图片 -> encoder_hidden_states"""
        blob = self._preprocess(img)
        return self.enc_sess.run([self.enc_output], {self.enc_input: blob})[0]

    def decode_step(self, input_ids: np.ndarray,
                    enc_hidden: np.ndarray) -> np.ndarray:
        """单步解码 -> logits [1, seq_len, vocab]"""
        logits = self.dec_sess.run(
            [self.dec_output],
            {self.dec_inputs[0]: input_ids,
             self.dec_inputs[1]: enc_hidden})[0]
        return logits

    def __call__(self, img: np.ndarray,
                 return_ids: bool = False) -> str:
        """
        识别公式图片为 LaTeX。

        Args:
            img: BGR 或灰度公式区域图片
            return_ids: 是否同时返回 token ids

        Returns:
            LaTeX 字符串 (或 (latex, ids))
        """
        if isinstance(img, (str, Path)):
            img = cv2.imread(str(img))
            if img is None:
                raise FileNotFoundError(f'无法读取图片: {img}')

        t0 = time.time()
        enc_hidden = self.encode(img)

        # 自回归贪婪解码
        ids = [self.bos_id]
        for _ in range(self.max_new_tokens):
            input_ids = np.array([ids], dtype=np.int64)
            logits = self.decode_step(input_ids, enc_hidden)
            next_id = int(np.argmax(logits[0, -1, :]))
            if next_id == self.eos_id:
                break
            ids.append(next_id)

        self.last_time = time.time() - t0
        latex = self._decode_fn(ids)

        if return_ids:
            return latex, ids
        return latex

    # -- 批量 ---------------------------------------------------------------
    def recognize_batch(self, images: List[np.ndarray]) -> List[str]:
        return [self(img) for img in images]

    # -- 可视化辅助 ---------------------------------------------------------
    @staticmethod
    def draw(img: np.ndarray, latex: str) -> np.ndarray:
        """在图片上方绘制识别结果。"""
        vis = img.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        try:
            from PIL import Image, ImageDraw, ImageFont
            font = None
            for cand in (r'C:\Windows\Fonts\consola.ttf',
                         r'C:\Windows\Fonts\msyh.ttc'):
                if os.path.exists(cand):
                    font = ImageFont.truetype(cand, 18)
                    break
            pil = Image.new('RGB', (max(vis.shape[1], 600), 30), 'white')
            d = ImageDraw.Draw(pil)
            d.text((5, 5), latex[:120], fill='black', font=font)
            bar = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            vis = np.vstack([bar, vis])
        except Exception:
            pass
        return vis


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    rec = FormulaRecognizerONNX()
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        print(f'\n{target}')
        print(f'  -> {rec(target)!r}  ({rec.last_time:.2f}s)')
    else:
        print('用法: python formula_recognize_onnx.py <formula_image>')
