#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 PaddleOCR 的 PaddlePaddle 模型转换为纯 ONNX 模型
=====================================================

转换内容:
1. 检测模型 (DBNet)  -> dbnet_det.onnx
2. 识别模型 (CRNN/SVTR) -> crnn_rec.onnx
3. 方向分类模型 (可选) -> cls.onnx
4. 复制字符字典 ppocr_keys_v1.txt

所有输出保存到当前工作目录的 weights/ 下。
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import os as _os
from pathlib import Path as _Path

# --- 自动定位项目根, 使 Path.cwd() 指向项目根 (shim) ---
def _chdir_project_root():
    _p = _Path(__file__).resolve()
    for _parent in [_p] + list(_p.parents):
        if (_parent / 'weights').is_dir() and (_parent / 'func').is_dir():
            _os.chdir(_parent)
            return _parent
    return _Path.cwd()

_chdir_project_root()
# --- shim end ---
def convert_model(model_dir, save_file, input_shape_dict=None):
    """
    使用 paddle2onnx 将 Paddle 推理模型转换为 ONNX

    Args:
        model_dir: Paddle 推理模型目录 (含 inference.pdmodel / inference.pdiparams)
        save_file: 输出的 onnx 文件路径
        input_shape_dict: 可选，指定输入形状 (CLI 版本不支持，保留参数以兼容)

    Returns:
        bool: 是否成功
    """
    model_dir = Path(model_dir)
    pdmodel = model_dir / 'inference.pdmodel'

    if not pdmodel.exists():
        print(f"[ERROR] 找不到模型文件: {pdmodel}")
        return False

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # 定位 paddle2onnx 可执行文件
    exe = Path(sys.executable).parent / 'Scripts' / 'paddle2onnx.exe'
    if not exe.exists():
        exe = Path(sys.executable).parent / 'Scripts' / 'paddle2onnx'

    cmd = [
        str(exe),
        '--model_dir', str(model_dir),
        '--model_filename', 'inference.pdmodel',
        '--params_filename', 'inference.pdiparams',
        '--save_file', save_file,
        '--opset_version', '11',
        '--enable_onnx_checker', 'True',
    ]

    print(f"\nCommand: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if os.path.exists(save_file):
        size = os.path.getsize(save_file)
        print(f"[OK] 转换成功: {save_file} ({size:,} bytes)")
        return True
    else:
        print(f"[ERROR] 转换失败，未生成文件: {save_file}")
        return False


def main():
    print("=" * 70)
    print("PaddleOCR -> ONNX 模型转换")
    print("=" * 70)

    base_dir = Path.cwd()
    print(f"\n工作目录: {base_dir}")

    # PaddleOCR 缓存的模型位置
    paddleocr_cache = Path(os.path.expanduser('~')) / '.paddleocr' / 'whl'
    site_packages = Path(sys.prefix) / 'Lib' / 'site-packages'

    # 模型路径
    det_dir = paddleocr_cache / 'det' / 'ch' / 'ch_PP-OCRv4_det_infer'
    rec_dir = paddleocr_cache / 'rec' / 'ch' / 'ch_PP-OCRv4_rec_infer'
    cls_dir = paddleocr_cache / 'cls' / 'ch_ppocr_mobile_v2.0_cls_infer'
    dict_src = site_packages / 'paddleocr' / 'ppocr' / 'utils' / 'ppocr_keys_v1.txt'

    # 输出路径
    out_det = base_dir / 'weights' / 'text_line_detect' / 'dbnet_det.onnx'
    out_rec = base_dir / 'weights' / 'text_recognition' / 'crnn_rec.onnx'
    out_cls = base_dir / 'weights' / 'text_recognition' / 'cls.onnx'
    out_dict = base_dir / 'weights' / 'text_recognition' / 'ppocr_keys_v1.txt'

    print("\n模型源路径检查:")
    for name, p in [('检测模型', det_dir), ('识别模型', rec_dir), ('方向分类', cls_dir), ('字典文件', dict_src)]:
        status = "[OK]" if p.exists() else "[MISSING]"
        print(f"  {status} {name}: {p}")

    if not det_dir.exists() or not rec_dir.exists():
        print("\n[ERROR] 找不到 PaddleOCR 模型，请先运行 quick_ocr_test.py 下载模型")
        return 1

    # ============ 1. 转换检测模型 ============
    print("\n" + "=" * 70)
    print("步骤 1/4: 转换检测模型 (DBNet)")
    print("=" * 70)
    # 检测模型输入为动态尺寸: [batch, 3, H, W]
    det_ok = convert_model(det_dir, str(out_det),
                           input_shape_dict="{'x':[-1,3,-1,-1]}")

    # ============ 2. 转换识别模型 ============
    print("\n" + "=" * 70)
    print("步骤 2/4: 转换识别模型 (PP-OCRv4 Rec)")
    print("=" * 70)
    # PP-OCRv4 识别模型输入高度固定为 48，宽度动态: [batch, 3, 48, W]
    rec_ok = convert_model(rec_dir, str(out_rec),
                           input_shape_dict="{'x':[-1,3,48,-1]}")

    # ============ 3. 转换方向分类模型 ============
    print("\n" + "=" * 70)
    print("步骤 3/4: 转换方向分类模型 (可选)")
    print("=" * 70)
    cls_ok = False
    if cls_dir.exists():
        cls_ok = convert_model(cls_dir, str(out_cls),
                               input_shape_dict="{'x':[-1,3,48,192]}")
    else:
        print("[SKIP] 未找到方向分类模型")

    # ============ 4. 复制字典文件 ============
    print("\n" + "=" * 70)
    print("步骤 4/4: 复制字符字典")
    print("=" * 70)
    dict_ok = False
    if dict_src.exists():
        os.makedirs(out_dict.parent, exist_ok=True)
        shutil.copy2(dict_src, out_dict)
        print(f"[OK] 字典已复制: {out_dict} ({os.path.getsize(out_dict):,} bytes)")
        # 统计字符数
        with open(out_dict, 'r', encoding='utf-8') as f:
            lines = [l.rstrip('\n') for l in f]
        print(f"     字符数: {len(lines)}")
        dict_ok = True
    else:
        print(f"[ERROR] 找不到字典文件: {dict_src}")

    # ============ 汇总 ============
    print("\n" + "=" * 70)
    print("转换结果汇总")
    print("=" * 70)
    print(f"  检测模型 (DBNet):    {'[OK]' if det_ok else '[FAIL]'}  {out_det}")
    print(f"  识别模型 (Rec):      {'[OK]' if rec_ok else '[FAIL]'}  {out_rec}")
    print(f"  方向分类 (Cls):      {'[OK]' if cls_ok else '[SKIP]'}  {out_cls}")
    print(f"  字符字典:            {'[OK]' if dict_ok else '[FAIL]'}  {out_dict}")

    print("\n已生成的 ONNX 文件:")
    for root, dirs, files in os.walk(base_dir / 'weights'):
        for f in sorted(files):
            if f.endswith('.onnx'):
                fp = Path(root) / f
                print(f"  {fp} ({fp.stat().st_size:,} bytes)")

    return 0 if (det_ok and rec_ok and dict_ok) else 1


if __name__ == '__main__':
    sys.exit(main())
