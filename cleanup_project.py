#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
目录清理脚本
============

保留:
  A. 原有项目文件 (git 跟踪)
  B. 纯 ONNX 推理模块 (func/algorithm/*.py)
  C. 权重文件 (推理必需)
  D. 模型导出链路 (PyTorch 权重 + 导出脚本 + venv_legacy_tf)
  E. 每模块一个验证脚本
  F. 三份核心报告 + README

删除:
  - 调研脚本、失败尝试、重复测试
  - 被取代的早期实现
  - 未采用的方案 (EasyOCR / CnSTD / PaddleOCR wrapper)
  - 临时图片/结果、调研材料、冗余 FP32 权重

用法:
    python cleanup_project.py --dry-run    # 只打印，不删除
    python cleanup_project.py --apply      # 实际删除
"""

import sys
import shutil
from pathlib import Path

ROOT = Path.cwd()

# ---------------------------------------------------------------------------
# 要保留的根目录脚本 -> 移动到 onnx_pipeline/ 下
# 格式: 原文件名 -> 目标相对路径
# ---------------------------------------------------------------------------
PIPELINE_LAYOUT = {
    # --- 导出链路 ---
    'convert_paddleocr_to_onnx.py':
        'onnx_pipeline/export/1_text_ocr_paddleocr_to_onnx.py',
    'download_mfd_models.py':
        'onnx_pipeline/export/2_formula_detect_download.py',
    'test_and_export_mfd.py':
        'onnx_pipeline/export/2_formula_detect_export_onnx.py',
    'export_mfd_fp16.py':
        'onnx_pipeline/export/2_formula_detect_export_fp16.py',
    'download_unimernet.py':
        'onnx_pipeline/export/3_formula_recognize_download.py',
    'download_unimernet_weights.py':
        'onnx_pipeline/export/3_formula_recognize_download_weights.py',
    'make_unimernet_minimal.py':
        'onnx_pipeline/export/3_formula_recognize_make_minimal.py',
    'export_unimernet_onnx.py':
        'onnx_pipeline/export/3_formula_recognize_export_onnx.py',
    'quantize_unimernet_fp16.py':
        'onnx_pipeline/export/3_formula_recognize_quantize_fp16.py',
    # --- 验证脚本 ---
    'test_onnx_ocr_full.py':
        'onnx_pipeline/verify/verify_text_ocr.py',
    'test_formula_detect_onnx.py':
        'onnx_pipeline/verify/verify_formula_detect.py',
    'test_formula_recognize_onnx.py':
        'onnx_pipeline/verify/verify_formula_recognize.py',
    'verify_no_paddle_dependency.py':
        'onnx_pipeline/verify/verify_no_framework_deps.py',
}

# ---------------------------------------------------------------------------
# 要保留的根目录脚本 (留在根目录)
# ---------------------------------------------------------------------------
KEEP_ROOT_SCRIPTS = {
    'config.py', 'server.py', 'infer.py', 'download_weights.py',
    'prepare_data.py', 'request_test.py',
    'cleanup_project.py',               # 本脚本自身
}

# ---------------------------------------------------------------------------
# 要保留的根目录文档
# ---------------------------------------------------------------------------
KEEP_ROOT_DOCS = {
    'README.md',
    'PADDLEOCR_TO_ONNX_REPORT.md',       # 文本检测/识别转换报告
    'FORMULA_DETECT_MODERN_REPORT.md',   # 公式检测报告
    'UNIMERNET_ONNX_REPORT.md',          # 公式识别报告
}

# ---------------------------------------------------------------------------
# 要删除的目录
# ---------------------------------------------------------------------------
DELETE_DIRS = [
    'research_open_source',   # 调研材料 (官方源码已归位到 weights/formula_recognize/unimernet_src)
    'test_output',            # 临时测试输出
    'weights/onnx_ocr',       # 测试可视化 (脚本可重新生成)
    'weights/paddleocr',      # 早期 PaddleOCR 测试结果
    'weights/formula_detect/cnstd_mfd',              # CnSTD 方案未采用
    'weights/formula_recognize/onnx',                # FP32 (保留 FP16)
]

# ---------------------------------------------------------------------------
# 要删除的单个文件
# ---------------------------------------------------------------------------
DELETE_FILES = [
    # 临时图片 / 结果
    'test_ocr_quick.jpg', 'test_standalone.jpg', 'test_results.json',
    'result_easyocr.json', 'result_easyocr.md',
    # 源码压缩包 (已解压)
    'unimernet_src.tar.gz',
    # EasyOCR 路线遗留
    'requirements_open_source.txt',
    'func/algorithm/OpenSourcePaperOCR.py',
    'handler/open_source_photo_ocr_handler.py',
    # 被取代的早期实现
    'func/algorithm/TextLineDivision/dbnet_onnx_detector.py',
    'func/algorithm/TextRecognition/crnn_onnx_recognizer.py',
    'func/algorithm/TextRecognition/paddleocr_wrapper.py',
    # 未采用的 CnSTD 权重
    'weights/formula_detect/yolov7_tiny_mfd-pytorch.zip',
    # 冗余 FP32 权重 (config 使用 FP16)
    'weights/formula_detect/mfd_yolov8.onnx',
    # 早期/过期文档
    'EASYOCR_COMPARISON.md', 'IMPLEMENTATION_COMPLETE.md',
    'migration_guide.md', 'model_analysis.md',
    'ONNX_MODEL_STATUS.md', 'ONNX_OCR_IMPLEMENTATION_REPORT.md',
    'PADDLEOCR_TEST_RESULTS.md', 'PADDLEOCR_V5_MIGRATION.md',
    'PURE_ONNX_COMPLETE_SOLUTION.md', 'PURE_ONNX_MIGRATION.md',
    'PURE_ONNX_RECOGNIZER_GUIDE.md', 'PURE_ONNX_SOLUTION_SUMMARY.md',
    'QUICKSTART.md', 'README_OPEN_SOURCE_MIGRATION.md',
    'SUMMARY.md', 'TEST_REPORT.md', 'USE_PADDLEOCR_DIRECTLY.md',
    'FINAL_SUMMARY.md', 'FORMULA_STATUS_REPORT.md',
    'fix_pipeline_paths.py',   # shim 逻辑已整合进本脚本
]


def fmt(n):
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f'{n:.1f} {u}'
        n /= 1024
    return f'{n:.1f} TB'


# ---------------------------------------------------------------------------
# 让移入 onnx_pipeline/ 的脚本可从任意目录运行
# (这些脚本原本依赖 cwd = 项目根，用 Path.cwd() 定位 weights/)
# ---------------------------------------------------------------------------
_SHIM = '''
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
'''

_SHIM_TARGETS = [
    'export/1_text_ocr_paddleocr_to_onnx.py',
    'export/2_formula_detect_download.py',
    'export/2_formula_detect_export_onnx.py',
    'export/2_formula_detect_export_fp16.py',
    'export/3_formula_recognize_download.py',
    'export/3_formula_recognize_download_weights.py',
    'export/3_formula_recognize_make_minimal.py',
    'export/3_formula_recognize_export_onnx.py',
    'export/3_formula_recognize_quantize_fp16.py',
    'verify/verify_formula_detect.py',
    'verify/verify_formula_recognize.py',
]


def apply_shim(path):
    """给脚本注入项目根定位 shim。返回 True 表示已修改。"""
    text = path.read_text(encoding='utf-8')
    if '_chdir_project_root' in text:
        return False

    lines = text.split('\n')

    # 补齐别名 import
    need_os = 'import os as _os' not in text
    need_path = 'from pathlib import Path as _Path' not in text
    last = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith(('import ', 'from ')):
            last = i
    add = []
    if need_os:
        add.append('import os as _os')
    if need_path:
        add.append('from pathlib import Path as _Path')
    for j, a in enumerate(add):
        lines.insert(last + 1 + j, a)
    last += len(add)

    lines.insert(last + 1, _SHIM)
    path.write_text('\n'.join(lines), encoding='utf-8')
    return True


def dir_size(p):
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def main():
    dry = '--apply' not in sys.argv
    mode = '干跑 (DRY-RUN)' if dry else '实际删除'
    print('=' * 74)
    print(f'目录清理 — {mode}')
    print('=' * 74)

    # ---------- 1. 扫描根目录多余脚本 ----------
    print('\n【1】根目录多余脚本/文档')
    print('-' * 74)
    extra = []
    for p in sorted(ROOT.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        if name in PIPELINE_LAYOUT:
            continue                       # 待重组
        if p.suffix == '.py':
            if name not in KEEP_ROOT_SCRIPTS:
                extra.append(p)
        elif p.suffix == '.md':
            if name not in KEEP_ROOT_DOCS:
                extra.append(p)

    total_freed = 0
    for p in extra:
        total_freed += p.stat().st_size
        print(f'  删除  {p.name:52} {fmt(p.stat().st_size):>10}')
        if not dry:
            p.unlink()

    # ---------- 1b. 重组 ONNX 流程脚本 ----------
    print('\n【1b】重组 ONNX 流程脚本 -> onnx_pipeline/')
    print('-' * 74)
    for src_name, dst_rel in sorted(PIPELINE_LAYOUT.items(),
                                    key=lambda kv: kv[1]):
        src = ROOT / src_name
        dst = ROOT / dst_rel
        if not src.exists():
            print(f'  跳过  {src_name:52} (不存在)')
            continue
        print(f'  {src_name:48} -> {dst_rel}')
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    # 注入项目根定位 shim
    if not dry:
        shimmed = 0
        for rel in _SHIM_TARGETS:
            p = ROOT / 'onnx_pipeline' / rel
            if p.exists() and apply_shim(p):
                shimmed += 1
        print(f'  已为 {shimmed} 个脚本注入项目根定位 shim')

    # ---------- 2. 删除指定目录 ----------
    print('\n【2】多余目录')
    print('-' * 74)
    for d in DELETE_DIRS:
        p = ROOT / d
        if not p.exists():
            print(f'  跳过  {d:52} (不存在)')
            continue
        s = dir_size(p)
        total_freed += s
        print(f'  删除  {d:52} {fmt(s):>10}')
        if not dry:
            shutil.rmtree(p)

    # ---------- 3. 删除指定文件 ----------
    print('\n【3】多余文件')
    print('-' * 74)
    for f in DELETE_FILES:
        p = ROOT / f
        if not p.exists():
            print(f'  跳过  {f:52} (不存在)')
            continue
        s = p.stat().st_size
        total_freed += s
        print(f'  删除  {f:52} {fmt(s):>10}')
        if not dry:
            p.unlink()

    # ---------- 4. 清理 __pycache__ ----------
    print('\n【4】__pycache__ 缓存')
    print('-' * 74)
    caches = [p for p in ROOT.rglob('__pycache__') if p.is_dir()]
    cache_total = sum(dir_size(p) for p in caches)
    total_freed += cache_total
    print(f'  共 {len(caches)} 个目录, 合计 {fmt(cache_total)}')
    if not dry:
        for p in caches:
            shutil.rmtree(p, ignore_errors=True)
        print('  已全部删除')

    # ---------- 汇总 ----------
    print('\n' + '=' * 74)
    print(f'{"预计" if dry else "已"}释放: {fmt(total_freed)}')
    print('=' * 74)

    if dry:
        print('\n这是干跑，未做任何改动。')
        print('确认无误后执行:  python cleanup_project.py --apply')
    else:
        print('\n清理完成。剩余结构:')
        show_tree(ROOT)


def name_display(p):
    return p.name


def show_tree(root, prefix='', depth=0, max_depth=2):
    if depth > max_depth:
        return
    try:
        items = sorted([p for p in root.iterdir()
                        if p.name not in ('.git', '.workbuddy-ai')],
                       key=lambda x: (x.is_file(), x.name))
    except PermissionError:
        return
    for p in items:
        if p.is_dir():
            s = dir_size(p)
            print(f'{prefix}{p.name}/  ({fmt(s)})')
            if depth < max_depth:
                show_tree(p, prefix + '    ', depth + 1, max_depth)
        else:
            print(f'{prefix}{p.name}  ({fmt(p.stat().st_size)})')


if __name__ == '__main__':
    main()
