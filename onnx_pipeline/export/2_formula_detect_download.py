#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载并测试现代公式检测模型 (MFD)
=================================

候选:
  A. CnSTD yolov7_tiny_mfd  (公开, 体积小, PyTorch -> ONNX)
  B. MinerU yolo_v8_ft.pt   (333MB, YOLOv8, ultralytics 可直接导出 ONNX)

本脚本先只做「下载 + 体积确认」，不立即转换。
"""

import os
import sys
import zipfile
import requests
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
OUT_DIR = Path.cwd() / 'weights' / 'formula_detect'
OUT_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0'}


def download(url, dest, desc=''):
    print(f'\n下载: {desc or url}')
    print(f'  URL : {url}')
    print(f'  DEST: {dest}')
    try:
        r = requests.get(url, stream=True, timeout=120, headers=HEADERS)
        if r.status_code != 200:
            print(f'  [FAIL] HTTP {r.status_code}')
            return False
        total = int(r.headers.get('content-length', 0))
        print(f'  大小: {total/1024/1024:.1f} MB')
        done = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        print(f'\r  进度: {pct:5.1f}%  ({done/1024/1024:.1f} MB)',
                              end='', flush=True)
        print(f'\n  [OK] 已保存 ({done/1024/1024:.1f} MB)')
        return True
    except Exception as e:
        print(f'  [ERROR] {type(e).__name__}: {e}')
        return False


def main():
    print('=' * 74)
    print('下载现代公式检测模型 (MFD)')
    print('=' * 74)

    results = {}

    # ---- A. CnSTD MFD (YOLOv7-tiny) ----
    print('\n' + '-' * 74)
    print('A. CnSTD MFD (yolov7_tiny_mfd) - PyTorch 版, 公开')
    print('-' * 74)
    url_a = ('https://huggingface.co/breezedeus/cnstd-cnocr-models/resolve/main/'
             'models/cnstd/1.2/yolov7_tiny_mfd-pytorch.zip')
    dest_a = OUT_DIR / 'yolov7_tiny_mfd-pytorch.zip'
    ok_a = download(url_a, dest_a, 'CnSTD MFD yolov7_tiny (PyTorch)')
    results['cnstd_mfd'] = ok_a

    if ok_a:
        # 解压查看内容
        extract_a = OUT_DIR / 'cnstd_mfd'
        extract_a.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(dest_a) as z:
                z.extractall(extract_a)
            print(f'\n  解压内容 ({extract_a}):')
            for p in sorted(extract_a.rglob('*')):
                if p.is_file():
                    print(f'    {p.relative_to(extract_a)}  ({p.stat().st_size:,} bytes)')
        except Exception as e:
            print(f'  [WARN] 解压失败: {e}')

    # ---- B. MinerU YOLOv8 MFD ----
    print('\n' + '-' * 74)
    print('B. MinerU/PDF-Extract-Kit MFD (yolo_v8_ft.pt)')
    print('-' * 74)
    url_b = ('https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/'
             'models/MFD/YOLO/yolo_v8_ft.pt')
    dest_b = OUT_DIR / 'yolo_v8_ft.pt'
    ok_b = download(url_b, dest_b, 'MinerU MFD YOLOv8 (333MB)')
    results['mineru_mfd'] = ok_b

    # ---- 汇总 ----
    print('\n' + '=' * 74)
    print('下载结果')
    print('=' * 74)
    for k, v in results.items():
        print(f'  {k:16} {"[OK]" if v else "[FAIL]"}')

    print(f'\n文件清单 ({OUT_DIR}):')
    for p in sorted(OUT_DIR.rglob('*')):
        if p.is_file():
            print(f'  {p.relative_to(OUT_DIR)}  ({p.stat().st_size/1024/1024:.1f} MB)')


if __name__ == '__main__':
    main()
