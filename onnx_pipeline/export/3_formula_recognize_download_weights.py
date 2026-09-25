#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 UniMERNet-tiny 权重 (410MB)
"""

import os
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
OUT = Path.cwd() / 'weights' / 'formula_recognize' / 'unimernet_tiny'
OUT.mkdir(parents=True, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0'}


def main():
    print('=' * 74)
    print('下载 UniMERNet-tiny 权重')
    print('=' * 74)

    targets = [
        ('wanderkid/unimernet_tiny', 'pytorch_model.pth', 'unimernet_tiny.pth'),
        ('opendatalab/PDF-Extract-Kit-1.0',
         'models/MFR/unimernet_tiny/pytorch_model.pth',
         'unimernet_tiny_pdfkit.pth'),
    ]

    for repo, path, dest_name in targets:
        dest = OUT / dest_name
        if dest.exists() and dest.stat().st_size > 300_000_000:
            print(f'\n[SKIP] 已存在: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)')
            continue

        url = f'https://huggingface.co/{repo}/resolve/main/{path}'
        print(f'\n下载: {repo}/{path}')
        print(f'  -> {dest}')
        try:
            r = requests.get(url, stream=True, timeout=180, headers=HEADERS)
            if r.status_code != 200:
                print(f'  [FAIL] HTTP {r.status_code}')
                continue
            total = int(r.headers.get('content-length', 0))
            print(f'  大小: {total/1e6:.1f} MB')
            done = 0
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(1024 * 512):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total and done % (20 * 1024 * 1024) < 1024 * 512:
                            print(f'  {done/1e6:6.1f} / {total/1e6:.1f} MB '
                                  f'({done/total*100:5.1f}%)')
            print(f'  [OK] 完成 ({done/1e6:.1f} MB)')
        except Exception as e:
            print(f'  [ERROR] {type(e).__name__}: {e}')

    print('\n文件清单:')
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            print(f'  {p.name:40} {p.stat().st_size/1e6:8.2f} MB')


if __name__ == '__main__':
    main()
