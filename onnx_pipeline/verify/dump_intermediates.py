#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中间结果可视化
==============

保存完整推理链路的中间产物, 供人工核对:

  DBNet (文本检测)
    ├── 01_det_input.jpg          送进检测模型的缩放后输入
    ├── 02_det_prob_map.jpg       DBNet 输出的概率图 (热力图)
    ├── 03_det_bitmap.jpg         二值化后的 bitmap
    ├── 04_det_boxes_raw.jpg      膨胀前的原始四边形框
    └── 05_det_boxes_final.jpg    排序后的最终文本框

  文本识别
    └── 06_rec_crops/             每个文本框的裁剪图

  公式检测 (YOLOv8 MFD)
    ├── 10_mfd_input.jpg          letterbox 后的输入
    ├── 11_mfd_boxes_raw.jpg      NMS 后未去重的框
    ├── 12_mfd_after_dedup.jpg    跨类去重后的框
    ├── 13_mfd_final.jpg          切分修复后的最终框
    └── 14_mfd_split_detail.jpg   被切分的框 (原框 vs 子框)

  公式识别 (UniMERNet)
    └── 20_formula_crops/         每个公式区域的裁剪图

用法:
    python onnx_pipeline/verify/dump_intermediates.py [image_path]
"""

import os
import sys
import json
import time
import importlib.util
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / 'weights' / 'onnx_ocr' / 'intermediates'


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def save_prob_heatmap(prob: np.ndarray, path: Path, title: str = ''):
    """概率图 -> 伪彩色热力图 (JET), 右侧带色条。"""
    p = np.clip(prob, 0, 1)
    u8 = (p * 255).astype(np.uint8)
    heat = cv2.applyColorMap(u8, cv2.COLORMAP_JET)

    # 加标题条
    H, W = heat.shape[:2]
    bar = np.full((34, W, 3), 255, np.uint8)
    txt = f'{title}  min={prob.min():.3f} max={prob.max():.3f} mean={prob.mean():.3f}'
    cv2.putText(bar, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 2)
    out = np.vstack([bar, heat])
    cv2.imwrite(str(path), out)


def draw_quads(img, boxes, scores=None, color=(0, 255, 0), thickness=3,
               label_prefix=''):
    """在图上绘制四边形框 (boxes: [N,4,2])"""
    vis = img.copy()
    if boxes is None or len(boxes) == 0:
        return vis
    for i, box in enumerate(boxes):
        pts = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, thickness)
        if label_prefix:
            x, y = int(box[0][0]), int(box[0][1])
            txt = f'{label_prefix}{i+1}'
            if scores is not None and i < len(scores):
                txt += f' {scores[i]:.2f}'
            cv2.putText(vis, txt, (x, max(y - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


def draw_rects(img, results, color_map=None, thickness=3, show_index=True):
    """绘制 [x1,y1,x2,y2] 矩形框"""
    vis = img.copy()
    if color_map is None:
        color_map = {'embedding': (0, 165, 255), 'isolated': (0, 200, 0)}
    for i, r in enumerate(results):
        x1, y1, x2, y2 = [int(v) for v in r['box']]
        c = color_map.get(r.get('type', ''), (0, 0, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, thickness)
        if show_index:
            label = f"#{i+1}"
            if 'type' in r:
                label += f" {r['type'][:4]}"
            if 'score' in r:
                label += f" {r['score']:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return vis


def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'test.jpg'
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'[ERROR] 无法读取: {img_path}')
        return 1
    H, W = img.shape[:2]

    OUT.mkdir(parents=True, exist_ok=True)
    for d in ['06_rec_crops', '20_formula_crops']:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print('中间结果可视化')
    print('=' * 78)
    print(f'\n输入: {img_path}  {W}x{H}')
    print(f'输出: {OUT}')

    report = {'image': str(img_path), 'size': [W, H]}

    # ==================== 1. DBNet 文本检测 ====================
    print('\n' + '=' * 78)
    print('【1】DBNet 文本检测')
    print('=' * 78)

    eng_mod = load_mod('onnx_ocr_engine',
                       ROOT / 'func/algorithm/onnx_ocr_engine.py')
    eng = eng_mod.OnnxOCREngine(use_gpu=False, verbose=False)

    t0 = time.time()
    text_results, eng_debug = eng(img, return_crops=True, return_debug=True)
    dt_total = time.time() - t0
    d = eng_debug['det']

    print(f"\n  原图尺寸      : {d['orig_shape']}")
    print(f"  缩放后尺寸    : {d['resized_shape']}  (limit={d['limit_side_len']})")
    print(f"  缩放比例      : {d['scale_ratio']:.4f}")
    print(f"  概率图尺寸    : {d['prob_map'].shape}")
    print(f"  概率值        : min={d['prob_min']:.4f} max={d['prob_max']:.4f} "
          f"mean={d['prob_mean']:.4f}")
    print(f"  二值化阈值    : {d['thresh']}")
    print(f"  前景占比      : {d['bitmap_ratio']*100:.2f}%")
    print(f"  框阈值/膨胀比 : {d['box_thresh']} / {d['unclip_ratio']}")
    print(f"  检测框数      : {d['num_boxes']} -> 裁剪 {d['num_crops']} "
          f"-> 最终 {d['num_final']}")
    print(f"  检测耗时      : {d['det_time']:.3f}s  (全流程 {dt_total:.3f}s)")

    # 1) 缩放后输入
    resized, _ = eng._resize_image_type0(img, eng.det_limit_side_len)
    cv2.imwrite(str(OUT / '01_det_input.jpg'), resized)
    print(f'\n  [保存] 01_det_input.jpg  {resized.shape[1]}x{resized.shape[0]}')

    # 2) 概率图热力图
    save_prob_heatmap(d['prob_map'], OUT / '02_det_prob_map.jpg',
                      'DBNet probability map')
    print(f"  [保存] 02_det_prob_map.jpg")

    # 3) bitmap
    bm = (d['bitmap'] * 255).astype(np.uint8)
    cv2.imwrite(str(OUT / '03_det_bitmap.jpg'), bm)
    print(f"  [保存] 03_det_bitmap.jpg  前景 {bm.mean()/255*100:.2f}%")

    # 4) 原始框 (未排序)
    vis_raw = draw_quads(img, d['boxes'], d['scores'],
                         color=(0, 0, 255), label_prefix='R')
    cv2.imwrite(str(OUT / '04_det_boxes_raw.jpg'), vis_raw)
    print(f"  [保存] 04_det_boxes_raw.jpg  {len(d['boxes'])} 个框")

    # 5) 最终框 (排序后)
    final_boxes = np.array([r['box'] for r in text_results], dtype=np.int32)
    final_scores = [r['score'] for r in text_results]
    vis_final = draw_quads(img, final_boxes, final_scores,
                           color=(0, 255, 0), label_prefix='')
    cv2.imwrite(str(OUT / '05_det_boxes_final.jpg'), vis_final)
    print(f"  [保存] 05_det_boxes_final.jpg  {len(final_boxes)} 个框")

    # 6) 识别裁剪图
    for i, r in enumerate(text_results):
        crop = r.get('crop')
        if crop is None:
            continue
        cv2.imwrite(str(OUT / '06_rec_crops' / f'{i+1:03d}.jpg'), crop)
    print(f"  [保存] 06_rec_crops/  {len(text_results)} 张")

    report['dbnet'] = {
        'orig_shape': d['orig_shape'],
        'resized_shape': d['resized_shape'],
        'scale_ratio': d['scale_ratio'],
        'prob_shape': list(d['prob_map'].shape),
        'prob_min': d['prob_min'], 'prob_max': d['prob_max'],
        'prob_mean': d['prob_mean'],
        'thresh': d['thresh'],
        'bitmap_ratio': d['bitmap_ratio'],
        'box_thresh': d['box_thresh'],
        'unclip_ratio': d['unclip_ratio'],
        'num_boxes': d['num_boxes'],
        'num_crops': d['num_crops'],
        'num_final': d['num_final'],
        'det_time': d['det_time'],
        'dropped': d.get('dropped', []),
        'boxes': [r['box'] for r in text_results],
        'texts': [r['text'] for r in text_results],
    }

    # ==================== 2. MFD 公式检测 ====================
    print('\n' + '=' * 78)
    print('【2】MFD 公式检测 (YOLOv8)')
    print('=' * 78)

    fd_mod = load_mod('formula_detect_onnx',
                      ROOT / 'func/algorithm/formula_detect_onnx.py')
    det = fd_mod.FormulaDetectorONNX(use_gpu=False, verbose=False)

    t0 = time.time()
    formula_results, fd_debug = det(img, return_debug=True)
    dt_fd = time.time() - t0

    print(f"\n  输入尺寸      : {fd_debug['input_shape']}")
    print(f"  letterbox     : imgsz={fd_debug['imgsz']} "
          f"ratio={fd_debug['letterbox_ratio']:.4f} pad={fd_debug['letterbox_pad']}")
    print(f"  模型输出      : {fd_debug['raw_output_shape']}")
    print(f"  原始框        : {len(fd_debug['raw_boxes'])}")
    print(f"  去重后        : {len(fd_debug['after_dedup'])}")
    print(f"  最终框        : {len(fd_debug['final_boxes'])}")
    print(f"  耗时          : {dt_fd:.3f}s")

    n_emb = sum(1 for r in formula_results if r['type'] == 'embedding')
    n_iso = sum(1 for r in formula_results if r['type'] == 'isolated')
    print(f"  类别          : embedding={n_emb} isolated={n_iso}")

    # 切分记录
    sr = fd_debug['split_records']
    print(f"\n  切分修复      : {len(sr)} 个框被切分")
    for i, rec in enumerate(sr, 1):
        print(f"    [{i}] 原框 {[int(v) for v in rec['original']]} "
              f"高={rec['height']:.0f}px (限 {rec['limit']:.0f}px) "
              f"-> {rec['n_split']} 个子框")
        for s in rec['children']:
            print(f"         {[int(v) for v in s]} 高={s[3]-s[1]:.0f}px")

    # 10) letterbox 输入
    lb, _, _ = det._letterbox(img, det.imgsz)
    cv2.imwrite(str(OUT / '10_mfd_input.jpg'), lb)
    print(f"\n  [保存] 10_mfd_input.jpg  {lb.shape[1]}x{lb.shape[0]}")

    # 11) 原始框
    vis_r = draw_rects(img, fd_debug['raw_boxes'])
    cv2.imwrite(str(OUT / '11_mfd_boxes_raw.jpg'), vis_r)
    print(f"  [保存] 11_mfd_boxes_raw.jpg  {len(fd_debug['raw_boxes'])} 个")

    # 12) 去重后
    vis_d = draw_rects(img, fd_debug['after_dedup'])
    cv2.imwrite(str(OUT / '12_mfd_after_dedup.jpg'), vis_d)
    print(f"  [保存] 12_mfd_after_dedup.jpg  {len(fd_debug['after_dedup'])} 个")

    # 13) 最终
    vis_f = draw_rects(img, formula_results)
    cv2.imwrite(str(OUT / '13_mfd_final.jpg'), vis_f)
    print(f"  [保存] 13_mfd_final.jpg  {len(formula_results)} 个")

    # 14) 切分详情对比图
    if sr:
        tiles = []
        for i, rec in enumerate(sr, 1):
            ox1, oy1, ox2, oy2 = [int(v) for v in rec['original']]
            orig = img[oy1:oy2, ox1:ox2]
            # 原框
            o_vis = orig.copy()
            cv2.putText(o_vis, f'#{i} ORIGINAL h={rec["height"]:.0f}px',
                        (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            tiles.append(o_vis)
            # 子框
            for j, s in enumerate(rec['children'], 1):
                sx1, sy1, sx2, sy2 = [int(v) for v in s]
                sub = img[max(0, sy1):sy2, max(0, sx1):sx2]
                if sub.size == 0:
                    continue
                sv = sub.copy()
                cv2.putText(sv, f'#{i}.{j} SPLIT h={sy2-sy1}px',
                            (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 150, 0), 2)
                tiles.append(sv)
            tiles.append(np.full((10, orig.shape[1], 3), 200, np.uint8))

        # 统一宽度拼接
        maxw = max(t.shape[1] for t in tiles)
        padded = []
        for t in tiles:
            if t.shape[1] < maxw:
                t = cv2.copyMakeBorder(t, 0, 0, 0, maxw - t.shape[1],
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded.append(t)
        cv2.imwrite(str(OUT / '14_mfd_split_detail.jpg'), np.vstack(padded))
        print(f"  [保存] 14_mfd_split_detail.jpg  ({len(sr)} 组对比)")

    report['mfd'] = {
        'input_shape': fd_debug['input_shape'],
        'imgsz': fd_debug['imgsz'],
        'letterbox_ratio': fd_debug['letterbox_ratio'],
        'letterbox_pad': fd_debug['letterbox_pad'],
        'raw_output_shape': fd_debug['raw_output_shape'],
        'n_raw': len(fd_debug['raw_boxes']),
        'n_dedup': len(fd_debug['after_dedup']),
        'n_final': len(formula_results),
        'n_embedding': n_emb,
        'n_isolated': n_iso,
        'split_records': [
            {'original': [float(v) for v in r['original']],
             'height': r['height'], 'limit': r['limit'],
             'n_split': r['n_split'],
             'children': [[float(v) for v in c] for c in r['children']]}
            for r in sr
        ],
        'raw_boxes': fd_debug['raw_boxes'],
        'final_boxes': formula_results,
        'time': dt_fd,
    }

    # ==================== 3. 公式识别裁剪 ====================
    print('\n' + '=' * 78)
    print('【3】公式识别裁剪')
    print('=' * 78)

    rec_mod = load_mod('formula_recognize_onnx',
                       ROOT / 'func/algorithm/formula_recognize_onnx.py')
    rec = rec_mod.FormulaRecognizerONNX(use_gpu=False, verbose=False)

    crops_meta = []
    for i, r in enumerate(formula_results, 1):
        x1, y1, x2, y2 = [int(v) for v in r['box']]
        crop = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        if crop.size == 0:
            continue
        fn = f'{i:03d}_{r["type"]}.jpg'
        cv2.imwrite(str(OUT / '20_formula_crops' / fn), crop)
        crops_meta.append({'idx': i, 'type': r['type'],
                           'box': r['box'], 'file': fn,
                           'score': r['score']})
    print(f"\n  [保存] 20_formula_crops/  {len(crops_meta)} 张")

    report['formula_crops'] = crops_meta

    # ==================== 保存报告 ====================
    rp = OUT / 'intermediates_report.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 78)
    print('完成')
    print('=' * 78)
    print(f'\n报告: {rp}')
    print(f'\n产物清单:')
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name != 'intermediates_report.json':
            rel = p.relative_to(OUT)
            if p.parent.name in ('06_rec_crops', '20_formula_crops'):
                continue
            print(f'  {str(rel):40} {p.stat().st_size/1024:8.1f} KB')
    for d in ['06_rec_crops', '20_formula_crops']:
        n = len(list((OUT / d).glob('*.jpg')))
        print(f'  {d}/{" "*(40-len(d)-1)}{n} 张')
    return 0


if __name__ == '__main__':
    sys.exit(main())
