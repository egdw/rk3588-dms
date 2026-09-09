"""后处理: Ultralytics YOLOv8 / YOLO11 检测输出解码 + NMS。

语义与浏览器 parseDetections ( driver-inference.js:1999 ) 一致:

- 输出 [1, A, N] (attr-major, Ultralytics 默认导出) 或 [1, N, A] 自动判别
- A >= num_classes + 5 时, 第 4 行是 objectness(class_start=5), 否则 class_start=4
- 每个候选取类别 argmax, 置信度 = max_class_score [* objectness]
- 坐标 cx,cy,w,h: max(值) <= 2 视为归一化坐标, 乘 input_size
- NMS 按类别分组, IoU 阈值默认 0.45

差异说明(有意为之):
- 浏览器把类别先映射成统一 key(safe/phone/...)再按 key 分组 NMS; 本模块按
  class_id 分组 —— 当一个 class_id 只映射一个 key 时两者等价。需要完全复刻
  浏览器行为时用 group_key 参数传入映射函数。
- 浏览器在解析阶段就做统一 key 过滤(ACTIVE_DETECTION_KEYS); 本模块输出全部
  类别的原始检测, 过滤交给 Detector/上层, 便于 compare_outputs.py 做全量对比。

另外提供 Rockchip Model Zoo 风格三分支输出([1, 64+nc, H, W] x3)的 DFL 解码,
供按 Rockchip 推荐方式重导 ONNX 时使用(默认导出不需要)。
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

RegMax = 16  # YOLOv8/v11 DFL 回归分布 bins


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """轴对齐框 IoU。"""
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(
    detections: List[Tuple[List[float], float, int]],
    iou_threshold: float = 0.45,
    group_key: Optional[Callable[[int], object]] = None,
) -> List[Tuple[List[float], float, int]]:
    """按置信度降序做 NMS。

    detections: [(bbox[x1,y1,x2,y2], confidence, class_id), ...]
    group_key:  可选, class_id -> 分组键(浏览器按统一 key 分组时的映射)。
    """
    groups: dict[object, List[Tuple[List[float], float, int]]] = {}
    for det in detections:
        key = group_key(det[2]) if group_key else det[2]
        groups.setdefault(key, []).append(det)

    kept: List[Tuple[List[float], float, int]] = []
    for items in groups.values():
        candidates = sorted(items, key=lambda d: d[1], reverse=True)
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [c for c in candidates if iou(best[0], c[0]) <= iou_threshold]
    return kept


def decode_ultralytics_output(
    output: np.ndarray,
    num_classes: int,
    input_size: int | Tuple[int, int] = 640,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_candidates: int = 300,
    group_key: Optional[Callable[[int], object]] = None,
) -> List[Tuple[List[float], float, int]]:
    """解码 Ultralytics 默认导出的单输出 ONNX/RKNN 输出。

    output: 形如 [1, A, N] / [1, N, A] / [A, N] / [N, A] 的数组, 已经过
            Sigmoid(类别分数)与 DFL(框坐标)处理 —— 即图内已含后处理。
    返回:   [(bbox 输入像素坐标, confidence, class_id), ...] (NMS 后, 置信度降序)
    """
    if isinstance(input_size, (tuple, list)):
        input_size = int(input_size[0])

    arr = np.asarray(output)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"输出必须是二维属性矩阵(可选带 batch 维), 实际 shape={np.shape(output)}")

    dim0, dim1 = arr.shape
    attr_major = dim0 < dim1          # [A, N]: 属性少, 候选多
    attr_count = dim0 if attr_major else dim1
    candidate_count = dim1 if attr_major else dim0
    mat = arr if attr_major else arr.T  # 统一成 [A, N] 布局

    if attr_count < 4 + num_classes:
        raise ValueError(
            f"属性数 {attr_count} 少于 4+{num_classes}, 输出与类别数不匹配(shape={arr.shape})"
        )

    has_objectness = attr_count >= num_classes + 5
    class_start = 5 if has_objectness else 4
    # 类别分数已过 Sigmoid(Ultralytics 默认导出); 若个别导出未过 Sigmoid,
    # 分数会普遍偏小, 由上层 compare 工具发现, 这里不猜测。
    class_scores = mat[class_start : class_start + num_classes, :]  # [nc, N]
    best_ids = np.argmax(class_scores, axis=0)                       # [N]
    best_scores = class_scores[best_ids, np.arange(candidate_count)] # [N]
    if has_objectness:
        best_scores = best_scores * mat[4, :]

    keep_mask = best_scores >= confidence_threshold
    if not keep_mask.any():
        return []

    xs = mat[0, keep_mask]
    ys = mat[1, keep_mask]
    ws = mat[2, keep_mask]
    hs = mat[3, keep_mask]
    scores = best_scores[keep_mask]
    ids = best_ids[keep_mask]

    # 归一化坐标判定: 浏览器用 max(cx,cy,w,h) <= 2 判断
    if max(float(xs.max()), float(ys.max()), float(ws.max()), float(hs.max())) <= 2.0:
        xs = xs * input_size
        ys = ys * input_size
        ws = ws * input_size
        hs = hs * input_size

    x1 = xs - ws / 2.0
    y1 = ys - hs / 2.0
    x2 = xs + ws / 2.0
    y2 = ys + hs / 2.0

    candidates = [
        ([float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])], float(scores[i]), int(ids[i]))
        for i in range(len(scores))
        if ws[i] > 0 and hs[i] > 0
    ]
    if len(candidates) > max_candidates:
        candidates = sorted(candidates, key=lambda d: d[1], reverse=True)[:max_candidates]

    result = nms(candidates, iou_threshold=iou_threshold, group_key=group_key)
    result.sort(key=lambda d: d[1], reverse=True)
    return result


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定 Sigmoid(避免 exp 溢出警告)。"""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def dfl_decode(box_dist: np.ndarray) -> np.ndarray:
    """DFL: 分布回归 -> 距离标量。

    box_dist: [..., 4*RegMax]
    返回:     [..., 4] (l, t, r, b 距离, 单位=当前特征图格子)
    """
    *lead, channels = box_dist.shape
    if channels != 4 * RegMax:
        raise ValueError(f"DFL 输入通道应为 {4 * RegMax}, 实际 {channels}")
    dist = box_dist.reshape(*lead, 4, RegMax)
    probs = _softmax(dist, axis=-1)
    bins = np.arange(RegMax, dtype=np.float32)
    return (probs * bins).sum(axis=-1)


def decode_rknn_modelzoo_branches(
    outputs: Sequence[np.ndarray],
    num_classes: int,
    input_size: int = 640,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> List[Tuple[List[float], float, int]]:
    """解码 Rockchip Model Zoo 风格 YOLOv8 导出(删除 Detect 尾部后处理)。

    输出为 3 个分支特征图 [1, 64+nc, H, W] (H=80/40/20 @640), 顺序不限。
    框通道未经 DFL, 类别通道未经 Sigmoid —— 都在本函数 CPU 侧完成。
    """
    strides: List[int] = []
    feats: List[np.ndarray] = []
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"分支输出应为 [1, C, H, W], 实际 {np.shape(out)}")
        channels, feat_h, feat_w = arr.shape
        if channels != 4 * RegMax + num_classes:
            raise ValueError(
                f"分支通道数 {channels} != 4*{RegMax}+{num_classes}={4 * RegMax + num_classes}, "
                "输出可能不是 Model Zoo 风格导出"
            )
        strides.append(input_size // feat_h)
        feats.append(arr)
    if len(feats) != 3:
        raise ValueError(f"期望 3 个尺度的分支输出, 实际 {len(feats)} 个")

    candidates: List[Tuple[List[float], float, int]] = []
    for arr, stride in zip(feats, strides):
        channels, feat_h, feat_w = arr.shape
        box_dist = arr[: 4 * RegMax, :, :]                      # [64, H, W]
        cls_scores = _stable_sigmoid(arr[4 * RegMax :, :, :])   # Sigmoid, [nc, H, W]
        best_ids = np.argmax(cls_scores, axis=0)                # [H, W]
        best_scores = cls_scores.max(axis=0)                    # [H, W]
        keep = best_scores >= confidence_threshold
        if not keep.any():
            continue

        # DFL: [64,H,W] -> [H,W,64] -> [H,W,4]
        dist = dfl_decode(box_dist.transpose(1, 2, 0)) * stride  # [H, W, 4] 像素距离
        ys_idx, xs_idx = np.nonzero(keep)
        for gy, gx in zip(ys_idx, xs_idx):
            cx = (gx + 0.5) * stride
            cy = (gy + 0.5) * stride
            l, t, r, b = dist[gy, gx]
            x1, y1 = cx - l, cy - t
            x2, y2 = cx + r, cy + b
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            candidates.append(
                ([float(x1), float(y1), float(x2), float(y2)], float(best_scores[gy, gx]), int(best_ids[gy, gx]))
            )

    result = nms(candidates, iou_threshold=iou_threshold)
    result.sort(key=lambda d: d[1], reverse=True)
    return result
