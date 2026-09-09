"""预处理: 与浏览器 driver-inference.js preprocess() 逐位对齐的 letterbox。

浏览器实现( driver-inference.js:1972 ):
  1. 640x640 画布整幅填充 #808080 (即 128 灰, 不是 Ultralytics 默认的 114)
  2. scale = min(640/w, 640/h), 等比缩放后居中贴图(canvas 双线性插值)
  3. RGB, /255, NCHW float32 喂给 ONNX Runtime Web

RKNN 侧:
  - 喂 RKNN 的是 uint8 HWC RGB(letterbox 结果), 归一化交给模型内
    mean=0/std=255 完成, 数值上与浏览器 /255 一致。
  - 参考路径(onnxruntime 对比)用 float32 NCHW 版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - 环境问题在入口处统一提示
    raise ImportError("preprocess 需要 opencv-python: pip install opencv-python") from exc

DEFAULT_PAD = (128, 128, 128)


@dataclass
class PreprocessResult:
    image: np.ndarray        # uint8 HWC RGB, 尺寸 = (dst_h, dst_w)
    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int

    def to_float_nchw(self, dtype=np.float32) -> np.ndarray:
        """浏览器等价的 float32 NCHW [0,1] 张量(参考推理/对比用)。"""
        arr = self.image.astype(dtype) / 255.0
        return arr.transpose(2, 0, 1)[None, ...]


def letterbox(
    frame: np.ndarray,
    size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] | List[int] = DEFAULT_PAD,
    interpolation=cv2.INTER_LINEAR,
) -> PreprocessResult:
    """等比缩放 + 居中 + 灰底填充。

    frame: BGR 或 RGB 都可 —— 本函数不换通道顺序, 调用方负责最终顺序。
    返回 uint8 HWC 图与逆映射所需的 scale/pad。
    """
    src_h, src_w = frame.shape[:2]
    dst_w, dst_h = int(size[0]), int(size[1])
    if src_w < 1 or src_h < 1:
        raise ValueError(f"非法输入尺寸: {frame.shape}")

    canvas = np.empty((dst_h, dst_w, 3), dtype=np.uint8)
    canvas[:, :] = pad_color

    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = src_w * scale
    new_h = src_h * scale
    pad_x = (dst_w - new_w) / 2.0
    pad_y = (dst_h - new_h) / 2.0

    resized = cv2.resize(frame, (max(1, round(new_w)), max(1, round(new_h))), interpolation=interpolation)
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized

    # 逆映射使用的 pad 取实际整像素起点(与贴图一致), scale 不变
    return PreprocessResult(
        image=canvas,
        scale=scale,
        pad_x=float(x0),
        pad_y=float(y0),
        src_w=src_w,
        src_h=src_h,
        dst_w=dst_w,
        dst_h=dst_h,
    )


def unletterbox_box(bbox, prep: PreprocessResult):
    """模型输入坐标(640x640 系) -> 原图坐标, 并裁剪到原图范围。

    浏览器实现( driver-inference.js:2050-2054 ):
      (x - pad) / scale, 并 clamp 到 [0, 原尺寸]
    """
    x1, y1, x2, y2 = bbox
    sx1 = (x1 - prep.pad_x) / prep.scale
    sy1 = (y1 - prep.pad_y) / prep.scale
    sx2 = (x2 - prep.pad_x) / prep.scale
    sy2 = (y2 - prep.pad_y) / prep.scale
    return [
        max(0.0, min(sx1, prep.src_w)),
        max(0.0, min(sy1, prep.src_h)),
        max(0.0, min(sx2, prep.src_w)),
        max(0.0, min(sy2, prep.src_h)),
    ]


def unletterbox_point(point, prep: PreprocessResult) -> Tuple[float, float]:
    x, y = point
    return (
        max(0.0, min((x - prep.pad_x) / prep.scale, prep.src_w)),
        max(0.0, min((y - prep.pad_y) / prep.scale, prep.src_h)),
    )
