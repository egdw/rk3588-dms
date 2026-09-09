"""rk3588_dms runtime: RKNN 推理运行时(预处理/后处理/模型封装/检测器)。

设计约束(与浏览器 driver-inference.js 逐位对齐):
- letterbox 640x640, 灰底 (128,128,128) —— 浏览器 preprocess() 用 #808080
- RGB 顺序, /255 归一化(通过 RKNN mean=0/std=255 实现)
- 输出解析兼容 [1, 4+nc, N] 与 [1, N, 4+nc] 两种主序
- 通用置信度 0.25 / phone 0.20, NMS IoU 0.45
"""

from .detection import Detection, DetectionResult
from .preprocess import letterbox, unletterbox_box
from .postprocess import decode_ultralytics_output, decode_rknn_modelzoo_branches, nms, iou
from .model import RKNNModel, RuntimeMode, NPU_STATUS, probe_npu_environment
from .base_detector import RknnYoloDetector, load_config, find_project_root
from .chaitanya_detector import ChaitanyaDetector

__all__ = [
    "Detection",
    "DetectionResult",
    "letterbox",
    "unletterbox_box",
    "decode_ultralytics_output",
    "decode_rknn_modelzoo_branches",
    "nms",
    "iou",
    "RKNNModel",
    "RuntimeMode",
    "NPU_STATUS",
    "probe_npu_environment",
    "RknnYoloDetector",
    "load_config",
    "find_project_root",
    "ChaitanyaDetector",
]
