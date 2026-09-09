"""统一检测结果数据结构。

所有模型的 Detector 返回同一种 DetectionResult, 避免"每个模型一套 JSON"。
字段与任务要求的示例对齐:
  {model, timestamp, inference_ms, detections:[{class_id, class_name, confidence, bbox}]}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Detection:
    """单个检测框。bbox 为原图像素坐标 [x1, y1, x2, y2]。"""

    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    unified_key: Optional[str] = None  # 浏览器 UNIFIED_CLASSES 映射后的 key(如 phone/safe), 无映射时为 None

    def to_dict(self) -> dict:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 1) for v in self.bbox],
            **({"unified_key": self.unified_key} if self.unified_key else {}),
        }


@dataclass
class DetectionResult:
    """一次完整推理的输出。"""

    model: str
    timestamp: float = field(default_factory=time.time)
    inference_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    source_size: List[int] = field(default_factory=lambda: [0, 0])
    detections: List[Detection] = field(default_factory=list)
    backend: str = ""  # npu / simulator / onnxruntime

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["detections"] = [d.to_dict() for d in self.detections]
        payload["detection_count"] = self.detection_count
        for key in ("inference_ms", "preprocess_ms", "postprocess_ms"):
            payload[key] = round(float(payload[key]), 2)
        return payload
