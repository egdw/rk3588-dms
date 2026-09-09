"""统一 YOLO->RKNN 检测器基类。

三个模型(chaitanya/soham/coco)共用同一接口与同一 DetectionResult 结构:

    detector = ChaitanyaDetector(config_path=..., mode="device")
    result = detector.infer(frame_bgr)      # OpenCV BGR 帧
    result.to_dict()                       # 统一 JSON

输出格式自适应(两种都在 2026 年的导出习惯里出现):
  A. Ultralytics 默认导出: 单输出 [1, 4+nc, N], 图内已含 DFL+Sigmoid
  B. Rockchip Model Zoo 风格重导出: 3 个分支 [1, 64+nc, H, W], CPU 侧解码

判定规则: 输出数 == 3 且每个通道数 == 64+nc -> B; 单个三维输出 -> A。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("base_detector 需要 opencv-python") from exc

from .detection import Detection, DetectionResult
from .model import RKNNModel
from .postprocess import decode_rknn_modelzoo_branches, decode_ultralytics_output
from .preprocess import letterbox, unletterbox_box

# rk3588_dms/ 包根
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# 项目根(存在 package.json / backend/server.py 的目录)
PROJECT_ROOT = PACKAGE_ROOT.parent


def find_project_root(start: Optional[Path] = None) -> Path:
    """从 start(默认本文件位置)向上找项目根标识。"""
    cursor = (start or PROJECT_ROOT).resolve()
    for candidate in [cursor, *cursor.parents]:
        if (candidate / "package.json").exists() and (candidate / "backend").is_dir():
            return candidate
        if (candidate / "backend" / "server.py").exists():
            return candidate
    return PROJECT_ROOT


class RknnYoloDetector:
    """配置驱动的 RKNN YOLO 检测器基类。"""

    model_name = "base"

    def __init__(
        self,
        model_config: Dict,
        model_path: str | Path,
        mode: str = "auto",
        config_root: Optional[Path] = None,
        core: str = "auto",
        confidence_threshold: Optional[float] = None,
    ):
        self.classes: List[str] = list(model_config["classes"])
        self.unified_classes: Dict[str, str] = dict(model_config.get("unified_classes") or {})
        self.input_size = tuple(model_config.get("input_size", [640, 640]))
        self.pad_color = tuple(model_config.get("letterbox_pad_color", [128, 128, 128]))
        self.confidence_threshold = float(
            confidence_threshold
            if confidence_threshold is not None
            else model_config.get("confidence_threshold", 0.25)
        )
        self.iou_threshold = float(model_config.get("iou_threshold", 0.45))

        root = find_project_root(config_root)
        path = Path(model_path)
        if not path.is_absolute():
            path = (root / path).resolve()
        self.model_path = path

        self.model = RKNNModel(path, mode=mode, core=core)
        self.model_name = self.model_name or "base"

    # ------------------------------------------------------------------ infer
    def infer(self, frame_bgr: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        prep = letterbox(frame_bgr, self.input_size, pad_color=self.pad_color)
        prep_ms = (time.perf_counter() - started) * 1000

        # RKNN 输入: RGB HWC uint8; BGR->RGB 在这里完成, /255 由模型内 mean/std 承担
        rgb = cv2.cvtColor(prep.image, cv2.COLOR_BGR2RGB)
        outputs = self.model.infer(rgb)
        infer_ms = self.model.last_inference_ms

        started = time.perf_counter()
        raw = self._decode(outputs)
        detections = [
            Detection(
                class_id=class_id,
                class_name=self.classes[class_id] if 0 <= class_id < len(self.classes) else str(class_id),
                confidence=score,
                bbox=unletterbox_box(box, prep),
                unified_key=self.unified_classes.get(
                    self.classes[class_id] if 0 <= class_id < len(self.classes) else "", None
                ),
            )
            for box, score, class_id in raw
        ]
        post_ms = (time.perf_counter() - started) * 1000

        return DetectionResult(
            model=self.model_name,
            inference_ms=infer_ms,
            preprocess_ms=prep_ms,
            postprocess_ms=post_ms,
            source_size=[prep.src_w, prep.src_h],
            detections=detections,
            backend=self.model.backend,
        )

    def _decode(self, outputs: List[np.ndarray]):
        """自动判别输出格式并解码为 [(bbox, score, class_id), ...]。"""
        nc = len(self.classes)
        size = int(self.input_size[0])

        if len(outputs) == 3:
            try:
                return decode_rknn_modelzoo_branches(
                    outputs, nc, input_size=size,
                    confidence_threshold=self.confidence_threshold,
                    iou_threshold=self.iou_threshold,
                )
            except ValueError:
                pass  # 不是 modelzoo 布局, 落到单输出路径

        if len(outputs) == 1:
            return decode_ultralytics_output(
                outputs[0], nc, input_size=size,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
            )

        raise ValueError(
            f"模型输出数量 {len(outputs)} 既不是 1(Ultralytics 默认)也不是 3(Model Zoo 风格), "
            "请先用 tools/inspect_model.py 检查导出方式"
        )

    def close(self) -> None:
        self.model.close()

    def __enter__(self) -> "RknnYoloDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_config(config_path: Optional[str | Path] = None) -> Dict:
    """加载 rk3588_dms/config/dms.json; 相对路径基于项目根解析。"""
    path = Path(config_path) if config_path else PACKAGE_ROOT / "config" / "dms.json"
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
