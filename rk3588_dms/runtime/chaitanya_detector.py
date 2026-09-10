"""chaitanya_best(YOLOv8n, 5 类驾驶行为目标)RKNN 检测器。

第一个打通 RKNN 链路的验证模型。类别与阈值取自浏览器
driver-inference.js:126 / :5 (与 config/dms.json 保持一致):

  classes   = Cigarette / Drinking / Eating / Phone / Seatbelt
  conf      = 0.25 (通用), NMS IoU = 0.45
  letterbox = 640x640, 灰底 128, RGB/255

注意: 浏览器端 chaitanya 的 Phone 输出会被丢弃(phone 只认 COCO 模型);
本检测器输出全部类别, 由上层服务/对比工具决定取舍 —— 便于一致性对比。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base_detector import PACKAGE_ROOT, RknnYoloDetector, load_config


class ChaitanyaDetector(RknnYoloDetector):
    model_name = "chaitanya"

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        mode: str = "auto",
        rknn_path: Optional[str | Path] = None,
        core: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ):
        config = load_config(config_path or PACKAGE_ROOT / "config" / "dms.json")
        model_config = config["runtime"]["models"]["chaitanya"]
        rknn = rknn_path or model_config.get("rknn") or "rk3588_dms/models/rknn/chaitanya_best_fp.rknn"
        super().__init__(
            model_config=model_config,
            model_path=rknn,
            mode=mode,
            core=core,
            confidence_threshold=confidence_threshold,
        )
