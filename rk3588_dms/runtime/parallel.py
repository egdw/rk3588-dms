"""多模型 NPU 核并行执行器。

每个 detector 持有独立 RKNNLite 实例, 核绑定由各自 config 的 npu_core
字段决定(chaitanya=0 / soham=1 / coco=2)。RKNN 推理在 librknnrt C 层
执行并释放 GIL, 不同模型绑定不同 NPU 核后用线程即可真并行, 无需多进程
(共享摄像头帧内存, 供后续原生 DMS 服务直接复用)。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .base_detector import RknnYoloDetector


class ParallelDetectorGroup:
    """一组绑定不同 NPU 核的检测器, infer_all() 线程级并行。"""

    def __init__(self, detectors: Dict[str, RknnYoloDetector]):
        if not detectors:
            raise ValueError("detectors 不能为空")
        self.detectors = detectors
        self._pool = ThreadPoolExecutor(max_workers=len(detectors))
        self.last_wall_ms = 0.0

    def infer_all(self, frame_bgr) -> Dict[str, "object"]:
        """同一帧并行送入所有模型, 返回 {model_name: DetectionResult}。"""
        started = time.perf_counter()
        futures = {
            name: self._pool.submit(detector.infer, frame_bgr)
            for name, detector in self.detectors.items()
        }
        results = {name: future.result() for name, future in futures.items()}
        self.last_wall_ms = (time.perf_counter() - started) * 1000.0
        return results

    @property
    def backends(self) -> Dict[str, str]:
        return {name: det.model.backend for name, det in self.detectors.items()}

    @property
    def cores(self) -> Dict[str, str]:
        return {name: str(det.model.core) for name, det in self.detectors.items()}

    def close(self) -> None:
        for detector in self.detectors.values():
            detector.close()
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "ParallelDetectorGroup":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def build_group(
    model_names: List[str],
    mode: str = "device",
    core_override: str | None = None,
    paths: Dict[str, str] | None = None,
):
    """按 config 构建并行组(延迟导入避免循环依赖)。

    paths: 可选 {model_name: rknn 路径} 覆盖 config 路径(如 FP/INT8 变体切换)。
    """
    from test_image import build_detector

    paths = paths or {}
    detectors = {}
    for name in model_names:
        detectors[name] = build_detector(name, paths.get(name), mode, core=core_override)
        print(f"[INIT] {name}: core={detectors[name].model.core} "
              f"backend={detectors[name].model.backend} "
              f"init={detectors[name].model.init_ms:.0f}ms")
    return ParallelDetectorGroup(detectors)
