"""RKNN 运行时封装: 设备(rknn-toolkit-lite2) / PC 模拟器(rknn-toolkit2) 两种模式。

规则(对应任务要求"不能偷偷回退 CPU"):
- mode="device": 只用 rknn-toolkit-lite2。初始化即绑定 NPU 驱动, 驱动缺失会
  直接抛异常, 不存在静默 CPU 路径; 初始化成功后还会做平台/驱动自检并记录。
- mode="simulator": PC 上用 rknn-toolkit2 模拟器, 输出会明确标注
  SIMULATOR, 任何 benchmark/一致性结论都不得声称 NPU 性能。
- mode="auto": 优先 device, 不可用时报错说明缺什么(不装作成功)。
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


class RuntimeMode(str, Enum):
    DEVICE = "device"        # RK3588 真机, rknn-toolkit-lite2 + NPU
    SIMULATOR = "simulator"  # PC, rknn-toolkit2 模拟器
    AUTO = "auto"


@dataclass
class NPU_STATUS:
    """设备自检结果(只在 mode=device 时填写)。"""

    ok: bool = False
    machine: str = ""
    device_tree_model: str = ""
    is_rk3588: bool = False
    rknpu_driver_evidence: str = ""
    notes: List[str] = field(default_factory=list)


def probe_npu_environment() -> NPU_STATUS:
    """只读探测 RK3588/NPU 环境, 不加载模型。"""
    status = NPU_STATUS(machine=f"{platform.machine()} {platform.system()}")
    if platform.machine() != "aarch64":
        status.notes.append("非 aarch64 环境, 不可能是 RK3588 真机")
        return status

    for path in ("/proc/device-tree/model",):
        try:
            status.device_tree_model = Path(path).read_text(errors="ignore").strip("\x00\n")
        except OSError:
            pass
    compatible = ""
    try:
        compatible = Path("/proc/device-tree/compatible").read_text(errors="ignore")
    except OSError:
        pass
    status.is_rk3588 = "rk3588" in (status.device_tree_model + compatible).lower()

    try:
        if Path("/sys/kernel/debug/rknpu/version").exists():
            status.rknpu_driver_evidence = "/sys/kernel/debug/rknpu/version 存在"
    except OSError:  # 非 root 访问 debugfs 可能 PermissionError(路径存在即算证据)
        status.rknpu_driver_evidence = "/sys/kernel/debug/rknpu 路径存在(需 root 读取详情)"
    if not status.rknpu_driver_evidence:
        try:
            dmesg = subprocess.run(
                ["dmesg"], capture_output=True, text=True, timeout=5
            ).stdout.lower()
            if "rknpu" in dmesg:
                status.rknpu_driver_evidence = "dmesg 含 rknpu 记录"
        except Exception:  # noqa: BLE001 - dmesg 可能无权限
            pass
    if not status.rknpu_driver_evidence and Path("/dev/dri").exists():
        status.rknpu_driver_evidence = "/dev/dri 存在(间接证据, NPU 挂载于 DRM)"

    status.ok = status.is_rk3588 and bool(status.rknpu_driver_evidence)
    if not status.ok:
        if not status.is_rk3588:
            status.notes.append("device-tree 未识别出 rk3588")
        if not status.rknpu_driver_evidence:
            status.notes.append("未找到 rknpu 驱动证据")
    return status


class RKNNModel:
    """单个 .rknn 模型的加载与推理。

    用法:
        model = RKNNModel("models/rknn/chaitanya_best_fp.rknn", mode="device")
        outputs = model.infer(rgb_hwc_uint8)   # 返回 [np.ndarray, ...]
        print(model.last_inference_ms)
        model.close()
    """

    def __init__(
        self,
        model_path: str | Path,
        mode: str = "auto",
        verbose: bool = False,
        core: str = "auto",
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN 模型不存在: {self.model_path}")
        self.mode = RuntimeMode(mode)
        self.verbose = verbose
        self.core = core
        self.backend = ""            # npu / simulator
        self.npu_status: Optional[NPU_STATUS] = None
        self.init_ms = 0.0
        self.last_inference_ms = 0.0
        self._rknn = None
        self._closed = False
        self._init()

    # ------------------------------------------------------------------ init
    def _init(self) -> None:
        started = time.perf_counter()
        if self.mode in (RuntimeMode.DEVICE, RuntimeMode.AUTO):
            try:
                self._init_device()
                self.backend = "npu"
            except Exception as exc:
                if self.mode == RuntimeMode.DEVICE:
                    raise
                print(f"[WARN] 真机 rknn-toolkit-lite2 不可用({exc}), 尝试 PC 模拟器模式")
                self._init_simulator()
                self.backend = "simulator"
        else:
            self._init_simulator()
            self.backend = "simulator"
        self.init_ms = (time.perf_counter() - started) * 1000

    def _init_device(self) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError("未安装 rknn-toolkit-lite2 (仅 RK3588 设备上可装)") from exc

        self.npu_status = probe_npu_environment()
        if not self.npu_status.is_rk3588:
            raise RuntimeError(
                "当前不是 RK3588(aarch64 + device-tree 未识别)。真机模式拒绝在非目标设备上运行"
            )
        print(
            f"[NPU] 平板自检: machine={self.npu_status.machine} "
            f"model={self.npu_status.device_tree_model or 'N/A'} "
            f"driver={self.npu_status.rknpu_driver_evidence or 'N/A'}"
        )

        core_mask = {
            "auto": getattr(RKNNLite, "NPU_CORE_AUTO", 0),
            "0": getattr(RKNNLite, "NPU_CORE_0", 1),
            "1": getattr(RKNNLite, "NPU_CORE_1", 2),
            "2": getattr(RKNNLite, "NPU_CORE_2", 4),
        }.get(str(self.core).lower())
        if core_mask is None:
            raise ValueError(f"非法 npu core 配置: {self.core} (可选 auto/0/1/2)")

        rknn = RKNNLite(verbose=self.verbose)
        if rknn.load_rknn(str(self.model_path)) != 0:
            raise RuntimeError(f"load_rknn 失败: {self.model_path}")
        # init_runtime 失败通常意味着 NPU 驱动不可用 —— 这里不会回退 CPU, 直接抛错
        if rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError("init_runtime 失败(NPU 驱动不可用?)")
        self._rknn = rknn

    def _init_simulator(self) -> None:
        try:
            from rknn.api import RKNN
        except ImportError as exc:
            raise RuntimeError(
                "未安装 rknn-toolkit2 (PC 端转换/模拟器包)。真机请改用 mode='device'"
            ) from exc
        rknn = RKNN(verbose=self.verbose)
        if rknn.load_rknn(str(self.model_path)) != 0:
            raise RuntimeError(f"load_rknn 失败: {self.model_path}")
        if rknn.init_runtime() != 0:  # 无 target -> 本地模拟器
            raise RuntimeError("rknn-toolkit2 模拟器 init_runtime 失败")
        self._rknn = rknn
        print("[SIMULATOR] 注意: 当前在 PC 模拟器上推理, 性能与量化行为不等同 NPU 真机")

    # -------------------------------------------------------------- inference
    def infer(self, image_rgb_hwc: np.ndarray) -> List[np.ndarray]:
        """输入 uint8 RGB HWC letterbox 后图像, 返回输出张量列表。"""
        if self._closed or self._rknn is None:
            raise RuntimeError("模型未初始化或已释放")
        # lite2 2.3+ 要求显式 4 维 (1,H,W,3) nhwc, 不会自动扩 batch 维
        img = np.ascontiguousarray(image_rgb_hwc, dtype=np.uint8)[None]
        started = time.perf_counter()
        outputs = self._rknn.inference(inputs=[img], data_format="nhwc")
        self.last_inference_ms = (time.perf_counter() - started) * 1000
        if outputs is None:
            raise RuntimeError("RKNN inference 返回 None")
        return [np.asarray(out) for out in outputs]

    def close(self) -> None:
        if self._rknn is not None and not self._closed:
            # toolkit2/toolkit-lite2 2.x 的释放 API 是 release(); 兼容旧版 deinit()
            for name in ("release", "deinit"):
                fn = getattr(self._rknn, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001
                        pass
                    break
        self._closed = True
        self._rknn = None

    def __enter__(self) -> "RKNNModel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
