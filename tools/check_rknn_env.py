#!/usr/bin/env python3
"""RKNN 开发/运行环境检查(PC 与 RK3588 两种角色)。

用法:
  python tools/check_rknn_env.py            # 自动判断: aarch64 -> 设备侧检查, 否则 PC 侧
  python tools/check_rknn_env.py --role pc
  python tools/check_rknn_env.py --role device

PC 检查: Python / rknn-toolkit2 / onnx / onnxruntime / numpy / opencv
设备检查: Python / rknn-toolkit-lite2 / librknnrt(RKNN Runtime) / NPU 驱动 / 架构

输出逐项 PASS / WARNING / FAIL, 结尾给出总结论。只做只读探测, 不安装任何东西。
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

RESULTS = {"PASS": [], "WARNING": [], "FAIL": []}


def record(level: str, name: str, detail: str) -> None:
    RESULTS[level].append((name, detail))
    print(f"[{level:<7}] {name}: {detail}")


def check_python(min_version=(3, 8)) -> None:
    version = sys.version_info
    ok = version >= min_version
    record(
        "PASS" if ok else "FAIL",
        "Python",
        f"{platform.python_version()} ({'>= 3.8' if ok else '低于 3.8'})",
    )


def try_import(module_name: str, display: str, required: bool, hint: str = "") -> bool:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        record("PASS", display, f"已安装 {version}")
        return True
    except ImportError:
        record("FAIL" if required else "WARNING", display, f"未安装。{hint}")
        return False


def check_pc() -> int:
    print(f"== PC 端环境(转换/模拟器) ==  platform={platform.platform()}")
    if platform.machine() not in ("x86_64", "AMD64", "arm64", "aarch64"):
        record("WARNING", "架构", f"{platform.machine()} 不是 rknn-toolkit2 常见支持架构")
    else:
        record("PASS", "架构", f"{platform.machine()}")

    check_python()

    has_toolkit2 = try_import(
        "rknn", "rknn-toolkit2", required=False,
        hint="转换模型必需: pip install rknn-toolkit2 (版本与板上 runtime 匹配)",
    )
    try_import("onnx", "onnx", required=False, hint="模型结构检查: pip install onnx")
    try_import(
        "onnxruntime", "onnxruntime", required=False,
        hint="一致性对比的参考端: pip install onnxruntime",
    )
    try_import("numpy", "numpy", required=True)
    try_import("cv2", "opencv-python", required=True, hint="pip install opencv-python")

    for tool in ("ffmpeg",):
        found = shutil.which(tool)
        record("PASS" if found else "WARNING", tool, f"{found or '未找到(仅影响数据准备)'}")

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "Driver-Monitoring-System" / "public" / "static" / "models"
    expected = ["chaitanya_best.onnx", "soham_best.onnx", "yolov8n_coco.onnx"]
    missing = [name for name in expected if not (models_dir / name).exists()]
    if missing:
        record("WARNING", "浏览器 ONNX 模型", f"{models_dir} 缺少: {', '.join(missing)}")
    else:
        record("PASS", "浏览器 ONNX 模型", f"{models_dir} 三个模型齐全")

    print()
    if RESULTS["FAIL"]:
        print("结论: FAIL —— 存在必需项缺失, 按上面提示补齐后重试")
        return 1
    if not has_toolkit2:
        print("结论: WARNING —— 基础依赖可用, 但 rknn-toolkit2 未装, 无法执行模型转换")
        return 0
    print("结论: PASS —— PC 端可执行模型检查/转换/模拟器对比")
    return 0


def check_device() -> int:
    print(f"== RK3588 设备端环境 ==  platform={platform.platform()}")
    check_python()

    if platform.machine() != "aarch64":
        record("FAIL", "设备架构", f"当前架构 {platform.machine()}, 不是 aarch64 —— 这不是 RK3588 设备")
    else:
        record("PASS", "设备架构", "aarch64")

    model = ""
    try:
        model = Path("/proc/device-tree/model").read_text(errors="ignore").strip("\x00\n")
    except OSError:
        pass
    record("PASS" if "rk3588" in model.lower() else "WARNING", "设备型号", model or "无法读取 /proc/device-tree/model")

    try_import(
        "rknnlite", "rknn-toolkit-lite2", required=True,
        hint="板上推理必需: pip install rknn-toolkit-lite2 (版本与 PC 端 rknn-toolkit2 匹配)",
    )

    runtime = None
    for candidate in ("/usr/lib/librknnrt.so", "/usr/lib/aarch64-linux-gnu/librknnrt.so"):
        if Path(candidate).exists():
            runtime = candidate
            break
    if runtime:
        try:
            output = subprocess.run(["strings", runtime], capture_output=True, text=True, timeout=10).stdout
            versions = [line.strip() for line in output.splitlines() if line.strip().startswith("librknnrt version")]
            record("PASS", "RKNN Runtime", f"{runtime} {versions[0] if versions else ''}")
        except Exception as exc:  # noqa: BLE001
            record("PASS", "RKNN Runtime", f"{runtime} (版本读取失败: {exc})")
    else:
        record("FAIL", "RKNN Runtime", "未找到 librknnrt.so (rknn-toolkit-lite2 自带或系统安装)")

    # NPU 驱动证据
    npu_evidence = ""
    try:
        debugfs_version = Path("/sys/kernel/debug/rknpu/version")
        if debugfs_version.exists():
            try:
                npu_evidence = f"RKNPU {debugfs_version.read_text(errors='ignore').strip()}"
            except PermissionError:
                npu_evidence = "/sys/kernel/debug/rknpu/version 存在(非 root 只能确认路径)"
    except OSError:
        pass
    if not npu_evidence:
        try:
            dmesg = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=5).stdout
            if "RKNPU" in dmesg or "rknpu" in dmesg:
                npu_evidence = "dmesg 含 rknpu 记录"
        except Exception:  # noqa: BLE001
            pass
    record("PASS" if npu_evidence else "WARNING", "NPU 驱动", npu_evidence or "未直接探测到(可能需要 root; 以实际推理为准)")

    try_import("numpy", "numpy", required=True)
    try_import("cv2", "opencv-python", required=True, hint="pip install opencv-python")

    print()
    if RESULTS["FAIL"]:
        print("结论: FAIL —— 设备端存在必需项缺失")
        return 1
    print("结论: PASS —— 设备端可运行 test_image / benchmark / test_camera")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RKNN 环境检查")
    parser.add_argument("--role", choices=["auto", "pc", "device"], default="auto")
    args = parser.parse_args()

    role = args.role
    if role == "auto":
        role = "device" if platform.machine() == "aarch64" else "pc"
        print(f"(自动判断角色: {role})\n")

    return check_device() if role == "device" else check_pc()


if __name__ == "__main__":
    sys.exit(main())
