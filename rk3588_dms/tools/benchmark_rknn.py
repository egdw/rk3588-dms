#!/usr/bin/env python3
"""RKNN 模型推理基准测试。

用法:
  python rk3588_dms/tools/benchmark_rknn.py                          # 默认 chaitanya FP
  python rk3588_dms/tools/benchmark_rknn.py --model path/to/x.rknn --model-name chaitanya
  python rk3588_dms/tools/benchmark_rknn.py --warmup 20 --runs 200

默认 NPU core = RKNN_NPU_CORE_AUTO(第一阶段不手动分配 core)。
输出: min / max / average / p50 / p95 / 理论 FPS, 以及 NPU 运行环境信息。
结果同时写入 logs/rknn/benchmark_*.json。

注意: 在 PC 模拟器模式下输出的耗时不具备 NPU 参考意义, 报告会显式标注。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="RKNN benchmark(warmup+runs 统计)")
    parser.add_argument("--model", help=".rknn 路径(缺省读 config)")
    parser.add_argument("--model-name", default="chaitanya", choices=["chaitanya", "soham", "coco"])
    parser.add_argument("--mode", default="auto", choices=["auto", "device", "simulator"])
    parser.add_argument("--core", default=None, help="NPU core: auto/0/1/2 (缺省用 config 的 npu_core)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    # 复用 test_image 的构建逻辑
    sys.path.insert(0, str(PACKAGE_ROOT))
    from test_image import build_detector  # noqa: E402
    from runtime.base_detector import load_config  # noqa: E402

    config = load_config(PACKAGE_ROOT / "config" / "dms.json")
    model_config = config["runtime"]["models"][args.model_name]
    input_size = model_config.get("input_size", [640, 640])

    detector = build_detector(args.model_name, args.model, args.mode, core=args.core)

    # 随机图像即可: benchmark 只测计算耗时, 不关心语义
    rng = np.random.default_rng(42)
    dummy = rng.integers(0, 255, size=(input_size[1], input_size[0], 3), dtype=np.uint8)

    print("=" * 64)
    print(f"model        : {detector.model_path}")
    print(f"runtime      : {'RK3588 NPU' if detector.model.backend == 'npu' else 'PC SIMULATOR (非 NPU!)'}")
    print(f"npu core     : {args.core}")
    print(f"input size   : {input_size[0]}x{input_size[1]}")
    print(f"warmup/runs  : {args.warmup}/{args.runs}")
    print("=" * 64)

    for _ in range(max(0, args.warmup)):
        detector.model.infer(dummy)

    samples_ms: list[float] = []
    for _ in range(max(0, args.runs)):
        detector.model.infer(dummy)
        samples_ms.append(detector.model.last_inference_ms)

    if not samples_ms:
        print("[FAIL] 没有有效采样", file=sys.stderr)
        detector.close()
        return 1

    avg = statistics.fmean(samples_ms)
    report = {
        "model": str(detector.model_path),
        "backend": detector.model.backend,
        "npu_core": args.core,
        "input_size": input_size,
        "warmup": args.warmup,
        "runs": len(samples_ms),
        "min_ms": round(min(samples_ms), 3),
        "max_ms": round(max(samples_ms), 3),
        "average_ms": round(avg, 3),
        "p50_ms": round(percentile(samples_ms, 50), 3),
        "p95_ms": round(percentile(samples_ms, 95), 3),
        "theoretical_fps": round(1000.0 / avg, 1) if avg > 0 else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"Average inference : {report['average_ms']} ms")
    print(f"min / max         : {report['min_ms']} / {report['max_ms']} ms")
    print(f"P50 / P95         : {report['p50_ms']} / {report['p95_ms']} ms")
    print(f"Theoretical FPS   : {report['theoretical_fps']}")
    if detector.model.backend != "npu":
        print("!! 以上数据来自 PC 模拟器, 不代表 RK3588 NPU 性能 !!")

    log_dir = Path(config["paths"].get("conversion_logs", "logs/rknn"))
    if not log_dir.is_absolute():
        log_dir = (PACKAGE_ROOT.parent / log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"benchmark_{args.model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report saved: {out}")

    detector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
