#!/usr/bin/env python3
"""多模型 NPU 三核并行基准。

对比同一帧送入多个模型时的两种执行方式:
  顺序    逐个推理, 总延迟 = 各模型之和
  并行    各模型绑定不同 NPU 核(config npu_core), 线程级并行, 总延迟 ≈ 最慢模型

用法(需 2 个及以上已转换的 .rknn, 缺哪个模型自动跳过):
  python rk3588_dms/tools/benchmark_parallel.py --mode device --runs 100
  python rk3588_dms/tools/benchmark_parallel.py --models chaitanya,soham
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="多模型 NPU 核并行基准")
    parser.add_argument("--models", default="chaitanya,soham,coco",
                        help="逗号分隔的模型名(默认全部; 只测已存在 .rknn 的模型)")
    parser.add_argument("--mode", default="device", choices=["auto", "device", "simulator"])
    parser.add_argument("--core", default=None, help="覆盖核绑定(默认按 config: 0/1/2)")
    parser.add_argument("--variant", default="fp",
                        help="fp|int8, 或逗号分隔按 --models 顺序逐模型指定(如 int8,fp,int8);"
                             "把 config 路径里的 _fp 替换为 _int8(缺文件自动跳过)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--image", help="测试图片(缺省用随机噪声帧, 只测耗时)")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("[FAIL] 缺少 opencv-python", file=sys.stderr)
        return 2

    from runtime.base_detector import load_config
    from runtime.parallel import build_group

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    config = load_config(PACKAGE_ROOT / "config" / "dms.json")

    variants = args.variant.split(",")
    available = []
    paths: dict[str, Path] = {}
    for index, name in enumerate(requested):
        variant = variants[index] if index < len(variants) else "fp"
        rknn_path = (PACKAGE_ROOT.parent / config["runtime"]["models"][name]["rknn"]).resolve()
        if variant == "int8":
            rknn_path = Path(str(rknn_path).replace("_fp.rknn", "_int8.rknn"))
        if rknn_path.exists():
            available.append(name)
            paths[name] = rknn_path
        else:
            print(f"[SKIP] {name}: 未找到 {rknn_path.name} (先转换)")
    if len(available) < 2:
        print("[FAIL] 可用模型少于 2 个, 无法对比并行收益", file=sys.stderr)
        return 1

    if args.image:
        frame = cv2.imread(str(Path(args.image).expanduser()), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[FAIL] 图片无法读取: {args.image}", file=sys.stderr)
            return 1
    else:
        frame = np.random.default_rng(7).integers(0, 255, (480, 640, 3), dtype=np.uint8)

    group = build_group(available, mode=args.mode, core_override=args.core,
                        paths={k: str(v) for k, v in paths.items()})
    print("=" * 64)
    print(f"models: {available} | cores: {group.cores} | backends: {group.backends}")
    print(f"warmup {args.warmup} / runs {args.runs}")
    print("=" * 64)

    for _ in range(max(0, args.warmup)):
        group.infer_all(frame)

    # 顺序基准: 逐个推理(同一线程), 总延迟 = 之和
    sequential_samples = []
    for _ in range(max(1, args.runs)):
        t0 = time.perf_counter()
        for detector in group.detectors.values():
            detector.infer(frame)
        sequential_samples.append((time.perf_counter() - t0) * 1000.0)

    # 并行基准
    parallel_samples = []
    for _ in range(max(1, args.runs)):
        group.infer_all(frame)
        parallel_samples.append(group.last_wall_ms)

    per_model = {name: [] for name in available}
    for _ in range(max(1, min(args.runs, 50))):
        results = group.infer_all(frame)
        for name, result in results.items():
            per_model[name].append(result.inference_ms)

    seq_avg = statistics.fmean(sequential_samples)
    par_avg = statistics.fmean(parallel_samples)
    print(f"顺序执行  : avg {seq_avg:7.2f} ms  p50 {statistics.median(sequential_samples):7.2f} ms")
    for name, samples in per_model.items():
        if samples:
            print(f"  [{name}] 单核 avg {statistics.fmean(samples):6.2f} ms "
                  f"(p50 {statistics.median(samples):6.2f} ms)")
    print(f"三核并行  : avg {par_avg:7.2f} ms  p50 {statistics.median(parallel_samples):7.2f} ms")
    print(f"加速比    : {seq_avg / par_avg:.2f}x   理论 FPS(并行): {1000.0 / par_avg:.1f}")
    if group.backends and set(group.backends.values()) != {"npu"}:
        print("!! 包含非 NPU 后端, 数据仅供参考 !!")

    # 检出 sanity(最后一帧)
    results = group.infer_all(frame)
    for name, result in results.items():
        print(f"  [{name}] 检出 {result.detection_count}: "
              + (", ".join(f"{d.class_name} {d.confidence:.2f}" for d in result.detections[:3]) or "none"))

    group.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
