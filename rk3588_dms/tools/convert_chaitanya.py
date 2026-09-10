#!/usr/bin/env python3
"""chaitanya_best.onnx -> chaitanya_best_fp.rknn 转换脚本(PC 端, rknn-toolkit2)。

用法:
  python rk3588_dms/tools/convert_chaitanya.py                 # FP 模型(第一轮必须)
  python rk3588_dms/tools/convert_chaitanya.py --int8          # 后续阶段(INT8 量化)
  python rk3588_dms/tools/convert_chaitanya.py --source path/to/other.onnx

行为:
  1. 从 config/dms.json 读取 onnx_candidates, 自动选择第一个存在的文件
  2. 转换目标平台 rk3588, 打印 输入模型/输出模型/平台/量化模式/输入尺寸
  3. 转换日志写入 logs/rknn/
  4. 失败时抛出明确异常退出(不吞异常, 不生成假 rknn)

依赖(仅 PC): pip install rknn-toolkit2 onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from runtime.base_detector import find_project_root, load_config  # noqa: E402


class Tee:
    """stdout 同步写入日志文件。"""

    def __init__(self, log_path: Path):
        self._stdout = sys.stdout
        self._file = log_path.open("w", encoding="utf-8")

    def write(self, text: str) -> int:
        self._stdout.write(text)
        return self._file.write(text)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def find_source_onnx(candidates: List[str], root: Path) -> Optional[Path]:
    for raw in candidates:
        path = Path(raw)
        if not path.is_absolute():
            path = (root / path).resolve()
        if path.exists():
            return path
    return None


def precheck_onnx(path: Path) -> List[str]:
    """尽力而为的转换前预检(缺 onnx 包时跳过, 不阻塞)。返回告警列表。"""
    warnings: List[str] = []
    try:
        import onnx
    except ImportError:
        warnings.append("未安装 onnx 包, 跳过结构预检(建议: pip install onnx)")
        return warnings

    try:
        model = onnx.load(str(path))
        graph = model.graph
        ops: set[str] = {node.op_type for node in graph.node}
        if any("NMS" in op or "NonMaxSuppression" in op for op in ops):
            warnings.append("ONNX 图内包含 NMS —— 通常来自 export(nms=True), 必须重新导出")
        for value in list(graph.input):
            dims = value.type.tensor_type.shape.dim
            if any((d.HasField("dim_param") and d.dim_param) for d in dims):
                warnings.append(f"输入 {value.name} 存在动态维度, RKNN 转换前应固定为 1x3x640x640")
        opsets = {o.domain or "ai.onnx": o.version for o in model.opset_import}
        if opsets.get("ai.onnx", 0) < 12:
            warnings.append(f"opset={opsets.get('ai.onnx')} 低于 12, rknn-toolkit2 兼容性风险较高")
    except Exception as exc:  # noqa: BLE001 - 预检失败不等于转换失败
        warnings.append(f"结构预检异常(忽略): {exc}")
    return warnings


def validate_in_simulator(rknn, config, root: Path, image_path: Path) -> bool:
    """转换后在模拟器(无 target)上用内存模型跑一张图, 与浏览器 ONNX 对比。

    toolkit2 2.3+ 不允许 load_rknn 的模型进模拟器, 因此必须在 build 之后、
    同一个 RKNN 对象上 init_runtime() 验证 —— 本函数在导出成功后调用。
    """
    try:
        import cv2
        import onnxruntime  # noqa: F401
        import numpy as np
    except ImportError as exc:
        print(f"[WARN] 跳过模拟器验证(缺依赖: {exc} —— pip install onnxruntime opencv-python)")
        return False

    from runtime.base_detector import load_config as _load_config
    from runtime.detection import Detection
    from runtime.postprocess import decode_rknn_modelzoo_branches, iou
    from runtime.preprocess import letterbox, unletterbox_box
    from tools.compare_outputs import OnnxReference, match_detections

    model_config = _load_config(config)["runtime"]["models"]["chaitanya"]
    classes: List[str] = list(model_config["classes"])
    ref_path = root / "Driver-Monitoring-System/public/static/models/chaitanya_best.onnx"
    if not ref_path.exists():
        print(f"[WARN] 跳过模拟器验证(找不到参考 ONNX: {ref_path})")
        return False
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[WARN] 跳过模拟器验证(图片无法读取: {image_path})")
        return False

    prep = letterbox(frame, tuple(model_config["input_size"]),
                     pad_color=tuple(model_config["letterbox_pad_color"]))
    rgb = cv2.cvtColor(prep.image, cv2.COLOR_BGR2RGB)

    print("--> 模拟器验证: init_runtime()(无 target) + 单图推理")
    if rknn.init_runtime() != 0:
        print("[WARN] 模拟器 init_runtime 失败, 跳过验证(不影响导出的 .rknn 与真机验证)")
        return False
    outputs = rknn.inference(inputs=[np.ascontiguousarray(rgb)], data_format="nhwc")
    raw = decode_rknn_modelzoo_branches(
        [np.asarray(o) for o in outputs], len(classes),
        input_size=int(model_config["input_size"][0]),
        confidence_threshold=float(model_config.get("confidence_threshold", 0.25)),
        iou_threshold=float(model_config.get("iou_threshold", 0.45)),
    )
    cand_dets = [
        Detection(class_id=c, class_name=classes[c] if 0 <= c < len(classes) else str(c),
                  confidence=float(s), bbox=unletterbox_box(b, prep))
        for b, s, c in raw
    ]
    ref_dets = OnnxReference(ref_path, model_config).infer(frame)
    pairs, ref_extra, cand_extra = match_detections(ref_dets, cand_dets, 0.5)

    print(f"    参考(浏览器 ONNX): {len(ref_dets)} 检出 | RKNN 模拟器: {len(cand_dets)} 检出")
    max_delta = 0.0
    consistent = True
    for ref_det, cand_det in pairs:
        delta = abs(ref_det.confidence - cand_det.confidence)
        max_delta = max(max_delta, delta)
        box_iou = iou(ref_det.bbox, cand_det.bbox)
        if ref_det.class_id != cand_det.class_id or box_iou < 0.9:
            consistent = False
        print(f"    {ref_det.class_name:<10} conf {ref_det.confidence:.4f} vs {cand_det.confidence:.4f}"
              f" (Δ{delta:.4f})  IoU={box_iou:.4f}")
    if ref_extra:
        consistent = False
        print(f"    仅参考有: {[d.class_name for d in ref_extra]}")
    if cand_extra:
        consistent = False
        print(f"    仅 RKNN 有: {[d.class_name for d in cand_extra]}")

    if consistent and max_delta <= 0.02:
        print(f"[PASS] 模拟器验证通过: {len(pairs)} 对全匹配, 最大 Δconf={max_delta:.4f} (<=0.02)")
        return True
    print(f"[WARN] 模拟器验证存在差异: 配对 {len(pairs)}, 最大 Δconf={max_delta:.4f} —— 请人工核对上方明细")
    return False


def release_rknn(rknn) -> None:
    """兼容释放: toolkit2 2.x 是 release(), 旧版/变体可能是 deinit()。"""
    for name in ("release", "deinit"):
        fn = getattr(rknn, name, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - 释放失败不影响转换结果
                print(f"[WARN] rknn.{name}() 释放异常: {exc}")
            return
    print("[WARN] RKNN 对象没有 release/deinit 方法, 跳过释放")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="chaitanya ONNX -> RKNN(FP 优先) 转换")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "dms.json"))
    parser.add_argument("--source", help="覆盖配置, 直接指定 ONNX 路径")
    parser.add_argument("--int8", action="store_true", help="生成 INT8 量化模型(默认 FP)")
    parser.add_argument("--output", help="覆盖输出 .rknn 路径")
    parser.add_argument("--validate-image", help="导出后在模拟器上用该图片验证(与浏览器 ONNX 对比, 需 onnxruntime+opencv)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    root = find_project_root(PACKAGE_ROOT)
    conv = config["conversion"]["chaitanya"]
    platform = config.get("target_platform", "rk3588")
    input_size = conv.get("input_size", [640, 640])

    # 1. 定位源 ONNX ------------------------------------------------------
    if args.source:
        source = Path(args.source).expanduser()
        if not source.exists():
            print(f"[FAIL] 指定的 --source 不存在: {source}", file=sys.stderr)
            return 1
    else:
        source = find_source_onnx(conv.get("onnx_candidates", []), root)
    if source is None:
        print("[FAIL] 未找到 chaitanya ONNX。已尝试以下候选路径(相对项目根 %s):" % root, file=sys.stderr)
        for candidate in conv.get("onnx_candidates", []):
            print(f"       - {candidate}", file=sys.stderr)
        print(
            "提示: RKNN 转换首选 rknn_source 重导出 ONNX。若缺失, 先运行:\n"
            "      python rk3588_dms/tools/export_onnx_from_pt.py "
            "--pt Driver-Monitoring-System/models/chaitanya/best.pt "
            "--out rk3588_dms/models/onnx/rknn_source/chaitanya_best_rknn.onnx --style rknn\n"
            "      或用 --source 显式指定其它 ONNX。",
            file=sys.stderr,
        )
        return 1

    # 2. 输出与日志路径 ----------------------------------------------------
    # 第一轮强制 FP(非 INT8): --int8 显式开启, 属于后续阶段
    quantization = args.int8
    out_name = conv.get("output_name_int8" if quantization else "output_name", "chaitanya_best_fp.rknn")
    out_dir = (root / config["paths"]["rknn_outdir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else out_dir / out_name

    log_dir = (root / config["paths"].get("conversion_logs", "logs/rknn")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"convert_chaitanya_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    tee = Tee(log_path)
    sys.stdout = tee

    print("=" * 68)
    print("chaitanya_best -> RKNN 转换")
    print(f"  input model      : {source}")
    print(f"  output model     : {output}")
    print(f"  target platform  : {platform}")
    print(f"  quantization     : {'INT8' if quantization else 'FP (no quantization)'}")
    print(f"  input size       : {input_size[0]}x{input_size[1]}")
    print(f"  mean/std         : {conv.get('mean_values')} / {conv.get('std_values')}")
    print(f"  log              : {log_path}")
    print("=" * 68)

    for warning in precheck_onnx(source):
        print(f"[WARN] {warning}")

    # 3. 依赖检查 ----------------------------------------------------------
    try:
        from rknn.api import RKNN
    except ImportError:
        print(
            "[FAIL] 未安装 rknn-toolkit2。转换只能在 PC(x86_64 Linux/Windows)上进行:\n"
            "       pip install rknn-toolkit2  (版本需与板上 rknn-toolkit-lite2 匹配, 见 README)",
            file=sys.stderr,
        )
        sys.stdout = tee._stdout
        tee.close()
        return 2

    # 4. 转换 --------------------------------------------------------------
    try:
        rknn = RKNN(verbose=True)
    except ModuleNotFoundError as exc:
        if "pkg_resources" in str(exc):
            raise RuntimeError(
                "rknn-toolkit2 依赖 pkg_resources(由 setuptools 提供), "
                "而 Python 3.12+ 的 venv 默认不再安装 setuptools。"
                "请在当前环境执行: pip install 'setuptools<81' 后重试"
            ) from exc
        raise

    try:
        mean_values = conv.get("mean_values", [[0, 0, 0]])
        std_values = conv.get("std_values", [[255, 255, 255]])
        print("\n--> rknn.config(target_platform=%s)" % platform)
        if rknn.config(mean_values=mean_values, std_values=std_values, target_platform=platform) != 0:
            raise RuntimeError("rknn.config 失败")

        print(f"--> rknn.load_onnx({source})")
        if rknn.load_onnx(model=str(source)) != 0:
            raise RuntimeError("rknn.load_onnx 失败(检查上文 precheck 警告: 动态 shape/NMS/opset)")

        if quantization:
            dataset = (root / conv.get("quant_dataset", "")).resolve()
            if not dataset.exists():
                raise FileNotFoundError(f"INT8 量化需要校准列表文件: {dataset} (格式: 每行一张图片绝对路径)")
            print(f"--> rknn.build(do_quantization=True, dataset={dataset})")
            if rknn.build(do_quantization=True, dataset=str(dataset)) != 0:
                raise RuntimeError("rknn.build(do_quantization=True) 失败")
        else:
            print("--> rknn.build(do_quantization=False)")
            if rknn.build(do_quantization=False) != 0:
                raise RuntimeError("rknn.build(do_quantization=False) 失败")

        print(f"--> rknn.export_rknn({output})")
        if rknn.export_rknn(str(output)) != 0:
            raise RuntimeError("rknn.export_rknn 失败")

        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"导出文件异常: {output}")

        if args.validate_image:
            # 必须在 deinit 之前: toolkit2 2.3+ 模拟器只能跑 build 后的内存模型
            validate_in_simulator(rknn, args.config, root, Path(args.validate_image).expanduser())

        elapsed = time.perf_counter()
        size_mb = output.stat().st_size / 1024 / 1024
        print("=" * 68)
        print("转换成功")
        print(f"  output : {output}  ({size_mb:.1f} MB)")
        print(f"  下一步 : python rk3588_dms/test_image.py --model {output} --image <test.jpg>")
        print("=" * 68)
        return 0
    except Exception as exc:  # noqa: BLE001 - 明确抛出, 不吞
        print(f"\n[FAIL] 转换失败: {exc}", file=sys.stderr)
        raise
    finally:
        release_rknn(rknn)
        sys.stdout = tee._stdout
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
