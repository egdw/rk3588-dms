#!/usr/bin/env python3
"""ONNX 参考 vs RKNN 输出一致性对比(STEP 4)。

两种工作模式:

A. PC 直跑(默认): 参考 = onnxruntime 跑浏览器原 ONNX; RKNN = rknn-toolkit2
   模拟器(或连板 device 模式)跑转换后 .rknn。两边共用同一套 letterbox/
   解码代码, 差异只反映模型转换本身。
   python rk3588_dms/tools/compare_outputs.py --model-name chaitanya

B. 真机 JSON 回灌: RK3588 板上先用 test_image.py --json 逐张导出结果,
   再拷回 PC 对比(模拟器量化行为与真机有差别, 真机数据更可信)。
   python rk3588_dms/tools/compare_outputs.py --model-name chaitanya \
       --rknn-json-dir board_results/

输入图片目录默认 testdata/dms/(>=20 张, 覆盖正常驾驶/打电话/喝水/吃东西/安全带)。
输出:
  logs/rknn/compare_<model>_<ts>/        每张图一个 JSON
  docs/rknn/chaitanya_validation.md      汇总报告(类别一致率/IoU/置信度差/异常样本)

依赖: onnxruntime, opencv-python, numpy (+rknn-toolkit2 模式A)
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np  # noqa: E402

from runtime.base_detector import find_project_root, load_config  # noqa: E402
from runtime.detection import Detection  # noqa: E402
from runtime.postprocess import decode_ultralytics_output, iou  # noqa: E402
from runtime.preprocess import letterbox, unletterbox_box  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------- reference
class OnnxReference:
    """浏览器等价路径: onnxruntime + 相同 letterbox/解码。"""

    def __init__(self, onnx_path: Path, model_config: Dict):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if len(self.session.get_outputs()) != 1:
            raise ValueError(
                f"参考端 {onnx_path.name} 不是标准单输出模型({len(self.session.get_outputs())} 个输出)——"
                "参考端必须是浏览器/Ultralytics 默认导出"
            )
        self.input_name = self.session.get_inputs()[0].name
        self.classes: List[str] = list(model_config["classes"])
        self.input_size = tuple(model_config.get("input_size", [640, 640]))
        self.pad_color = tuple(model_config.get("letterbox_pad_color", [128, 128, 128]))
        self.conf = float(model_config.get("confidence_threshold", 0.25))
        self.iou_thr = float(model_config.get("iou_threshold", 0.45))
        self.unified = dict(model_config.get("unified_classes") or {})

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        import cv2

        prep = letterbox(frame_bgr, self.input_size, pad_color=self.pad_color)
        tensor = cv2.cvtColor(prep.image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[None, ...]
        outputs = self.session.run(None, {self.input_name: tensor})
        raw = decode_ultralytics_output(
            outputs[0], len(self.classes), input_size=self.input_size[0],
            confidence_threshold=self.conf, iou_threshold=self.iou_thr,
        )
        dets = []
        for box, score, class_id in raw:
            name = self.classes[class_id] if 0 <= class_id < len(self.classes) else str(class_id)
            dets.append(
                Detection(
                    class_id=class_id, class_name=name, confidence=float(score),
                    bbox=unletterbox_box(box, prep), unified_key=self.unified.get(name),
                )
            )
        return dets


# --------------------------------------------------------------- rknn side
def rknn_detections_from_json(path: Path) -> List[Detection]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dets = []
    for d in payload.get("detections", []):
        dets.append(
            Detection(
                class_id=int(d["class_id"]), class_name=d["class_name"],
                confidence=float(d["confidence"]), bbox=[float(v) for v in d["bbox"]],
                unified_key=d.get("unified_key"),
            )
        )
    return dets


class RknnLive:
    """现场跑 RKNN(模拟器或真机), 复用统一 Detector。"""

    def __init__(self, model_name: str, model_path: Optional[str], mode: str):
        from test_image import build_detector

        self.detector = build_detector(model_name, model_path, mode)

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        return self.detector.infer(frame_bgr).detections


class RknnStyleOnnx:
    """PC 预检: 用 onnxruntime 跑 Rockchip 风格三分支 ONNX,
    并经过与 RKNN 完全相同的 letterbox/decode_rknn_modelzoo_branches 路径。
    用于在 rknn-toolkit2 转换前验证重导出 ONNX 的数值正确性。"""

    def __init__(self, model_name: str, onnx_path: Path):
        import onnxruntime as ort

        from runtime.base_detector import load_config

        config = load_config(PACKAGE_ROOT / "config" / "dms.json")
        model_config = config["runtime"]["models"][model_name]
        self.classes: List[str] = list(model_config["classes"])
        self.unified = dict(model_config.get("unified_classes") or {})
        self.input_size = tuple(model_config.get("input_size", [640, 640]))
        self.pad_color = tuple(model_config.get("letterbox_pad_color", [128, 128, 128]))
        self.conf = float(model_config.get("confidence_threshold", 0.25))
        self.iou_thr = float(model_config.get("iou_threshold", 0.45))

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        import cv2

        from runtime.postprocess import decode_rknn_modelzoo_branches
        from runtime.preprocess import letterbox, unletterbox_box

        prep = letterbox(frame_bgr, self.input_size, pad_color=self.pad_color)
        rgb = cv2.cvtColor(prep.image, cv2.COLOR_BGR2RGB)
        # 重导出图内无归一化(RKNN 转换由 mean=0/std=255 承担), 这里手动 /255 等价
        tensor = (rgb[None].transpose(0, 3, 1, 2)).astype(np.float32) / 255.0
        outputs = self.session.run(None, {self.input_name: tensor})
        raw = decode_rknn_modelzoo_branches(
            outputs, len(self.classes), input_size=self.input_size[0],
            confidence_threshold=self.conf, iou_threshold=self.iou_thr,
        )
        dets = []
        for box, score, class_id in raw:
            name = self.classes[class_id] if 0 <= class_id < len(self.classes) else str(class_id)
            dets.append(
                Detection(
                    class_id=class_id, class_name=name, confidence=float(score),
                    bbox=unletterbox_box(box, prep), unified_key=self.unified.get(name),
                )
            )
        return dets


# --------------------------------------------------------------- matching
def match_detections(
    reference: List[Detection], candidate: List[Detection], iou_threshold: float = 0.5
) -> Tuple[List[Tuple[Detection, Detection]], List[Detection], List[Detection]]:
    """按 class_id + IoU 贪心匹配。返回 (配对, 参考多余, 候选多余)。"""
    pairs: List[Tuple[Detection, Detection]] = []
    used: set[int] = set()
    for ref in sorted(reference, key=lambda d: -d.confidence):
        best_j, best_iou = -1, iou_threshold
        for j, cand in enumerate(candidate):
            if j in used or cand.class_id != ref.class_id:
                continue
            score = iou(ref.bbox, cand.bbox)
            if score >= best_iou:
                best_iou, best_j = score, j
        if best_j >= 0:
            used.add(best_j)
            pairs.append((ref, candidate[best_j]))
    cand_extra = [c for j, c in enumerate(candidate) if j not in used]
    ref_extra = [r for r in reference if not any(r is rr for rr, _ in pairs)]
    return pairs, ref_extra, cand_extra


# --------------------------------------------------------------- main flow
def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX vs RKNN 输出一致性对比")
    parser.add_argument("--model-name", default="chaitanya", choices=["chaitanya", "soham", "coco"])
    parser.add_argument("--images", help="测试图片目录/单张图片(默认 testdata/dms)")
    parser.add_argument("--onnx", help="参考 ONNX 路径(默认浏览器模型路径)")
    parser.add_argument("--rknn", help="RKNN 模型路径(模式A)")
    parser.add_argument("--mode", default="simulator", choices=["auto", "device", "simulator"])
    parser.add_argument("--rknn-json-dir", help="模式B: 板上导出的 JSON 目录")
    parser.add_argument("--candidate-onnx", help="模式C: 用 onnxruntime 预检 Rockchip 风格重导出 ONNX(与 RKNN 同解码路径)")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--conf-tolerance", type=float, default=0.05,
                        help="配对样本置信度差超过该值记入异常样本")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张(0=全部)")
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[FAIL] 缺少 opencv-python", file=sys.stderr)
        return 2
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("[FAIL] 模式A/B 都需要 onnxruntime 作为参考端: pip install onnxruntime", file=sys.stderr)
        return 2

    config = load_config(PACKAGE_ROOT / "config" / "dms.json")
    root = find_project_root(PACKAGE_ROOT)
    model_config = config["runtime"]["models"][args.model_name]

    # 图片集 ---------------------------------------------------------------
    images_root = Path(args.images) if args.images else (root / config["paths"].get("test_images", "testdata/dms"))
    if images_root.is_file():
        image_paths = [images_root]
    else:
        image_paths = sorted(
            p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print(
            f"[FAIL] 测试图片为空: {images_root}\n"
            "请先准备 >=20 张覆盖 正常驾驶/打电话/喝水/吃东西/安全带 的图片放入该目录",
            file=sys.stderr,
        )
        return 1

    # 参考 ONNX(固定为浏览器标准单输出模型, 不受 conversion 候选顺序影响) -------------
    browser_names = {
        "chaitanya": "chaitanya_best.onnx",
        "soham": "soham_best.onnx",
        "coco": "yolov8n_coco.onnx",
    }
    onnx_path = Path(args.onnx) if args.onnx else (
        root / "Driver-Monitoring-System/public/static/models" / browser_names[args.model_name]
    )
    if not onnx_path.exists():
        print(f"[FAIL] 找不到参考 ONNX(浏览器标准模型): {onnx_path}。可用 --onnx 指定", file=sys.stderr)
        return 1

    reference = OnnxReference(onnx_path, model_config)

    # RKNN 端 --------------------------------------------------------------
    rknn_live: Optional[RknnLive] = None
    json_dir: Optional[Path] = None
    candidate: object = None
    if args.candidate_onnx:
        candidate_path = Path(args.candidate_onnx)
        if not candidate_path.exists():
            print(f"[FAIL] --candidate-onnx 不存在: {candidate_path}", file=sys.stderr)
            return 1
        candidate = RknnStyleOnnx(args.model_name, candidate_path)
    elif args.rknn_json_dir:
        json_dir = Path(args.rknn_json_dir)
        if not json_dir.is_dir():
            print(f"[FAIL] --rknn-json-dir 不是目录: {json_dir}", file=sys.stderr)
            return 1
    else:
        rknn_live = RknnLive(args.model_name, args.rknn, args.mode)

    out_dir = (root / config["paths"].get("conversion_logs", "logs/rknn")).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / f"compare_{args.model_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "fully_matched": 0,
        "pairs": 0,
        "class_consistent_pairs": 0,
        "iou_values": [],
        "conf_diffs": [],
        "ref_only": 0,
        "rknn_only": 0,
        "outliers": [],
    }

    for image_path in image_paths:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] 跳过无法解码的图片: {image_path}")
            continue
        ref_dets = reference.infer(frame)

        if candidate is not None:
            rknn_dets = candidate.infer(frame)
        elif rknn_live is not None:
            rknn_dets = rknn_live.infer(frame)
        else:
            json_path = json_dir / f"{image_path.stem}.json"
            if not json_path.exists():
                matches = sorted(json_dir.glob(f"{image_path.stem}*.json"))
                json_path = matches[0] if matches else None
            if json_path is None:
                print(f"[WARN] 板上结果缺少 {image_path.stem}.json, 跳过")
                continue
            rknn_dets = rknn_detections_from_json(json_path)

        pairs, ref_extra, rknn_extra = match_detections(ref_dets, rknn_dets, args.match_iou)
        image_outliers: List[str] = []
        conf_diffs = []
        ious = []
        class_consistent = 0
        for ref, cand in pairs:
            if ref.class_id == cand.class_id:
                class_consistent += 1
            ious.append(iou(ref.bbox, cand.bbox))
            diff = abs(ref.confidence - cand.confidence)
            conf_diffs.append(diff)
            if diff > args.conf_tolerance:
                image_outliers.append(
                    f"{ref.class_name}: conf {ref.confidence:.3f} vs {cand.confidence:.3f} (Δ{diff:.3f})"
                )
        if ref_extra:
            image_outliers.append(f"参考有而 RKNN 无: {[d.class_name for d in ref_extra]}")
        if rknn_extra:
            image_outliers.append(f"RKNN 有而参考无: {[d.class_name for d in rknn_extra]}")

        matched = not ref_extra and not rknn_extra and class_consistent == len(pairs) and not any(
            d > args.conf_tolerance for d in conf_diffs
        )

        payload = {
            "image": str(image_path),
            "matched": matched,
            "reference": [d.to_dict() for d in ref_dets],
            "rknn": [d.to_dict() for d in rknn_dets],
            "pairs": [
                {
                    "class": ref.class_name,
                    "class_consistent": ref.class_id == cand.class_id,
                    "iou": round(ious[i], 4),
                    "conf_reference": round(ref.confidence, 4),
                    "conf_rknn": round(cand.confidence, 4),
                    "conf_diff": round(conf_diffs[i], 4),
                }
                for i, (ref, cand) in enumerate(pairs)
            ],
            "outliers": image_outliers,
        }
        (run_dir / f"{image_path.stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        stats["images"] += 1
        stats["fully_matched"] += int(matched)
        stats["pairs"] += len(pairs)
        stats["class_consistent_pairs"] += class_consistent
        stats["iou_values"].extend(ious)
        stats["conf_diffs"].extend(conf_diffs)
        stats["ref_only"] += len(ref_extra)
        stats["rknn_only"] += len(rknn_extra)
        if image_outliers:
            stats["outliers"].append({"image": image_path.name, "issues": image_outliers})
        print(f"[{stats['images']}/{len(image_paths)}] {image_path.name}: "
              f"ref={len(ref_dets)} rknn={len(rknn_dets)} pairs={len(pairs)} matched={matched}")

    if rknn_live is not None:
        rknn_live.detector.close()

    # 汇总 ------------------------------------------------------------------
    def _mean(values: List[float]) -> Optional[float]:
        return round(statistics.fmean(values), 4) if values else None

    summary = {
        "model": args.model_name,
        "reference_onnx": str(onnx_path),
        "rknn_source": (
            str(args.candidate_onnx) if args.candidate_onnx
            else str(args.rknn_json_dir) if json_dir
            else str(args.rknn or "config默认")
        ),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "images": stats["images"],
        "fully_matched_images": stats["fully_matched"],
        "detection_pairs": stats["pairs"],
        "class_consistency": (
            f"{stats['class_consistent_pairs']}/{stats['pairs']}"
            if stats["pairs"] else "n/a"
        ),
        "mean_iou": _mean(stats["iou_values"]),
        "mean_abs_conf_diff": _mean(stats["conf_diffs"]),
        "ref_only_detections": stats["ref_only"],
        "rknn_only_detections": stats["rknn_only"],
        "outliers": stats["outliers"],
    }

    console = json.dumps({k: v for k, v in summary.items() if k != "outliers"},
                         ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print(console)

    # 汇总 markdown ---------------------------------------------------------
    doc_dir = (root / "docs" / "rknn").resolve()
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / f"{args.model_name}_validation.md"
    lines = [
        f"# {args.model_name} RKNN 一致性验证报告",
        "",
        f"- 生成时间: {summary['timestamp']}",
        f"- 参考(Reference): `{summary['reference_onnx']}` (onnxruntime, 与浏览器同预处理/解码)",
        f"- RKNN 来源: {summary['rknn_source']}",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 |",
        "| ---- | ---- |",
        f"| 测试图片数 | {summary['images']} |",
        f"| 完全匹配图片数 | {summary['fully_matched_images']} |",
        f"| 检测配对数 | {summary['detection_pairs']} |",
        f"| 类别一致率 | {summary['class_consistency']} |",
        f"| bbox 平均 IoU | {summary['mean_iou']} |",
        f"| 置信度平均绝对差 | {summary['mean_abs_conf_diff']} |",
        f"| 仅参考检出 | {summary['ref_only_detections']} |",
        f"| 仅 RKNN 检出 | {summary['rknn_only_detections']} |",
        "",
        "## 异常样本",
        "",
    ]
    if summary["outliers"]:
        for item in summary["outliers"][:50]:
            lines.append(f"### {item['image']}")
            lines.extend(f"- {issue}" for issue in item["issues"])
            lines.append("")
    else:
        lines.append("无(所有样本在容差内一致)。")
    lines += [
        "## 判定建议",
        "",
        "- 类别一致率 100% 且 平均 IoU >= 0.9 且 置信度差 <= 0.02: 可进入真机验证",
        "- 出现整类丢失(仅参考检出成批出现): 检查导出方式/量化",
        "- 本报告由 `rk3588_dms/tools/compare_outputs.py` 生成, 原始数据见 "
        f"`{run_dir.relative_to(root)}`",
        "",
    ]
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"汇总报告: {doc_path}")
    print(f"明细目录: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
