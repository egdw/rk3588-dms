#!/usr/bin/env python3
"""RK3588 单图推理测试(完全独立, 不接网页/后端)。

用法(RK3588 板上):
  python rk3588_dms/test_image.py --model rk3588_dms/models/rknn/chaitanya_best_fp.rknn --image test.jpg
  python rk3588_dms/test_image.py --image test.jpg                  # 默认 chaitanya FP 模型
  python rk3588_dms/test_image.py --model-name soham --image a.jpg  # 换 soham 类别表

PC 上(无 NPU)自动进入 rknn-toolkit2 模拟器模式, 输出会明确标注 SIMULATOR,
模拟器结果只用于核对 shape/解码逻辑, 不代表 NPU 性能。

依赖: opencv-python, numpy; 板上另需 rknn-toolkit-lite2, PC 另需 rknn-toolkit2。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))


def build_detector(model_name: str, model_path, mode: str, core: str = "auto"):
    from runtime.base_detector import load_config

    config = load_config(PACKAGE_ROOT / "config" / "dms.json")
    model_config = config["runtime"]["models"][model_name]
    rknn_path = model_path or model_config.get("rknn")
    if model_name == "chaitanya":
        from runtime.chaitanya_detector import ChaitanyaDetector

        return ChaitanyaDetector(mode=mode, rknn_path=rknn_path, core=core)
    from runtime.base_detector import RknnYoloDetector

    class _Detector(RknnYoloDetector):
        model_name = model_name

    return _Detector(model_config=model_config, model_path=rknn_path, mode=mode, core=core)


def main() -> int:
    parser = argparse.ArgumentParser(description="RKNN 单图推理测试")
    parser.add_argument("--model", help=".rknn 路径(缺省用 config/dms.json 中该模型的 rknn 路径)")
    parser.add_argument("--model-name", default="chaitanya", choices=["chaitanya", "soham", "coco"])
    parser.add_argument("--image", required=True, help="测试图片路径")
    parser.add_argument("--mode", default="auto", choices=["auto", "device", "simulator"])
    parser.add_argument("--json", help="把结果写入 JSON 文件(供 compare_outputs.py)")
    parser.add_argument("--show", action="store_true", help="OpenCV 弹窗显示检测结果(需要桌面环境)")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.exists():
        print(f"[FAIL] 图片不存在: {image_path}", file=sys.stderr)
        return 1

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[FAIL] 缺少 opencv-python: pip install opencv-python", file=sys.stderr)
        return 2

    detector = build_detector(args.model_name, args.model, args.mode)
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[FAIL] 图片解码失败: {image_path}", file=sys.stderr)
        return 1

    result = detector.infer(frame)

    runtime_note = (
        "RK3588 NPU (rknn-toolkit-lite2)"
        if detector.model.backend == "npu"
        else "PC SIMULATOR (rknn-toolkit2, 非 NPU!)"
    )
    print("=" * 64)
    print(f"model        : {detector.model_path}")
    print(f"input        : {image_path}  ({result.source_size[0]}x{result.source_size[1]})")
    print(f"runtime      : {runtime_note}")
    if detector.model.npu_status is not None:
        st = detector.model.npu_status
        print(f"device check : machine={st.machine} model={st.device_tree_model or 'N/A'}")
        print(f"npu driver   : {st.rknpu_driver_evidence or 'N/A'}")
        for note in st.notes:
            print(f"  [WARN] {note}")
    print(f"model init   : {detector.model.init_ms:.1f} ms")
    print(f"preprocess   : {result.preprocess_ms:.1f} ms")
    print(f"inference    : {result.inference_ms:.1f} ms")
    print(f"postprocess  : {result.postprocess_ms:.1f} ms")
    print(f"detections   : {result.detection_count}")
    for det in result.detections:
        box = ", ".join(f"{v:.0f}" for v in det.bbox)
        print(f"  {det.class_name:<12s} {det.confidence:.2f}  [{box}]")
        if det.unified_key:
            print(f"      -> unified key: {det.unified_key}")
    print("=" * 64)

    if args.json:
        payload = result.to_dict()
        payload["image"] = str(image_path)
        payload["runtime_backend"] = detector.model.backend
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 已写入: {args.json}")

    if args.show:
        canvas = frame.copy()
        for det in result.detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                canvas, f"{det.class_name} {det.confidence:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
        cv2.imshow("rknn dms test", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    detector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
