#!/usr/bin/env python3
"""从 .pt 权重导出 RKNN 转换用 ONNX。

两种风格:

  rknn(默认)   Rockchip rknn_model_zoo 推荐方式: 3 个分支输出
               [1, 64+nc, 80/40/20, ...], 图内不含 DFL softmax / Sigmoid / NMS,
               解码在 CPU 侧完成(runtime/postprocess.decode_rknn_modelzoo_branches)。
  standard     Ultralytics 默认导出: 单输出 [1, 4+nc, 8400](图内已含 DFL+Sigmoid)。

用法:
  # Rockchip 方式重导出两个自定义模型(RKNN 转换首选源)
  python rk3588_dms/tools/export_onnx_from_pt.py \
      --pt Driver-Monitoring-System/models/chaitanya/best.pt \
      --out rk3588_dms/models/onnx/rknn_source/chaitanya_best_rknn.onnx

  # 标准 COCO 模型(浏览器用)
  python rk3588_dms/tools/export_onnx_from_pt.py \
      --pt yolov8n.pt --style standard \
      --out Driver-Monitoring-System/public/static/models/yolov8n_coco.onnx

依赖(项目 .venv-rknn): ultralytics torch onnx onnxslim
"""

from __future__ import annotations

import argparse
import shutil
import sys
import types
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))


def export_rknn_style(pt_path: Path, out_path: Path, opset: int, imgsz: int) -> None:
    """Rockchip 推荐: Detect 头只输出 [box_dist(64), cls_logits(nc)] 分支特征图。"""
    import torch
    from ultralytics import YOLO

    yolo = YOLO(str(pt_path))
    model = yolo.model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.fuse()

    head = model.model[-1]
    for attr in ("cv2", "cv3", "nl", "nc"):
        if not hasattr(head, attr):
            raise RuntimeError(f"Detect 头缺少属性 {attr!r}, ultralytics 版本可能不兼容")
    nc, nl = int(head.nc), int(head.nl)

    def forward_branches(self, x):
        outs = []
        for i in range(self.nl):
            box = self.cv2[i](x[i])  # [B, 4*reg_max=64, H, W] 原始分布 logits
            cls = self.cv3[i](x[i])  # [B, nc, H, W] 原始类别 logits(未 Sigmoid)
            outs.append(torch.cat([box, cls], dim=1))
        return outs

    head.forward = types.MethodType(forward_branches, head)

    dummy = torch.zeros(1, 3, imgsz, imgsz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False: 使用 legacy TorchScript 导出器 —— 尊重 opset_version 且默认内嵌权重
    # (torch 2.14 新导出器会把权重写到 .onnx.data 外部文件且强制 opset>=17)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=[f"out{i}" for i in range(nl)],
        dynamo=False,
    )
    _embed_external_data(out_path)
    print(f"[OK] rknn 风格导出: {out_path}  (nc={nc}, 期望分支通道 64+{nc}={64 + nc})")


def _embed_external_data(out_path: Path) -> None:
    """若导出器把权重写成外部 .data 文件, 合并回单个 .onnx 并删除外部文件。"""
    import onnx

    model = onnx.load(str(out_path), load_external_data=True)
    external_refs = {
        init.data_location == onnx.TensorProto.EXTERNAL for init in model.graph.initializer
    }
    if True in external_refs or (out_path.parent / f"{out_path.name}.data").exists():
        onnx.save(model, str(out_path))  # 默认内嵌
        external = out_path.parent / f"{out_path.name}.data"
        if external.exists():
            external.unlink()
            print(f"     已合并外部权重文件并删除: {external.name}")


def export_standard(pt_path: Path, out_path: Path, opset: int, imgsz: int) -> None:
    """Ultralytics 默认导出 [1, 4+nc, N], 供浏览器/参考端使用。"""
    import tempfile
    from ultralytics import YOLO

    yolo = YOLO(str(pt_path))
    with tempfile.TemporaryDirectory() as tmp:
        produced = yolo.export(
            format="onnx", imgsz=imgsz, opset=opset, simplify=True,
            dynamic=False, half=False, batch=1, device="cpu",
        )
        produced = Path(produced)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), out_path)
    print(f"[OK] 标准导出: {out_path}")


def verify_onnx(out_path: Path, style: str, expected_nc: int | None) -> None:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    print(f"     输入: {inp.name} {inp.shape} {inp.type}")
    dummy = np.random.random((1, 3, 640, 640)).astype(np.float32)
    outputs = session.run(None, {inp.name: dummy})
    for index, out in enumerate(session.get_outputs()):
        shape = out.shape
        channels = shape[1] if len(shape) in (3, 4) else None
        note = ""
        if expected_nc and channels is not None:
            if style == "rknn":
                note = " (64+nc ✓)" if channels == 64 + expected_nc else f" (期望 {64 + expected_nc}!)"
            else:
                note = " (4+nc ✓)" if channels == 4 + expected_nc else f" (期望 {4 + expected_nc}!)"
        print(f"     输出[{index}]: {out.name} {shape}{note}")


def get_num_classes(pt_path: Path) -> int:
    from ultralytics import YOLO

    return int(YOLO(str(pt_path)).model.model[-1].nc)


def main() -> int:
    parser = argparse.ArgumentParser(description=".pt -> ONNX (rknn/standard 风格)")
    parser.add_argument("--pt", required=True, help=".pt 权重路径(yolov8n.pt 不存在时 ultralytics 自动下载)")
    parser.add_argument("--out", required=True, help="输出 ONNX 路径")
    parser.add_argument("--style", default="rknn", choices=["rknn", "standard"])
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-verify", action="store_true", help="跳过 onnxruntime 验证")
    args = parser.parse_args()

    pt_path = Path(args.pt).expanduser()
    if not pt_path.exists():
        print(f"[FAIL] .pt 不存在: {pt_path}", file=sys.stderr)
        return 1
    out_path = Path(args.out).expanduser()

    nc = get_num_classes(pt_path)
    print(f"模型: {pt_path}  nc={nc}  style={args.style}  opset={args.opset}  imgsz={args.imgsz}")

    if args.style == "rknn":
        export_rknn_style(pt_path, out_path, args.opset, args.imgsz)
    else:
        export_standard(pt_path, out_path, args.opset, args.imgsz)

    if not out_path.exists() or out_path.stat().st_size == 0:
        print("[FAIL] 导出产物缺失", file=sys.stderr)
        return 1
    print(f"     大小: {out_path.stat().st_size / 1024 / 1024:.2f} MB")

    if not args.no_verify:
        verify_onnx(out_path, args.style, nc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
