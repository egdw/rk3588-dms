# models/onnx —— 转换用 ONNX 工作目录

本目录**不存放**浏览器正在使用的模型；浏览器模型保持原位只读：

```text
Driver-Monitoring-System/public/static/models/*.onnx   ← 浏览器使用, 勿动
```

## 当前内容（2026-09-09）

- `rknn_source/` —— **RKNN 转换首选源**（Rockchip 推荐三分支导出，opset 12，
  图内无 DFL/Sigmoid类别/NMS），已由 `tools/export_onnx_from_pt.py --style rknn`
  生成并通过与浏览器标准 ONNX 的 FP32 数值一致性校验（Δconf=0.0000, IoU=1.0000）：
  - `chaitanya_best_rknn.onnx`（11.52 MB，源 `models/chaitanya/best.pt`，[1,69,80/40/20]×3）
  - `soham_best_rknn.onnx`（9.93 MB，源 `models/soham/best.pt`，[1,72,…]×3）
  - `yolov8n_coco_rknn.onnx`（12.07 MB，源本目录 `yolov8n.pt`，[1,144,…]×3）
- `yolov8n.pt` —— 官方 Ultralytics YOLOv8n COCO 基础权重（6.5 MB，重导出/复现用）。
- `browser/` —— 预留：浏览器模型的手工副本（跨机器转换时使用）。
- `dataset.txt` —— INT8 量化校准列表（后续阶段使用）。

## 重新生成

```bash
.venv-rknn/Scripts/python.exe rk3588_dms/tools/export_onnx_from_pt.py \
  --pt Driver-Monitoring-System/models/chaitanya/best.pt \
  --out rk3588_dms/models/onnx/rknn_source/chaitanya_best_rknn.onnx --style rknn
```

转换产物（.rknn）输出到 `../rknn/`。
