# 模型资产盘点（STEP 1 · 2026-09-09 真实数据版）

> 信息来源：**代码实证 + 模型文件实际解析**（`rk3588_dms/tools/inspect_model.py`，
> 零依赖 protobuf 解析 + ultralytics 元数据），不凭文件名猜测。
> 4 个缺失文件已确认来源 `https://github.com/AlbatrossC/Driver-Monitoring-System`（main 分支）
> 并下载到项目约定路径；`yolov8n_coco.onnx` 不在上游仓库，已由官方 `yolov8n.pt` 重新导出补齐。

## 1. 盘点总表（文件级字段为实测）

| 模型 | 当前路径（相对项目根） | 大小 | 输入尺寸 | 输入格式 | 输出 shape | 类别数 | 架构 | 当前调用位置 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| `chaitanya_best.onnx` | `Driver-Monitoring-System/public/static/models/chaitanya_best.onnx` | 12,241,395 B | 1×3×640×640 固定 | float32 NCHW, RGB, /255, letterbox 灰底128 | `[1, 9, 8400]`（4+5） | 5：Cigarette/Drinking/Eating/Phone/Seatbelt（实测 names） | YOLOv8n（.pt 内 C2f 块 + base `yolov8n.pt`；Ultralytics 8.3.251 导出，opset 11，nms=False，dynamic=False） | `driver-inference.js:102-106/:3311`；RKNN 重导出源 `models/chaitanya/best.pt`（6,226,730 B） |
| `soham_best.onnx` | `Driver-Monitoring-System/public/static/models/soham_best.onnx` | 10,570,055 B | 1×3×640×640 固定 | 同上 | `[1, 12, 8400]`（4+8） | 8：Distracted/Drinking/Drowsy/Eating/PhoneUse/SafeDriving/Seatbelt/Smoking（实测 names） | **YOLO11n**（.pt 内 C3k2 块 + `yolo11n.yaml`；同 8.3.251 导出，opset 11） | 默认单模型模式；RKNN 重导出源 `models/soham/best.pt`（5,450,067 B，与 fly.md 历史记录一致） |
| `yolov8n_coco.onnx` | `Driver-Monitoring-System/public/static/models/yolov8n_coco.onnx` | 12.26 MB（本次由官方 `yolov8n.pt` 导出） | 1×3×640×640 固定 | 同上 | `[1, 84, 8400]` | 80（COCO，只用 `cell phone`=cls 67） | 标准 YOLOv8n COCO（opset 12，无 NMS） | 手机检测唯一来源 + ROI ±22° 复检（`:2079/:2162`） |
| `face_landmarker.task` | `vendor/mediapipe/face_landmarker.task` | 3.58 MB | MediaPipe VIDEO | task bundle | 478 关键点 + 变换矩阵 + blendshapes | — | MediaPipe FaceLandmarker | `driver-inference.js:67-72/:1805/:1843`（150ms 循环）——KEEP IN BROWSER |
| 6DRepNet 权重 | `server.py:54-58` 查找序（本副本仍无 .pth） | — | 256→224 人脸裁切 | ImageNet mean/std | pitch/yaw/roll | — | RepVGG-B1g2(deploy)+GAP+FC→6D | `server.py:495`，浏览器 650ms 限频 |

## 2. 浏览器两个 ONNX 的实测迁移特征（inspect_model 输出）

| 特征 | chaitanya_best | soham_best |
| ---- | ---- | ---- |
| 生成器 | pytorch 2.8.0（Ultralytics 8.3.251 导出） | 同左 |
| opset | **11** | **11** |
| 内置 NMS | 否（args: nms=False） | 否 |
| DFL | 是（Softmax×1, Reshape×5 在图内） | 是（Softmax×2, Reshape×8；另 MatMul×2 为 v11 头特性） |
| 动态 shape | 否 | 否 |
| 节点数 / 参数量 | 259 / 3,006,623 | 350 / 2,583,712 |
| 主要算子 | Conv×64, Sigmoid×58(SiLU), Split×9 | Conv×88, Sigmoid×78, Split×11, MatMul×2 |

**结论**：两个浏览器 ONNX 都是标准 Ultralytics 导出（固定 640、无 NMS、输出 attr-major
`[1,4+nc,8400]`），与 `driver-inference.js parseDetections` 的解析约定完全吻合；
opset 11 可被 rknn-toolkit2 加载，但按用户指示 **RKNN 转换不直接使用它们**，
而是从 `best.pt` 按 Rockchip 推荐方式重导出（见下节）。

## 3. RKNN 专用重导出（已生成，PC 数值验证通过）

`rk3588_dms/tools/export_onnx_from_pt.py --style rknn`（Rockchip rknn_model_zoo 推荐）：
Detect 头只输出 `[1, 64+nc, 80/40/20]` 三分支原始 logits（图内无 DFL softmax、
无类别 Sigmoid、无 NMS），解码在 CPU 侧由
`runtime/postprocess.decode_rknn_modelzoo_branches` 完成。

| 产物（`rk3588_dms/models/onnx/rknn_source/`） | 来源 .pt | 输出 | 大小 | opset |
| ---- | ---- | ---- | ---- | ---- |
| `chaitanya_best_rknn.onnx` | `models/chaitanya/best.pt` | 3×`[1,69,80/40/20,80/40/20]` | 11.52 MB | 12 |
| `soham_best_rknn.onnx` | `models/soham/best.pt` | 3×`[1,72,...]` | 9.93 MB | 12 |
| `yolov8n_coco_rknn.onnx` | `rk3588_dms/models/onnx/yolov8n.pt`（官方 COCO） | 3×`[1,144,...]` | 12.07 MB | 12 |

（soham 图内残留的 Softmax×1 属 YOLO11 PSA 注意力模块，非 DFL，RKNN 支持。）

**PC 数值一致性（onnxruntime, face-test.png, 同 letterbox/解码路径）**：

| 模型 | 标准导出 vs rknn 风格重导出 |
| ---- | ---- |
| chaitanya | 检出 Cigarette 0.1153/0.1153，**Δconf=0.0000，IoU=1.0000**（低阈值 0.04 与 0.10 下均一致；生产阈值 0.25 两侧均无检出，一致） |
| soham | 4 检出全配对（SafeDriving 0.6659 / PhoneUse×2 / Smoking 0.1120），**Δconf=0.0000，IoU=1.0000** |
| coco | 2 检出全配对（0.4825/0.2140），**Δconf=0.0000，IoU=1.0000** |

即：重导出链路（RKNN 将采用的模型图 + 解码器）与浏览器链路在 FP32 下数值等价；
后续 RKNN 转换后的任何偏差都将只来自 RKNN 编译/量化本身——这正是我们想要的对照基线。
正式 ≥20 张测试集对比见 `chaitanya_validation.md`（当前仅 1 张健全性样本，测试集待补）。

## 4. 浏览器端解析约定（RKNN 后处理已逐位对齐）

来自 `driver-inference.js:1999 parseDetections`（详见 `current_dataflow.md` 第 4-5 节）：
letterbox 灰底 **128**、RGB/255；`[1,4+nc,8400]`/`[1,N,A]` 双主序自动判别；
坐标归一化判定 `max(值)<=2`；阈值 0.25（phone 0.20）；NMS IoU 0.45；
phone 只认 coco 模型输出。

## 5. 转换源优先级（本次已按用户指示调整）

`config/dms.json` `conversion.*.onnx_candidates` 现为：
1. `rk3588_dms/models/onnx/rknn_source/*_rknn.onnx`（**RKNN 转换首选**，已生成）
2. `Driver-Monitoring-System/public/static/models/*.onnx`（浏览器模型，仅回退）

重导出命令（记录于 config `reexport_command`）：

```bash
python rk3588_dms/tools/export_onnx_from_pt.py \
  --pt Driver-Monitoring-System/models/chaitanya/best.pt \
  --out rk3588_dms/models/onnx/rknn_source/chaitanya_best_rknn.onnx --style rknn
# soham / coco 同理（coco 用 rk3588_dms/models/onnx/yolov8n.pt）
```

## 6. MediaPipe / 6DRepNet 结论（不变）

- `face_landmarker.task`：KEEP IN BROWSER（分析见 `mediapipe_migration.md`）。
- 6DRepNet：第一阶段不迁移（分析见 `6drepnet_migration.md`；权重 `.pth` 仍缺）。
