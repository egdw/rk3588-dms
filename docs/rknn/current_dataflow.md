# 当前 DMS 数据流（RKNN 迁移前基线）

> 本文档基于对项目源码的实际阅读整理（2026-09-09），是 RK3588 NPU 迁移改造的基线记录。
> 阅读范围：`driver-inference.js`（3628 行）、`backend/server.py`（1555 行）、`index.html`、`backend/README.md`、`fly.md`。
> 结论中的行号均指向当前文件版本，供后续对照。

## 1. 总体架构（现状）

所有 AI 推理与状态融合目前都在 RK3588 的 Chromium 浏览器内完成：

```text
USB Camera
    ↓ getUserMedia (浏览器独占)
#cameraVideo 原生播放 (320x180@8fps 采集, 画面持续渲染)
    ↓ 两个异步循环
┌─────────────────────────────────────────────────────┐
│ 浏览器 (Chromium)                                    │
│                                                     │
│ cameraLoop 每 500ms:                                │
│   preprocess (letterbox 640x640, 灰底, RGB/255)     │
│   → ONNX Runtime Web (WebGPU 或 WASM)               │
│     · soham_best.onnx      (默认单模型模式)          │
│     · chaitanya_best.onnx  (双模型融合模式)          │
│     · yolov8n_coco.onnx    (始终加载, 手机检测专用)   │
│   → parseDetections → suppressModelDrowsy           │
│   → detectPhoneAroundFace (COCO ROI ±22° TTA)       │
│                                                     │
│ cameraFaceLoop 每 150ms:                            │
│   MediaPipe FaceLandmarker (vendor/mediapipe/)      │
│   → 头姿/闭眼/视线 → head_down/drowsy/gaze_off      │
│   → 6DRepNet 后端 HTTP 推理 (优先, 失败回退关键点)    │
│                                                     │
│ 融合: suppressOverlaps(NMS) + 手动控制覆盖           │
│   → 报警确认时序 (1500/1000/1800ms)                  │
│   → 3s 报警保持 + 音频播报                           │
│   → UI 更新 (全部在浏览器)                           │
└─────────────────────────────────────────────────────┘
    ↑↓ HTTP                    ↑↓ MQTT over WebSocket
Python 后端 (server.py)        外部 mosquitto (192.168.2.13:8084)
  · /api/manual-control          · 仅订阅控制命令
  · /api/head-pose/6drepnet        vision-sentinel/control
  · 静态文件/模型/数据集 API     · 不输出检测结果
```

**关键结论：当前系统没有检测结果 WebSocket 输出通道**。所有检测结果只进浏览器 UI，不上报、不推送。MQTT WebSocket 仅用于接收手动控制命令（演示模式）。后端 `server.py` 是纯 HTTP（`ThreadingHTTPServer`），不提供 WebSocket 服务。

## 2. 视频来源与摄像头占用

- 浏览器通过 `navigator.mediaDevices.getUserMedia` 打开摄像头（`driver-inference.js:3454`）：
  - 目标 320×180，上限 424×240，8 fps，facingMode user，无音频。
- 视频元素原生连续播放；透明叠加画布 `#detectionCanvas` 异步更新检测框，推理不阻塞视频刷新。
- **摄像头被浏览器独占**。若 Python 原生服务同时打开 `/dev/video0` 会 `Device busy`。第一阶段单机测试时需关闭浏览器占用；集成阶段必须引入 Camera Capture Service 单点采集分发（见 `rk3588_dms/README.md` 集成章节）。

## 3. 模型资产与加载

| 模型 | 浏览器路径（相对站点根） | 用途 | 类别 |
| ---- | ---- | ---- | ---- |
| `soham_best.onnx` | `Driver-Monitoring-System/public/static/models/soham_best.onnx` | 状态检测（默认模式） | 8 类: Distracted/Drinking/Drowsy/Eating/PhoneUse/SafeDriving/Seatbelt/Smoking |
| `chaitanya_best.onnx` | `Driver-Monitoring-System/public/static/models/chaitanya_best.onnx` | 行为目标（融合模式） | 5 类: Cigarette/Drinking/Eating/Phone/Seatbelt |
| `yolov8n_coco.onnx` | `Driver-Monitoring-System/public/static/models/yolov8n_coco.onnx` | 手机检测唯一来源 + ROI 复检 | COCO 80 类，只用 `cell phone` |
| `face_landmarker.task` | `vendor/mediapipe/face_landmarker.task` | 人脸关键点/头姿/虹膜 | MediaPipe 478 点 |
| 6DRepNet 权重 | `6DRepNet-master/**.pth`（后端查找） | 头部姿态 pitch/yaw/roll | 回归 6D |

- ONNX Runtime Web 从本地 `/vendor/ort/` 加载：跨域隔离时 `ort.webgpu.min.js`（webgpu+wasm），否则 `ort.wasm.min.js`（纯 wasm）。WASM 线程数上限 4。
- `selectedModels()`（`driver-inference.js:743`）保证 **coco 模型永远加载**（手机检测唯一来源）。
- 后端另注册两个 `.pt` 训练权重：`Driver-Monitoring-System/models/{soham,chaitanya}/best.pt`（评估/再训练用）。

## 4. 预处理（必须逐位复刻）

`driver-inference.js:1972 preprocess()`：

1. 画布 640×640 先填充 **`#808080`（128 灰，不是 Ultralytics 默认 114）**。
2. `scale = min(640/w, 640/h)`，等比缩放居中贴图（无 Ultralytics 的 half-pixel 取整差异，浏览器 canvas 自带插值）。
3. 像素 **RGB 顺序**、除以 255、NCHW float32。
4. 手机 ROI 复检用同样手法但带 ±22° 旋转，输入也是 640×640。

RKNN 侧如需与浏览器输出严格对齐，letterbox 填充值必须用 128。

## 5. 输出解析（`parseDetections`, `driver-inference.js:1999`）

兼容两种布局，自动判别：

- 取第一个输出张量，要求 dims 为 `[1, A, N]`（attr-major，Ultralytics 默认导出）或 `[1, N, A]`。
- `hasObjectness = attrCount >= numClasses + 5`（即第 5 行是 objectness 时 classStart=5，否则 4）。
- 每个候选取类别分数 argmax 作为置信度（乘 objectness，若有）。
- 坐标 cx,cy,w,h：若 `max(值) <= 2` 则视为归一化坐标 ×640。
- 类别映射到统一 key（`UNIFIED_CLASSES`）；只有 `ACTIVE_DETECTION_KEYS`（safe/drowsy/head_down/gaze_off/phone）参与展示。
- **phone 类只接受 coco 模型输出**（`driver-inference.js:2043`），旧模型的 Phone/PhoneUse 高误报输出被丢弃。
- 置信度阈值：通用 0.25，phone 0.20。
- NMS：按统一 key 分组，IoU>0.45 抑制（`suppressOverlaps`）。

## 6. 人脸通道（MediaPipe + 6DRepNet）

`analyzeHeadPose()`（`driver-inference.js:1843`）：

- FaceLandmarker VIDEO 模式，GPU delegate 优先失败回退 CPU；选主脸（面积/可见性/居中/肤色证据综合评分）。
- 6DRepNet：人脸框 256×256 JPEG dataURL → POST `/api/head-pose/6drepnet`，最短间隔 650ms；连续失败 2 次禁用并回退关键点头姿。后端为 PyTorch CPU 推理（`server.py:495`）。
- 产出三类检测（均带 faceBox）：
  - `head_down`：pitch≥16° 起算线性置信度（range 18°），阈值 0.56；
  - `drowsy`：完全闭眼（眼部置信度 ≥0.75）持续 300ms；
  - `gaze_off`：虹膜偏移+头姿综合 ≥0.62 确认 400ms，释放阈值 0.34，保持 500ms。
- 模型来源的 `Drowsy` 抑制规则：眼部明显睁开（<0.52）时丢弃；闭眼时要求置信度 ≥0.6（`suppressModelDrowsy`）。

## 7. 状态融合与报警（浏览器 JS）

`infer()`（`driver-inference.js:3294`）合并顺序：

```text
merged = applyManualControl(
           suppressOverlaps(
             suppressModelDrowsy(ONNX 检测) + 人脸通道检测))
```

报警确认时序（`updateResults`, `driver-inference.js:3111`）：

| key | 确认时长 | 漏检宽限 | 说明 |
| ---- | ---- | ---- | ---- |
| drowsy | 1000ms | 700ms | 所有来源统一 1 秒 |
| gaze_off | 1800ms | 700ms | |
| 其他（head_down/phone） | 1500ms | phone 2200ms | |

- 报警确认后保持显示 3000ms（ALERT_HOLD_MS）。
- 音频播报优先级 drowsy(4) > head_down(3) > gaze_off(2) > phone(1)，同 key 冷却 4500ms。
- 疲劳指数 = drowsy 置信度×100，≥35 报警；专注度 = max(safe, 1−danger)，<75 报警。
- 事件时间线同 key 限频 5 秒一条。

**迁移约束**：以上阈值和时序是比赛现场调优结果，任何后端化改造不得改动数值。

## 8. 外部接口清单（不可破坏）

| 接口 | 方向 | 端点/地址 | 现状 |
| ---- | ---- | ---- | ---- |
| 手动控制轮询 | 浏览器→后端 | `GET/POST /api/manual-control`（800ms 轮询） | 保留 |
| MQTT 控制 | 浏览器←mosquitto | `wss://192.168.2.13:8084/mqtt` 订阅 `vision-sentinel/control`（浏览器内置裸 MQTT 3.1.1 编解码） | 保留 |
| 6DRepNet 推理 | 浏览器→后端 | `POST /api/head-pose/6drepnet` | 第一阶段保留 |
| 静态资源 | 浏览器→后端 | `/index.html`、`/vendor/**`、`Driver-Monitoring-System/**` | 保留 |
| 标注录制 | 浏览器→后端 | `POST /api/annotation-tracks` → `storage/annotation-tracks/*.json` | 保留 |
| 后端服务 | — | `start-backend.sh`，HTTP 8000 / HTTPS 8443（自签） | 保留 |

浏览器可用 URL 参数 `?mqttWs=...&mqttTopic=...` 临时覆盖 MQTT 地址。

## 9. 迁移基线结论

1. 第一阶段只迁 **三个 YOLO ONNX → RKNN**，在 RK3588 NPU 后台跑通 `chaitanya_best` 单图；MediaPipe 与 6DRepNet 明确不动（浏览器保留）。
2. 原生服务输出通道将是**新增** `/ws/dms/native` WebSocket（复用现有事件格式的能力受限于"当前本就没有结果输出"这一事实，因此第一阶段只做独立验证程序，不接 UI、不动浏览器代码）。
3. 摄像头独占是集成阶段的已知冲突点，第一阶段测试程序独立运行即可。
4. 状态融合（第 7 节）第一阶段继续留在浏览器；原生服务只输出 RawDetection（模型原始结果）。
5. 预处理对齐要点：灰底 128、RGB、/255、letterbox 居中（无取整偏移）。
