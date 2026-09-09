# MediaPipe FaceLandmarker 迁移分析（本轮 KEEP IN BROWSER）

> 结论先行：**本轮明确保留在浏览器**（KEEP IN BROWSER），不做任何迁移改动。
> 本文记录现状与未来难点，避免后续误判工作量。

## 1. 现状

- 资产：`vendor/mediapipe/face_landmarker.task`（3.58 MB task bundle，已本地化，
  不依赖外网 CDN）+ `vendor/mediapipe/wasm/`（MediaPipe wasm 运行时）+
  `vision_bundle.mjs`；
- 调用：`driver-inference.js:1805 ensureHeadPoseModel()` ——
  `FaceLandmarker.createFromOptions`，VIDEO 模式，GPU delegate 优先、CPU 兜底，
  `numFaces=1`，`outputFacialTransformationMatrixes=true`，
  `outputFaceBlendshapes=true`；
- 频率：`cameraFaceLoop` 每 150ms（≈6.7Hz），与 500ms 的 YOLO 循环解耦；
- 产出（`analyzeHeadPose`, `driver-inference.js:1843`）：
  - `head_down`（低头）置信度（矩阵 pitch + 关键点几何 + 6DRepNet 融合）；
  - `drowsy`（完全闭眼 ≥0.75 持续 300ms）；
  - `gaze_off`（虹膜偏移 + 头姿综合 ≥0.62）；
  - 全部遥测 UI（头姿角度、睁闭眼、虹膜方向线、眼动波形、关键区域裁切）。

## 2. 该 .task 内部是什么

FaceLandmarker task bundle 是 MediaPipe 图 + 内嵌 TFLite 模型的序列化组合，
核心子模型：

1. **Face Detector**（约 1MB 级 blazeface 变体）：整图人脸检测 + ROI；
2. **Face Landmark**（478 点稠密网格模型）：对 ROI 回归关键点；
3. **Face Blendshapes**（52 维表情系数头）：可选开启（本项目开启）；
4. **Facial Transformation Matrix**（几何求解器 + 头部模型顶点）：输出 4×4 矩阵。

`.task` 文件本质是 zip 容器；如需确认内嵌 tflite，可
`unzip -l vendor/mediapipe/face_landmarker.task` 查看（不修改原文件）。

## 3. 未来迁移难点（如果做）

| 难点 | 说明 |
| ---- | ---- |
| 模型获取 | task bundle 内嵌 tflite 需解包；MediaPipe 官方不直接发布裸 ONNX，需要走 tflite→ONNX→RKNN 或用替代模型 |
| 多模型流水线 | detector→ROI crop→landmark 是两级模型 + 坐标变换，RKNN 侧要自己编排 |
| Tracking | VIDEO 模式的帧间跟踪（光流/平滑）在 MediaPipe 内部，迁移后需自研或降级为逐帧检测 |
| Blendshapes | 52 维表情系数头是额外子模型，浏览器端只用了眨眼两项，迁移价值低 |
| 关键点几何 | 478 点的眼睑/虹膜几何计算（EAR、虹膜局部坐标系）是纯 CPU 数学，可平移，但要逐位对齐阈值 |
| 后处理链 | 本项目 12+ 个阈值（0.56/0.75/0.62/0.34…）与 150ms 节奏是调优结果，迁移后需完整回归 |

## 4. 建议

- 短期（本项目内）：**保持浏览器**。当前浏览器负载瓶颈是三个 YOLO ONNX + 渲染，
  FaceLandmarker 在 wasm/GPU 下开销相对小（150ms 周期、单人脸）；
- 中期可选项：若 NPU 化后浏览器仍卡，优先考虑把"landmark 检测"换成 RKNN 侧
  等效关键点模型，把 478 点坐标推给浏览器只做绘制——但这是新的一整条验证链路，
  必须像 chaitanya 一样走完 单图→对比→真机→摄像头 四步，不允许直接替换；
- 任何情况下 `face_landmarker.task` 文件与浏览器加载路径不动。
