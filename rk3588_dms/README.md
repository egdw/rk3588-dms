# rk3588_dms —— RK3588 原生 NPU 推理(第一阶段)

把浏览器里三个 YOLO ONNX(`chaitanya_best` / `soham_best` / `yolov8n_coco`)逐步迁移到
RK3588 NPU 后台推理的独立模块。**不改动现有浏览器推理代码**；第一阶段只做
`chaitanya_best` 的 RKNN 链路验证(单图正确识别 + 输出 bbox/class/confidence/latency)。

- 当前 DMS 数据流基线: `docs/rknn/current_dataflow.md`
- 模型盘点: `docs/rknn/model_inventory.md`
- 第一阶段结论: `docs/rknn/phase1_report.md`

## 目录

```text
rk3588_dms/
├── README.md                 本文件
├── config/dms.json           全部配置(目标平台/路径/阈值/类别表)
├── models/
│   ├── onnx/                 转换用 ONNX 工作目录(见下方"模型来源")
│   └── rknn/                 转换产物 chaitanya_best_fp.rknn 等
├── runtime/                  推理运行时(预处理/后处理/模型封装/检测器)
├── tools/
│   ├── inspect_model.py      ONNX 结构检查(也可用根目录 tools/model_inspect.py)
│   ├── convert_chaitanya.py  ONNX -> RKNN 转换(PC 端)
│   ├── benchmark_rknn.py     推理基准(板上 NPU 或 PC 模拟器)
│   └── compare_outputs.py    ONNX vs RKNN 一致性对比
├── test_image.py             RK3588 单图推理测试(完全独立)
├── test_camera.py            摄像头实时测试(OpenCV 采集, 预留 MPP/RGA)
└── tests/                    纯 numpy/cv2 单元测试(不依赖 rknn)
```

所有路径在 `config/dms.json` 中相对项目根配置, 脚本自动定位项目根, **无硬编码绝对路径**;
Windows / Linux 均可运行(真机推理只能在 RK3588 上)。

## 一、PC 端(模型转换)

### 1. 环境准备

```bash
# 建议独立虚拟环境(Python 3.8~3.12, 以 rknn-toolkit2 官方支持为准)
python -m venv .venv-rknn
# Windows: .venv-rknn\Scripts\activate    Linux: source .venv-rknn/bin/activate

pip install rknn-toolkit2 onnx onnxruntime opencv-python numpy
# Python 3.12+ 的 venv 默认不装 setuptools, 而 rknn-toolkit2 依赖 pkg_resources:
pip install "setuptools<81"
# 若报 "compiled using NumPy 1.x" 类错误(旧版 toolkit2 与 numpy 2 不兼容):
pip install "numpy<2"
# rknn-toolkit2 完整安装方式(含依赖)见官方仓库:
#   https://github.com/airockchip/rknn-toolkit2  (packages 目录下对应平台的 whl)

# 环境自检
python tools/check_rknn_env.py --role pc
```

版本匹配原则: PC 端 rknn-toolkit2 版本必须 >= 且尽量等于板上 librknnrt/rknn-toolkit-lite2
版本(转换时会写入版本信息, 大版本不匹配可能加载失败)。

### 2. 模型来源与 RKNN 专用重导出(已完成)

RKNN 转换**不依赖浏览器 ONNX**。按 Rockchip YOLOv8/YOLO11 推荐方式, 从 `.pt`
权重重导出"三分支" ONNX(Detect 头只输出 `[1,64+nc,H,W]` 原始 logits, 图内无
DFL softmax / Sigmoid / NMS), 已生成于 `rk3588_dms/models/onnx/rknn_source/`:

```bash
# 如需重新生成(环境: 项目 .venv-rknn, 见下)
.venv-rknn/Scripts/python.exe rk3588_dms/tools/export_onnx_from_pt.py \
    --pt Driver-Monitoring-System/models/chaitanya/best.pt \
    --out rk3588_dms/models/onnx/rknn_source/chaitanya_best_rknn.onnx --style rknn
# soham_best_rknn.onnx / yolov8n_coco_rknn.onnx 同理(coco 源: rk3588_dms/models/onnx/yolov8n.pt)
```

浏览器模型保持原位只读(`Driver-Monitoring-System/public/static/models/*.onnx`);
`yolov8n_coco.onnx` 不在上游仓库, 已由官方 `yolov8n.pt` 标准导出补齐(640, opset 12)。
`tools/convert_chaitanya.py` 的源查找顺序(config `onnx_candidates`)已是
**rknn_source 优先, 浏览器 ONNX 仅作回退**。

实测校验(2026-09-09, onnxruntime, 详见 `docs/rknn/model_inventory.md` 第 3 节):
三个 rknn_source ONNX 与浏览器标准 ONNX 在 FP32 下 **Δconf=0.0000 / IoU=1.0000**。

PC 导出环境(已装在项目内 venv, 未动系统 Python):

```bash
python -m venv .venv-rknn
.venv-rknn/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-rknn/Scripts/python.exe -m pip install ultralytics onnx onnxslim onnxruntime opencv-python numpy onnxscript
```

模型结构检查(零依赖, 无 onnx 包也可运行):

```bash
python tools/model_inspect.py Driver-Monitoring-System/public/static/models/chaitanya_best.onnx
```

### 3. 转换(第一轮强制 FP, 不做 INT8)

```bash
python rk3588_dms/tools/convert_chaitanya.py
# 明确打印: 输入模型 / 输出模型 / 目标平台 rk3588 / 量化模式 FP / 输入尺寸 640x640
# 日志: logs/rknn/convert_chaitanya_*.log
# 产物: rk3588_dms/models/rknn/chaitanya_best_fp.rknn

# INT8 属后续阶段(需先准备 dataset.txt 校准图片列表, 见"INT8 量化"章节):
# python rk3588_dms/tools/convert_chaitanya.py --int8
```

### 4. PC 端一致性预检(可选但强烈建议)

准备测试集(见 `testdata/dms/README.md`, ≥20 张)后:

```bash
# 模式C: 转换前预检 —— 用 onnxruntime 跑 rknn_source ONNX(与 RKNN 同解码路径)
python rk3588_dms/tools/compare_outputs.py --model-name chaitanya \
    --candidate-onnx rk3588_dms/models/onnx/rknn_source/chaitanya_best_rknn.onnx

# 模式A: 转换后 —— RKNN = rknn-toolkit2 模拟器
python rk3588_dms/tools/compare_outputs.py --model-name chaitanya
# 参考 = onnxruntime 跑浏览器标准 ONNX(与浏览器同预处理/解码)
# 汇总报告: docs/rknn/chaitanya_validation.md
```

注意: PC 模拟器与真机量化行为有差异, 真机结论以模式 B(板上导出 JSON 回灌)为准。

## 二、RK3588 端(真机验证)

### 1. 环境准备

```bash
# 板上(Python 3.9~3.11 视系统而定)
pip install rknn-toolkit-lite2 opencv-python numpy
# librknnrt.so 通常由系统固件/厂商提供(/usr/lib/librknnrt.so),
# 也可从 rknn-toolkit2 仓库 rknpu2/runtime/Linux/librknn_api/aarch64/ 安装

python tools/check_rknn_env.py --role device   # 期望全部 PASS
```

### 2. 同步文件到板子

最小集合(相对项目根):

```text
rk3588_dms/            整个目录(含 models/rknn/chaitanya_best_fp.rknn)
tools/check_rknn_env.py
testdata/dms/          测试图片(可选)
```

```bash
rsync -av --exclude __pycache__ rk3588_dms/ <user>@<rk3588>:/home/<user>/dms/rk3588_dms/
```

### 3. 单图测试(STEP 5, 本阶段停止线)

```bash
python rk3588_dms/test_image.py \
  --model rk3588_dms/models/rknn/chaitanya_best_fp.rknn \
  --image testdata/dms/phone_01.jpg
```

输出示例(必须看到 runtime 为 NPU 而非 SIMULATOR):

```text
model        : .../chaitanya_best_fp.rknn
input        : phone_01.jpg  (1920x1080)
runtime      : RK3588 NPU (rknn-toolkit-lite2)
inference    : 13.4 ms
detections   : 2
  Phone        0.94  [120, 80, 340, 420]
  Seatbelt     0.87  [300, 210, 510, 620]
```

板上批量导出 JSON 供 PC 对比:

```bash
for img in testdata/dms/*.jpg; do
  python rk3588_dms/test_image.py --image "$img" \
    --json "board_results/$(basename "$img" .jpg).json"
done
# 拷回 PC 后: python rk3588_dms/tools/compare_outputs.py --rknn-json-dir board_results/
```

### 4. Benchmark(STEP 6)

```bash
python rk3588_dms/tools/benchmark_rknn.py --model-name chaitanya --warmup 20 --runs 200
# 输出 min/max/average/p50/p95/理论 FPS, 默认 NPU_CORE_AUTO(第一阶段不手动分核)
# 结果 JSON 存 logs/rknn/benchmark_*.json
```

### 5. 摄像头实时测试(STEP 7)

```bash
python rk3588_dms/test_camera.py --camera /dev/video0
# 每 5 秒输出: camera FPS / inference FPS / 端到端延迟 / 检测结果
# Ctrl+C 安全退出; 读帧失败有明确日志与有限重试
```

⚠️ 摄像头独占: 浏览器 `getUserMedia` 与本程序不能同时占用 `/dev/video0`。
单独测试时先不要打开网页检测页。

### 6. INT8 量化(后续阶段, FP 全部通过后再做)

1. 准备校准集 100~300 张真实驾驶舱图片, 生成 `rk3588_dms/models/onnx/dataset.txt`
   (每行一张图片的**绝对路径**, 覆盖不同光线/姿态/手机/安全带/低头);
2. `python rk3588_dms/tools/convert_chaitanya.py --int8`
   → 产物 `chaitanya_best_int8.rknn`;
3. 重跑 `compare_outputs.py` 对比 FP vs INT8;
4. **精度明显下降则比赛版本继续用 FP**, 不为速度牺牲识别稳定性。

## 三、集成阶段预告(本阶段不实施)

- **摄像头共享**: 浏览器与 Python 不能同时开摄像头。方案(按优先级):
  1. Camera Capture Service 单点采集(Python 打开 /dev/video0), 通过
     MJPEG/WS 分发给浏览器显示 + 本模块推理;
  2. 或复用现有视频链路, 浏览器继续显示, 原生服务只做推理(需 v4l2 分配共享)。
  第一阶段不解决, 单独测试即可。
- **WebSocket 输出**: 新增 `/ws/dms/native` 推送
  `{"type":"dms_result","source":"rk3588_rknn","models":{...}}`;
  当前系统本没有结果输出通道, 属新增能力, 不破坏现有 MQTT/HTTP。
- **browser/native 双模式**: 配置 `dms.inference_mode`, browser=现状不动,
  native=浏览器不加载三个 YOLO、只连 WS; 切换不需重编译。
- **状态融合/报警时序留在浏览器**(阈值是比赛调优结果, 不迁移不改动)。
- MediaPipe FaceLandmarker、6DRepNet 不迁移(分析见 docs/rknn/ 对应文档)。

## 四、常用命令速查

```bash
python tools/check_rknn_env.py                          # 自动识别 PC/设备
python tools/model_inspect.py <model.onnx>              # 模型结构检查
python rk3588_dms/tools/convert_chaitanya.py            # FP 转换
python rk3588_dms/test_image.py --image x.jpg           # 单图(板上)
python rk3588_dms/tools/benchmark_rknn.py               # 基准
python rk3588_dms/test_camera.py --camera /dev/video0   # 摄像头
python rk3588_dms/tools/compare_outputs.py              # 一致性对比
python -m unittest discover -s rk3588_dms/tests -v      # 单元测试(纯 numpy/cv2)
```

## 五、日志

- `logs/rknn/` —— 转换日志、benchmark JSON、对比明细
- `logs/dms/` —— 预留给原生 DMS 服务运行日志(模型加载/初始化耗时/摄像头事件/
  推理异常/每 5 秒统计, 不逐帧刷屏)
