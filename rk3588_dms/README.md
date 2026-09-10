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

### 6. INT8 量化(已完成并真机验证, 2026-09-10)

校准集: 比赛视频(1280x720@60fps, 115s)均匀抽帧 320 张 → 274 张校准
(letterbox 640x640 灰底128, 与推理预处理一致, 保证 toolkit 内部 resize 为恒等)
+ 46 张测试集(已入仓 `testdata/dms/`)。校准图与 dataset.txt 留在真机本地
(dataset.txt 为机器相关绝对路径, 已 gitignore)。

```bash
python rk3588_dms/tools/convert_chaitanya.py --int8    # 用 config 的 quant_dataset
# FP vs INT8 真机对比(--rknn=FP 参考, --rknn2=INT8 对照, 46 张测试图):
python rk3588_dms/tools/compare_outputs.py --model-name chaitanya \
  --images testdata/dms \
  --rknn rk3588_dms/models/rknn/chaitanya_best_fp.rknn \
  --rknn2 rk3588_dms/models/rknn/chaitanya_best_int8.rknn --mode device
```

**实测结果**(A588 板真 NPU, 报告 `docs/rknn/chaitanya_int8_vs_fp.md`):

| 指标 | FP RKNN | INT8 RKNN |
| ---- | ---- | ---- |
| 平均推理 | 41.7 ms | **20.4 ms (2.05x)** |
| P50 / P95 | 41.2 / 46.5 ms | 20.1 / 23.3 ms |
| 理论 FPS | 24.0 | **49.1** |
| 模型大小 | 7.0 MB | 4.2 MB |
| 精度(46 张真实座舱图) | 基线 | 类别一致率 10/10, 平均 IoU 0.9825, 平均 \|Δconf\| 0.0277, 零漏检零多检 |

唯一差异样本: test_0148 Cigarette 0.571(FP) vs 0.667(INT8) —— INT8 更自信,
两者均远超 0.25 阈值, 判定行为不变。**INT8 可用**; 比赛默认仍配置为 FP
(保守), 切换只需把 `config/dms.json` 对应模型的 `rknn` 路径改为
`*_int8.rknn`。三模型并行若叠加 INT8, 预计并行墙钟 ~40ms。

## 三、NPU 三核并行与量化(已实现并实测)

RK3588 有 3 个 NPU 核。`config/dms.json` 中每个模型通过 `npu_core` 字段绑核
(chaitanya=0 / soham=1 / coco=2, 全局 `runtime.npu_core` 为默认值),
`runtime.parallel.ParallelDetectorGroup` 用线程级并行同时推理(RKNN 在 C 层释放
GIL, 无需多进程, 可直接共享摄像头帧), `tools/benchmark_parallel.py` 一键实测,
`--variant int8` / `--variant int8,fp,int8` 可测任意 FP/INT8 组合。

### 真机实测汇总(A588 板, 2026-09-10, face-test.png, warmup10/runs100)

| 部署形态 | 单核耗时 | 三核并行墙钟 | 理论 FPS |
| ---- | ---- | ---- | ---- |
| 全 FP | 48.1 / 51.6 / 51.1 ms | 74.9 ms (2.34x) | 13.4 |
| 全 INT8 | **18.8 / 22.0 / 20.4 ms** | **43.8 ms (2.29x)** | **22.8** |
| 混合 int8,fp,int8(soham 保 FP) | 21.0 / 49.3 / 21.0 ms | 68.1 ms (1.86x) | 14.7 |

### INT8 量化精度(46 张真实座舱图, FP 与 INT8 均在真 NPU 上对比)

| 模型 | 类别一致率 | 平均 IoU | 平均 \|Δconf\| | 漏检/多检 | 结论 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| chaitanya | 10/10 (100%) | 0.9825 | 0.0277 | 0 / 0 | 优秀, 可直接用 INT8 |
| coco | 67/67 (100%) | 0.99 | 0.0156 | 5 / 6 | 优秀(阈值边缘正常抖动) |
| soham | 108/108 (100%) | 0.9464 | 0.0853 | 20 / 2 | SafeDriving 置信度整体下移; 涉警类别(Drowsy 0.5→0.38)均低于浏览器 0.6 门槛, 判定行为不变 |

报告: `docs/rknn/chaitanya_int8_vs_fp.md`、`docs/rknn/soham_int8_vs_fp.md`、`docs/rknn/coco_int8_vs_fp.md`。

**部署建议**: 全 INT8(43.8ms / 22.8 FPS)为首选; 若对 soham 状态显示的平滑度有
顾虑, 用混合 `int8,fp,int8`(68.1ms / 14.7 FPS)也完全达标。切换只需改
`config/dms.json` 各模型的 `rknn` 路径(`*_fp.rknn` ↔ `*_int8.rknn`)。
比赛默认配置当前保留 FP。

## 四、原生 DMS 服务(已实现, 2026-09-10)

`rk3588_dms/service/dms_service.py` —— 摄像头独占采集 + MJPEG 推流 +
三模型 INT8 三核并行推理 + WebSocket 检测推送 + 板端报警音频, 单进程:

```bash
# 板上启动(真摄像头; taskset 绑大核 4-7: 墙钟 ~60ms→~48ms 且方差收窄)
taskset -c 4-7 .venv/bin/python rk3588_dms/service/dms_service.py
# 无摄像头调试(静态图当帧源)
.venv/bin/python rk3588_dms/service/dms_service.py --test-image testdata/dms/test_0001.jpg
```

| 端点 | 说明 |
| ---- | ---- |
| `GET /video.mjpg` | MJPEG 流(multipart/x-mixed-replace), 浏览器显示用 |
| `WS /ws/dms/native` | 推送 `{type:"detections", models:{chaitanya/soham/coco:[{originalClass,key,confidence,box(归一化)}]}, frame, inference_ms}`(约 12fps, 与浏览器 parseDetections 输出同构) |
| `POST /alert {key}` | 板端 ffplay 播放报警 mp3(音量 100, 每 key 冷却 4.5s 与浏览器一致) |
| `GET /health` | 摄像头/推理/客户端状态 |

### 浏览器接入(页面与样式零改动)

`driver-inference.js` 增加 `?infer=native` 模式(缺省仍为纯浏览器推理, 随时回退):

- **视频**: MJPEG 帧 → 隐藏 canvas `captureStream()` → 原有 `#cameraVideo`——
  显示/object-fit 映射/MediaPipe VIDEO 模式/关键区域裁切全部原样工作;
- **推理**: YOLO 检测来自 WS 缓存(坐标按当前帧尺寸反归一化), 阈值过滤/
  统一 key/phone 归属/NMS/drowsy 抑制/手动控制/报警时序/模型选择器全沿用
  浏览器原有逻辑; MediaPipe 人脸通道照旧在浏览器;
- **音频**: 报警触发时 POST key 给服务, 板端 ffplay 音量 100 播放,
  触发节奏(优先级/冷却)沿用浏览器逻辑;
- 一致性证据: test_0001 帧 soham SafeDriving 板端 0.8481 vs 浏览器 FP ONNX 0.848。

```text
打开: http://<板子IP>:8000/index.html?infer=native#/live-detection
回退: 去掉 ?infer=native 即恢复纯浏览器推理(什么都不用改)
```

⚠️ 注意: native 模式请用 **http** 打开页面(https 页面会拦截 http 的 MJPEG/WS,
浏览器会给出明确提示); 服务地址可用 `?nativeBase=http://IP:8600` 覆盖。
已知差异: 浏览器端"手机 ROI ±22° 复检"在 native 模式不可用(依赖浏览器端
COCO 会话), 手机检测由板端全帧 COCO 承担, 极小目标召回略低于纯浏览器模式。

## 五、集成边界(明确不做/后续)

- 状态融合/报警时序留在浏览器(阈值是比赛调优结果, 不迁移不改动);
- MediaPipe FaceLandmarker、6DRepNet 不迁移(浏览器/后端保留, 见 docs/rknn/ 分析);
- 后续可选: MPP/RGA 零拷贝采集替换 OpenCV、service 常驻 systemd、HTTPS 反代。

## 六、常用命令速查

```bash
python tools/check_rknn_env.py                          # 自动识别 PC/设备
python tools/model_inspect.py <model.onnx>              # 模型结构检查
python rk3588_dms/tools/convert_chaitanya.py            # FP 转换
python rk3588_dms/test_image.py --image x.jpg           # 单图(板上)
python rk3588_dms/tools/benchmark_rknn.py               # 基准
python rk3588_dms/tools/benchmark_parallel.py           # 三模型三核并行基准
.venv/bin/python rk3588_dms/service/dms_service.py      # 原生 DMS 服务(板上)
python rk3588_dms/test_camera.py --camera /dev/video0   # 摄像头
python rk3588_dms/tools/compare_outputs.py              # 一致性对比
python -m unittest discover -s rk3588_dms/tests -v      # 单元测试(纯 numpy/cv2)
```

## 七、日志

- `logs/rknn/` —— 转换日志、benchmark JSON、对比明细
- `logs/dms/` —— 预留给原生 DMS 服务运行日志(模型加载/初始化耗时/摄像头事件/
  推理异常/每 5 秒统计, 不逐帧刷屏)
