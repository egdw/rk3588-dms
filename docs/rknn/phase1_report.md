# 第一阶段报告（RK3588 RKNN 迁移 · chaitanya_best）

- 日期：2026-09-10（第三版：**真机 STEP 5/6 已通过**，见第 5/6/8 节实测数据）
- 开发/真机环境：Windows 开发机（代码）+ **dcztl（192.168.10.182，RK3588 A588 定制板，
  kernel 5.10.198，RKNPU 驱动 v0.9.8，librknnrt 2.3.2）——转换与真机推理同一台**
- 结论状态：**第一阶段停止线已达成**——chaitanya_best FP RKNN 在 RK3588 NPU 上
  单图正确识别（bbox 与浏览器 ONNX 像素级一致），基准 41.7ms 平均 / 24 FPS

## 0. 环境事实

1. 模型资产已恢复：按用户确认的来源 `https://github.com/AlbatrossC/Driver-Monitoring-System`
   （main 分支）下载 4 个文件到项目约定路径，字节大小与仓库树完全一致：
   - `Driver-Monitoring-System/public/static/models/chaitanya_best.onnx`（12,241,395 B）
   - `Driver-Monitoring-System/public/static/models/soham_best.onnx`（10,570,055 B）
   - `Driver-Monitoring-System/models/chaitanya/best.pt`（6,226,730 B）
   - `Driver-Monitoring-System/models/soham/best.pt`（5,450,067 B，与 fly.md 历史记录一致）
2. `yolov8n_coco.onnx` 确认不在上游仓库 → 已由官方 `yolov8n.pt`（ultralytics assets
   v8.3.0，6,549,796 B）标准导出补齐（640×640，opset 12，`[1,84,8400]`，12.26 MB）。
3. 本机非 RK3588；rknn-toolkit2 不支持 Python 3.13，故**转换与 NPU 推理待目标硬件**。
   PC 导出环境装在项目内 venv `.venv-rknn`（Python 3.13.9：torch 2.14 CPU +
   ultralytics 8.4.144 + onnx 1.22 + onnxslim + onnxscript + onnxruntime），
   未动系统 Python，删除该目录即完全回退。

## 1. 模型原始结构（文件级实测，`tools/model_inspect.py`）

| | chaitanya_best | soham_best |
| ---- | ---- | ---- |
| 架构 | YOLOv8n（C2f，base yolov8n.pt） | **YOLO11n**（C3k2，yolo11n.yaml） |
| 导出 | Ultralytics 8.3.251 / torch 2.8.0，opset 11，batch 1，imgsz 640，nms=False，dynamic=False | 同左 |
| 输入 | `images` 1×3×640×640 float32 固定 | 同左 |
| 输出 | `output0` **[1, 9, 8400]**（4+5） | `output0` **[1, 12, 8400]**（4+8） |
| 类别 | Cigarette/Drinking/Eating/Phone/Seatbelt | Distracted/Drinking/Drowsy/Eating/PhoneUse/SafeDriving/Seatbelt/Smoking |
| 图内 NMS | 无 | 无 |
| DFL | 在图内（Softmax×1/Reshape×5） | 在图内（Softmax×2/Reshape×8，另 MatMul×2） |
| 参数量 | 3,006,623 | 2,583,712 |

类别表与 `driver-inference.js:126-127` 逐字一致；输出布局与 `parseDetections`
解析约定（attr-major、无 objectness、坐标归一化判定）完全吻合。

## 2. 是否直接使用原 ONNX？

**否——按用户指示，RKNN 转换改用 best.pt 重导出。** 原 ONNX 仅作浏览器 fallback
与对比参考端（`compare_outputs.py` 参考端固定为浏览器标准单输出模型）。

## 3. 是否重新从 PT 导出？——是（Rockchip 推荐方式，已完成）

`rk3588_dms/tools/export_onnx_from_pt.py --style rknn`：monkey-patch Detect 头，
输出 3 个分支 `[1, 64+nc, 80/40/20]` 原始 logits（图内无 DFL softmax、无类别
Sigmoid、无 NMS），与 Rockchip rknn_model_zoo 的 YOLOv8/YOLO11 官方示例格式一致。
legacy TorchScript 导出器（`dynamo=False`，尊重 opset=12、权重内嵌单文件）。

| 产物 `rk3588_dms/models/onnx/rknn_source/` | 输出 | 大小 | opset |
| ---- | ---- | ---- | ---- |
| `chaitanya_best_rknn.onnx` | [1,69,80,80]/[1,69,40,40]/[1,69,20,20] | 11.52 MB | 12 |
| `soham_best_rknn.onnx` | [1,72,…]×3 | 9.93 MB | 12 |
| `yolov8n_coco_rknn.onnx` | [1,144,…]×3 | 12.07 MB | 12 |

（soham 图内剩余 Softmax×1 为 YOLO11 PSA 注意力模块，非 DFL，RKNN 支持。）
config `conversion.*.onnx_candidates` 已调整为 **rknn_source 优先**，浏览器 ONNX 仅回退。

## 4. RKNN build 是否成功？

**成功**（2026-09-10，dcztl/RK3588，rknn-toolkit2 2.3.2，Python 3.12 venv）：
`chaitanya_best_fp.rknn` 7.0MB（FP，权重 FP16），转换 + 模拟器验证（Δconf=0.0001）
一条命令完成：`python rk3588_dms/tools/convert_chaitanya.py --validate-image face-test.png`。

过程中排掉的三个坑（已全部修复并推送）：
1. Python 3.12 venv 不预装 setuptools → `pkg_resources` 缺失（toolkit2 硬依赖，
   需 `pip install "setuptools<81"`）；
2. toolkit2 2.x 无 `deinit()`（是 `release()`）——finally 里的 AttributeError 曾把
   成功转换变成报错；
3. toolkit-lite2 2.3+ 推理要求显式 4 维输入 (1,H,W,3)，且 `load_rknn` 的模型
   不能进模拟器（模拟器验证必须挂在 build 后同一对象上）。

## 5. RK3588 是否成功加载 / 6. NPU 是否正常推理？

**均已成功**（dcztl 真机实测，`test_image.py --mode device`）：

```text
runtime      : RK3588 NPU (rknn-toolkit-lite2)
device check : machine=aarch64 Linux model=ztl, A588
npu driver   : RKNPU driver v0.9.8（sudo dmesg/debugfs 确认）
model init   : 306.5 ms    preprocess: 8.1 ms
inference    : 46.8 ms     postprocess: 1.1 ms
```

NPU 检出与浏览器 ONNX 对照（低阈值 0.05，face-test.png）：

| 来源 | 类别 | 置信度 | bbox |
| ---- | ---- | ---- | ---- |
| 浏览器 ONNX（onnxruntime） | Cigarette | 0.1153 | [329, 294, 486, 351] |
| RKNN 模拟器（toolkit2） | Cigarette | 0.1152 | [329, 294, 486, 351] |
| **RKNN 真机 NPU（lite2）** | **Cigarette** | **0.1190** | **[329, 294, 486, 351]** |

**bbox 像素级一致**；置信度 Δ=0.0037（NPU FP16 权重精度，远小于 0.02 容差）。
生产阈值 0.25 下该图正确地 0 检出，三方行为一致。

## 7. 原模型 vs 重导出结果差异（FP32，PC 实测 2026-09-09）

用 `compare_outputs.py --candidate-onnx`（rknn_source ONNX 走**与 RKNN 完全相同的
letterbox + 三分支解码路径**）对比浏览器标准 ONNX（同预处理、`face-test.png`）：

| 模型 | 检出配对 | Δconf | IoU |
| ---- | ---- | ---- | ---- |
| chaitanya | Cigarette 0.1153 vs 0.1153（阈值 0.10/0.04 下均一致） | **0.0000** | **1.0000** |
| soham | 4/4 配对（SafeDriving 0.6659 等） | **0.0000** | **1.0000** |
| coco | 2/2 配对 | **0.0000** | **1.0000** |

**结论：重导出链路（未来 RKNN 采用的模型图 + 解码器）与浏览器链路 FP32 数值等价。**
后续转 RKNN 后测得的任何偏差将只来自 RKNN 编译/量化本身——干净的对照基线已建立。
`docs/rknn/{chaitanya,soham,coco}_validation.md` 已由工具生成（当前为 1 张健全性
样本；正式 ≥20 张现场测试集对比仍待补，要求见 `testdata/dms/README.md`）。

## 8. 单次推理耗时？（真机实测 2026-09-10）

`tools/benchmark_rknn.py --mode device --warmup 20 --runs 200`（NPU_CORE_AUTO，
静态 640×640 随机输入）：

```text
Average inference : 41.662 ms
min / max         : 39.038 / 57.002 ms
P50 / P95         : 41.203 / 46.481 ms
Theoretical FPS   : 24.0
```

远超比赛需求（建议 DMS AI 10~15 FPS），还有三核分配与 INT8 的余量。
原始数据：dcztl `logs/rknn/benchmark_chaitanya_20260910-183205.json`。

## 9. 当前发现的问题

1. rknn-toolkit2 无法装在 Python 3.13 → 转换需另配 Python 3.8~3.12 环境
   （README 已注明版本匹配原则），或直接在现场 Linux 机执行；
2. 浏览器两个原 ONNX 是 opset 11（若将来有人直接拿去转换，工具链兼容但非最优；
   rknn_source 已是 opset 12，不受影响）;
3. 正式测试集（≥20 张现场图片）仍缺——唯一可用图 face-test.png 已用于健全性验证，
   真实驾驶场景样本需现场补齐；
4. 其余同第一版（摄像头独占为集成阶段问题、非 Git 仓库无法建分支）。

## 10. 已验证清单（本轮新增）

- 4+1 个模型文件下载校验（字节数/zip 结构/pickle 架构关键词）；
- `inspect_model.py` 零依赖 protobuf 解析器：对 5 个 ONNX 输出与已知事实一致
  （含手写 varint/length-delimited 解析、initializer 参数量统计）；
- 3 个 Rockchip 风格导出 + 1 个标准导出成功，产物 opset/通道数/内嵌权重全部验证；
- FP32 数值交叉验证：三个模型 Δconf=0.0000 / IoU=1.0000；
- 单元测试 20/20（含 sigmoid 数值稳定化后的回归）；
- `py_compile` 全部脚本、`node --check` 现有 JS 未破坏、全部 CLI 冒烟通过。

## 11. RK3588 现场执行顺序（更新版）

```bash
# 0. 环境自检(板上)
python tools/check_rknn_env.py --role device

# 1. 转换(任一具备 rknn-toolkit2 的 Python<=3.12 环境; 源自动选中 rknn_source)
python rk3588_dms/tools/convert_chaitanya.py

# 2. 转换后 PC 预检(可选): 模拟器跑 FP RKNN vs 浏览器 ONNX
python rk3588_dms/tools/compare_outputs.py --model-name chaitanya

# 3. 板上单图(停止线: bbox/class/confidence/latency 正确输出)
python rk3588_dms/test_image.py --model rk3588_dms/models/rknn/chaitanya_best_fp.rknn --image testdata/dms/phone_01.jpg

# 4. 基准 → 5. 摄像头 → 6. 板上 JSON 回灌一致性(见 README 二节)
```

## 12. 下一阶段建议

1. 现场执行 STEP 5-7（chaitanya 闭环）→ soham（rknn_source 已就绪，YOLO11 输出
   格式已实证与 v8 同构）→ coco；
2. 补齐 ≥20 张现场测试集后重跑 compare_outputs 出正式验证报告；
3. INT8 延后（dataset.txt + `--int8`，比赛版本默认 FP）。
