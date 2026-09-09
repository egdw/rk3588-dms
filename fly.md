# fly.md - 状态检测项目长期记忆

> **强制约定：后续 Codex 在处理本项目任何任务前，必须先阅读本文件。**
>
> 本文件是 `/Users/skyblue/Desktop/mainnn/状态检测` 项目根目录下的权威长期记忆。完成重要代码、配置、数据库、测试、启动流程或文档变更后，必须同步更新本文末尾的“最近修改”。

## 项目概览

- 项目路径：`/Users/skyblue/Desktop/mainnn/状态检测`（Windows 开发副本：`D:\project\dms`，Git 仓库名 `rk3588-dms`）
- 项目性质：本地全栈司机状态检测开发平台。前端保持无构建、原生 HTML/CSS/JavaScript；Python 后端负责持久化、数据加工和 ML 任务编排；浏览器端可直接运行现成 ONNX 模型完成图片与摄像头推理。
- 当前主应用：`智证先锋 | 司机状态检测模型训练与评估平台`，内部服务名仍沿用 `Vision Sentinel backend`；前端必须完全对齐 `stitch_/` 中 Vision Sentinel 四页 UI：控制台、数据集管理、实时检测演示、系统设置；页面内容承载数据集、标注、加工、训练、分析、导出部署完整流程，并接入 Python 标准库后端 API。
- 附带原型：`stitch_/` 下保存 Vision Sentinel 相关静态 HTML 原型和设计文档，是当前 UI 对齐依据。
- Git 状态：2026-09-09 起为 Git 仓库（用户明确要求创建，仓库名 `rk3588-dms`，本地目录 `D:\project\dms` 未改名）。默认分支 `main`；远程 `origin = https://github.com/egdw/rk3588-dms.git`（**私有仓库**，2026-09-09 经用户确认由 gh CLI 创建并推送）。gh CLI 位于 `%LOCALAPPDATA%\Programs\gh-cli\bin\gh.exe`（用户级免安装解压版，非系统安装）。规范见下文“分支与提交规则”。
- 本文件状态：本目录 `fly.md` 为当前项目权威记忆。此前文件中提到迁移到上级目录的说明已作废。

## 项目结构

```text
.
├── app.js
├── .gitignore
├── backend/
│   ├── README.md
│   └── server.py
├── Driver-Monitoring-System/
│   ├── public/static/models/*.onnx
│   ├── models/*
│   └── public/demo-images/*
├── driver-inference.js
├── fly.md
├── index.html
├── package.json
├── start-backend.sh
├── storage/
│   ├── annotations/
│   ├── datasets/
│   ├── exports/
│   ├── jobs/
│   ├── models/
│   └── processed/
├── styles.css
└── stitch_/
    ├── vision_sentinel/DESIGN.md
    ├── vision_sentinel_1/code.html
    ├── vision_sentinel_2/code.html
    ├── vision_sentinel_3/code.html
    └── vision_sentinel_4/code.html
```

关键文件说明：

- `index.html`：主入口页面，采用 `stitch_` Vision Sentinel 四页式结构：控制台、数据集管理、实时检测演示、系统设置；保留后端联动所需控件 ID。
- `styles.css`：主页面样式，必须遵守 `stitch_/vision_sentinel/DESIGN.md`：72px 玻璃顶栏、Geist 字体、`#f8fafc` 背景、白色 32px Bento 卡片、蓝色渐变按钮、柔和蓝色阴影和状态色浅底标签。
- `app.js`：主页面交互逻辑，包含四页 hash 切换、后端健康检查、数据集创建/选择、样本上传、图片/视频预览、检测框绘制与回显、数据加工配置、训练/评估/导出任务、任务轮询以及真实训练指标和产物渲染。
- `driver-inference.js`：实时检测页的浏览器端真实推理模块；直接读取下载项目内两个 ONNX 权重，支持图片和摄像头、双模型融合/单模型切换、640×640 预处理、置信度过滤、NMS、中文检测框、风险指标和事件时间线。摄像头采用原生视频连续播放与透明检测框异步覆盖，默认使用状态检测单模型流畅模式。
- `Driver-Monitoring-System/`：用户下载的 AlbatrossC Driver Monitoring System 完整项目，本项目只读复用其中 `public/static/models/soham_best.onnx` 与 `chaitanya_best.onnx` 以及演示图片；不要覆盖、移动或删除该目录中的原始模型资源。
- `.gitignore`：忽略 Python 缓存和 `storage/` 下的运行时数据库、上传数据、模型、导出文件和任务日志。
- `backend/server.py`：Python 标准库 HTTP 后端，提供静态文件服务和 `/api/...` 接口；实现数据集、样本上传/读取、标注、FFmpeg 清洗与统一尺寸、训练/验证/测试集划分、数据增强、YOLO 数据集生成、Ultralytics 训练/评估/导出、模型注册和任务日志。
- `backend/README.md`：后端启动、API 调用和真实训练接入说明。
- `package.json`：npm 脚本包装入口，无第三方依赖；用于 `npm start`、`npm run check` 等常用命令。
- `start-backend.sh`：后端启动脚本，内置项目绝对路径，默认监听 `0.0.0.0:8000`，同时提示本机和局域网访问地址。
- `start-backend.sh` 启动前会检查 Python、项目目录、后端入口和端口占用；若默认端口已有 Vision Sentinel 后端在运行则提示本机与局域网访问地址并退出，若被其他程序占用则默认自动切换到后续可用端口。
- `storage/`：后端运行时数据目录，保存 SQLite、上传样本、加工输出、模型、导出文件和任务日志。
- `stitch_/vision_sentinel/DESIGN.md`：Vision Sentinel 原型设计系统说明。
- `stitch_/vision_sentinel_*/code.html`：Vision Sentinel 静态页面原型，包含控制台、数据集管理、实时检测演示和系统设置页面。

## 技术栈

主应用：

- HTML
- CSS
- 原生 JavaScript
- npm scripts：只作为启动/检查命令包装层，不引入前端框架、构建工具或第三方包。
- ONNX Runtime Web 1.17.0：实时检测页由 jsDelivr 在浏览器运行时加载；模型权重在本地 `Driver-Monitoring-System/public/static/models/`，首次推理需要加载约 22 MB 权重，不需要 Python ML 依赖。
- Python 标准库后端：`http.server`、`sqlite3`、`threading`、`subprocess`
- FFmpeg/ffprobe：真实图片质量检测、图片缩放填充、视频抽帧和训练集增强。
- SQLite 元数据存储：`storage/vision_sentinel.sqlite3`
- 无前端框架、无构建工具；`package.json` 只包装启动和静态检查命令。
- 后端不自动安装 ML 依赖；真实 YOLO 训练、评估和非 PT 格式导出会调用项目环境已有的 `yolo` 或 `python -m ultralytics`，缺失时任务必须明确失败，禁止生成伪结果或伪模型。
- 当前系统解释器为 `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`（Python 3.11.9），项目尚无 `.venv`，当前未安装 Ultralytics/PyTorch；因此 FFmpeg 加工、浏览器 ONNX 推理和 PT 原样导出可用，训练、评估以及 ONNX/TensorRT/TFLite 转换暂不可执行。

`stitch_/` 原型：

- 单文件 HTML 原型
- Tailwind CDN
- Google Fonts
- Material Symbols
- 原生内联 JavaScript

新增技术栈时必须同步记录：

- 语言和运行时版本
- 包管理器和锁文件
- 框架、主要库与原因
- 构建/预览/部署工具
- 默认端口和环境变量

## 启动方式

前端无需安装依赖、无需构建。

仅预览静态前端：

```sh
open /Users/skyblue/Desktop/mainnn/状态检测/index.html
```

启动后端与静态文件服务：

```sh
/Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
```

也可以用 npm 启动：

```sh
cd /Users/skyblue/Desktop/mainnn/状态检测 && npm start
```

启动脚本支持的常用环境变量：

```sh
VISION_SENTINEL_PORT=8010 /Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
VISION_SENTINEL_HOST=127.0.0.1 /Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
VISION_SENTINEL_AUTO_PORT=0 /Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
VISION_SENTINEL_OPEN_BROWSER=1 /Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
PYTHON_BIN=/path/to/python3 /Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
```

完整绝对路径命令：

```sh
cd /Users/skyblue/Desktop/mainnn/状态检测 && python3 /Users/skyblue/Desktop/mainnn/状态检测/backend/server.py
```

默认本机地址：

```text
http://127.0.0.1:8000/index.html
```

默认局域网地址由启动脚本自动输出，通常类似：

```text
http://192.168.x.x:8000/index.html
```

健康检查：

```sh
curl http://127.0.0.1:8000/api/health
```

注意：

- 运行 `open` 这类 GUI 命令前，如当前工具环境要求提权，需要先按 Codex 工具规则申请确认。
- 默认监听 `0.0.0.0` 便于同一 Wi-Fi/局域网访问；如只希望本机访问，使用 `VISION_SENTINEL_HOST=127.0.0.1` 启动。
- macOS 防火墙可能拦截局域网入站连接；如弹出 Python 网络访问确认，需要允许，或在系统设置中允许对应 Python 接收局域网连接。
- 不要为了预览而安装依赖。
- 如需安装 Ultralytics/PyTorch 等真实训练依赖，必须先说明解释器、虚拟环境、安装目录、缓存/工具目录，并等待用户明确确认。

## 数据库与数据

- 当前使用 SQLite：`storage/vision_sentinel.sqlite3`，由 `backend/server.py` 首次启动时自动创建。
- 当前无 ORM、无迁移框架。
- 后端 API 见 `backend/README.md`，主要包括 `/api/datasets`、`/api/datasets/{id}/upload`、`/api/annotations`、`/api/process-jobs`、`/api/train-jobs`、`/api/evaluate-jobs`、`/api/export-jobs`、`/api/jobs/{id}`、`/api/models`。
- 分析图表不再显示演示指标：没有真实训练/评估结果时显示空状态；有结果后读取任务产生的 loss、mAP、Precision、Recall、F1、PR 曲线和混淆矩阵。
- 当前前端“上传图片/视频”会调用后端上传接口并持久化到 `storage/datasets/{dataset_id}/`。
- 后端上传接口会把文件保存到 `storage/datasets/{dataset_id}/` 并登记到 SQLite。
- 标注页通过 `/api/samples/{id}/file` 加载原始样本，通过归一化坐标绘制/保存检测框；同一样本和任务类型再次保存时替换旧标注。
- 数据加工使用不可变任务目录 `storage/processed/{dataset_id}/{job_id}/`，包含真实 YOLO `images/`、`labels/`、`dataset.yaml`、质量检查信息和增强样本；不删除原始上传文件。
- 后端启动时自动注册 `Driver-Monitoring-System/models/soham/best.pt` 与 `models/chaitanya/best.pt` 两个现成权重，供真实评估、PT 导出或后续安装 Ultralytics 后继续训练。
- 实时检测页当前只保留摄像头入口，摄像头帧只在浏览器内推理，不经过数据集上传 API，也不会写入 SQLite；如需保存为训练样本，应使用控制台或数据集管理页的“导入样本”。

如果新增数据库，必须记录：

- 数据库类型和版本
- 连接配置位置
- 本地启动方式
- 迁移、回滚、种子命令
- 数据备份和危险操作注意事项

## 测试与验证

当前没有独立单元测试框架；使用 npm 包装 JavaScript 语法检查和 Python 编译检查，并结合隔离临时数据库的后端管线测试与浏览器人工回归。

统一检查命令：

```sh
cd /Users/skyblue/Desktop/mainnn/状态检测 && npm run check
```

后端静态验证：

```sh
python3 -m py_compile backend/server.py
```

人工验证清单：

- 打开 `index.html` 后页面正常展示。
- 点击“导入样本”可触发文件选择；选择文件后总样本和类别分布变化。
- 未安装 Ultralytics 时，点击“开始训练”应明确显示依赖缺失并将任务标记为失败；安装完成后，训练必须产生真实 `best.pt` 后才可成功。
- 在目标检测模式下拖动鼠标/触控应生成真实检测框；整图分类模式不要求检测框。
- 点击“保存标注”后应将当前样本、类别、任务类型和归一化检测框写入后端，重新打开样本时能够回显。
- 点击“上一张”“下一张”后样本游标随当前数据集样本切换。
- 打开 `#live-detection`，选择 `Driver-Monitoring-System/public/demo-images/1_drowsy_phone.jpg` 后，应出现“使用手机”和“疲劳/瞌睡”中文检测框及置信度；首次双模型加载较慢，缓存后的推理应明显加快。
- 在窄屏宽度下检查布局不重叠、按钮和文字不溢出。
- 启动 `python3 backend/server.py` 后访问 `/api/health` 返回 `{ "ok": true }` 及 `ffmpeg_processing`、`ultralytics_training`、`browser_onnx_inference` 能力状态。
- 创建数据集、上传样本、保存标注、启动加工/训练/评估/导出任务后，可通过 `/api/jobs/{job_id}` 查看状态和日志。

如果新增测试体系，必须记录：

- 单元测试命令
- 集成测试命令
- 端到端测试命令
- lint/格式化/类型检查命令
- 哪类变更必须运行哪些验证

## 分支与提交规则

- 2026-09-09 起当前目录为 Git 仓库（用户明确要求创建，仓库名 `rk3588-dms`，默认分支 `main`）。
- 默认分支：`main`；功能分支命名规范：`codex/<主题>`（如 RKNN 迁移用 `codex/rk3588-rknn-dms`）。
- 提交信息格式：Conventional Commits（`feat:`/`fix:`/`docs:`/`chore:` + 中文简述）。
- 提交前必须检查 `git status`，保护用户已有改动；小步提交，新代码集中，不随意格式化整仓。
- 不改写历史、不强制推送；模型大文件（*.onnx/*.pt）当前随仓库提交，`*.rknn` 与运行时数据（`storage/`、`logs/`、`board_results/`、`.venv-rknn/`）已忽略。
- 远程仓库：`egdw/rk3588-dms`（GitHub 私有）。创建新的远程/公开化现有仓库属于对外发布动作，必须先经用户明确确认。
- PR/Review、合并策略、发布或回滚流程：暂无（单人本地仓库），启用时补充。

## 项目约定

- 每次处理任务前先阅读 `/Users/skyblue/Desktop/mainnn/状态检测/fly.md`。
- 修改前先确认相关文件和现有实现，避免凭空假设。
- 优先保持静态、轻量、无依赖的项目形态；只有用户明确需要或收益明显时才引入工具链。
- 主应用中文界面文案应保持简洁、工作台风格，不写成营销落地页。
- UI 变更需照顾移动端响应式，不让文字、按钮、面板互相重叠。
- 对 `stitch_/` 的改动应视作原型资源改动；主页面 UI 需要按该目录的 Vision Sentinel 设计系统保持一致。
- 完成重要代码、配置、数据库、测试、启动流程或文档变更后，必须更新“最近修改”。

## 禁止事项

遵守用户在 AGENTS.md 中给出的安全规则：

- 禁止批量删除文件或目录。
- 禁止使用递归或强制删除命令，包括 `rm -rf`、`rm -R`。
- 禁止使用 `find ... -delete`。
- 删除任何物理文件前，必须先向用户确认；一次只能删除一个明确路径的具体文件。
- 禁止删除 `.git` 文件夹或项目根目录重要配置文件。
- 禁止未经确认安装、升级或删除 Python 包、浏览器运行时或系统级工具。
- 禁止直接使用系统 Python 环境安装依赖，例如 `pip install`、`python -m pip install`、`python -m playwright install`、`brew install`。
- 如果确实需要安装依赖，必须先说明解释器、虚拟环境、安装目录、缓存/工具目录，并等待用户明确确认。
- 禁止未经说明和确认修改 macOS 系统设置。
- 禁止回滚、覆盖或删除用户已有改动，除非用户明确要求。

## Codex 工作流程

1. 阅读本文件。
2. 扫描与任务相关的项目结构、配置和文档。
3. 修改前简要说明要编辑哪些内容。
4. 使用项目现有风格和最小必要改动完成任务。
5. 运行与变更范围匹配的验证；没有自动化测试时执行人工/静态检查并说明。
6. 重要变更完成后更新本文“最近修改”。
7. 最终回复说明改动内容、验证结果和无法验证的部分。

## 最近修改

- 2026-09-09（第三批）：按用户明确要求初始化 Git 仓库 `rk3588-dms`（此前本目录不是 Git 仓库）。默认分支 `main`，本地配置 `core.quotepath=false`、`core.autocrlf=false`（按原样存储行尾，不做平台转换）。`.gitignore` 补充 `storage/`（后端运行时 SQLite/上传数据/任务日志，与本文“数据库与数据”章节记载的忽略意图对齐）。`fly.md` 更新“项目概览 Git 状态”与“分支与提交规则”（默认分支 main、功能分支 `codex/<主题>`、Conventional Commits 中文简述、不改写历史、推送远程前必须用户确认）。首包含全部项目源码、浏览器模型（onnx/pt）、rknn_source 重导出 ONNX 与文档；忽略项：`*.rknn`、`storage/`、`logs/*`、`board_results/`、`.venv-rknn/`、证书与日志。尚无远程仓库。
- 2026-09-09（第二批）：按用户确认的来源 `github.com/AlbatrossC/Driver-Monitoring-System`（main）补齐 4 个模型文件到约定路径（chaitanya_best.onnx 12,241,395 B、soham_best.onnx 10,570,055 B、models/chaitanya/best.pt 6,226,730 B、models/soham/best.pt 5,450,067 B，字节数与仓库树一致；.pt 实测架构 chaitanya=YOLOv8n/C2f、soham=YOLO11n/C3k2）。`yolov8n_coco.onnx` 确认不在上游仓库，由官方 yolov8n.pt（ultralytics assets v8.3.0）标准导出补齐至浏览器路径（640，opset 12，[1,84,8400]）。新增 `rk3588_dms/tools/export_onnx_from_pt.py`：按 Rockchip rknn_model_zoo 推荐方式从 best.pt 重导出三分支 ONNX（[1,64+nc,80/40/20]，图内无 DFL softmax/类别 Sigmoid/NMS，opset 12，legacy 导出器保证权重内嵌），已生成 chaitanya_best_rknn/soham_best_rknn/yolov8n_coco_rknn 三个产物到 `rk3588_dms/models/onnx/rknn_source/`；`config/dms.json` 转换候选顺序改为 rknn_source 优先、浏览器 ONNX 仅回退。`tools/inspect_model.py` 增加零依赖 protobuf 解析回退（无 onnx 包可读 opset/算子/shape/参数量），实测两浏览器 ONNX：opset 11、固定 640、无 NMS、输出 [1,9,8400]/[1,12,8400]，类别名与 driver-inference.js 完全一致；`runtime/postprocess.py` Sigmoid 改数值稳定实现。`compare_outputs.py` 新增 --candidate-onnx 模式（rknn_source ONNX 走与 RKNN 相同的解码路径做转换前预检），参考端固定为浏览器标准 ONNX。FP32 数值交叉验证（face-test.png）：三个模型 Δconf=0.0000、IoU=1.0000。PC 导出环境装于项目内 venv `.venv-rknn`（torch 2.14 CPU + ultralytics 8.4.144 + onnx/onnxslim/onnxscript/onnxruntime），未动系统 Python，已加入 .gitignore。rknn-toolkit2 转换与真机 NPU 推理仍待目标硬件（本机 Windows/Python 3.13 无 rknn-toolkit2 发行版）；单元测试 20/20、py_compile、node --check 全部复验通过。浏览器推理保持为 fallback，未改动现有业务代码。
- 2026-09-09：新增 RK3588 原生 NPU 推理迁移第一阶段交付（全部为新增文件，未改动任何现有业务代码；仅向 `.gitignore` 追加 rknn 产物忽略规则并更新本记录）。新增 `rk3588_dms/`（README 部署说明、`config/dms.json`、`runtime/` 预处理-后处理-模型封装-ChaitanyaDetector、`tools/` 转换/基准/对比脚本、`test_image.py`/`test_camera.py`、`tests/` 17 个单元测试）、根目录 `tools/model_inspect.py`（ONNX 结构检查）与 `tools/check_rknn_env.py`（PC/设备环境自检）、`docs/rknn/`（current_dataflow 数据流基线、model_inventory 模型盘点、6drepnet/mediapipe 迁移分析、chaitanya_validation 待运行占位、phase1_report 阶段报告）、`testdata/dms/README.md`、`logs/rknn`/`logs/dms` 日志目录。后处理语义与浏览器 `driver-inference.js` 逐位对齐（letterbox 灰底 128、`[1,4+nc,8400]` 双主序解析、置信度 0.25/0.20、NMS IoU 0.45），并支持 Rockchip Model Zoo 三分支解码。已验证：全部新增 .py 通过 py_compile；`python -m unittest discover -s rk3588_dms/tests -v` 17/17 通过；`tools/check_rknn_env.py --role pc` 实跑正常；`node --check app.js driver-inference.js` 与 `py_compile backend/server.py` 复验通过。当前工作副本中 `Driver-Monitoring-System/` 为空、三个 ONNX 与 best.pt 不在场，本机也非 RK3588 且未安装 rknn-toolkit2（遵守本文件禁令），因此模型转换与真机推理未执行、未伪造结果；真机执行顺序见 `docs/rknn/phase1_report.md` 第 11 节。浏览器端 ONNX 推理保留为 fallback，未删除任何现有推理代码。
- 2026-09-02：按用户反馈修正眼部射线不会随眼神方向和位置变化的问题。`driver-inference.js` 将眼部视线可视化改为基于每只眼的局部坐标系实时计算：眼角连线作为水平轴、上下眼睑连线作为垂直轴，虹膜中心相对眼眶中心的偏移决定主画面蓝色方向线的起点、方向和长度；中心注视时显示短圆点提示，偏移时绘制短蓝色动态方向线。同时将侧栏状态文案改为“动态虹膜方向线”，明确它是 2D 方向估计而非专业三维眼动追踪。`backend/README.md` 同步更新输出与边界说明。未安装依赖，未修改 ONNX 权重、摄像头权限、后端 API、SQLite 数据或训练流程。
- 2026-09-02：根据用户最新截图再次修正实时检测主画面人脸叠加。`driver-inference.js` 撤销此前对摄像头叠加层加入的水平镜像补偿，因当前 `#cameraVideo` 没有 CSS 镜像，关键点、眼部轮廓、头姿坐标轴和危险行为框现在均按视频原始坐标直接绘制，避免点位被翻到人脸反方向；同时移除主画面眼部“视线射线”绘制，因为现有实现只是基于虹膜偏移的状态估计，不具备真实三维视线射线含义。保留蓝色五官关键点、蓝色眼部轮廓、蓝色头姿辅助线，以及内部视线偏移状态判断。`backend/README.md` 同步更新说明。`npm run check` 通过。未安装依赖，未修改 ONNX 权重、摄像头权限、后端 API、SQLite 数据或训练流程。
- 2026-09-02：修复用户截图反馈的五官关键点没有贴合人脸问题。`driver-inference.js` 将摄像头模式的主画面叠加坐标从 `cover` 缩放改为与 `#cameraVideo` 一致的 `object-fit: contain` 映射，并新增统一的源坐标到画布坐标转换函数，对摄像头画面执行水平镜像补偿；五官关键点、眼部轮廓、视线射线、头姿坐标轴和危险行为框现在共用同一套坐标映射。`backend/README.md` 同步补充叠加层 contain 映射与前置摄像头镜像补偿说明。`npm run check` 通过。未安装依赖，未修改 ONNX 权重、摄像头权限、后端 API、SQLite 数据或训练流程。
- 2026-09-02：按用户进一步要求将蓝色五官关键点、眼部轮廓、视线线和头姿辅助线固定为“直接贴在主摄像头画面人脸上”的呈现方式。`driver-inference.js` 停止向侧栏人脸裁切、左右眼裁切和眼部波形小画布绘制，所有面部可视化只通过主 `detectionCanvas` 叠加在原始视频人脸位置；同时将视线射线旧黄色配色统一改为蓝色系。`styles.css` 隐藏侧栏单独的人脸/眼睛预览画布并调整驾驶员状态网格为单列。`backend/README.md` 同步说明面部状态不再通过侧栏单独预览展示。`npm run check` 通过。未安装依赖，未修改 ONNX 权重、摄像头权限、后端 API、SQLite 数据或训练流程。
- 2026-09-02：按用户参考图调整实时检测主画面叠加效果。`driver-inference.js` 移除未使用的人脸绿色角标框函数，安全类结果（正常驾驶、安全带等）不再在主画面绘制绿色/青绿色识别框；保留危险行为红色报警框，并将 MediaPipe Face Landmarker 的面部状态叠加改为蓝色五官关键点、蓝色眼部轮廓、蓝色视线射线和蓝色系头姿坐标辅助线。`backend/README.md` 同步说明当前输出为危险行为框 + 五官蓝色关键点。未安装依赖，未修改 ONNX 权重、摄像头权限、后端 API、SQLite 数据或训练流程。
- 2026-09-01：补齐实时检测页的独立低头检测。`driver-inference.js` 新增浏览器端 MediaPipe Face Landmarker 头姿模块，基于人脸姿态矩阵和关键点计算 `head_down` 置信度，并要求连续超过阈值后才触发报警；低头检测结果会参与检测框、关键区域、顶部状态条、驾驶专注度、报警保留、事件时间线和标注类别联动，同时保留原 YOLO `Distracted` 为“分心驾驶”而不再直接冒充低头。`index.html` 将顶部“低头分心”状态项改为监听 `head_down`。`backend/README.md` 同步说明低头检测依赖浏览器端人脸关键点/姿态模型，首次加载需要网络。未安装 Python 或 npm 依赖，未修改 ONNX 权重、摄像头权限、后端 API、SQLite 数据或训练流程。
- 2026-08-31：按用户截图调整实时检测页左侧监测栏高度。`styles.css` 新增最终侧栏尺寸覆盖，将“驾驶员状态”卡从内容少却被拉高的布局收紧为 190px，并同步将“疲劳监测”“驾驶专注度”分别调整为 156px 和 178px，实时识别结果承接剩余空间；同时压实驾驶员检测对象模块和两张监测卡内部间距，减少左侧大面积空白。`npm run check` 通过。未修改 HTML 结构、后端 API、SQLite 数据、ONNX 推理、模型文件或摄像头权限。
- 2026-08-31：修复用户从 `file:///Users/.../index.html` 直接打开页面时出现浏览器原始 `Failed to fetch` 的问题。`app.js` 新增本地文件预览检测：在 `file://` 模式下优先探测 `http://127.0.0.1:8000/api/health`，若后端在线则自动跳转到 `http://127.0.0.1:8000/index.html` 并保留当前 hash；若后端未启动，所有 API 网络异常统一转为中文提示，明确要求运行 `/Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh`。`npm run check` 通过。未修改页面 UI、后端 API、SQLite 数据、ONNX 推理、模型文件或摄像头权限。
- 2026-08-31：按用户要求补齐控制台首页中“所有状态识别和数据标注”的展示。`index.html` 将原先大面积空白的“异常检测趋势”主卡改为“状态识别与数据标注”总览，覆盖正常驾驶、打电话、闭眼/睡眠、低头分心、左顾右盼、抽烟/饮食、未系安全带 7 类状态，并加入整图分类、目标检测框、标注步骤、标注对象和训练指标小图区域；保留 `lossChart` 节点以兼容既有真实训练曲线渲染。`app.js` 新增控制台标注百分比和进度条的真实数据联动。`styles.css` 新增控制台总览专属布局，使状态识别和标注流程填满中间区域、减少空白。`npm run check` 通过。未修改后端 API、SQLite 数据、ONNX 推理、摄像头权限或模型文件。
- 2026-08-31：按用户要求进一步强化实时检测页所有红色报警状态。`driver-inference.js` 将画面中危险类别检测框统一改为红色粗框，安全类仍保留原安全色；`styles.css` 新增最终报警覆盖，顶部异常状态项使用更强红底、红边、左侧红条、“报警”徽标和轻微脉冲，左侧风险读数、驾驶员异常徽标、疲劳/专注报警指标、识别结果报警行和关键区域报警卡均改为更醒目的红色层级与阴影。`npm run check` 通过。未修改报警阈值、ONNX 模型、摄像头权限、后端 API、SQLite 数据或训练流程。
- 2026-08-31：按用户截图要求移除实时检测页左侧“驾驶员状态”卡片底部的“头部姿态风险”和“眼睛疲劳风险”两个小块。`index.html` 删除 `.dms-risk-grid` 可见结构；`driver-inference.js` 对应 `headStateValue`、`eyeStateValue` 本就有空值保护，因此保留推理、顶部状态栏、疲劳监测、专注度、识别结果和关键区域裁切逻辑不变。`npm run check` 通过。未修改 ONNX 模型、摄像头权限、后端 API、SQLite 数据或训练流程。
- 2026-08-31：按用户要求增强实时检测顶部状态栏的红色报警显示。`driver-inference.js` 将顶部状态项也接入已有 3 秒报警保留结果，避免异常状态短暂抖动后立即消失；`styles.css` 在文件尾部新增最终覆盖规则，扩大顶部状态项列宽，允许状态条横向滚动而不是裁切，红色异常项改为浅红底、红色边框、红色文字、阴影和“报警”胶囊标记，同时加宽“停止摄像头”红色按钮。`npm run check` 通过。未修改 ONNX 模型、摄像头权限、后端 API、SQLite 数据或训练流程。
- 2026-08-31：按用户要求调整实时检测页的关键区域与报警展示。`index.html` 将“关键区域”裁切面板从摄像头画面右上角移入左侧“实时识别结果”卡片顶部，使真实裁切画面在结果列表位置展示；`driver-inference.js` 新增 `ALERT_HOLD_MS = 3000`，危险类别报警结果和关键区域裁切在检测短暂消失后继续保留 3 秒，减少识别抖动导致的闪烁；`styles.css` 为结果卡内关键区域增加静态布局和列表高度限制，避免继续使用旧的视频绝对定位。`npm run check` 通过。未修改模型权重、后端 API、SQLite 数据、摄像头采集权限或训练流程。
- 2026-08-31：按用户要求将默认启动改为局域网可访问。`start-backend.sh` 默认 `VISION_SENTINEL_HOST` 从 `127.0.0.1` 改为 `0.0.0.0`，启动时同时输出本机地址 `http://127.0.0.1:{port}/index.html` 和自动探测到的局域网地址；浏览器自动打开和健康检查仍使用本机回环地址，避免直接打开 `0.0.0.0`。`backend/server.py` 直接运行时也默认监听 `0.0.0.0`；`backend/README.md` 和本文同步补充局域网访问、只允许本机访问的 `VISION_SENTINEL_HOST=127.0.0.1` 用法，以及 macOS 防火墙可能需要允许 Python 入站连接的提示。已验证 `bash -n start-backend.sh`、`npm run check`，并用临时端口 `8128` 非沙箱启动 `0.0.0.0` 后访问 `/api/health` 正常返回；临时服务已停止，未安装依赖、未修改系统设置。
- 2026-08-31：按用户要求将主应用可见品牌名从“视觉哨兵”改为“智证先锋”。`index.html` 同步更新浏览器标题、顶部品牌 `aria-label`、导航左侧品牌文字和页脚版权文字；仅调整展示文案，保留原图标、布局、导航、ONNX 推理、后端 API、SQLite 数据和模型文件不变。
- 2026-08-31：按用户要求分别增加实时检测侧栏前三张卡的高度。`styles.css` 为驾驶员状态、疲劳监测、驾驶专注度分别设置 `246px / 184px / 202px` 高度下限，并按各自内容独立调整对象预览、风险项、统计框、进度条和底部内边距；内容较多时允许自然增高，监测卡仍保持 `12px / 14px` 连续行距，不引入弹性空白行。侧栏前三行改为内容高度、结果卡承接剩余空间；较矮桌面允许侧栏滚动，高而窄的桌面视口利用原来闲置的纵向空间，结果空状态同步适应剩余高度。`npm run check` 通过；浏览器验证 `1160×1180`、`1920×1080`、`1440×900` 下前三卡完整显示，`1440×900` 下整页和侧栏均无溢出；`1366×768` 下可滚动访问完整侧栏，`390×844` 下保留原有移动布局且卡片无内部裁切。使用已有启动脚本恢复本机 `8000` 端口预览，未安装依赖、未启动摄像头、未修改模型推理、后端或数据库逻辑。
- 2026-08-30：消除实时检测侧栏“疲劳监测”和“驾驶专注度”卡片中部的大段纵向空白。`styles.css` 将两张卡片的弹性第二行改为内容高度，取消明细区底部对齐，使标题、右上主读数与下方明细保持连续 `10px` 间距；宽屏高视口下将两张监测卡改为按内容收紧，并把释放的高度分配给“实时识别结果”卡。`npm run check` 通过；浏览器在 `1920×1080` 与 `1440×900` 下验证两张监测卡无中部空洞、页面无横向或纵向溢出，实时识别结果区域同步增大。未启动摄像头，未修改 ONNX 推理、后端 API、SQLite 数据或模型文件。
- 2026-08-30：按用户“白色背景都不要”的要求，将系统配置页改为透明网格工程台。`styles.css` 仅针对 `settings` 页面移除检测阈值、API 管理、配置菜单、诊断面板以及加工/训练/评估/导出卡片的白色背景与浮层阴影，保留细线框分区；按钮、表头、输入框和内部指标使用浅蓝灰半透明底，聚焦状态继续保留蓝色反馈。浏览器在 `1920×1080` 下确认配置页可见内容中无纯白或白色半透明面板、无横向溢出及卡片裁切，并在 `1440×900` 下复核无横向溢出和内部裁切。未修改 HTML 结构、后端 API、SQLite 数据、训练流程、ONNX 推理、摄像头或模型文件。
- 2026-08-30：按最新三张参考图统一控制台、数据集管理与系统配置页面的桌面布局。`index.html` 移除控制台“完整开发流程”整块；`styles.css` 将三页统一到最大 `1880px` 工作区、相同的 28px 内容起点和 44px 标题基线，控制台收束为统计卡、趋势图与当前数据集三行结构，数据集页重新分配统计卡、数据表单、类别分布与近期样本的高度和列宽，系统配置页改为同宽的紧凑工程配置台并同步压实侧栏、阈值控制台、API 表格及下方加工/训练模块。`npm run check` 通过；浏览器在 `1920×1080` 下验证控制台与数据集均完整铺满首屏、无横向/纵向溢出或卡片内部裁切，系统配置无横向溢出并通过自然纵向滚动展示完整功能；在 `1440×900` 下再次验证三页无横向溢出。未修改后端 API、SQLite 数据、训练流程、ONNX 推理、摄像头或模型文件。
- 2026-08-30：重新平衡实时检测侧栏四块 UI 的高度占比。`styles.css` 将此前“前三块固定高度、识别结果无限承接剩余空间”改为带内容下限的比例网格，驾驶员状态、疲劳监测、驾驶专注度、实时识别结果约占侧栏 `27% / 20% / 22% / 28%`；驾驶员详情在中部稳定居中，疲劳与专注的次级指标贴合卡片底部，识别结果继续保留可滚动的完整结果空间但不再产生巨型空白。`npm run check` 通过；浏览器在 `1920×1080` 和 `2560×1440` 下验证四卡均无内部裁切，整页无横向或纵向溢出。未启动摄像头，未修改推理、后端、数据库或模型逻辑。
- 2026-08-30：根据最新截图再次修正实时检测侧栏的视觉重心。`styles.css` 将驾驶员状态、疲劳监测、驾驶专注度三块信息卡收敛为 `218px / 138px / 154px`，并将“实时识别结果”设置为至少 `240px` 且自动填满侧栏剩余高度；结果标题增至 `20px`，计数徽标、空状态和结果列表同步放大。`npm run check` 已通过；未启动摄像头，未修改推理、后端、数据库或模型逻辑。
- 2026-08-30：继续优化实时检测页在超宽桌面视口下的空间利用与监测信息布局。`styles.css` 为横向桌面模式解除实时页旧的 `1880px` 最大宽度和 `680–740px` 最大高度限制，使工作台从 64px 顶部导航下方开始，横向与纵向完整填满可用视口；内部保留 8px 安全边距，摄像头画面仍按原比例完整呈现。监测信息侧栏改为驾驶员、疲劳、专注三块按内容高度排列，实时识别结果承接剩余高度并允许真实结果列表铺满，消除卡片被等比例拉伸产生的分散空白。顶部驾驶状态栏统一隐藏“本地推理/未检出/危险行为”等辅助小字，并在宽屏下放大到 80px 高度、27px 图标、13px 主文字和 48px 摄像头按钮。`npm run check` 通过；浏览器验证辅助小字全部隐藏、状态栏及四张侧栏卡无内部裁切，当前测试视口无横向或纵向溢出。未启动摄像头，未修改 ONNX 推理、后端 API、SQLite 数据或模型文件。
- 2026-08-30：压缩实时检测页待机状态下的过量空白。`styles.css` 新增最终桌面覆盖，将实时页从旧的 `100vh` 强制拉伸改为内容受控工作台：页面宽度放宽到 `min(1880px, 100vw - 32px)`，宽屏工作区高度限制在 `680–740px`、中等桌面限制在 `640–680px`，主区固定为 64px 状态栏加弹性摄像头画面；左侧驾驶员、疲劳、专注和识别结果四卡改用紧凑比例与更小内边距，消除卡片内部无意义留白。摄像头视频和检测画布继续使用 `object-fit: contain`，不会裁切原始画面、检测框或右侧真实关键区域截取。`npm run check` 通过；浏览器在 `1160×1180` 下实测工作区由约 1034px 降至 680px，四张侧栏卡 `scrollHeight` 均未超过 `clientHeight`，整页无横向溢出。验证未启动摄像头，未读取用户画面；未修改 ONNX 推理、后端 API、SQLite 数据或模型文件。
- 2026-08-30：完善实时检测页的真实状态呈现、异常红色报警和关键区域截取。`driver-inference.js` 沿用既有判定阈值：疲劳指数达到 35、危险类别置信度达到 0.25 或专注评分低于 75 时，将对应主读数、置信度、状态估计和危险识别结果切换为红色报警；无检测输入时保持中性。新增由 ONNX 实际检测框驱动的关键区域裁切，直接从当前摄像头/图片原始帧按检测框坐标裁出最多 3 个不同类别并显示在画面右侧，无真实检测框时整块隐藏；裁切只发生在浏览器端，不上传原始画面。`index.html` 新增关键区域面板，并将静态“在线/已锁定/驾驶员一号”改为随摄像头与检测结果变化的“待机/监测中/异常”和“未检出/目标已检出”，避免暗示未实现的人脸身份识别。`styles.css` 新增浅红报警块、红色数值/进度条和深色关键区域检查器。`npm run check` 通过；89 个 HTML ID 无重复且新增 5 个关键 ID 均存在；浏览器在 `1160×1180` 下验证待机态无红色误报、关键面板默认隐藏、页面无横向或纵向溢出。为保护隐私，验证时未擅自开启用户摄像头。
- 2026-08-30：按用户参考图重排实时检测页“疲劳监测”和“驾驶专注度”两张卡。`styles.css` 将“风险指数/专注评分”主读数从标题下方的整行模块移到卡片右上角，与左侧图标和标题同排；下方继续完整保留疲劳置信度、运行状态、安全状态估计和安全带检测，并增加窄屏宽度收敛规则。未修改读数 ID、动态状态文案或 ONNX 推理更新逻辑。`npm run check` 通过；浏览器在 `1160×1180` 下验证两处主读数均位于右上角、两张卡无内部裁切，整页无横向或纵向溢出。
- 2026-08-30：按用户截图去除实时检测页底部“检测样本标注”和“事件时间线”两块可见区域。`styles.css` 将 `.annotation-panel` 与 `.timeline-card` 从布局中完全隐藏，并把右侧主区由四行改为“驾驶状态栏 + 弹性摄像头监测画面”两行，使监测画面自动填满释放空间；保留原 DOM 节点和 ID 供既有推理/标注脚本兼容，未删除数据处理逻辑。`npm run check` 通过；浏览器在 `1160×1180` 下验证两块区域均不占空间，监测画面与左侧监测栏底部对齐，页面无横向或纵向溢出。未修改摄像头、ONNX 推理、后端 API、SQLite 数据或模型文件。
- 2026-08-30：按用户截图精简实时检测页左侧监测栏。`index.html` 清空头部姿态、眼睛疲劳、疲劳监测和驾驶专注度四处初始“待检测/等待检测”文案，将疲劳与专注两张卡的半圆仪表替换为无装饰的数值指标块；`driver-inference.js` 在尚无有效检测结果时保持这些状态文案为空，获得结果后继续显示真实风险状态；`styles.css` 新增指标块、空状态隐藏和侧栏行高/宽度最终规则，重新分配驾驶员、疲劳、专注和识别结果四卡比例。`npm run check` 通过；浏览器在 `1160×1180` 下验证实时页无“待检测/等待检测”文本、无 `.gauge` 节点、无整页横向/纵向溢出，侧栏与主区底部对齐，四张侧栏卡均无内部裁切。未修改摄像头、ONNX 推理结果、后端 API、SQLite 数据或模型文件。
- 2026-08-30：按用户要求移除实时检测页状态栏中的“检测图片”入口，并优化摄像头优先的整体 UI 比例。`index.html` 删除可见的图片检测按钮与文件输入入口，将待命提示和事件时间线文案改为“启动/打开摄像头”；`driver-inference.js` 将实时推理初始化条件从必须存在图片输入改为必须存在摄像头按钮，并让图片输入监听变为可选，避免移除入口后摄像头模块不启动；`styles.css` 新增摄像头单按钮状态栏、主画面等宽约束、待命卡片居中、实时页隐藏页脚和中等窗口侧栏紧凑比例，修复旧 16:9 画面规则导致状态栏与检测画面宽度不一致的问题。`npm run check` 通过；浏览器验证实时检测页无“检测图片”可见文本、无 `#inferenceImageInput`、无横向/纵向溢出，状态栏与检测画面同宽，待命提示在画面内居中，左侧四张卡无内部裁切。未修改 ONNX 模型、摄像头推理算法、后端 API、SQLite 数据或训练逻辑。
- 2026-08-30：按用户提供的侧栏参考图优化实时检测页左侧监测栏。`index.html` 将“驾驶员状态”标题图标从人脸替换为人员图标；`styles.css` 新增实时页侧栏最终覆盖规则，将桌面端侧栏宽度提高到 `clamp(318px, 22vw, 388px)`，四张侧栏卡分别调整为更接近参考图的宽松高度与浅色卡片风格，驾驶员状态卡改为大图标标题、居中检测对象模块和两列风险项，疲劳/专注仪表改为自然流式布局并保留用户要求的红色半圆仪表弧，实时识别结果卡改为更清爽的空状态；同时为 `#live-detection` 增加桌面锚点偏移，避免固定导航遮挡侧栏顶部。`npm run check` 通过；浏览器验证当前实时检测页左侧四张卡无内部裁切、无横向溢出。未修改摄像头、ONNX 推理、后端 API、SQLite 数据或模型文件。
- 2026-08-30：按用户参考图重构“系统设置”页为“系统配置”工程配置台。`index.html` 将设置页改为顶部大标题与操作按钮、左侧配置菜单和诊断卡、右侧检测阈值控制台与 API 管理表格，并在下方保留数据加工、模型训练、评估指标、混淆矩阵、导出部署和任务日志的原功能控件及关键 ID；`styles.css` 新增设置页专属网格背景、硬边卡片、左侧菜单激活态、诊断进度、API 配额表、蓝/紫/红三色阈值滑杆和响应式布局，并修复 `#settings` 直接访问时标题被固定导航遮挡的问题。`npm run check` 通过；浏览器验证设置页无横向溢出、标题完整可见、关键功能控件 ID 未缺失。未修改后端 API、SQLite 数据、ONNX 推理、摄像头或模型文件。
- 2026-08-30：优化全站顶部导航栏视觉。`styles.css` 新增最终导航覆盖规则，将四页顶栏统一为 64px 高度、深蓝渐变品牌标识、浅灰分段导航容器和紧凑胶囊式激活态；移除实时检测页旧样式叠加产生的粗蓝外框与底部蓝色指示线，保留键盘焦点的轻量可访问描边。同步调整移动端固定导航位置和页面顶部留白。未修改页面结构、后端 API、数据库、ONNX 推理、摄像头或模型文件。
- 2026-08-30：按用户要求将实时检测页“疲劳监测”和“驾驶专注度”两张卡的半圆仪表弧由继承的深蓝色改为明确的告警红 `#dc2626`；仅修改 `styles.css` 仪表弧颜色，浅灰底轨、数字、状态标签、推理逻辑和数据更新方式保持不变。
- 2026-08-30：完成四个主页面的最终桌面端空间占比与内部裁切优化。`styles.css` 将控制台趋势图改为“标题 + 弹性图表”网格，避免图表固定高度越界；数据集统计卡收敛装饰光晕范围并压实类别分布行高，消除统计卡横向溢出和类别卡纵向裁切；系统设置页将检测阈值区提高到 194px，并让阈值卡、说明文字和滑杆按剩余高度自适应；实时检测页保持左侧遥测、顶部状态、中央监测画面、标注工具和事件时间线完整铺满首屏。浏览器在 `1784×1079` 下逐页检查控制台、数据集管理、实时检测、系统设置：四页均无整页横向/纵向溢出，全部 Bento 卡片 `scrollWidth`/`scrollHeight` 均未超过容器；系统设置与实时检测截图复核通过。`npm run check` 通过，启动脚本确认服务地址为 `http://127.0.0.1:8000/index.html`。未修改摄像头、ONNX 推理、后端 API、SQLite 数据、模型权重或训练逻辑。
- 2026-08-30：优化实时检测页左侧监测模块的元素占比。`styles.css` 将桌面端驾驶员状态、疲劳监测、驾驶专注度、实时识别结果四张卡从按剩余高度等比分配改为内容高度 `202/164/176/178px`，取消风险项、统计项和进度条的自动贴底拉伸，改为连续紧凑排列，并限制识别结果列表最大高度。浏览器在当前 `1160×1180` 视口实测四卡高度与规则一致，各卡 `scrollHeight` 均小于可用高度，无文字/仪表裁切，主画面保持完整；`npm run check` 通过。未修改摄像头、ONNX 推理、后端 API、数据库或模型文件。
- 2026-08-30：按用户要求移除实时检测页顶部介绍区域。`styles.css` 隐藏“驾驶监测系统在线 / 实时驾驶状态监测 / 本地模型说明”整块 `live-heading`，并将实时页网格改为单行 `workspace`，使驾驶状态栏、摄像头画面、左侧监测卡、标注栏和事件时间线直接从内容区顶部开始。已在当前服务页面刷新验证，介绍区不可见、横向无溢出；`npm run check` 通过。未修改摄像头、ONNX 推理、后端 API、数据库或模型文件。
- 2026-08-29：修正实时检测页整体比例。`styles.css` 将桌面结构从“标题占左列、主画面跨两行”恢复为“96px 整行标题 + 下方左右工作区”，左侧监测栏使用 `clamp(264px, 19vw, 320px)` 稳定宽度，右侧主区固定分配 82px 状态栏、弹性监测画面、68px 标注栏和 148px 事件时间线，避免中央画面横向/纵向过度拉伸以及底部模块被压扁；较矮桌面使用 790px 最小工作高度，允许页面自然纵向滚动而不裁切内容。窄屏补充锚点滚动偏移，修复直接访问 `#live-detection` 时标题被双层固定导航遮住。浏览器已在 `1695×983` 和 `760×1000` 下截图验证，模块对齐且无重叠；`npm run check` 通过。未修改推理、摄像头、后端 API、数据库或模型文件。
- 2026-08-29：补齐平台从上传样本到可训练数据集、训练、评估和导出的真实流程。`index.html`、`app.js`、`styles.css` 增加图片/视频样本预览、归一化检测框绘制/回显、整图分类与目标检测切换、清洗/增强/视频抽帧配置、已注册模型选择和真实指标空状态；`driver-inference.js` 与样本预览联动。`backend/server.py` 新增样本文件/标注读取接口，使用 FFmpeg/ffprobe 实现去重、亮度/模糊检测、640×640 等比例填充、视频抽帧、按类别稳定划分训练/验证/测试集、训练集增强和 YOLO 标签/`dataset.yaml` 生成；训练必须调用真实 Ultralytics 并产出 `best.pt`，评估必须解析真实验证指标，PT 导出复制真实权重，ONNX/TensorRT/TFLite 必须由真实导出命令生成，所有缺失能力均明确失败，不再写入模拟指标或伪模型。启动时自动注册下载项目中的两个 `.pt` 权重。`backend/README.md` 同步更新。验证包括 `npm run check`、60 个前端依赖 ID 完整性检查、隔离临时数据库的真实加工管线（1 个原样本 + 1 个增强样本、YOLO 标签和 YAML）、真实 PT 导出（5,450,067 字节）、`1600×900` 四页无横向/纵向溢出且设置卡片无裁切，以及 `/api/health` 能力检查。当前解释器为 `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3` 3.11.9，项目无 `.venv` 且未安装 Ultralytics/PyTorch；遵守安全规则未自动安装，因此 FFmpeg 加工、浏览器 ONNX 推理和 PT 导出可用，真实训练/评估/格式转换需用户确认后在项目 `.venv` 中安装依赖。
- 2026-08-29：按用户截图扩充实时检测页上半部分空白区域。`styles.css` 将桌面端实时页改为两列两行布局：左上保留页面标题、左下保留四个监测卡，右侧状态栏、检测画面、标注栏和事件时间线跨越上下两行，从而直接利用原先标题右侧的空白；在 `1280×900` 下检测画面由 368px 增高到 466px，同时修正专注度进度标签继承全局网格样式导致的内部高度膨胀。浏览器验证 `1280×900` 无页面溢出或卡片裁切，`760×1000` 继续保持单列且无横向溢出。未修改摄像头、ONNX 推理、后端 API、数据库或模型文件。
- 2026-08-29：继续优化实时驾驶状态监测页的元素位置与空间分配。`styles.css` 将桌面端实时页高度由视口减 152px 调整为减 112px，更充分使用首屏；主区域重新分配状态栏、检测画面、标注栏与事件时间线行高；左侧驾驶员状态、疲劳监测、驾驶专注度和实时识别结果改为按内容权重弹性分配，补足驾驶员卡最小高度，并让风险指标、底部统计和识别结果自然贴合卡片上下边界；修复 900px 以下旧规则叠加导致主画面仍保持桌面宽度的问题。已用浏览器验证 `1280×900`、`1600×900`、`2048×1186` 均无页面溢出或卡片内容裁切，`760×1000` 下主画面、状态条、标注栏、时间线和侧栏均严格单列显示；`npm run check` 通过。未修改摄像头、ONNX 推理、后端 API、数据库或模型文件。
- 2026-08-29：根据用户最终提供的浅色实时监测参考图，重新确定实时页视觉方向并覆盖此前深色监控舱皮肤。`styles.css` 将实时页改为白色/浅灰网格背景、深蓝文字、青蓝强调色、细边框与轻阴影，保留左侧四组驾驶员遥测、顶部状态条、中央大画面、底部紧凑标注与事件时间线；`index.html` 将“检测图片/启动摄像头”移入状态栏右端，使信息层级与参考图一致；`app.js` 增加 `file://` 预览时的离线图标降级，避免 Material Symbols 无法联网时显示 `wifi_tethering`、`face`、`center_focus_strong` 等英文图标名，并修复直接访问 `#live-detection` 时固定导航遮挡标题的问题。1600×900 服务页实测无纵向滚动、无模块重叠；未修改推理算法、后端 API、数据库和模型文件。
- 2026-08-29：按用户提供的专业驾驶员监测系统参考图重构“实时驾驶状态监测”页面。`index.html` 将实时页调整为左侧驾驶员遥测、顶部八项驾驶行为状态条、中央摄像头画面、底部紧凑标注与事件时间线的 DMS 监测舱布局；新增头部姿态风险、眼睛疲劳风险和驾驶员锁定状态。`driver-inference.js` 将现有 ONNX 推理结果同步到顶部状态项及头部/闭眼风险读数，仍使用本地 `soham_best.onnx`/`chaitanya_best.onnx`，未伪造独立人脸身份或精确头姿数据。`styles.css` 为实时页增加深色网格化专业监控视觉、检测框角标、紧凑仪表和 900px 响应式断点；在 1160×1180 与 1600×900 视口实测均保持左侧遥测和中央画面完整显示，1600×900 无页面纵向滚动。未修改其他三个页面、后端 API、数据库或模型文件。
- 2026-08-29：按用户要求将平台视觉从偏演示感的浅色大圆角风格收敛为更专业的管理后台风格。`styles.css` 新增全局专业化皮肤：卡片圆角由 32px 压到 12px、内部模块圆角压到 8px，主色改为沉稳蓝，背景改为中性浅灰蓝，阴影减弱，顶部导航、按钮、输入框、统计卡、图表、实时检测画面和任务日志统一为更克制的运营控制台视觉。未改变 `index.html` 结构、`app.js`/`driver-inference.js` 推理逻辑、后端 API 或模型文件。
- 2026-08-29：按用户截图去除实时检测画面上的三处悬浮控件：左上角模型就绪状态、右上角推理耗时标签、画面中的推理方案/本地权重选择条。`styles.css` 使用实时检测画面内的覆盖样式隐藏这些元素，保留 `driver-inference.js` 默认“状态检测模型（流畅）”推理逻辑、摄像头检测、图片检测和右侧检测结果显示不变；未删除 DOM、未移动模型文件、未安装依赖。
- 2026-08-29：按用户截图重新分配实时检测页空间，优先完整显示摄像头画面。`styles.css` 新增最终大屏覆盖规则：检测画面高度使用 `clamp(520px, 61vh, 720px)`，摄像头继续以 `object-fit: contain` 等比完整显示；数据标注区改为横向紧凑工具条并固定 68px，事件时间线固定 124px、内部列表限高 76px并滚动；模型方案选择器移动到画面顶部，避免遮挡底部人脸和检测框。已用浏览器按用户截图近似视口 `2536×1234` 验证：检测画面 720px、标注区 68px、时间线 124px，页面无横向或纵向溢出。
- 2026-08-29：优化摄像头实时检测卡顿。将摄像头画面从“每次推理后重绘一帧”改为原生 `<video>` 持续播放，透明 `canvas` 只异步更新检测框，推理不再决定视频刷新率；默认推理方案由双模型融合改为约 168 毫秒的状态检测单模型流畅模式，双模型保留为可选精细模式；摄像头采集目标调整为 960×540、24 帧/秒，推理间隔 650 毫秒；复用 640×640 预处理画布和约 5 MB Float32 输入缓冲区，减少内存分配与垃圾回收；检测结果 DOM 仅在结果变化时重绘，同类时间线事件限频为 5 秒一次。同步更新 `backend/README.md`。未安装新依赖。
- 2026-08-29：将用户下载的 `Driver-Monitoring-System` 现成模型真实嵌入平台。新增 `driver-inference.js`，直接加载 `public/static/models/soham_best.onnx` 与 `chaitanya_best.onnx`，实现浏览器端图片/摄像头推理、双模型融合与单模型切换、640×640 预处理、置信度过滤、NMS、中文类别映射、检测框绘制、疲劳/专注指标、安全带状态和事件时间线；`index.html` 重构实时检测页媒体画布与控制区，`styles.css` 补充响应式布局，`app.js` 移除旧模拟司机节点依赖，`package.json` 将新脚本加入统一 JS 检查，`backend/README.md` 补充现成模型使用说明。未安装任何 Python 或 npm 依赖，模型权重保留在原下载目录。已用仓库示例图 `1_drowsy_phone.jpg` 实测识别“使用手机 89%”与“疲劳/瞌睡 76%”；首次双模型初始化约 13 秒，缓存后双模型约 345 毫秒、单模型约 168 毫秒；`2048×1050` 下实时检测页无横向或纵向溢出。
- 2026-08-28：新增 npm 脚本入口。新增 `package.json`，仅作为无依赖命令包装层，不安装或引入第三方包；支持 `npm start`/`npm run dev` 调用 `start-backend.sh` 启动平台，`npm run open` 启动并自动打开浏览器，`npm run check` 串联 `node --check app.js` 与 `python3 -m py_compile backend/server.py`，`npm run health` 请求默认后端健康检查。同步更新 `backend/README.md` 和本文项目结构、技术栈、启动方式、验证命令。
- 2026-08-28：优化平台启动体验。`start-backend.sh` 保持原有绝对路径入口不变，新增 Python 命令、项目目录、后端入口和端口占用检查；若默认端口已运行 Vision Sentinel 后端，会直接提示访问地址并退出；若端口被其他程序占用，默认自动切换到后续可用端口；新增 `VISION_SENTINEL_AUTO_PORT=0` 禁用自动换端口、`VISION_SENTINEL_OPEN_BROWSER=1` 启动后自动打开浏览器、`PYTHON_BIN=/path/to/python3` 指定解释器。同步更新 `backend/README.md` 和本文启动方式；已验证 `bash -n start-backend.sh`、`python3 -m py_compile backend/server.py`，并用临时端口 `8127` 真实启动后访问 `/api/health` 正常返回，测试进程已停止。
- 2026-08-28：按用户最新截图继续优化四页子模块比例。`styles.css` 调整大屏覆盖层：控制台压低 KPI 与趋势图区、修正完整开发流程子卡文字垂直分布；数据集管理压缩表单/类别分布卡片高度并放大近期样本区，减少中部空白；实时检测重新分配视频、标注、时间线与右侧仪表/类别卡高度，避免仪表底部内容被裁切；系统设置细化阈值、加工、训练、指标、混淆矩阵、导出和任务日志的内部间距。已用浏览器在 `2048×1050` 视口检查控制台、数据集管理、实时检测演示、系统设置四页，整页无横向或纵向滚动，主要内容完整可见；已验证 `node --check app.js` 和 `python3 -m py_compile backend/server.py`。
- 2026-08-28：按用户截图修正控制台和数据集管理两页的大屏缩放比例。`styles.css` 新增 1600px×900px 以上针对 `dashboard` 与 `dataset` 的整屏行高分配：控制台改为标题、KPI、趋势/当前数据集、完整流程四行铺满视口；数据集页改为标题、三张统计卡、数据集表单/类别分布、近期样本四行铺满视口，并放大标题、KPI/统计数字、图表、样本卡和流程卡比例，避免内容集中在上方导致底部大面积空白。已验证 `2048×1200` 下控制台、数据集管理、实时检测演示、系统设置四页均无横向溢出、无纵向溢出、无内部裁切；`node --check app.js` 和 `python3 -m py_compile backend/server.py` 通过。
- 2026-08-28：修复新版 Python 启动后端时报 `ModuleNotFoundError: No module named 'cgi'` 的问题。`backend/server.py` 移除已废弃的 `cgi.FieldStorage` 依赖，改用 Python 标准库 `email.parser.BytesParser` 解析 `multipart/form-data` 上传文件，保持无第三方依赖；`backend/README.md` 补充上传接口不依赖 `cgi` 的兼容说明。已验证 `python3 -m py_compile backend/server.py`、multipart 文件解析自检、临时端口 `127.0.0.1:8011` 启动后 `/api/health` 正常返回；测试服务已停止。
- 2026-08-27：继续按用户截图反馈调整所有网页格式。`styles.css` 取消普通桌面端页面硬裁切，改为“内容完整优先、必要时页面纵向滚动”，避免 1280×720 等较矮视口下卡片内容被隐藏；同时新增 1600px×900px 以上大屏单屏模式，扩大主内容宽度到 `min(1660px, calc(100vw - 56px))`，重新分配控制台、数据集管理、实时检测演示、系统设置四页网格行高，压实实时检测仪表、事件时间线、设置页阈值、加工比例、导出和任务日志区域。已验证 `node --check app.js`、`python3 -m py_compile backend/server.py`；浏览器实测 2048×1050 下四页均无横向溢出、无纵向溢出、无内部裁切，默认 1280×720 下四页无横向溢出且内部裁切计数为 0。
- 2026-08-27：按用户要求继续调整四页“一页完整显示”和中文化。`styles.css` 新增桌面端固定视口布局：顶部栏压缩到 56px，四个页面在 1280×720 下均固定在一屏内，页面本身不产生纵向滚动或横向溢出；数据集页类别分布改为四列紧凑直显，7 类状态全部在卡片内可见，样本预览同步压缩。`index.html` 将可见英文品牌、账号、图例、实时检测状态、事件、指标、设置页分组、导出格式和页脚全部改为中文；`app.js` 将动态样本状态、媒体类型、任务类型、任务状态、类别说明、图表坐标和导出提示中文化，并保留中文模型名到后端模型文件名的映射。已验证 `node --check app.js`、`python3 -m py_compile backend/server.py`、39 个 JS 依赖 ID 全部存在、`/api/health` 正常；浏览器检查控制台、数据集管理、实时检测演示、系统设置四页均一屏完整显示，数据集 7 类分布全部直显。
- 2026-08-27：按用户最新要求“stitch_ 完全使用这个文件夹中的 UI”重构主前端。`index.html` 改为完全围绕 `stitch_/vision_sentinel_1..4/code.html` 的四页导航与页面结构：控制台、数据集管理、实时检测演示、系统设置；`styles.css` 全量重写为 `stitch_/vision_sentinel/DESIGN.md` 的浅色 SaaS/Bento 系统，包括 72px 玻璃顶栏、Geist 字体、`#f8fafc` 背景、白色 32px 圆角卡片、柔和蓝色阴影、蓝色渐变按钮和状态色浅底标签；`app.js` 新增 hashchange 页面切换，并让数据集页上传入口复用后端上传流程。已验证 `node --check app.js`、`python3 -m py_compile backend/server.py`、39 个 JS 依赖 ID 均存在、`/api/health` 正常；浏览器检查四个 hash 页面均可独立显示，桌面与窄屏均无横向溢出。
- 2026-08-27：优化完善司机状态检测模型训练与评估平台。当前 `状态检测` 根目录前端文件曾缺失，本次重建 `index.html`、`styles.css`、`app.js` 为七大流程模块工作台：工作台、数据集管理、数据标注、数据加工、模型训练、结果分析、模型导出/部署；前端接入后端 `/api/health`、`/api/datasets`、`/api/datasets/{id}/upload`、`/api/datasets/{id}/summary`、`/api/datasets/{id}/samples`、`/api/process-jobs`、`/api/train-jobs`、`/api/export-jobs`、`/api/jobs/{id}`、`/api/jobs`、`/api/models`。后端新增 GET `/api/datasets/{id}/samples`、GET `/api/jobs` 和 HEAD 静态文件检查支持，用于前端展示样本列表、最近任务和页面入口健康检查。平台现在支持真实创建数据集、上传样本、保存标注、启动加工/训练/导出任务并轮询日志状态；已验证 `index.html` 返回 200、`/api/health` 正常、`/api/datasets` 正常。
- 2026-08-27：新增 `start-backend.sh` 后端启动脚本，内置项目绝对路径 `/Users/skyblue/Desktop/mainnn/状态检测`，默认启动 `http://127.0.0.1:8000/index.html`，并支持通过 `VISION_SENTINEL_PORT` 改端口；同步更新 `backend/README.md` 和本文启动方式，补充完整绝对路径启动命令。
- 2026-08-27：按用户要求新增真实训练版后端骨架。新增 `backend/server.py`，使用 Python 标准库提供 HTTP API、SQLite 元数据、静态文件服务、文件上传、标注保存、数据加工任务、YOLO 训练任务、评估任务、模型导出任务和任务日志；新增 `backend/README.md` 记录启动方式与 API 示例；新增 `storage/` 运行时目录用于数据集、标注、加工产物、模型、导出文件、任务日志和 SQLite；新增 `.gitignore` 忽略运行时产物。后端不会自动安装依赖，训练任务会优先调用当前环境已有的 `yolo` 命令或 `python -m ultralytics`，缺失时任务失败并写明原因。已验证 `python3 -m py_compile backend/server.py` 与 `/api/health`。当前前端尚未接入真实后端 API。
- 2026-08-27：按用户要求将“缩放比例下方显示完整”的修正扩展到四个页面。`styles.css` 新增桌面端统一页面缩放变量 `--desktop-scale: 0.9`，并进一步压缩顶栏高度、页面 padding、KPI 卡片、图表、数据集样本图、实时检测视频区、右侧仪表盘、设置侧栏、阈值卡、流程步骤和分析区高度，目标是在同一浏览器缩放下让控制台、数据集管理、实时检测演示、系统设置四页下方内容都更完整显示；未引入依赖，未改变四页导航、模拟数据或交互逻辑。
- 2026-08-26：根据用户截图反馈调整桌面端页面缩放观感，避免控制台在当前浏览器缩放比例下下方内容被截断。`styles.css` 收紧全局容器间距、卡片圆角和 padding、KPI 卡片高度、标题字号、按钮高度、趋势图高度、节点行与告警卡片尺寸，使控制台下方“近期严重告警”区域在同样缩放下更完整显示；未改变四视图结构、交互逻辑、技术栈或数据模拟方式。
- 2026-08-26：按用户澄清，将主前端从单个长流程页面改为与 `stitch_/vision_sentinel_1..4/code.html` 一致的四视图结构：控制台、数据集管理、实时检测演示、系统设置。四个视图共用同一套 Vision Sentinel 顶部导航、`#f8fafc` 背景、Geist 字体、白色 Bento 卡片、32px 圆角、柔和蓝色阴影、蓝色渐变按钮和统一 footer；各视图内容分别补齐趋势/节点/告警、数据集搜索/样本卡/类别分布、实时视频/事件时间线/疲劳与专注仪表、阈值/API/交互/训练评估配置。`app.js` 新增导航 hash 切换和设置滑块联动，保留原训练、标注、导入和图表模拟能力。
- 2026-08-26：按当前真实存在的 `stitch_/` 目录重写主页面 UI，对齐 `stitch_/vision_sentinel/DESIGN.md` 和 `vision_sentinel_*` 页面。主页面由深色编辑器风格切换为 Vision Sentinel 的浅色 SaaS/Bento 风格：72px 玻璃顶栏、Geist 字体、`#f8fafc` 背景、白色 32px 圆角 Bento 卡片、蓝色渐变主按钮、柔和蓝色阴影、状态色浅底标签、数据集管理/实时检测/训练分析模块化布局。`app.js` 动态类别和图表色值同步切换为 Vision Sentinel 色板。同步修正本文档中旧的 `stitch_3d/` 记录为 `stitch_/`。
- 2026-08-26：按用户要求补全“司机状态检测完整流程”的网页实现。新增顶部完整训练流程面板，覆盖获取数据集、数据标注、数据加工、模型训练、结果分析、导出部署六个阶段；新增“一键演示流程”前端交互，自动推进阶段、模拟样本导入、标注增加、数据划分、训练启动和结果分析刷新。该实现仍保持纯前端展示，无后端、无真实训练服务、无新增依赖。
- 2026-08-26：按用户要求将主页面进一步收紧为与 `stitch_3d/3d_desktop/code.html` 一致的编辑器布局与色彩系统。`index.html` 改为固定 260px 左侧项目栏、固定 64px 顶部导航、中央全屏深色视口、悬浮工具条、左侧 Hierarchy/Dataset 面板、右侧 Properties/Training/Evaluation 面板和底部 120px 时间轴；`styles.css` 改用 Gaussian3D 设计 token（`#131313`、`#1c1b1b`、`#201f1f`、`#2a2a2a`、`#353534`、`#e5e2e1`、`#c1c6d7`、`#8b90a0`、`#414755`、`#adc6ff`、`#00eefc` 等）并实现同款玻璃拟态、网格视口、固定面板、时间轴关键帧和 Material Symbols 图标风格；`app.js` 动态类别和图表颜色同步改为该色板。仍无新增构建工具、后端、数据库或本地安装依赖。
- 2026-08-26：为主页面新增训练结果分析展示：Loss/Accuracy 曲线、ROC/AUC 曲线、Precision、Recall、F1-score、mAP 指标解释卡片和类别混淆矩阵；`开始训练` 完成后会前端模拟刷新曲线和评估分数。补回主页面引用的 `styles.css`，保持项目为无后端、无依赖的静态前端展示。
- 2026-08-26：依照现有 `stitch_3d/` Gaussian3D 原型修改主前端网页。重构 `index.html` 为深色视觉工作台布局，包含左侧 Pipeline 侧栏、顶部阶段导航、中央标注视口、底部时间轴和右侧训练/评估检查器；重写 `styles.css` 为 Gaussian3D 风格的深色玻璃拟态、网格视口、霓虹状态色和响应式布局；调整 `app.js` 类别颜色以匹配新视觉系统。项目仍保持静态 HTML/CSS/原生 JavaScript，无新增依赖、构建工具、数据库或测试框架。
- 2026-08-26：更新 `fly.md` 为 `/Users/skyblue/Desktop/mainnn/状态检测` 项目根目录权威长期记忆；扫描确认当前项目为静态前端，主入口为 `index.html`，交互在 `app.js`，样式在 `styles.css`，并补充 `stitch_3d/` Gaussian3D 原型、无 Git、无数据库、无测试框架、启动/验证方式、安全禁令和后续维护流程。
- 2026-08-26：此前记录曾声明权威长期记忆迁移到 `/Users/skyblue/Desktop/mainnn/fly.md`；本次按用户要求恢复本项目根目录 `fly.md` 为权威记忆，该旧迁移说明作废。
- 2026-08-26：按用户澄清，将司机状态检测网页整理为单独项目目录 `状态检测/`，并将项目记忆文件放入 `状态检测/fly.md`；同步更新项目路径、启动路径和项目结构说明。
- 2026-08-26：将项目长期记忆文件改放到独立目录 `fly/fly.md`，并更新文档中的记忆文件路径和项目结构说明。
- 2026-08-26：按用户要求创建独立目录 `project-memory/`，并将项目长期记忆文件移动到 `project-memory/fly.md`；同步更新记忆文件中的路径和项目结构说明。
- 2026-08-26：创建纯前端司机状态检测训练台展示页。新增 `index.html`、`styles.css`、`app.js`，实现 Roboflow 风格的工作台布局，包含数据集类别分布、驾驶舱标注画布、司机状态类别、训练进度模拟和评估指标展示；明确项目无需后端。
- 2026-08-26：创建项目长期记忆 `fly.md`。扫描确认当前项目目录为空，未发现 Git 仓库、源码、配置、启动方式、技术栈、数据库、测试命令、分支规则或现有项目文档。
