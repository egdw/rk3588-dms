# Vision Sentinel 后端

这是司机状态检测模型训练与评估平台的本地后端。HTTP、SQLite、任务队列和文件管理使用 Python 标准库；图片/视频加工调用本机 FFmpeg，训练、评估和模型格式转换调用项目环境中的 Ultralytics。后端不会自动安装任何依赖，也不会用模拟结果冒充真实训练或评估。

后端上传接口使用标准库 `email.parser` 解析 `multipart/form-data`，不依赖已废弃并在新版 Python 中移除的 `cgi` 模块。

## 启动

推荐使用启动脚本：

```sh
/Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
```

启动脚本会自动检查 Python、项目目录、后端入口和端口占用情况。默认监听 `0.0.0.0:8000`，同一局域网设备可通过本机局域网 IP 访问；如果 8000 已经是本平台在运行，会直接提示本机和局域网访问地址并退出；如果 8000 被其他程序占用，会自动切换到后续可用端口。

也可以使用 npm 包装命令启动，不需要安装依赖：

```sh
cd /Users/skyblue/Desktop/mainnn/状态检测 && npm start
```

npm 可用命令：

```sh
npm start
npm run dev
npm run open
npm run check
npm run health
```

完整绝对路径命令：

```sh
cd /Users/skyblue/Desktop/mainnn/状态检测 && python3 /Users/skyblue/Desktop/mainnn/状态检测/backend/server.py
```

如果要换端口：

```sh
VISION_SENTINEL_PORT=8010 /Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
```

如果只允许本机访问：

```sh
VISION_SENTINEL_HOST=127.0.0.1 /Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
```

如果不想自动换端口：

```sh
VISION_SENTINEL_AUTO_PORT=0 /Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
```

如果希望启动后自动打开浏览器：

```sh
VISION_SENTINEL_OPEN_BROWSER=1 /Users/skyblue/Desktop/mainnn/状态检测/start-backend.sh
```

默认本机地址：

```text
http://127.0.0.1:8000
```

局域网访问地址以启动脚本输出为准，通常类似：

```text
http://192.168.x.x:8000/index.html
```

本机浏览器打开：

```text
http://127.0.0.1:8000/index.html
```

注意：如果 macOS 防火墙拦截 Python 入站连接，需要在系统弹窗中允许，或在系统设置中允许对应 Python 接收局域网连接。

## 数据存储

后端会写入项目内 `storage/` 目录：

- `storage/vision_sentinel.sqlite3`：SQLite 元数据
- `storage/datasets/`：上传的图片/视频
- `storage/annotations/`：预留标注文件目录
- `storage/processed/`：数据加工输出与 `dataset.yaml`
- `storage/models/`：训练输出模型目录
- `storage/exports/`：导出文件
- `storage/jobs/`：任务日志

## API

### 健康检查

```sh
curl http://127.0.0.1:8000/api/health
```

### 创建数据集

```sh
curl -X POST http://127.0.0.1:8000/api/datasets \
  -H "Content-Type: application/json" \
  -d '{"name":"司机状态数据集","source":"驾驶舱摄像头","description":"正常、通话、闭眼、分心、安全带等状态"}'
```

### 上传样本

```sh
curl -X POST http://127.0.0.1:8000/api/datasets/{dataset_id}/upload \
  -F "files=@/path/to/image.jpg"
```

### 保存标注

```sh
curl -X POST http://127.0.0.1:8000/api/annotations \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id":"ds_xxx",
    "sample_id":"sample_xxx",
    "task_type":"detection",
    "label":"打电话",
    "boxes":[{"label":"手机","x":0.62,"y":0.33,"w":0.12,"h":0.18}]
  }'
```

标注框坐标为相对原图的归一化中心点格式 `x/y/w/h`。同一样本同一任务类型再次保存时会替换旧标注。

样本预览与标注读取：

```text
GET /api/samples/{sample_id}/file
GET /api/samples/{sample_id}/annotation
```

### 数据加工

```sh
curl -X POST http://127.0.0.1:8000/api/process-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id":"ds_xxx",
    "options":{
      "target_size":640,
      "quality_filter":true,
      "augment":true,
      "video_fps":1,
      "splits":{"train":70,"val":20,"test":10}
    }
  }'
```

加工任务会真实执行 SHA-256 去重、FFmpeg 视频抽帧、亮度/模糊度检测、等比缩放与黑边填充、按类别分层划分、YOLO 标签生成，并为训练集生成一份轮换的提亮/夜间/模糊/镜像增强样本。被质量筛掉和判定重复的加工产物会保留在该次运行目录的 `rejected/` 与 `duplicates/`，不会静默删除原始上传文件。

### 模型训练

```sh
curl -X POST http://127.0.0.1:8000/api/train-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id":"ds_xxx",
    "model":"Driver-Monitoring-System/models/soham/best.pt",
    "image_size":640,
    "epochs":50,
    "batch_size":16,
    "learning_rate":0.001,
    "pretrained":true
  }'
```

训练前必须先完成一次成功的数据加工。训练接口会调用当前环境已有的 `yolo` 命令；如果没有，则只在当前 Python 可导入 Ultralytics 时使用 `python -m ultralytics`。训练完成后必须实际存在 `weights/best.pt` 才会登记模型，并从真实 `results.csv` 回写 Precision、Recall、F1、mAP 与训练历史。当前环境未安装 Ultralytics 时任务会明确失败，不会自动下载或伪造训练结果。

### 查询任务

```sh
curl http://127.0.0.1:8000/api/jobs/{job_id}
```

### 手动控制与 MQTT 桥接

实时检测页支持手动模式。手动模式下摄像头画面继续显示，但自动模型/人脸分析暂停，页面状态和报警只由手动指令决定。

HTTP 控制接口：

```sh
curl -k -X POST https://127.0.0.1:8443/api/manual-control \
  -H "Content-Type: application/json" \
  -d '{"mode":"manual","key":"head_down","source":"terminal"}'
```

常用状态：

```text
safe       正常
drowsy     闭眼
head_down  低头
gaze_off   视线偏离
phone      打电话
```

演示流程对应的 MQTT 命令：

```sh
mosquitto_pub -h 192.168.2.13 -t "vision-sentinel/control" -m '{"command":"manual_enter"}'
mosquitto_pub -h 192.168.2.13 -t "vision-sentinel/control" -m '{"command":"head_down"}'
mosquitto_pub -h 192.168.2.13 -t "vision-sentinel/control" -m '{"command":"drowsy"}'
mosquitto_pub -h 192.168.2.13 -t "vision-sentinel/control" -m '{"command":"phone"}'
mosquitto_pub -h 192.168.2.13 -t "vision-sentinel/control" -m '{"command":"manual_exit"}'
```

网页直接走 MQTT over WebSocket，不需要在检测服务器额外启用 `vision-sentinel-mqtt.service`。检测页默认连接：

```text
wss://192.168.2.13:8084/mqtt
```

并订阅：

```text
vision-sentinel/control
```

完整流程：

1. 浏览器打开检测页。
2. 检测页自动连接 `wss://192.168.2.13:8084/mqtt`。
3. 终端向 `192.168.2.13:1883` 发布 MQTT 命令，或者打开 `mqtt-websocket-demo.html` 从网页发布命令。
4. 检测页收到命令后调用本机 `/api/manual-control`，进入手动模式并切换状态。
5. `manual_exit` 后退出手动模式，恢复摄像头自动识别。

因为检测页部署后是 `https://...`，所以 MQTT WebSocket 使用 `wss://...`，避免 Chrome 的 mixed content 拦截。

可以通过 URL 参数临时改 WebSocket 地址和 topic：

```text
https://192.168.2.8:8443/?mqttWs=wss%3A%2F%2F192.168.2.13%3A8084%2Fmqtt&mqttTopic=vision-sentinel%2Fcontrol#live-detection
```

备用方案才需要在 Linux 服务器启用 MQTT 桥接服务：

```sh
sudo cp /home/ztl/code/vision-sentinel/deploy/vision-sentinel-mqtt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vision-sentinel-mqtt
```

桥接服务会调用服务器上的 `mosquitto_sub`，订阅 `vision-sentinel/control`，收到命令后转发到本机 `https://127.0.0.1:8443/api/manual-control`。如果控制接口设置了 `VISION_SENTINEL_CONTROL_TOKEN`，MQTT 桥接服务也需要配置同名环境变量。

### 评估模型

```sh
curl -X POST http://127.0.0.1:8000/api/evaluate-jobs \
  -H "Content-Type: application/json" \
  -d '{"model_id":"model_xxx","dataset_id":"ds_xxx"}'
```

评估会在最新加工数据的测试集上运行 Ultralytics `detect val`；小数据集没有测试集时依次回退到验证集、训练集，并记录实际使用的 split。只有从真实命令输出读取到指标后才会成功，同时登记 PR 曲线和混淆矩阵图片。

### 导出模型

```sh
curl -X POST http://127.0.0.1:8000/api/export-jobs \
  -H "Content-Type: application/json" \
  -d '{"model_id":"model_xxx","format":"onnx"}'
```

`.pt` 导出会复制真实权重；ONNX、TensorRT Engine 和 TFLite 会调用 Ultralytics 的真实转换命令，只有检测到对应产物后才会成功。系统不会再创建扩展名看似正确的说明文件冒充模型。

## 现成模型推理

“实时检测演示”已直接接入项目内下载的 `Driver-Monitoring-System` 模型，不需要执行训练，也不需要安装 Python 机器学习依赖：

- 状态检测模型：`Driver-Monitoring-System/public/static/models/soham_best.onnx`
- 行为目标模型：`Driver-Monitoring-System/public/static/models/chaitanya_best.onnx`
- 推理页面：`http://127.0.0.1:8000/index.html#live-detection`
- 输入方式：本地图片或浏览器摄像头
- 推理方案：默认“状态检测模型（流畅）”；也可切换双模型融合（精细）或行为目标模型
- 输出内容：危险行为中文检测框、五官蓝色关键点、蓝色眼部轮廓、蓝色动态虹膜方向线、蓝色头姿辅助线、置信度、独立低头检测、疲劳风险、专注度、安全带检测和事件时间线

模型权重从当前项目目录读取，图片和摄像头帧在浏览器本地推理，不上传到第三方服务器。摄像头画面直接由原生视频元素连续播放，检测叠加层以独立透明画布异步覆盖，因此检测间隔不会冻结视频。主画面不再绘制正常驾驶、安全带等安全类绿色识别框；危险行为仍保留红色报警框，面部状态通过蓝色五官关键点、蓝色眼部轮廓、蓝色动态虹膜方向线和蓝色系头姿辅助线直接贴在主摄像头画面的人脸上呈现，不再通过侧栏单独人脸/眼睛预览画面展示。摄像头叠加层使用与视频一致的 `object-fit: contain` 坐标映射，并按视频原始坐标直接绘制；因为当前视频元素没有做 CSS 水平镜像，叠加层也不再额外镜像，避免关键点被翻到人脸反方向。眼部“视线偏移”和主画面的动态方向线均基于每只眼的虹膜中心相对眼角、上下眼睑构成的局部眼部坐标系实时估计；该线用于 2D 方向提示，不等同于专业眼动仪的真实三维注视点。低头检测使用浏览器端 MediaPipe Face Landmarker 的人脸关键点和姿态矩阵生成 `head_down` 状态，与现成 YOLO 模型的 `Distracted` 类分开计算；连续超过阈值后才触发报警，减少瞬时抖动。浏览器推理运行库和头姿模型目前由 CDN 加载，因此首次使用需要网络；权重加载完成后会被浏览器缓存。该下载项目定位为学术/演示用途，不能直接作为量产车辆安全决策系统。

## 当前运行依赖与边界

- 数据集、真实标注、FFmpeg 加工、YOLO 标签、SQLite、任务日志、现成 ONNX 图片/摄像头推理：已实现。
- 真实 YOLO 训练、测试集评估、ONNX/TensorRT/TFLite 转换：代码已实现，但运行时必须在项目 `.venv` 中具备 Ultralytics/PyTorch及相应导出依赖。
- TensorRT Engine 通常需要 NVIDIA GPU、CUDA 和 TensorRT；Apple Silicon 环境不能仅靠安装 Python 包完成 Engine 导出。
- TFLite 导出可能需要 Ultralytics 按版本安装额外 TensorFlow/ONNX 转换依赖。
- 实时页面已加入独立 `head_down` 低头检测；`Distracted` 与 `Drowsy` 仍是现成 YOLO 模型的状态类别，不代表人脸身份识别或医学级眼部疲劳判断。
- 当前是本地单用户开发平台，不包含登录权限、多人协作、云端队列和量产车辆安全认证。
