# 6DRepNet RKNN 迁移分析（本阶段不实施）

> 结论先行：**第一阶段保持现状**（后端 `backend/server.py` PyTorch CPU 推理，浏览器 650ms
> 限频调用）。本文只做迁移可行性分析，供后续阶段决策。

## 1. 现状

- 代码位置：`6DRepNet-master/sixdrepnet/`（上游项目拷贝）；
- 权重：`6DRepNet_300W_LP_AFLW2000.pth`（本工作副本中不存在，现场机由
  `backend/server.py:415 sixdrepnet_weights_path()` 按候选顺序查找，或环境变量
  `SIXDREPNET_WEIGHTS` 指定）；
- 调用链：浏览器裁 256×256 人脸 JPEG → POST `/api/head-pose/6drepnet` →
  `predict_sixdrepnet_pose()`（`server.py:495`）→ `torch.no_grad()` + `detector.predict()`
  → 返回 pitch/yaw/roll；
- 浏览器侧：`driver-inference.js:1148 requestSixDRepNetPose`，最短间隔 650ms，
  连续失败 2 次自动禁用并回退 MediaPipe 关键点头姿；
- 失败容忍已内建：6DRepNet 不可用时系统照常运行（MediaPipe 兜底）。

## 2. 结构（读源码确认）

`6DRepNet-master/sixdrepnet/model.py`：

```text
input 1x3x224x224 (ImageNet mean/std 归一化, regressor.py transformations)
  → RepVGG-B1g2 backbone (deploy=True, 即重参数化后的纯 Conv+ReLU 结构)
  → AdaptiveAvgPool2d(1)
  → flatten
  → Linear(→6)                     ← 神经网络部分到此为止
  → compute_rotation_matrix_from_ortho6d   ← 6D 表征 → 3x3 旋转矩阵
  → 欧拉角分解(pitch/yaw/roll)
```

关键点：

- deploy 模式的 RepVGG 是**顺序 Conv/ReLU 堆叠**（训练态的多分支已折叠），
  是对 NPU 最友好的结构之一，无注意力/控制流/动态 shape；
- 输入预处理是 ImageNet mean/std（不是 YOLO 的 /255），RKNN 转换时
  mean_values=[[123.675,116.28,103.53]] std_values=[[58.395,57.12,57.375]]；
- 后处理（6D→矩阵→欧拉角）是少量矩阵运算，**必须留在 CPU**（RKNN 不适合
  atan2/三角函数链，也完全没有必要放进去）。

## 3. 建议迁移方案（后续阶段）

```text
浏览器(不变)                 RK3588 原生服务                     CPU (ARM)
────────────                ─────────────────────              ─────────────
人脸框裁切 256x256    →      224x224 Face ROI(中心裁切)  →
                            RepVGG Backbone              →
                            GAP + FC                     →    6D vector
                                                              rotation matrix
                                                              pitch/yaw/roll
                                                              (numpy, 微秒级)
                                                        ←     返回欧拉角
```

- ONNX 导出：`torch.onnx.export(model, dummy(1,3,224,224), opset=12)`，
  导出时把 forward 截断到 `linear_reg` 输出（6 维向量），欧拉角换算不进图；
- RKNN 转换与三个 YOLO 同一工具链（`convert_chaitanya.py` 可直接参数化复用）；
- 输出只有 6 个 float，通信开销可忽略。

## 4. 收益与风险评估

| 项 | 说明 |
| ---- | ---- |
| 收益 | 摘掉后端 PyTorch CPU 推理（当前是常驻 Python torch 进程，内存占用大）；头姿频率可从 ~1.5Hz 提到 10Hz+，`head_down` 判定更跟手 |
| 风险 | 低 —— 结构简单；失败已有 MediaPipe 兜底，回退路径现成 |
| 依赖 | 现场机需补 `.pth` 权重才能重导 ONNX（当前副本无权重） |
| 顺序 | 必须排在三个 YOLO 之后（STEP 12+），头姿不是比赛演示的卡顿瓶颈 |

## 5. 明确不做的事

- 不把 rotation matrix / atan2 / Euler conversion 塞进 RKNN 图；
- 本阶段不动 `backend/server.py` 的 `/api/head-pose/6drepnet` 接口；
- 不改浏览器 650ms 限频与失败禁用逻辑。
