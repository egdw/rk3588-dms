# testdata/dms —— 统一测试集（STEP 4）

一致性对比（`rk3588_dms/tools/compare_outputs.py`）与真机单图验证共用的图片集。

## 要求

- 数量：**≥20 张**（建议 30~50 张）；
- 来源：项目现有测试图片、训练样例、或从比赛实际视频中抽帧（`ffmpeg -i video.mp4 -vf fps=1 frames/%03d.jpg`）；
- 覆盖场景（按 chaitanya 的 5 类 + soham 相关状态）：
  - 正常驾驶（安全带可见）
  - 打电话（手持/贴耳，不同角度）
  - 喝水
  - 吃东西
  - 抽烟（如有素材）
  - 安全带特写/未系
- 命名建议：`normal_01.jpg`、`phone_01.jpg`、`drinking_01.jpg`、`eating_01.jpg`、`seatbelt_01.jpg` …
- 光线/分辨率尽量贴近现场摄像头（320×180~1080p 均可，工具会自动 letterbox）。

## 注意

- **不要把大视频文件提交进 Git**；抽帧后删源或放仓库外。
- 本目录仅放测试图片；板上批量导出的 JSON 放 `board_results/`（不入 Git）。

## INT8 校准集（后续阶段）

另建 `rk3588_dms/models/onnx/dataset.txt`，内容为 100~300 张校准图片的**绝对路径**，
覆盖不同光线/姿态/手机/安全带/低头。与本目录图片可以复用同一批素材。
