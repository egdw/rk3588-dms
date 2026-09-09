# models/rknn —— RKNN 转换产物

由 `rk3588_dms/tools/convert_chaitanya.py` 等生成，命名约定：

```text
chaitanya_best_fp.rknn      第一轮 FP(非量化)验证模型
chaitanya_best_int8.rknn    后续阶段 INT8 量化模型(需 dataset.txt)
soham_best_fp.rknn          第二个模型(第一阶段完成后)
yolov8n_coco_fp.rknn        第三个模型
```

`.rknn` 文件较大，不入 Git（`.gitignore` 建议追加 `*.rknn`）。
部署到 RK3588 时同步整个目录。
