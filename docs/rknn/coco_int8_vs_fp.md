# coco RKNN 一致性验证报告

- 生成时间: 2026-09-10T20:10:36
- 参考(Reference): `rk3588_dms/models/rknn/yolov8n_coco_fp.rknn` (FP RKNN 真机)
- RKNN 来源: rk3588_dms/models/rknn/yolov8n_coco_int8.rknn

## 汇总指标

| 指标 | 数值 |
| ---- | ---- |
| 测试图片数 | 46 |
| 完全匹配图片数 | 35 |
| 检测配对数 | 67 |
| 类别一致率 | 67/67 |
| bbox 平均 IoU | 0.99 |
| 置信度平均绝对差 | 0.0156 |
| 仅参考检出 | 5 |
| 仅 RKNN 检出 | 6 |

## 异常样本

### test_0015.jpg
- 参考有而 RKNN 无: ['tennis racket']

### test_0029.jpg
- tv: conf 0.260 vs 0.337 (Δ0.077)

### test_0043.jpg
- tv: conf 0.260 vs 0.376 (Δ0.116)
- RKNN 有而参考无: ['chair']

### test_0050.jpg
- RKNN 有而参考无: ['chair']

### test_0057.jpg
- chair: conf 0.258 vs 0.359 (Δ0.101)
- RKNN 有而参考无: ['laptop']

### test_0064.jpg
- tv: conf 0.302 vs 0.368 (Δ0.066)
- 参考有而 RKNN 无: ['laptop']

### test_0071.jpg
- tv: conf 0.302 vs 0.390 (Δ0.088)
- RKNN 有而参考无: ['chair']

### test_0078.jpg
- RKNN 有而参考无: ['tv', 'chair']

### test_0099.jpg
- 参考有而 RKNN 无: ['cell phone']

### test_0141.jpg
- toothbrush: conf 0.456 vs 0.394 (Δ0.062)
- 参考有而 RKNN 无: ['cell phone']

### test_0148.jpg
- 参考有而 RKNN 无: ['cell phone']

## 判定建议

- 类别一致率 100% 且 平均 IoU >= 0.9 且 置信度差 <= 0.02: 可进入真机验证
- 出现整类丢失(仅参考检出成批出现): 检查导出方式/量化
- 本报告由 `rk3588_dms/tools/compare_outputs.py` 生成, 原始数据见 `logs/rknn/compare_coco_20260910-201030`
