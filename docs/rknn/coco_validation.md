# coco RKNN 一致性验证报告

- 生成时间: 2026-09-09T18:24:06
- 参考(Reference): `D:\project\dms\Driver-Monitoring-System\public\static\models\yolov8n_coco.onnx` (onnxruntime, 与浏览器同预处理/解码)
- RKNN 来源: rk3588_dms/models/onnx/rknn_source/yolov8n_coco_rknn.onnx

## 汇总指标

| 指标 | 数值 |
| ---- | ---- |
| 测试图片数 | 1 |
| 完全匹配图片数 | 1 |
| 检测配对数 | 1 |
| 类别一致率 | 1/1 |
| bbox 平均 IoU | 1.0 |
| 置信度平均绝对差 | 0.0 |
| 仅参考检出 | 0 |
| 仅 RKNN 检出 | 0 |

## 异常样本

无(所有样本在容差内一致)。
## 判定建议

- 类别一致率 100% 且 平均 IoU >= 0.9 且 置信度差 <= 0.02: 可进入真机验证
- 出现整类丢失(仅参考检出成批出现): 检查导出方式/量化
- 本报告由 `rk3588_dms/tools/compare_outputs.py` 生成, 原始数据见 `logs\rknn\compare_coco_20260909-182406`
