# chaitanya RKNN 一致性验证报告

- 生成时间: 2026-09-10T19:39:51
- 参考(Reference): `rk3588_dms/models/rknn/chaitanya_best_fp.rknn` (FP RKNN 真机)
- RKNN 来源: rk3588_dms/models/rknn/chaitanya_best_int8.rknn

## 汇总指标

| 指标 | 数值 |
| ---- | ---- |
| 测试图片数 | 46 |
| 完全匹配图片数 | 45 |
| 检测配对数 | 10 |
| 类别一致率 | 10/10 |
| bbox 平均 IoU | 0.9825 |
| 置信度平均绝对差 | 0.0277 |
| 仅参考检出 | 0 |
| 仅 RKNN 检出 | 0 |

## 异常样本

### test_0148.jpg
- Cigarette: conf 0.571 vs 0.667 (Δ0.096)

## 判定建议

- 类别一致率 100% 且 平均 IoU >= 0.9 且 置信度差 <= 0.02: 可进入真机验证
- 出现整类丢失(仅参考检出成批出现): 检查导出方式/量化
- 本报告由 `rk3588_dms/tools/compare_outputs.py` 生成, 原始数据见 `logs/rknn/compare_chaitanya_20260910-193947`
