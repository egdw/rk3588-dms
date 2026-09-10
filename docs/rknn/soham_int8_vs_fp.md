# soham RKNN 一致性验证报告

- 生成时间: 2026-09-10T20:10:29
- 参考(Reference): `rk3588_dms/models/rknn/soham_best_fp.rknn` (FP RKNN 真机)
- RKNN 来源: rk3588_dms/models/rknn/soham_best_int8.rknn

## 汇总指标

| 指标 | 数值 |
| ---- | ---- |
| 测试图片数 | 46 |
| 完全匹配图片数 | 2 |
| 检测配对数 | 108 |
| 类别一致率 | 108/108 |
| bbox 平均 IoU | 0.9464 |
| 置信度平均绝对差 | 0.0853 |
| 仅参考检出 | 20 |
| 仅 RKNN 检出 | 2 |

## 异常样本

### test_0001.jpg
- SafeDriving: conf 0.848 vs 0.778 (Δ0.070)
- SafeDriving: conf 0.848 vs 0.662 (Δ0.186)

### test_0008.jpg
- SafeDriving: conf 0.848 vs 0.768 (Δ0.080)

### test_0015.jpg
- SafeDriving: conf 0.703 vs 0.434 (Δ0.269)

### test_0022.jpg
- SafeDriving: conf 0.848 vs 0.640 (Δ0.208)

### test_0029.jpg
- SafeDriving: conf 0.848 vs 0.715 (Δ0.133)
- SafeDriving: conf 0.673 vs 0.593 (Δ0.080)
- 参考有而 RKNN 无: ['SafeDriving']

### test_0036.jpg
- Drowsy: conf 0.500 vs 0.383 (Δ0.117)

### test_0043.jpg
- SafeDriving: conf 0.848 vs 0.795 (Δ0.054)
- SafeDriving: conf 0.703 vs 0.526 (Δ0.176)
- SafeDriving: conf 0.500 vs 0.428 (Δ0.072)

### test_0050.jpg
- SafeDriving: conf 0.848 vs 0.709 (Δ0.139)
- RKNN 有而参考无: ['Drowsy']

### test_0057.jpg
- SafeDriving: conf 0.848 vs 0.673 (Δ0.175)
- SafeDriving: conf 0.500 vs 0.308 (Δ0.192)

### test_0064.jpg
- 参考有而 RKNN 无: ['Drowsy', 'Drowsy', 'SafeDriving']

### test_0071.jpg
- SafeDriving: conf 0.848 vs 0.779 (Δ0.069)
- SafeDriving: conf 0.500 vs 0.432 (Δ0.068)

### test_0078.jpg
- Drowsy: conf 0.500 vs 0.357 (Δ0.143)

### test_0085.jpg
- SafeDriving: conf 0.848 vs 0.732 (Δ0.116)
- SafeDriving: conf 0.500 vs 0.271 (Δ0.229)
- Drowsy: conf 0.500 vs 0.563 (Δ0.063)

### test_0092.jpg
- Drowsy: conf 0.848 vs 0.673 (Δ0.175)
- Drowsy: conf 0.848 vs 0.678 (Δ0.170)
- Smoking: conf 0.703 vs 0.456 (Δ0.246)

### test_0099.jpg
- Smoking: conf 0.500 vs 0.317 (Δ0.183)

### test_0106.jpg
- Smoking: conf 0.500 vs 0.295 (Δ0.205)

### test_0113.jpg
- Smoking: conf 0.703 vs 0.461 (Δ0.241)
- RKNN 有而参考无: ['PhoneUse']

### test_0120.jpg
- Smoking: conf 0.703 vs 0.424 (Δ0.279)

### test_0127.jpg
- Smoking: conf 0.500 vs 0.365 (Δ0.135)
- PhoneUse: conf 0.389 vs 0.317 (Δ0.072)

### test_0134.jpg
- Smoking: conf 0.703 vs 0.490 (Δ0.213)

### test_0141.jpg
- 参考有而 RKNN 无: ['Smoking']

### test_0148.jpg
- SafeDriving: conf 0.848 vs 0.724 (Δ0.125)
- SafeDriving: conf 0.848 vs 0.791 (Δ0.057)
- PhoneUse: conf 0.611 vs 0.487 (Δ0.123)
- Smoking: conf 0.500 vs 0.448 (Δ0.052)
- 参考有而 RKNN 无: ['Drowsy']

### test_0155.jpg
- Smoking: conf 0.703 vs 0.448 (Δ0.254)
- PhoneUse: conf 0.611 vs 0.549 (Δ0.061)

### test_0162.jpg
- Smoking: conf 0.703 vs 0.485 (Δ0.218)

### test_0169.jpg
- SafeDriving: conf 0.848 vs 0.765 (Δ0.083)
- SafeDriving: conf 0.703 vs 0.590 (Δ0.113)

### test_0176.jpg
- SafeDriving: conf 0.848 vs 0.701 (Δ0.147)

### test_0183.jpg
- SafeDriving: conf 0.703 vs 0.625 (Δ0.078)
- SafeDriving: conf 0.500 vs 0.427 (Δ0.073)
- 参考有而 RKNN 无: ['PhoneUse']

### test_0190.jpg
- 参考有而 RKNN 无: ['PhoneUse', 'Drowsy']

### test_0197.jpg
- SafeDriving: conf 0.703 vs 0.492 (Δ0.210)
- SafeDriving: conf 0.500 vs 0.399 (Δ0.101)
- 参考有而 RKNN 无: ['PhoneUse']

### test_0204.jpg
- Drowsy: conf 0.500 vs 0.367 (Δ0.133)

### test_0211.jpg
- SafeDriving: conf 0.703 vs 0.511 (Δ0.191)
- SafeDriving: conf 0.500 vs 0.404 (Δ0.096)

### test_0218.jpg
- SafeDriving: conf 0.703 vs 0.423 (Δ0.280)
- SafeDriving: conf 0.703 vs 0.512 (Δ0.190)

### test_0232.jpg
- Drowsy: conf 0.440 vs 0.373 (Δ0.067)
- 参考有而 RKNN 无: ['PhoneUse']

### test_0239.jpg
- SafeDriving: conf 0.703 vs 0.603 (Δ0.099)

### test_0246.jpg
- Drowsy: conf 0.500 vs 0.316 (Δ0.184)

### test_0253.jpg
- 参考有而 RKNN 无: ['Drowsy']

### test_0267.jpg
- Drowsy: conf 0.500 vs 0.323 (Δ0.177)
- 参考有而 RKNN 无: ['Drowsy']

### test_0274.jpg
- Drowsy: conf 0.500 vs 0.290 (Δ0.210)
- 参考有而 RKNN 无: ['Drowsy']

### test_0281.jpg
- 参考有而 RKNN 无: ['Drowsy']

### test_0288.jpg
- 参考有而 RKNN 无: ['Drowsy']

### test_0295.jpg
- 参考有而 RKNN 无: ['Drowsy']

### test_0302.jpg
- 参考有而 RKNN 无: ['Drowsy']

### test_0309.jpg
- 参考有而 RKNN 无: ['Drowsy']

### test_0316.jpg
- SafeDriving: conf 0.703 vs 0.461 (Δ0.242)
- SafeDriving: conf 0.440 vs 0.490 (Δ0.050)
- 参考有而 RKNN 无: ['SafeDriving']

## 判定建议

- 类别一致率 100% 且 平均 IoU >= 0.9 且 置信度差 <= 0.02: 可进入真机验证
- 出现整类丢失(仅参考检出成批出现): 检查导出方式/量化
- 本报告由 `rk3588_dms/tools/compare_outputs.py` 生成, 原始数据见 `logs/rknn/compare_soham_20260910-201024`
