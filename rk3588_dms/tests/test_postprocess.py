"""postprocess 单元测试: 解码语义与浏览器 parseDetections 对齐。

运行: python -m unittest discover -s rk3588_dms/tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from runtime.postprocess import (
    decode_rknn_modelzoo_branches,
    decode_ultralytics_output,
    dfl_decode,
    iou,
    nms,
)

NC = 5  # chaitanya 类别数


def make_attr_major_matrix(boxes, num_classes=NC, with_objectness=False, normalized=False, n=8400):
    """构造 [A, N] 布局的合成输出。

    boxes: [(cx, cy, w, h, score, class_id), ...] 坐标为 640 输入像素(normalized=True 时为 0~1)
    """
    class_start = 5 if with_objectness else 4
    attr = class_start + num_classes
    mat = np.zeros((attr, n), dtype=np.float32)
    if with_objectness:
        mat[4, :] = 1.0
    for index, (cx, cy, w, h, score, class_id) in enumerate(boxes):
        scale = 1.0 / 640.0 if normalized else 1.0
        mat[0, index] = cx * scale
        mat[1, index] = cy * scale
        mat[2, index] = w * scale
        mat[3, index] = h * scale
        mat[class_start + class_id, index] = score
    return mat


class TestDecodeUltralytics(unittest.TestCase):
    def test_attr_major_pixels(self):
        mat = make_attr_major_matrix([(320, 320, 100, 200, 0.9, 3)])
        result = decode_ultralytics_output(mat[None], NC, input_size=640, confidence_threshold=0.25)
        self.assertEqual(len(result), 1)
        (box, score, class_id) = result[0]
        self.assertAlmostEqual(box[0], 270.0, places=4)
        self.assertAlmostEqual(box[1], 220.0, places=4)
        self.assertAlmostEqual(box[2], 370.0, places=4)
        self.assertAlmostEqual(box[3], 420.0, places=4)
        self.assertEqual(class_id, 3)
        self.assertAlmostEqual(score, 0.9, places=6)

    def test_candidate_major_equivalence(self):
        boxes = [(100, 200, 50, 80, 0.8, 1), (400, 300, 120, 60, 0.5, 0)]
        attr_major = make_attr_major_matrix(boxes)
        result_a = decode_ultralytics_output(attr_major[None], NC, input_size=640)
        result_b = decode_ultralytics_output(attr_major.T[None], NC, input_size=640)  # [1, N, A]
        self.assertEqual(len(result_a), len(result_b))
        for (ba, sa, ca), (bb, sb, cb) in zip(result_a, result_b):
            self.assertEqual(ca, cb)
            self.assertAlmostEqual(sa, sb, places=5)
            for va, vb in zip(ba, bb):
                self.assertAlmostEqual(va, vb, places=3)

    def test_objectness_multiplies(self):
        mat = make_attr_major_matrix([(320, 320, 100, 100, 0.8, 2)], with_objectness=True)
        mat[4, 0] = 0.5  # objectness
        result = decode_ultralytics_output(mat[None], NC, input_size=640)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][1], 0.4, places=6)

    def test_normalized_coordinates_auto_scaled(self):
        # 传入像素坐标, helper 按 normalized=True 除以 640 存成归一化值(均 <=2)
        mat = make_attr_major_matrix([(320, 320, 160, 320, 0.9, 0)], normalized=True)
        result = decode_ultralytics_output(mat[None], NC, input_size=640)
        self.assertEqual(len(result), 1)
        box = result[0][0]
        self.assertAlmostEqual(box[0], 320 - 80, places=3)  # (0.5 - 0.125) * 640 = 240

    def test_threshold_filters(self):
        mat = make_attr_major_matrix([(320, 320, 100, 100, 0.10, 0)])
        result = decode_ultralytics_output(mat[None], NC, input_size=640, confidence_threshold=0.25)
        self.assertEqual(result, [])

    def test_nms_suppresses_overlap(self):
        mat = make_attr_major_matrix([
            (320, 320, 100, 100, 0.9, 0),
            (322, 321, 100, 100, 0.8, 0),   # 同类高重叠 -> 应被抑制
            (100, 100, 50, 50, 0.7, 1),     # 不同类不受影响
        ])
        result = decode_ultralytics_output(mat[None], NC, input_size=640)
        self.assertEqual(len(result), 2)
        self.assertEqual({r[2] for r in result}, {0, 1})

    def test_attr_count_mismatch_raises(self):
        mat = np.zeros((6, 100), dtype=np.float32)  # 6 < 4+5
        with self.assertRaises(ValueError):
            decode_ultralytics_output(mat[None], NC, input_size=640)


class TestNmsHelpers(unittest.TestCase):
    def test_iou_identical(self):
        box = [0, 0, 10, 10]
        self.assertAlmostEqual(iou(box, box), 1.0)

    def test_iou_disjoint(self):
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_group_key_nms(self):
        dets = [
            ([0, 0, 10, 10], 0.9, 0),
            ([1, 1, 11, 11], 0.8, 1),   # 不同 class_id 但同一统一 key -> 应被分组抑制
        ]
        kept = nms(dets, 0.45, group_key=lambda cid: "phone")
        self.assertEqual(len(kept), 1)


class TestModelZooBranches(unittest.TestCase):
    def test_dfl_decode_one_hot(self):
        dist = np.zeros((1, 64), dtype=np.float32)
        # l=4, t=8, r=12, b=15 -> one-hot bins(其余 bin=0, softmax 后近似 one-hot)
        dist[0, 0 * 16 + 4] = 10.0
        dist[0, 1 * 16 + 8] = 10.0
        dist[0, 2 * 16 + 12] = 10.0
        dist[0, 3 * 16 + 15] = 10.0
        out = dfl_decode(dist)
        self.assertAlmostEqual(float(out[0, 0]), 4.0, delta=0.01)
        self.assertAlmostEqual(float(out[0, 1]), 8.0, delta=0.01)
        self.assertAlmostEqual(float(out[0, 2]), 12.0, delta=0.01)
        self.assertAlmostEqual(float(out[0, 3]), 15.0, delta=0.01)

    def test_branch_decode_recovers_box(self):
        # 在 stride=8 的 (gx=10, gy=10) 格子放一个目标, 类别 2, 分数 0.9
        nc = NC
        size = 640
        feats = []
        target_stride = 8
        for stride, feat_size in ((8, 80), (16, 40), (32, 20)):
            arr = np.full((64 + nc, feat_size, feat_size), -6.0, dtype=np.float32)  # sigmoid(-6)≈0.0025
            feats.append(arr)
        arr = feats[0]
        gx = gy = 10
        cx = (gx + 0.5) * target_stride
        cy = (gy + 0.5) * target_stride
        l = t = r = b = 4  # 整数距离(单位: 格子), one-hot DFL 精确还原
        for i, dist in enumerate((l, t, r, b)):
            arr[i * 16 + dist, gy, gx] = 20.0  # softmax 后近似 one-hot
        arr[64 + 2, gy, gx] = 2.2  # sigmoid(2.2)≈0.9

        result = decode_rknn_modelzoo_branches(feats, nc, input_size=size, confidence_threshold=0.5)
        self.assertEqual(len(result), 1)
        box, score, class_id = result[0]
        self.assertEqual(class_id, 2)
        self.assertGreater(score, 0.85)
        # DFL 距离单位是格子: 像素距离 = 格子数 * stride
        self.assertAlmostEqual(box[0], cx - l * target_stride, delta=0.3)
        self.assertAlmostEqual(box[1], cy - t * target_stride, delta=0.3)
        self.assertAlmostEqual(box[2], cx + r * target_stride, delta=0.3)
        self.assertAlmostEqual(box[3], cy + b * target_stride, delta=0.3)


if __name__ == "__main__":
    unittest.main()
