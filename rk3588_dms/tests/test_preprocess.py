"""preprocess 单元测试: letterbox 与浏览器 canvas 行为对齐。

运行: python -m unittest discover -s rk3588_dms/tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from runtime.preprocess import letterbox, unletterbox_box, unletterbox_point


class TestLetterbox(unittest.TestCase):
    def test_output_shape_and_pad_color(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        prep = letterbox(frame, (640, 640), pad_color=(128, 128, 128))
        self.assertEqual(prep.image.shape, (640, 640, 3))
        # 4:3 源 -> 640x480 内容, 上下各 80 行灰底 128
        self.assertTrue((prep.image[:79] == 128).all())
        self.assertTrue((prep.image[561:] == 128).all())
        self.assertTrue((prep.image[80:560] == 0).all())
        self.assertAlmostEqual(prep.scale, 640 / 640, places=6)

    def test_wide_frame(self):
        frame = np.zeros((360, 1280, 3), dtype=np.uint8)
        prep = letterbox(frame, (640, 640))
        self.assertAlmostEqual(prep.scale, 640 / 1280, places=6)  # 宽图受宽度约束
        # 宽图(16:4.5)缩放后 640x180: 内容满宽, 灰条在上下
        content_h = round(360 * prep.scale)  # 180
        top = round((640 - content_h) / 2)
        self.assertTrue((prep.image[: top - 1] == 128).all())
        self.assertTrue((prep.image[top + content_h + 1 :] == 128).all())
        self.assertTrue((prep.image[top : top + content_h] == 0).all())

    def test_inverse_mapping_roundtrip(self):
        frame = np.zeros((300, 700, 3), dtype=np.uint8)
        prep = letterbox(frame, (640, 640))
        # 原图 (10, 20)-(310, 220) 映射到输入再映射回来
        src_box = [10.0, 20.0, 310.0, 220.0]
        mapped = [
            src_box[0] * prep.scale + prep.pad_x,
            src_box[1] * prep.scale + prep.pad_y,
            src_box[2] * prep.scale + prep.pad_x,
            src_box[3] * prep.scale + prep.pad_y,
        ]
        back = unletterbox_box(mapped, prep)
        for a, b in zip(src_box, back):
            self.assertAlmostEqual(a, b, delta=1.5)  # resize 取整误差

    def test_point_inverse_clamped(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        prep = letterbox(frame, (640, 640))
        x, y = unletterbox_point((-50, -50), prep)
        self.assertEqual((x, y), (0.0, 0.0))

    def test_float_nchw_layout(self):
        frame = np.full((640, 640, 3), 255, dtype=np.uint8)
        prep = letterbox(frame, (640, 640))
        tensor = prep.to_float_nchw()
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertAlmostEqual(float(tensor.max()), 1.0, places=6)
        self.assertAlmostEqual(float(tensor.min()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
