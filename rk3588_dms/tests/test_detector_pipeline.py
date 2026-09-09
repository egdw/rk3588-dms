"""Detector 全链路接线测试: 伪造 RKNN 输出, 验证 letterbox→解码→逆映射→DetectionResult。

不依赖 rknn/rknnlite/onnxruntime —— RKNNModel 被桩替换。
运行: python -m unittest discover -s rk3588_dms/tests -v
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from runtime.base_detector import RknnYoloDetector
from runtime.detection import DetectionResult
from tests.test_postprocess import make_attr_major_matrix


class FakeRKNNModel:
    """返回预置输出的桩。"""

    def __init__(self, outputs):
        self.outputs = outputs
        self.backend = "npu"
        self.npu_status = None
        self.init_ms = 12.0
        self.last_inference_ms = 3.2

    def infer(self, image_rgb_hwc):
        assert image_rgb_hwc.dtype == np.uint8
        assert image_rgb_hwc.shape[2] == 3
        return self.outputs

    def close(self):
        pass


def build_detector(outputs) -> RknnYoloDetector:
    model_config = {
        "classes": ["Cigarette", "Drinking", "Eating", "Phone", "Seatbelt"],
        "unified_classes": {"Phone": "phone", "Seatbelt": "seatbelt"},
        "input_size": [640, 640],
        "letterbox_pad_color": [128, 128, 128],
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
    }
    detector = RknnYoloDetector.__new__(RknnYoloDetector)  # 跳过真实模型加载
    detector.classes = list(model_config["classes"])
    detector.unified_classes = dict(model_config["unified_classes"])
    detector.input_size = tuple(model_config["input_size"])
    detector.pad_color = tuple(model_config["letterbox_pad_color"])
    detector.confidence_threshold = model_config["confidence_threshold"]
    detector.iou_threshold = model_config["iou_threshold"]
    detector.model = FakeRKNNModel(outputs)
    return detector


class TestDetectorPipeline(unittest.TestCase):
    def test_end_to_end_detection(self):
        # 在 640x640 输入坐标里放一个 Phone(类别 3): 中心(320,240) 宽 160 高 120
        mat = make_attr_major_matrix([(320, 240, 160, 120, 0.9, 3)])
        detector = build_detector([mat[None]])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 4:3 源, letterbox 无灰条
        result = detector.infer(frame)

        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.model, "base")
        self.assertEqual(result.detection_count, 1)
        det = result.detections[0]
        self.assertEqual(det.class_name, "Phone")
        self.assertEqual(det.class_id, 3)
        self.assertAlmostEqual(det.confidence, 0.9, places=5)
        self.assertEqual(det.unified_key, "phone")
        # 逆映射回原图: 4:3(640x480) 进 640x640 画布, 上下各 80px 灰条
        self.assertAlmostEqual(det.bbox[0], 320 - 80, delta=1.0)
        self.assertAlmostEqual(det.bbox[1], 240 - 80 - 60, delta=1.0)
        self.assertAlmostEqual(det.bbox[2], 320 + 80, delta=1.0)
        self.assertAlmostEqual(det.bbox[3], 240 - 80 + 60, delta=1.0)
        self.assertEqual(result.source_size, [640, 480])
        self.assertEqual(result.backend, "npu")

    def test_result_dict_schema(self):
        mat = make_attr_major_matrix([(320, 320, 100, 100, 0.8, 4)])  # Seatbelt
        detector = build_detector([mat[None]])
        result = detector.infer(np.zeros((640, 640, 3), dtype=np.uint8))
        payload = result.to_dict()
        for key in ("model", "timestamp", "inference_ms", "detections", "detection_count"):
            self.assertIn(key, payload)
        det = payload["detections"][0]
        self.assertEqual(det["class_name"], "Seatbelt")
        self.assertEqual(det["unified_key"], "seatbelt")
        self.assertEqual(len(det["bbox"]), 4)

    def test_letterbox_pad_geometry(self):
        # 16:9 源图: letterbox 后上下灰条, 检测框应正确逆映射回 16:9 原图坐标
        mat = make_attr_major_matrix([(320, 320, 100, 100, 0.9, 0)])
        detector = build_detector([mat[None]])
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = detector.infer(frame)
        det = result.detections[0]
        # scale = 640/640 = 1(宽约束), pad_y = 140 -> 原图 y = 输入 y - 140
        self.assertAlmostEqual(det.bbox[1], 320 - 140 - 50, delta=1.0)
        self.assertAlmostEqual(det.bbox[3], 320 - 140 + 50, delta=1.0)
        self.assertAlmostEqual(det.bbox[0], 270, delta=1.0)


if __name__ == "__main__":
    unittest.main()
