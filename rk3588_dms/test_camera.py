#!/usr/bin/env python3
"""RKNN 摄像头实时推理测试(STEP 7, 完全独立)。

用法(RK3588 板上):
  python rk3588_dms/test_camera.py --camera /dev/video0
  python rk3588_dms/test_camera.py --camera 0 --model-name chaitanya   # OpenCV 索引也可以
  python rk3588_dms/test_camera.py --show                              # 带画面窗口

行为:
  - 第一阶段用 OpenCV 采集; 采集层已抽象为 CameraSource, 预留以后替换
    MPP(RGA) 零拷贝实现, 不在本阶段引入复杂原生代码。
  - 实时输出: camera FPS / inference FPS / 端到端延迟 / 检测结果(每 5 秒汇总一次)。
  - Ctrl+C 安全退出; 摄像头打不开/掉线/读帧失败都有明确日志与有限次重试,
    不会无提示死循环。

注意: 若浏览器正在用 getUserMedia 占用 USB 摄像头, 本程序会 Device busy ——
单独测试 RKNN 时请先关闭浏览器摄像头(或先不打开检测页)。
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

try:
    import cv2
except ImportError:
    print("[FAIL] 缺少 opencv-python: pip install opencv-python", file=sys.stderr)
    sys.exit(2)


# ------------------------------------------------------------- camera layer
@dataclass
class Frame:
    image: object            # BGR ndarray
    timestamp: float         # 采集时刻(perf_counter 秒)
    sequence: int = 0


class CameraSource:
    """采集层抽象: 当前 OpenCV 实现, 未来可替换 MPP/RGA 零拷贝实现。"""

    def __init__(self, device, width: int = 640, height: int = 480, fps: int = 30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._sequence = 0

    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> Optional[Frame]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class OpenCVCameraSource(CameraSource):
    """cv2.VideoCapture 实现(第一阶段)。"""

    def __init__(self, device, width=640, height=480, fps=30):
        super().__init__(device, width, height, fps)
        self.capture: Optional[cv2.VideoCapture] = None
        self.fail_streak = 0

    def open(self) -> bool:
        index = self.device if isinstance(self.device, int) else str(self.device)
        self.capture = cv2.VideoCapture(index, cv2.CAP_V4L2) if isinstance(index, str) and index.startswith("/") \
            else cv2.VideoCapture(index)
        if not self.capture.isOpened():
            print(f"[CAMERA][FAIL] 无法打开摄像头: {self.device} "
                  "(检查设备存在/权限/是否被浏览器 getUserMedia 独占)")
            return False
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        actual = (self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                  self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                  self.capture.get(cv2.CAP_PROP_FPS))
        print(f"[CAMERA] 已打开 {self.device}, 实际参数 w/h/fps = {actual}")
        self.fail_streak = 0
        return True

    def read(self) -> Optional[Frame]:
        if self.capture is None:
            return None
        ok, image = self.capture.read()
        if not ok or image is None:
            self.fail_streak += 1
            if self.fail_streak == 1:
                print("[CAMERA][WARN] 读帧失败, 开始重试(最多 50 次)...")
            elif self.fail_streak % 10 == 0:
                print(f"[CAMERA][WARN] 连续读帧失败 {self.fail_streak} 次")
            if self.fail_streak >= 50:
                print("[CAMERA][FAIL] 摄像头疑似掉线(连续 50 次读帧失败), 退出")
                return None
            time.sleep(0.05)
            return None
        self.fail_streak = 0
        self._sequence += 1
        return Frame(image=image, timestamp=time.perf_counter(), sequence=self._sequence)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            print("[CAMERA] 已释放")


# class MPPCameraSource(CameraSource):
#     """预留: RK3588 MPP + RGA 零拷贝采集(后续性能阶段实现, 接口保持不变)。"""
#     ...


# ------------------------------------------------------------------- stats
@dataclass
class LoopStats:
    started_at: float = field(default_factory=time.perf_counter)
    frames: int = 0
    inferences: int = 0
    inference_ms: List[float] = field(default_factory=list)
    last_report_at: float = field(default_factory=time.perf_counter)
    last_detections: List[str] = field(default_factory=list)
    end_to_end_ms: List[float] = field(default_factory=list)

    def note_inference(self, inference_ms: float, latency_ms: float, labels: List[str]) -> None:
        self.inferences += 1
        self.inference_ms.append(inference_ms)
        self.end_to_end_ms.append(latency_ms)
        self.last_detections = labels

    def maybe_report(self, interval: float = 5.0) -> bool:
        now = time.perf_counter()
        if now - self.last_report_at < interval:
            return False
        elapsed = now - self.started_at
        camera_fps = self.frames / elapsed if elapsed > 0 else 0.0
        infer_fps = self.inferences / elapsed if elapsed > 0 else 0.0
        avg = sum(self.inference_ms) / len(self.inference_ms) if self.inference_ms else 0.0
        e2e = sum(self.end_to_end_ms) / len(self.end_to_end_ms) if self.end_to_end_ms else 0.0
        print(
            f"[STAT] camera {camera_fps:5.1f} FPS | inference {infer_fps:5.1f} FPS | "
            f"avg infer {avg:6.1f} ms | e2e {e2e:6.1f} ms | detections: "
            + (", ".join(self.last_detections) if self.last_detections else "none")
        )
        self.last_report_at = now
        self.frames = 0
        self.inferences = 0
        self.inference_ms = []
        self.end_to_end_ms = []
        self.started_at = now
        return True


# -------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="RKNN 摄像头实时测试")
    parser.add_argument("--camera", default="/dev/video0", help="/dev/videoN 或 OpenCV 索引(如 0)")
    parser.add_argument("--model", help=".rknn 路径(缺省读 config)")
    parser.add_argument("--model-name", default="chaitanya", choices=["chaitanya", "soham", "coco"])
    parser.add_argument("--mode", default="auto", choices=["auto", "device", "simulator"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--infer-interval-ms", type=int, default=80,
                        help="推理最小间隔(默认 80ms ≈ 12.5 FPS, 比赛建议 AI 10~15 FPS)")
    parser.add_argument("--show", action="store_true", help="显示画面窗口(需要桌面)")
    args = parser.parse_args()

    camera_id = int(args.camera) if str(args.camera).isdigit() else args.camera

    from test_image import build_detector

    print("[INIT] 加载 RKNN 模型...")
    detector = build_detector(args.model_name, args.model, args.mode)
    print(f"[INIT] 模型就绪: {detector.model_path} (backend={detector.model.backend}, "
          f"init {detector.model.init_ms:.0f} ms)")

    camera = OpenCVCameraSource(camera_id, args.width, args.height, args.fps)
    if not camera.open():
        detector.close()
        return 1

    stop = {"flag": False}

    def request_stop(signum, frame):  # noqa: ANN001
        stop["flag"] = True
        print(f"\n[EXIT] 收到信号 {signum}, 正在安全退出...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    stats = LoopStats()
    last_infer_at = 0.0
    exit_code = 0
    print("[RUN] Ctrl+C 退出; 每 5 秒输出一次统计")

    try:
        while not stop["flag"]:
            frame = camera.read()
            if frame is None:
                # 连续失败到上限时 read 内部已提示; 这里判断是否真的掉线退出
                if camera.fail_streak >= 50:
                    exit_code = 3
                    break
                continue

            stats.frames += 1
            now = time.perf_counter()
            if now - last_infer_at < args.infer_interval_ms / 1000.0:
                continue
            last_infer_at = now

            result = detector.infer(frame.image)
            labels = [f"{d.class_name} {d.confidence:.2f}" for d in result.detections]
            latency_ms = (time.perf_counter() - frame.timestamp) * 1000.0
            stats.note_inference(result.inference_ms, latency_ms, labels)

            if args.show:
                canvas = frame.image.copy()
                for det in result.detections:
                    x1, y1, x2, y2 = (int(v) for v in det.bbox)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(canvas, f"{det.class_name} {det.confidence:.2f}",
                                (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("rknn camera test", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("[EXIT] 窗口退出键")
                    break

            stats.maybe_report(5.0)
    finally:
        camera.close()
        detector.close()
        if args.show:
            cv2.destroyAllWindows()
        print("[EXIT] 资源已释放" + ("" if exit_code == 0 else f" (exit={exit_code})"))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
