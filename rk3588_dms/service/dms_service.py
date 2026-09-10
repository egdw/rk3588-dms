#!/usr/bin/env python3
"""RK3588 原生 DMS 服务(集成阶段 STEP 13/14/15)。

职责:
  1. 摄像头独占采集(OpenCV, 预留 MPP/RGA)
  2. MJPEG 流       GET /video.mjpg        (浏览器显示)
  3. 三模型 INT8 三核并行推理, WebSocket /ws/dms/native 推送归一化检测
     (与浏览器 parseDetections 输出同构, 阈值过滤/NMS/融合仍在浏览器)
  4. 报警音频板端播放 POST /alert {key}     (ffplay, 音量 100)
  5. 健康检查       GET /health

只依赖: opencv-python numpy (+rknn-toolkit-lite2 真机)。WebSocket 服务端
为手写最小实现(仅广播文本帧), 不引入第三方包。

用法(RK3588 板上):
  .venv/bin/python rk3588_dms/service/dms_service.py            # 默认 config
  .venv/bin/python rk3588_dms/service/dms_service.py --camera /dev/video0 --infer-fps 12
浏览器:
  http://<板子IP>:8000/index.html?infer=native#/live-detection
  (native 模式请用 http 打开页面, 避免 https 页面的 mixed content 拦截
   MJPEG/WS; 需要时可给本服务挂 HTTPS 反代)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from runtime.base_detector import find_project_root, load_config  # noqa: E402

# 与浏览器 driver-inference.js ALERT_AUDIO_SOURCES 一致(文件在项目根)
ALERT_AUDIO_FILES = {
    "drowsy": "严禁疲劳驾驶，请立即休息.mp3",
    "gaze_off": "视线已偏离，请专注驾驶.mp3",
    "phone": "禁止拨打电话，安全第一.mp3",
    "head_down": "请立即抬头，观察前方.mp3",
}
ALERT_AUDIO_COOLDOWN_MS = 4500  # 与浏览器 ALERT_AUDIO_COOLDOWN_MS 一致

_stop = threading.Event()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------ camera
class CameraThread(threading.Thread):
    """独占采集, 维护最新帧; 打开失败/掉线自动重试并明确记日志。

    test_image: 调试用——循环把一张静态图当帧源(无摄像头环境验证全链路)。
    """

    def __init__(self, device, width, height, fps, test_image: str | None = None):
        super().__init__(daemon=True, name="camera")
        self.device, self.width, self.height, self.fps = device, width, height, fps
        self.test_image = test_image
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._capture = None
        self.fail_count = 0

    @property
    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    def latest(self):
        with self._lock:
            return (self._frame, self._frame_id) if self._frame is not None else (None, 0)

    def run(self) -> None:
        if self.test_image:
            image = cv2.imread(self.test_image, cv2.IMREAD_COLOR)
            if image is None:
                log(f"[CAMERA][FAIL] --test-image 无法读取: {self.test_image}")
                return
            log(f"[CAMERA] 测试图模式: {self.test_image} ({image.shape[1]}x{image.shape[0]}) 循环输出")
            interval = 1.0 / max(1, self.fps)
            while not _stop.is_set():
                with self._lock:
                    self._frame = image.copy()
                    self._frame_id += 1
                _stop.wait(interval)
            return
        backoff = 1.0
        while not _stop.is_set():
            if self._open():
                if self._loop():
                    return  # 正常停止
            if _stop.is_set():
                return
            self.fail_count += 1
            log(f"[CAMERA][WARN] 打开/读取失败(第 {self.fail_count} 次), {backoff:.0f}s 后重试: {self.device}")
            _stop.wait(backoff)
            backoff = min(backoff * 2, 10.0)

    def _open(self) -> bool:
        index = self.device if isinstance(self.device, int) else str(self.device)
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if isinstance(index, str) and index.startswith("/") \
            else cv2.VideoCapture(index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = cap
        actual = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))
        log(f"[CAMERA] 已打开 {self.device}, 实际 w/h/fps = {actual}")
        return True

    def _loop(self) -> bool:
        fails = 0
        while not _stop.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                fails += 1
                if fails == 1:
                    log("[CAMERA][WARN] 读帧失败, 重试中...")
                if fails >= 60:
                    log("[CAMERA][WARN] 连续 60 次读帧失败, 重新打开设备")
                    self._capture.release()
                    return False
                time.sleep(0.05)
                continue
            fails = 0
            with self._lock:
                self._frame = frame
                self._frame_id += 1
        self._capture.release()
        return True


# ------------------------------------------------------------------ audio
class AudioPlayer:
    """报警音频板端播放(ffplay, 音量 100), 每 key 冷却与浏览器一致。"""

    def __init__(self, root: Path):
        self.root = root
        self.ffplay = shutil.which("ffplay")
        self._lock = threading.Lock()
        self._last_played: dict[str, float] = {}
        self._current: subprocess.Popen | None = None
        if not self.ffplay:
            log("[AUDIO][WARN] 未找到 ffplay, 报警音频不可用(apt install ffmpeg)")

    def play(self, key: str) -> dict:
        now = time.time()
        with self._lock:
            if not self.ffplay or key not in ALERT_AUDIO_FILES:
                return {"played": False, "reason": "no ffplay or unknown key"}
            if now - self._last_played.get(key, 0) < ALERT_AUDIO_COOLDOWN_MS / 1000:
                return {"played": False, "reason": "cooldown"}
            file_path = self.root / ALERT_AUDIO_FILES[key]
            if not file_path.exists():
                return {"played": False, "reason": f"missing {file_path.name}"}
            if self._current and self._current.poll() is None:
                self._current.terminate()
            self._last_played[key] = now
            # -volume 100 = 最大音量; -nodisp 无窗口; -autoexit 播完即退
            self._current = subprocess.Popen(
                [self.ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", "-volume", "100", str(file_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            log(f"[AUDIO] 播放报警: {key} -> {file_path.name}")
            return {"played": True}


# ------------------------------------------------------------ websocket hub
class WSHub:
    """极简 WebSocket 广播: 客户端只收文本帧。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._queues: list["queue.SimpleQueue[str]"] = []

    def subscribe(self):
        import queue

        q = queue.SimpleQueue()
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    def broadcast(self, text: str) -> None:
        with self._lock:
            queues = list(self._queues)
        for q in queues:
            q.put(text)


WS_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1(client_key.encode() + WS_GUID).digest()).decode()


def ws_encode_text(payload: str) -> bytes:
    data = payload.encode("utf-8")
    header = bytearray([0x81])  # FIN + text
    length = len(data)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack(">H", length))
    else:
        header.append(127)
        header.extend(struct.pack(">Q", length))
    return bytes(header) + data


def ws_read_frame(conn) -> tuple[int, bytes] | None:
    """读一个客户端帧(只处理 close/ping; 数据帧返回原始 payload 供忽略)。"""
    try:
        head = conn.recv(2)
        if len(head) < 2:
            return None
        opcode = head[0] & 0x0F
        masked = bool(head[1] & 0x80)
        length = head[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", conn.recv(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", conn.recv(8))[0]
        mask = conn.recv(4) if masked else b""
        payload = b""
        while len(payload) < length:
            chunk = conn.recv(length - len(payload))
            if not chunk:
                return None
            payload += chunk
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload
    except (BlockingIOError, InterruptedError):
        return (-1, b"")  # 无数据(非阻塞模式)
    except OSError:
        return None


# ------------------------------------------------------------------ server
class ServiceState:
    camera: CameraThread
    audio: AudioPlayer
    hub = WSHub()
    detections_payload: dict = {"type": "detections", "models": {}, "frame": [0, 0], "inference_ms": 0}
    payload_lock = threading.Lock()
    stats = {"inferences": 0, "started": time.time()}


STATE = ServiceState()


def make_handler(state: ServiceState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # 静默常规访问日志
            pass

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/health":
                frame, frame_id = state.camera.latest()
                body = json.dumps({
                    "ok": True,
                    "camera": frame is not None,
                    "frame_id": frame_id,
                    "clients": len(state.hub._queues),
                    "inferences": state.stats["inferences"],
                    "uptime_s": round(time.time() - state.stats["started"], 1),
                }, ensure_ascii=False).encode()
                self.send_response(HTTPStatus.OK)
                self._cors()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/video.mjpg":
                self._mjpeg()
            elif path == "/ws/dms/native":
                self._websocket()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _mjpeg(self) -> None:
            self.send_response(HTTPStatus.OK)
            self._cors()
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_id = -1
            while not _stop.is_set():
                frame, frame_id = state.camera.latest()
                if frame is None or frame_id == last_id:
                    time.sleep(0.01)
                    continue
                last_id = frame_id
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                data = encoded.tobytes()
                try:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        def _websocket(self) -> None:
            client_key = self.headers.get("Sec-WebSocket-Key")
            if not client_key:
                self.send_error(HTTPStatus.BAD_REQUEST, "missing Sec-WebSocket-Key")
                return
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", ws_accept_key(client_key))
            self._cors()
            self.end_headers()

            conn = self.connection
            conn.setblocking(False)
            queue_obj = state.hub.subscribe()
            log(f"[WS] 客户端接入 ({self.client_address[0]}), 当前 {len(state.hub._queues)} 个")
            try:
                while not _stop.is_set():
                    try:
                        message = queue_obj.get(timeout=0.5)
                        conn.setblocking(True)
                        conn.sendall(ws_encode_text(message))
                        conn.setblocking(False)
                    except Exception:  # queue.Empty
                        pass
                    frame = ws_read_frame(conn)
                    if frame is None:
                        return
                    opcode, payload = frame
                    if opcode == 0x8:  # close
                        return
                    if opcode == 0x9:  # ping -> pong
                        conn.setblocking(True)
                        conn.sendall(b"\x8A\x00")
                        conn.setblocking(False)
            finally:
                state.hub.unsubscribe(queue_obj)
                log(f"[WS] 客户端断开, 剩余 {len(state.hub._queues)} 个")

        def do_POST(self) -> None:
            path = self.path.split("?")[0]
            if path != "/alert":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                payload = {}
            result = state.audio.play(str(payload.get("key") or ""))
            body = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(HTTPStatus.OK)
            self._cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


# --------------------------------------------------------------- inference
def inference_loop(state: ServiceState, model_names, mode, infer_fps) -> None:
    from runtime.parallel import build_group

    group = build_group(model_names, mode=mode)
    interval = 1.0 / max(1, infer_fps)
    last_frame_id = -1
    last_report = time.time()
    inferences = 0

    while not _stop.is_set():
        started = time.perf_counter()
        frame, frame_id = state.camera.latest()
        if frame is None or frame_id == last_frame_id:
            time.sleep(0.01)
            continue
        last_frame_id = frame_id

        results = group.infer_all(frame)
        h, w = frame.shape[:2]
        models_payload = {}
        for name, result in results.items():
            models_payload[name] = [
                {
                    "originalClass": det.class_name,
                    "key": det.unified_key or det.class_name,
                    "confidence": round(float(det.confidence), 4),
                    # 归一化坐标: 浏览器端乘以 videoWidth/Height, 与源帧像素一致
                    "box": [
                        round(det.bbox[0] / w, 5), round(det.bbox[1] / h, 5),
                        round(det.bbox[2] / w, 5), round(det.bbox[3] / h, 5),
                    ],
                }
                for det in result.detections
                if det.confidence >= 0.05  # 低于所有业务阈值的直接丢弃, 省带宽
            ]
        payload = {
            "type": "detections",
            "ts": time.time(),
            "frame_id": frame_id,
            "frame": [w, h],
            "inference_ms": round(group.last_wall_ms, 1),
            "models": models_payload,
        }
        with state.payload_lock:
            state.detections_payload = payload
        state.hub.broadcast(json.dumps(payload, ensure_ascii=False))
        inferences += 1
        state.stats["inferences"] = inferences

        now = time.time()
        if now - last_report >= 5.0:
            counts = {n: len(d) for n, d in models_payload.items()}
            log(f"[STAT] 推理 {inferences} 次 | 墙钟 {group.last_wall_ms:.1f} ms | 检出 {counts}")
            last_report = now

        elapsed = time.perf_counter() - started
        if elapsed < interval:
            time.sleep(interval - elapsed)

    group.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="RK3588 原生 DMS 服务(MJPEG+WS+音频)")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config" / "dms.json"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--infer-fps", type=int, default=None)
    parser.add_argument("--mode", default="device", choices=["auto", "device", "simulator"])
    parser.add_argument("--test-image", default=None, help="调试: 用静态图当帧源(无摄像头验证链路)")
    args = parser.parse_args()

    config = load_config(args.config)
    service_config = config.get("service", {})
    camera_config = config.get("camera", {})
    host = args.host
    port = args.port or int(service_config.get("port", 8600))
    device_arg = args.camera or camera_config.get("device", "/dev/video0")
    camera_device = int(device_arg) if str(device_arg).isdigit() else device_arg
    infer_fps = args.infer_fps or int(service_config.get("inference_fps_target", 12))
    model_names = service_config.get("models_enabled", ["chaitanya", "soham", "coco"])

    log_dir = (find_project_root(PACKAGE_ROOT) / "logs" / "dms")
    log_dir.mkdir(parents=True, exist_ok=True)

    STATE.camera = CameraThread(
        camera_device,
        int(camera_config.get("width", 640)), int(camera_config.get("height", 480)),
        int(camera_config.get("fps", 30)),
        test_image=args.test_image,
    )
    STATE.camera.start()
    STATE.audio = AudioPlayer(find_project_root(PACKAGE_ROOT))

    # HTTP 立即可用(health 反映摄像头状态); 推理线程自己等待首帧
    server = ThreadingHTTPServer((host, port), make_handler(STATE))
    infer_thread = threading.Thread(
        target=inference_loop, args=(STATE, model_names, args.mode, infer_fps),
        daemon=True, name="inference",
    )
    infer_thread.start()

    log(f"[SERVICE] http://{host}:{port}  (MJPEG /video.mjpg | WS /ws/dms/native | 健康检查 /health)")
    log(f"[SERVICE] 浏览器打开: http://<板子IP>:8000/index.html?infer=native#/live-detection")
    try:
        while not _stop.is_set():
            server.handle_request()
    except KeyboardInterrupt:
        log("[SERVICE] Ctrl+C, 正在退出...")
    finally:
        _stop.set()
        server.server_close()
        log("[SERVICE] 已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
