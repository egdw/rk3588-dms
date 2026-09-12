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
  .venv/bin/python rk3588_dms/service/dms_service.py            # 摄像头自动扫描, 可选 --camera /dev/videoN 指定
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
import socket
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
def candidate_camera_devices() -> list[str]:
    """所有 /dev/video* 候选, 按序号升序(数字节点优先, 排除 video-dec/enc 等命名节点)。"""
    import glob
    import re as _re

    paths = glob.glob("/dev/video*")

    def sort_key(path: str):
        m = _re.search(r"(\d+)$", path)
        return (0, int(m.group(1))) if m else (1, path)

    return sorted(paths, key=sort_key)


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
        # 指定路径优先; 打不开(设备不存在/被占/是 metadata 节点)则自动扫描所有 /dev/video*
        explicit = self.device if isinstance(self.device, str) and self.device.startswith("/") else None
        if explicit is None and isinstance(self.device, int):
            candidates: list = [self.device]
        else:
            candidates = ([explicit] if explicit else []) + candidate_camera_devices()
        seen: set = set()
        for index in candidates:
            key = str(index)
            if key in seen:
                continue
            seen.add(key)
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if isinstance(index, str) and index.startswith("/") \
                else cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()  # 读一帧验证: metadata/无法采集的节点在这里被淘汰
            if not ok or frame is None:
                cap.release()
                continue
            if explicit is not None and index != explicit:
                log(f"[CAMERA] 指定设备 {explicit} 打不开, 自动搜索命中 {index}")
            # UVC 摄像头默认 YUY2 常被限 10fps; 请求 MJPG 格式通常可解锁 30fps
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小驱动缓冲, 降低取帧滞后(不支持则忽略)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.device = index
            self._capture = cap
            actual = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))
            log(f"[CAMERA] 已打开 {index}(MJPG fourcc 已请求), 实际 w/h/fps = {actual}")
            return True
        return False

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

    def play(self, key: str, force: bool = False) -> dict:
        now = time.time()
        with self._lock:
            if not self.ffplay or key not in ALERT_AUDIO_FILES:
                return {"played": False, "reason": "no ffplay or unknown key"}
            if not force and now - self._last_played.get(key, 0) < ALERT_AUDIO_COOLDOWN_MS / 1000:
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


# ------------------------------------------------------------ mqtt command
# 板端直接订阅 MQTT(纯 stdlib TCP, 免浏览器 wss 自签证书问题):
# 命令 -> 后端 /api/manual-control(顶栏状态, 浏览器 800ms 轮询跟随) + 板端直接放报警音
MQTT_COMMAND_ALIASES = {
    "manual_enter": ("manual", ""), "enter": ("manual", ""), "manual": ("manual", ""),
    "clear": ("manual", ""),
    "manual_exit": ("auto", ""), "exit": ("auto", ""), "auto": ("auto", ""),
    "safe": ("manual", "safe"), "normal": ("manual", "safe"),
    "drowsy": ("manual", "drowsy"), "eye": ("manual", "drowsy"), "eye_closure": ("manual", "drowsy"),
    "head": ("manual", "head_down"), "head_down": ("manual", "head_down"),
    "gaze": ("manual", "gaze_off"), "gaze_off": ("manual", "gaze_off"),
    "phone": ("manual", "phone"), "phone_call": ("manual", "phone"), "call": ("manual", "phone"),
}


def decode_mqtt_command(payload: bytes):
    """返回 (mode, key) 或 None。兼容 {"command": "..."} / {"mode","key"} / 裸字符串。"""
    import json as _json

    text = payload.decode("utf-8", "ignore").strip()
    if not text:
        return None
    try:
        data = _json.loads(text)
    except ValueError:
        data = {"command": text}
    if not isinstance(data, dict):
        return None
    if str(data.get("mode", "")).strip().lower() in ("manual", "auto"):
        return (str(data.get("mode")).strip().lower(), str(data.get("key", "")).strip().lower())
    raw = str(data.get("command") or data.get("key") or data.get("state") or data.get("status") or "")
    command = raw.strip().lower().replace("\\", "")
    command = "_".join(part for part in command.replace("-", " ").split())
    return MQTT_COMMAND_ALIASES.get(command)


class MqttCommandThread(threading.Thread):
    """订阅 vision-sentinel/control, 命令驱动手动控制状态与板端报警音。"""

    def __init__(self, host, port, topic, backend_url, audio: AudioPlayer):
        super().__init__(daemon=True, name="mqtt-command")
        self.host, self.port, self.topic = host, port, topic
        self.backend_url = backend_url.rstrip("/")
        self.audio = audio
        self._socket = None

    def run(self) -> None:
        import urllib.request

        backoff = 2.0
        while not _stop.is_set():
            try:
                self._socket = socket.create_connection((self.host, self.port), timeout=10)
                self._mqtt_handshake()
                log(f"[MQTT] 已连接 {self.host}:{self.port}, 订阅 {self.topic}")
                backoff = 2.0
                self._read_loop()
            except Exception as exc:
                if _stop.is_set():
                    return
                log(f"[MQTT] 连接断开({exc}), {backoff:.0f}s 后重试")
                _stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    # ---- 极简 MQTT v311 客户端(仅 CONNECT/SUBSCRIBE/PUBLISH/PING) ----
    @staticmethod
    def _enc_len(n: int) -> bytes:
        out = b""
        while True:
            b = n % 128
            n //= 128
            out += bytes([b | 0x80 if n else b])
            if not n:
                return out

    @staticmethod
    def _mstr(s: str) -> bytes:
        raw = s.encode()
        return struct.pack(">H", len(raw)) + raw

    def _packet(self, body: bytes, typ: int) -> None:
        self._socket.sendall(bytes([typ]) + self._enc_len(len(body)) + body)

    def _mqtt_handshake(self) -> None:
        self._socket.settimeout(30)
        body = self._mstr("MQTT") + bytes([4, 2]) + struct.pack(">H", 60) + self._mstr("rk3588-dms-%d" % os.getpid())
        self._packet(body, 0x10)
        head, resp = self._recv_packet()
        if head is None or head >> 4 != 2 or not resp or resp[-1] != 0:
            raise RuntimeError(f"CONNACK 拒绝: {resp.hex() if resp else '空'}")
        self._packet(struct.pack(">H", 1) + self._mstr(self.topic) + bytes([1]), 0x82)  # QoS1 订阅
        head, resp = self._recv_packet()
        if head is None or head >> 4 != 9:
            raise RuntimeError(f"SUBACK 异常: {resp.hex() if resp else '空'}")

    def _recv_packet(self):
        """返回 (完整固定头字节, body) 或 (None, None)。"""
        header = self._recv_exact(1)
        if not header:
            return None, None
        mult, val = 1, 0
        while True:
            b = self._recv_exact(1)
            if b is None:
                return None, None
            val += (b[0] & 127) * mult
            mult *= 128
            if not b[0] & 128:
                break
        body = self._recv_exact(val)
        if body is None:
            return None, None
        return header[0], body

    def _recv_exact(self, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            chunk = self._socket.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _read_loop(self) -> None:
        import urllib.request

        while not _stop.is_set():
            head, body = self._recv_packet()
            if head is None:
                raise RuntimeError("连接被关闭")
            if head >> 4 != 3:  # 只关心 PUBLISH
                continue
            tl = struct.unpack(">H", body[0:2])[0]
            topic = body[2:2 + tl].decode("utf-8", "ignore")
            qos = (head >> 1) & 3
            payload_start = 2 + tl + (2 if qos else 0)
            payload = body[payload_start:]
            if qos == 1 and payload_start >= 4:
                pid = struct.unpack(">H", body[2 + tl:payload_start])[0]
                try:
                    self._packet(struct.pack(">H", pid), 0x40)  # PUBACK, 防 broker 重发
                except OSError:
                    pass
            if topic != self.topic:
                continue
            decoded = decode_mqtt_command(payload)
            if not decoded:
                log(f"[MQTT] 忽略无法识别的命令: {payload[:60]!r}")
                continue
            mode, key = decoded
            log(f"[MQTT] 命令: mode={mode} key={key or '-'}")
            try:
                request = urllib.request.Request(
                    f"{self.backend_url}/api/manual-control",
                    data=json.dumps({"mode": mode, "key": key, "source": "mqtt"}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                urllib.request.urlopen(request, timeout=5).read()
            except Exception as exc:
                log(f"[MQTT][WARN] 更新手动控制失败: {exc}")
            if key in ALERT_AUDIO_FILES:
                # 手动命令: 每条必响, 不走自动冷却
                self.audio.play(key, force=True)


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
class SharedJpeg:
    """同一帧只编码一次, 多个 MJPEG 客户端共享字节(此前每客户端各编一遍)。"""

    def __init__(self, camera: CameraThread, quality: int = 85):
        self.camera = camera
        self.quality = quality
        self._lock = threading.Lock()
        self._frame_id = -1
        self._data: bytes | None = None

    def get(self) -> bytes | None:
        frame, frame_id = self.camera.latest()
        if frame is None:
            return None
        with self._lock:
            if frame_id != self._frame_id:
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    return None
                self._data = encoded.tobytes()
                self._frame_id = frame_id
            return self._data


class ServiceState:
    camera: CameraThread
    audio: AudioPlayer
    hub = WSHub()
    jpeg: SharedJpeg
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
            last_data = None
            while not _stop.is_set():
                data = state.jpeg.get()
                if data is None or data is last_data:
                    time.sleep(0.005)
                    continue
                last_data = data
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
                # 服务端预过滤: 对齐浏览器阈值(phone 0.20 / 其余 0.25)并留 1 分缓冲,
                # 之下必被浏览器丢弃 —— 只省传输/解析, 不改变任何判定行为
                if det.confidence >= (0.19 if (det.unified_key or "") == "phone" else 0.24)
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
    # 默认 auto: 自动扫描 /dev/video* 找第一个真正能出图的节点; 显式传 --camera 则优先尝试它
    device_arg = args.camera or camera_config.get("device", "auto")
    camera_device = int(device_arg) if str(device_arg).isdigit() else (None if device_arg == "auto" else device_arg)
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
    STATE.jpeg = SharedJpeg(STATE.camera)
    STATE.audio = AudioPlayer(find_project_root(PACKAGE_ROOT))

    # MQTT 手动控制: 默认订阅 192.168.2.13:1883 的 vision-sentinel/control(DMS_MQTT_HOST= 空可关)
    mqtt_host = os.environ.get("DMS_MQTT_HOST", "192.168.2.13")
    if mqtt_host:
        MqttCommandThread(
            mqtt_host,
            int(os.environ.get("DMS_MQTT_PORT", "1883")),
            os.environ.get("DMS_MQTT_TOPIC", "vision-sentinel/control"),
            os.environ.get("DMS_BACKEND_URL", "http://127.0.0.1:8080"),
            STATE.audio,
        ).start()

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
