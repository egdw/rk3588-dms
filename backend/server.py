#!/usr/bin/env python3
"""Vision Sentinel backend.

This server intentionally uses only the Python standard library so it can run
before the project installs ML dependencies. If Ultralytics YOLO is available
in the active environment, training jobs will call it; otherwise the job fails
with an actionable message instead of silently simulating real training.
"""

from __future__ import annotations

import base64
import binascii
import json
import csv
import hashlib
import mimetypes
import os
import random
import re
import shutil
import sqlite3
import ssl
import subprocess
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "storage"
DB_PATH = STORAGE / "vision_sentinel.sqlite3"

DATASETS = STORAGE / "datasets"
ANNOTATIONS = STORAGE / "annotations"
PROCESSED = STORAGE / "processed"
MODELS = STORAGE / "models"
EXPORTS = STORAGE / "exports"
JOBS = STORAGE / "jobs"
ANNOTATION_TRACKS = STORAGE / "annotation-tracks"

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
SIXDREPNET_ROOT = ROOT / "6DRepNet-master"
SIXDREPNET_DEFAULT_WEIGHTS = [
    SIXDREPNET_ROOT / "6DRepNet_300W_LP_AFLW2000.pth",
    SIXDREPNET_ROOT / "output" / "snapshots" / "6DRepNet_300W_LP_AFLW2000.pth",
    ROOT / "Driver-Monitoring-System" / "models" / "6drepnet" / "6DRepNet_300W_LP_AFLW2000.pth",
]
BUNDLED_MODELS = {
    "model_bundled_soham": ROOT / "Driver-Monitoring-System/models/soham/best.pt",
    "model_bundled_chaitanya": ROOT / "Driver-Monitoring-System/models/chaitanya/best.pt",
}
SIXDREPNET_LOCK = threading.Lock()
SIXDREPNET_DETECTOR = None
SIXDREPNET_ERROR = ""

STATE_LABELS = [
    "Normal Driving",
    "Phone Call",
    "Eye Closure/Sleep",
    "Head-Down Distraction",
    "Looking Aside",
    "Smoking/Eating",
    "No Seatbelt",
]

CONTROL_KEY_ALIASES = {
    "normal": "safe",
    "safe": "safe",
    "0": "",
    "clear": "",
    "off": "",
    "none": "",
    "phone": "phone",
    "phone_call": "phone",
    "call": "phone",
    "drowsy": "drowsy",
    "eye": "drowsy",
    "eye_closure": "drowsy",
    "head": "head_down",
    "head_down": "head_down",
    "gaze": "gaze_off",
    "gaze_off": "gaze_off",
}
CONTROL_LABELS = {
    "safe": "Normal Driving",
    "phone": "Phone Call",
    "drowsy": "Eye Closure",
    "head_down": "Head Down",
    "gaze_off": "Gaze Offset",
}
CONTROL_LOCK = threading.Lock()
MANUAL_CONTROL = {
    "enabled": False,
    "key": "",
    "label": "",
    "confidence": 0.99,
    "source": "manual",
    "updated_at": 0,
    "expires_at": 0,
}

STATIC_LONG_CACHE_SUFFIXES = {
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".onnx",
    ".png",
    ".pth",
    ".pt",
    ".task",
    ".wasm",
    ".webp",
}


def now() -> int:
    return int(time.time())


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def ensure_storage() -> None:
    for directory in [STORAGE, DATASETS, ANNOTATIONS, PROCESSED, MODELS, EXPORTS, JOBS, ANNOTATION_TRACKS]:
        directory.mkdir(parents=True, exist_ok=True)


def safe_annotation_track_path(name: str) -> Path | None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.json", name):
        return None
    target = (ANNOTATION_TRACKS / name).resolve()
    if not str(target).startswith(str(ANNOTATION_TRACKS.resolve())):
        return None
    return target


def list_annotation_tracks() -> list[dict]:
    items = []
    for target in ANNOTATION_TRACKS.glob("*.json"):
        stat = target.stat()
        items.append({"name": target.name, "size": stat.st_size, "modified": int(stat.st_mtime)})
    return sorted(items, key=lambda item: item["modified"], reverse=True)


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    ensure_storage()
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              source TEXT NOT NULL,
              description TEXT NOT NULL DEFAULT '',
              created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS samples (
              id TEXT PRIMARY KEY,
              dataset_id TEXT NOT NULL,
              filename TEXT NOT NULL,
              path TEXT NOT NULL,
              media_type TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'raw',
              label TEXT,
              created_at INTEGER NOT NULL,
              FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );

            CREATE TABLE IF NOT EXISTS annotations (
              id TEXT PRIMARY KEY,
              dataset_id TEXT NOT NULL,
              sample_id TEXT,
              task_type TEXT NOT NULL,
              label TEXT NOT NULL,
              boxes_json TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              FOREIGN KEY(dataset_id) REFERENCES datasets(id),
              FOREIGN KEY(sample_id) REFERENCES samples(id)
            );

            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              type TEXT NOT NULL,
              status TEXT NOT NULL,
              dataset_id TEXT,
              model_id TEXT,
              progress INTEGER NOT NULL DEFAULT 0,
              message TEXT NOT NULL DEFAULT '',
              result_json TEXT NOT NULL DEFAULT '{}',
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS models (
              id TEXT PRIMARY KEY,
              dataset_id TEXT,
              name TEXT NOT NULL,
              framework TEXT NOT NULL,
              weights_path TEXT NOT NULL,
              metrics_json TEXT NOT NULL DEFAULT '{}',
              created_at INTEGER NOT NULL
            );
            """
        )
        for model_id, weights_path in BUNDLED_MODELS.items():
            if not weights_path.exists():
                continue
            model_name = "Ready State Detection Model" if "soham" in model_id else "Ready Action Object Model"
            conn.execute(
                """
                INSERT OR IGNORE INTO models
                  (id, dataset_id, name, framework, weights_path, metrics_json, created_at)
                VALUES (?, NULL, ?, 'ultralytics-yolo', ?, '{}', ?)
                """,
                (model_id, model_name, str(weights_path), now()),
            )


def row_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    for key in ["result_json", "metrics_json", "boxes_json"]:
        if key in data:
            try:
                data[key.replace("_json", "")] = json.loads(data.pop(key) or "{}")
            except json.JSONDecodeError:
                data[key.replace("_json", "")] = {}
    return data


def write_job_log(job_id: str, message: str) -> None:
    log_path = JOBS / f"{job_id}.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def update_job(job_id: str, *, status: str | None = None, progress: int | None = None, message: str | None = None, result: dict | None = None) -> None:
    fields = ["updated_at = ?"]
    values: list[object] = [now()]

    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if progress is not None:
        fields.append("progress = ?")
        values.append(max(0, min(progress, 100)))
    if message is not None:
        fields.append("message = ?")
        values.append(message)
        write_job_log(job_id, message)
    if result is not None:
        fields.append("result_json = ?")
        values.append(json.dumps(result, ensure_ascii=False))

    values.append(job_id)
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)


def create_job(job_type: str, dataset_id: str | None = None, model_id: str | None = None) -> str:
    job_id = make_id("job")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, type, status, dataset_id, model_id, progress, message, created_at, updated_at)
            VALUES (?, ?, 'queued', ?, ?, 0, '', ?, ?)
            """,
            (job_id, job_type, dataset_id, model_id, now(), now()),
        )
    return job_id


@dataclass
class TrainConfig:
    dataset_id: str
    model: str = "yolov8n.pt"
    image_size: int = 640
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    pretrained: bool = True


def dataset_summary(dataset_id: str) -> dict:
    with connect() as conn:
        dataset = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if not dataset:
            raise KeyError(dataset_id)
        total = conn.execute("SELECT COUNT(*) AS c FROM samples WHERE dataset_id = ?", (dataset_id,)).fetchone()["c"]
        annotated = conn.execute("SELECT COUNT(*) AS c FROM annotations WHERE dataset_id = ?", (dataset_id,)).fetchone()["c"]
        label_rows = conn.execute(
            "SELECT label, COUNT(*) AS c FROM annotations WHERE dataset_id = ? GROUP BY label ORDER BY c DESC",
            (dataset_id,),
        ).fetchall()
    return {
        "dataset": row_to_dict(dataset),
        "total_samples": total,
        "annotated_samples": annotated,
        "labels": {row["label"]: row["c"] for row in label_rows},
        "supported_labels": STATE_LABELS,
    }


def normalize_control_key(value: object) -> str:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        return ""
    if key in CONTROL_LABELS:
        return key
    if key in CONTROL_KEY_ALIASES:
        return CONTROL_KEY_ALIASES[key]
    raise ValueError(f"unsupported manual control key: {value}")


def manual_control_payload() -> dict:
    with CONTROL_LOCK:
        payload = dict(MANUAL_CONTROL)
    active = bool(payload["enabled"])
    expires_at = int(payload.get("expires_at") or 0)
    if active and expires_at and now() >= expires_at:
        with CONTROL_LOCK:
            MANUAL_CONTROL.update(
                {
                    "enabled": False,
                    "key": "",
                    "label": "",
                    "confidence": 0.99,
                    "source": "manual",
                    "updated_at": now(),
                    "expires_at": 0,
                }
            )
            payload = dict(MANUAL_CONTROL)
        active = False
    payload["active"] = active
    payload["mode"] = "manual" if active else "auto"
    payload["has_state"] = bool(active and payload["key"])
    payload["supported_keys"] = CONTROL_LABELS
    return payload


def set_manual_control(payload: dict) -> dict:
    raw_mode = str(payload.get("mode") or "").strip().lower()
    key = normalize_control_key(payload.get("key", payload.get("state", payload.get("status", ""))))
    if raw_mode == "auto":
        enabled = False
    elif raw_mode == "manual":
        enabled = True
    else:
        enabled = bool(payload.get("enabled", True))
        if not key:
            enabled = False
    confidence = max(0.0, min(float(payload.get("confidence", 0.99)), 1.0))
    ttl = int(payload.get("ttl_ms", payload.get("ttl", 0)) or 0)
    if 0 < ttl < 1000:
        ttl *= 1000
    expires_at = now() + max(0, ttl // 1000) if enabled and ttl else 0
    source = str(payload.get("source") or "remote").strip()[:48] or "remote"
    with CONTROL_LOCK:
        MANUAL_CONTROL.update(
            {
                "enabled": enabled,
                "key": key if enabled and key else "",
                "label": CONTROL_LABELS.get(key, "") if enabled and key else "",
                "confidence": confidence,
                "source": source,
                "updated_at": now(),
                "expires_at": expires_at,
            }
        )
    return manual_control_payload()


def control_token_authorized(headers) -> bool:
    expected = os.environ.get("VISION_SENTINEL_CONTROL_TOKEN", "")
    if not expected:
        return True
    auth = headers.get("Authorization", "")
    token = headers.get("X-Control-Token", "")
    return token == expected or auth == f"Bearer {expected}"


def path_to_url(path: Path) -> str:
    return "/" + path.resolve().relative_to(ROOT).as_posix()


def static_cache_control(target: Path) -> str:
    if target.name == "index.html" or target.suffix in {".css", ".html", ".js"}:
        return "no-cache"
    if target.suffix.lower() in STATIC_LONG_CACHE_SUFFIXES:
        return "public, max-age=604800, immutable"
    return "public, max-age=3600"


def sixdrepnet_weights_path() -> Path | None:
    configured = os.environ.get("SIXDREPNET_WEIGHTS")
    if configured:
        path = Path(configured).expanduser()
        return path if path.exists() else None
    for path in SIXDREPNET_DEFAULT_WEIGHTS:
        if path.exists():
            return path
    return None


def sixdrepnet_status() -> dict:
    weights = sixdrepnet_weights_path()
    return {
        "source_dir": str(SIXDREPNET_ROOT),
        "source_found": SIXDREPNET_ROOT.exists(),
        "weights_path": str(weights) if weights else "",
        "weights_found": weights is not None,
        "ready": SIXDREPNET_DETECTOR is not None,
        "error": SIXDREPNET_ERROR,
        "expected_weights": [str(path) for path in SIXDREPNET_DEFAULT_WEIGHTS],
    }


def get_sixdrepnet_detector():
    global SIXDREPNET_DETECTOR, SIXDREPNET_ERROR
    if SIXDREPNET_DETECTOR is not None:
        return SIXDREPNET_DETECTOR
    with SIXDREPNET_LOCK:
        if SIXDREPNET_DETECTOR is not None:
            return SIXDREPNET_DETECTOR
        weights = sixdrepnet_weights_path()
        if not SIXDREPNET_ROOT.exists():
            SIXDREPNET_ERROR = "6DRepNet-master source directory was not found."
            raise RuntimeError(SIXDREPNET_ERROR)
        if weights is None:
            SIXDREPNET_ERROR = "6DRepNet weights were not found. Put 6DRepNet_300W_LP_AFLW2000.pth in 6DRepNet-master or set SIXDREPNET_WEIGHTS."
            raise RuntimeError(SIXDREPNET_ERROR)

        package_root = str(SIXDREPNET_ROOT)
        package_dir = str(SIXDREPNET_ROOT / "sixdrepnet")
        for path in [package_root, package_dir]:
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from sixdrepnet import SixDRepNet

            SIXDREPNET_DETECTOR = SixDRepNet(gpu_id=-1, dict_path=str(weights))
            SIXDREPNET_ERROR = ""
            return SIXDREPNET_DETECTOR
        except Exception as exc:
            SIXDREPNET_ERROR = f"6DRepNet failed to load: {exc}"
            raise RuntimeError(SIXDREPNET_ERROR) from exc


def decode_data_url_image(data_url: str):
    try:
        header, encoded = data_url.split(",", 1)
    except ValueError as exc:
        raise ValueError("image must be a data URL") from exc
    if "base64" not in header:
        raise ValueError("image must be base64 encoded")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid base64 image") from exc

    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"Missing 6DRepNet image dependency: {exc}") from exc

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def predict_sixdrepnet_pose(data_url: str) -> dict:
    detector = get_sixdrepnet_detector()
    image = decode_data_url_image(data_url)
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"Missing 6DRepNet inference dependency: {exc}") from exc

    with SIXDREPNET_LOCK:
        with torch.no_grad():
            pitch, yaw, roll = detector.predict(image)
    return {
        "pitch": float(pitch[0]),
        "yaw": float(yaw[0]),
        "roll": float(roll[0]),
        "source": "6drepnet",
        "weights_path": sixdrepnet_status()["weights_path"],
    }


def run_logged(job_id: str, command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    write_job_log(job_id, "$ " + " ".join(command))
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if completed.stdout:
        write_job_log(job_id, completed.stdout)
    if completed.stderr:
        write_job_log(job_id, completed.stderr)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "Command execution failed").strip().splitlines()[-1]
        raise RuntimeError(detail)
    return completed


def probe_dimensions(path: Path) -> tuple[int, int]:
    if not FFPROBE:
        raise RuntimeError("ffprobe was not found, so image or video dimensions cannot be read.")
    completed = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Could not read media dimensions: {path.name}")
    streams = json.loads(completed.stdout or "{}").get("streams") or []
    if not streams:
        raise RuntimeError(f"No usable visual frame found in file: {path.name}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def inspect_image_quality(path: Path) -> dict[str, float | bool]:
    if not FFMPEG:
        raise RuntimeError("ffmpeg was not found, so image quality inspection cannot run.")
    completed = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(path),
            "-vf",
            "signalstats,metadata=print:file=-,blurdetect",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    output = completed.stdout + "\n" + completed.stderr
    brightness_match = re.search(r"lavfi\.signalstats\.YAVG=([\d.]+)", output)
    blur_match = re.search(r"blur mean:\s*([\d.]+)", output)
    brightness = float(brightness_match.group(1)) if brightness_match else 128.0
    blur = float(blur_match.group(1)) if blur_match else 0.0
    return {
        "brightness": round(brightness, 3),
        "blur": round(blur, 3),
        "too_dark": brightness < 22.0,
        "too_blurry": blur > 15.0,
    }


def normalized_split_config(payload: dict) -> dict[str, float]:
    values = {
        "train": max(0.0, float(payload.get("train", 0.7))),
        "val": max(0.0, float(payload.get("val", 0.2))),
        "test": max(0.0, float(payload.get("test", 0.1))),
    }
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Training, validation, and test split ratios must sum to more than 0.")
    return {key: value / total for key, value in values.items()}


def assign_splits(rows: list[dict], ratios: dict[str, float]) -> dict[str, str]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["annotation_label"]), []).append(row)
    assignments: dict[str, str] = {}
    for label, items in grouped.items():
        random.Random(f"vision-sentinel:{label}").shuffle(items)
        count = len(items)
        train_count = max(1, round(count * ratios["train"]))
        val_count = round(count * ratios["val"])
        if count >= 3 and ratios["val"] > 0:
            val_count = max(1, val_count)
        if train_count + val_count > count:
            val_count = max(0, count - train_count)
        for index, item in enumerate(items):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            assignments[str(item["id"])] = split
    return assignments


def transform_boxes(boxes: list[dict], width: int, height: int, target_size: int) -> list[dict]:
    scale = min(target_size / width, target_size / height)
    scaled_width = width * scale
    scaled_height = height * scale
    pad_x = (target_size - scaled_width) / 2
    pad_y = (target_size - scaled_height) / 2
    transformed = []
    for box in boxes:
        x = min(max(float(box.get("x", 0.5)), 0.0), 1.0)
        y = min(max(float(box.get("y", 0.5)), 0.0), 1.0)
        w = min(max(float(box.get("w", 1.0)), 0.001), 1.0)
        h = min(max(float(box.get("h", 1.0)), 0.001), 1.0)
        transformed.append(
            {
                "label": str(box.get("label") or ""),
                "x": (x * width * scale + pad_x) / target_size,
                "y": (y * height * scale + pad_y) / target_size,
                "w": w * width * scale / target_size,
                "h": h * height * scale / target_size,
            }
        )
    return transformed


def write_yolo_label(path: Path, class_id: int, boxes: list[dict]) -> None:
    lines = [
        f"{class_id} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}"
        for box in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_yaml(target: Path) -> Path:
    yaml_path = target / "dataset.yaml"
    names = "\n".join(
        f"  {index}: {json.dumps(label, ensure_ascii=False)}" for index, label in enumerate(STATE_LABELS)
    )
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {target}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def process_media(
    job_id: str,
    source: Path,
    output: Path,
    *,
    target_size: int,
    video_fps: float | None = None,
) -> list[Path]:
    if not FFMPEG:
        raise RuntimeError("ffmpeg was not found, so real dataset processing cannot run.")
    video_filter = []
    if video_fps is not None:
        video_filter.append(f"fps={video_fps:g}")
    video_filter.extend(
        [
            f"scale={target_size}:{target_size}:force_original_aspect_ratio=decrease",
            f"pad={target_size}:{target_size}:(ow-iw)/2:(oh-ih)/2:color=black",
        ]
    )
    run_logged(
        job_id,
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            ",".join(video_filter),
            "-q:v",
            "2",
            *( ["-frames:v", "1"] if video_fps is None else [] ),
            str(output),
        ],
    )
    if video_fps is None:
        return [output] if output.exists() else []
    return sorted(output.parent.glob(output.name.replace("%06d", "*")))


def create_augmentation(job_id: str, source: Path, strategy: str) -> Path:
    filters = {
        "bright": "eq=brightness=0.08:contrast=1.03",
        "night": "eq=brightness=-0.16:contrast=1.12:saturation=0.82",
        "blur": "gblur=sigma=1.0",
        "flip": "hflip",
    }
    suffixes = {"bright": "bright", "night": "night", "blur": "blur", "flip": "flip"}
    target = source.with_name(f"{source.stem}_aug_{suffixes[strategy]}{source.suffix}")
    run_logged(
        job_id,
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-vf", filters[strategy], "-q:v", "2", str(target)],
    )
    return target


def process_dataset(job_id: str, dataset_id: str, options: dict | None = None) -> None:
    try:
        options = options or {}
        target_size = max(320, min(int(options.get("target_size", 640)), 1280))
        ratios = normalized_split_config(options.get("splits") or {})
        augment = bool(options.get("augment", True))
        quality_filter = bool(options.get("quality_filter", True))
        video_fps = max(0.1, min(float(options.get("video_fps", 1.0)), 10.0))
        update_job(job_id, status="running", progress=3, message="Reading samples, annotations, and processing parameters.")

        with connect() as conn:
            sample_rows = conn.execute(
                """
                SELECT s.*, a.task_type, a.label AS annotation_label, a.boxes_json
                FROM samples s
                LEFT JOIN annotations a ON a.id = (
                  SELECT id FROM annotations
                  WHERE sample_id = s.id
                  ORDER BY created_at DESC LIMIT 1
                )
                WHERE s.dataset_id = ?
                ORDER BY s.created_at, s.id
                """,
                (dataset_id,),
            ).fetchall()
        rows = [dict(row) for row in sample_rows if row["annotation_label"] in STATE_LABELS]
        if not rows:
            raise RuntimeError("No annotated samples are available for processing. Upload images or videos and complete annotations first.")

        run_dir = PROCESSED / dataset_id / job_id
        for split in ["train", "val", "test"]:
            (run_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (run_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        rejected_dir = run_dir / "rejected"
        duplicate_dir = run_dir / "duplicates"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        duplicate_dir.mkdir(parents=True, exist_ok=True)

        assignments = assign_splits(rows, ratios)
        seen_hashes: set[str] = set()
        report = {
            "processed": 0,
            "augmented": 0,
            "duplicates": 0,
            "quality_rejected": 0,
            "unannotated": len(sample_rows) - len(rows),
            "video_frames": 0,
            "splits": {"train": 0, "val": 0, "test": 0},
            "quality": [],
        }
        strategies = ["bright", "night", "blur", "flip"]

        for row_index, row in enumerate(rows):
            source = Path(str(row["path"]))
            if not source.exists():
                write_job_log(job_id, f"Skipping missing file: {source}")
                continue
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest in seen_hashes:
                report["duplicates"] += 1
                continue
            seen_hashes.add(digest)
            split = assignments[str(row["id"])]
            image_dir = run_dir / "images" / split
            label_dir = run_dir / "labels" / split
            is_video = str(row["media_type"]).startswith("video/")
            output_name = f"{row['id']}_%06d.jpg" if is_video else f"{row['id']}.jpg"
            output_files = process_media(
                job_id,
                source,
                image_dir / output_name,
                target_size=target_size,
                video_fps=video_fps if is_video else None,
            )
            width, height = probe_dimensions(source)
            raw_boxes = json.loads(row["boxes_json"] or "[]")
            if row["task_type"] == "classification" or not raw_boxes:
                raw_boxes = [{"label": row["annotation_label"], "x": 0.5, "y": 0.5, "w": 1.0, "h": 1.0}]
            transformed_boxes = transform_boxes(raw_boxes, width, height, target_size)
            class_id = STATE_LABELS.index(str(row["annotation_label"]))

            for frame_index, output_file in enumerate(output_files):
                quality = inspect_image_quality(output_file)
                quality["sample"] = output_file.name
                if len(report["quality"]) < 100:
                    report["quality"].append(quality)
                if quality_filter and (quality["too_dark"] or quality["too_blurry"]):
                    shutil.move(str(output_file), rejected_dir / output_file.name)
                    report["quality_rejected"] += 1
                    continue
                output_digest = hashlib.sha256(output_file.read_bytes()).hexdigest()
                if output_digest in seen_hashes and is_video:
                    shutil.move(str(output_file), duplicate_dir / output_file.name)
                    report["duplicates"] += 1
                    continue
                seen_hashes.add(output_digest)
                label_file = label_dir / f"{output_file.stem}.txt"
                write_yolo_label(label_file, class_id, transformed_boxes)
                report["processed"] += 1
                report["splits"][split] += 1
                if is_video:
                    report["video_frames"] += 1

                if augment and split == "train":
                    strategy = strategies[(row_index + frame_index) % len(strategies)]
                    augmented_file = create_augmentation(job_id, output_file, strategy)
                    augmented_boxes = [dict(box) for box in transformed_boxes]
                    if strategy == "flip":
                        for box in augmented_boxes:
                            box["x"] = 1.0 - float(box["x"])
                    write_yolo_label(label_dir / f"{augmented_file.stem}.txt", class_id, augmented_boxes)
                    report["augmented"] += 1
                    report["splits"][split] += 1

            progress = 8 + round(((row_index + 1) / len(rows)) * 82)
            update_job(job_id, progress=progress, message=f"Processed {row_index + 1}/{len(rows)} annotated source files.")

        if report["splits"]["train"] == 0:
            raise RuntimeError("The training split is empty after quality filtering. Check annotations or disable quality filtering and try again.")
        yaml_path = write_dataset_yaml(run_dir)
        result = {
            "dataset_yaml": str(yaml_path),
            "dataset_url": path_to_url(yaml_path),
            "run_dir": str(run_dir),
            "target_size": target_size,
            "requested_splits": ratios,
            "before_samples": len(sample_rows),
            "after_samples": report["processed"] + report["augmented"],
            "report": report,
        }
        update_job(job_id, status="succeeded", progress=100, message="Real dataset processing complete. YOLO dataset generated.", result=result)
    except Exception as exc:
        update_job(job_id, status="failed", message=f"Dataset processing failed: {exc}")


def find_yolo_command() -> list[str] | None:
    yolo = shutil.which("yolo")
    if yolo:
        return [yolo]
    python = shutil.which("python3") or shutil.which("python")
    if python and subprocess.run([python, "-c", "import ultralytics"], capture_output=True).returncode == 0:
        return [python, "-m", "ultralytics"]
    return None


def latest_processed_dataset(dataset_id: str) -> tuple[Path, dict]:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT result_json FROM jobs
            WHERE type = 'process' AND dataset_id = ? AND status = 'succeeded'
            ORDER BY updated_at DESC LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
    if not row:
        raise RuntimeError("No successful dataset processing job exists yet. Run dataset processing first.")
    result = json.loads(row["result_json"] or "{}")
    yaml_path = Path(result.get("dataset_yaml") or "")
    if not yaml_path.exists():
        raise RuntimeError("The processed dataset.yaml file does not exist. Run dataset processing again.")
    train_images = list((yaml_path.parent / "images" / "train").glob("*"))
    train_labels = list((yaml_path.parent / "labels" / "train").glob("*.txt"))
    if not train_images or not train_labels:
        raise RuntimeError("The processed training images or YOLO labels are empty.")
    return yaml_path, result


def parse_training_results(run_dir: Path) -> dict:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {"source": "ultralytics", "history": []}
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"source": "ultralytics", "history": []}

    def metric(row: dict, candidates: list[str]) -> float | None:
        normalized = {key.strip(): value for key, value in row.items()}
        for candidate in candidates:
            value = normalized.get(candidate)
            if value not in (None, ""):
                return float(value)
        return None

    history = []
    for index, row in enumerate(rows):
        history.append(
            {
                "epoch": index + 1,
                "loss": metric(row, ["train/box_loss", "val/box_loss"]),
                "precision": metric(row, ["metrics/precision(B)", "metrics/precision"]),
                "recall": metric(row, ["metrics/recall(B)", "metrics/recall"]),
                "map50": metric(row, ["metrics/mAP50(B)", "metrics/mAP50"]),
                "map5095": metric(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"]),
            }
        )
    last = history[-1]
    precision = last.get("precision")
    recall = last.get("recall")
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    artifacts = {}
    for name, key in [("PR_curve.png", "pr_curve"), ("confusion_matrix.png", "confusion_matrix")]:
        artifact = run_dir / name
        if artifact.exists():
            artifacts[key] = path_to_url(artifact)
    return {
        "source": "ultralytics",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": last.get("map50"),
        "map5095": last.get("map5095"),
        "history": history,
        "artifacts": artifacts,
    }


def train_yolo(job_id: str, config: TrainConfig) -> None:
    try:
        update_job(job_id, status="running", progress=5, message="Preparing YOLO training configuration.")
        dataset_yaml, _ = latest_processed_dataset(config.dataset_id)
        command_base = find_yolo_command()
        if not command_base:
            raise RuntimeError("Ultralytics is not installed in the current project environment. Install it in the project .venv before real training.")

        run_name = f"vision_sentinel_{job_id}"
        processed_root = dataset_yaml.parent
        if any((processed_root / "images" / "test").glob("*")):
            evaluation_split = "test"
        elif any((processed_root / "images" / "val").glob("*")):
            evaluation_split = "val"
        else:
            evaluation_split = "train"
        command = [
            *command_base,
            "detect",
            "train",
            f"data={dataset_yaml}",
            f"model={config.model}",
            f"imgsz={config.image_size}",
            f"epochs={config.epochs}",
            f"batch={config.batch_size}",
            f"lr0={config.learning_rate}",
            f"pretrained={str(config.pretrained).lower()}",
            f"project={MODELS}",
            f"name={run_name}",
            "exist_ok=True",
        ]
        update_job(job_id, progress=12, message="Starting training command: " + " ".join(command))

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        progress = 12
        assert process.stdout is not None
        for line in process.stdout:
            write_job_log(job_id, line)
            if "Epoch" in line or "/" in line:
                progress = min(progress + 1, 94)
                update_job(job_id, progress=progress, message=line.strip()[:240])

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"YOLO training command exited with code {return_code}. Check storage/jobs/{job_id}.log")

        run_dir = MODELS / run_name
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            raise RuntimeError("Training finished, but weights/best.pt was not generated.")
        metrics = parse_training_results(run_dir)
        model_id = make_id("model")
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO models (id, dataset_id, name, framework, weights_path, metrics_json, created_at)
                VALUES (?, ?, ?, 'ultralytics-yolo', ?, ?, ?)
                """,
                (
                    model_id,
                    config.dataset_id,
                    run_name,
                    str(best),
                    json.dumps(metrics, ensure_ascii=False),
                    now(),
                ),
            )
            conn.execute("UPDATE jobs SET model_id = ? WHERE id = ?", (model_id, job_id))
        update_job(
            job_id,
            status="succeeded",
            progress=100,
            message="Training complete. Model weights registered.",
            result={"model_id": model_id, "weights_path": str(best), "run_dir": str(run_dir)},
        )
    except Exception as exc:
        update_job(job_id, status="failed", message=f"Training failed: {exc}")


def evaluate_model(job_id: str, model_id: str, dataset_id: str | None = None) -> None:
    try:
        update_job(job_id, status="running", progress=10, message="Preparing real test-set evaluation.")
        with connect() as conn:
            model = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if not model:
                raise KeyError(model_id)
        resolved_dataset_id = dataset_id or model["dataset_id"]
        if not resolved_dataset_id:
            raise RuntimeError("This ready model is not bound to a dataset. Select a processed dataset before evaluation.")
        dataset_yaml, _ = latest_processed_dataset(str(resolved_dataset_id))
        processed_root = dataset_yaml.parent
        if any((processed_root / "images" / "test").glob("*")):
            evaluation_split = "test"
        elif any((processed_root / "images" / "val").glob("*")):
            evaluation_split = "val"
        else:
            evaluation_split = "train"
        command_base = find_yolo_command()
        if not command_base:
            raise RuntimeError("Ultralytics is not installed in the current project environment, so real evaluation cannot run.")
        weights = Path(model["weights_path"])
        if not weights.exists():
            raise RuntimeError("Model weights file does not exist.")
        evaluation_root = MODELS / "evaluations" / model_id
        evaluation_name = job_id
        command = [
            *command_base,
            "detect",
            "val",
            f"model={weights}",
            f"data={dataset_yaml}",
            f"split={evaluation_split}",
            f"project={evaluation_root}",
            f"name={evaluation_name}",
            "plots=True",
            "exist_ok=True",
        ]
        update_job(job_id, progress=20, message="Starting Ultralytics test-set evaluation.")
        completed = run_logged(job_id, command)
        output = completed.stdout + "\n" + completed.stderr
        matches = re.findall(
            r"^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            output,
            flags=re.MULTILINE,
        )
        if not matches:
            raise RuntimeError("Evaluation ran, but real metrics could not be read from Ultralytics output.")
        precision, recall, map50, map5095 = [float(value) for value in matches[-1]]
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        evaluation_dir = evaluation_root / evaluation_name
        artifacts = {}
        for name, key in [("PR_curve.png", "pr_curve"), ("confusion_matrix.png", "confusion_matrix")]:
            artifact = evaluation_dir / name
            if artifact.exists():
                artifacts[key] = path_to_url(artifact)
        previous_metrics = json.loads(model["metrics_json"] or "{}")
        metrics = {
            **previous_metrics,
            "source": "ultralytics-test",
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": map50,
            "map5095": map5095,
            "evaluated_dataset_id": resolved_dataset_id,
            "evaluated_split": evaluation_split,
            "artifacts": {**previous_metrics.get("artifacts", {}), **artifacts},
        }
        with connect() as conn:
            conn.execute("UPDATE models SET metrics_json = ? WHERE id = ?", (json.dumps(metrics, ensure_ascii=False), model_id))
        update_job(job_id, status="succeeded", progress=100, message="Real test-set evaluation complete.", result=metrics)
    except Exception as exc:
        update_job(job_id, status="failed", message=f"Evaluation failed: {exc}")


def export_model(job_id: str, model_id: str, export_format: str) -> None:
    try:
        update_job(job_id, status="running", progress=20, message=f"Preparing {export_format} export.")
        with connect() as conn:
            model = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if not model:
                raise KeyError(model_id)
        weights = Path(model["weights_path"])
        export_dir = EXPORTS / model_id
        export_dir.mkdir(parents=True, exist_ok=True)
        if not weights.exists():
            raise RuntimeError("Model weights file does not exist.")
        normalized_format = export_format.lower().lstrip(".")
        target = export_dir / f"{model['name']}.{normalized_format}"
        if normalized_format == "pt":
            shutil.copy2(weights, target)
        else:
            command_base = find_yolo_command()
            if not command_base:
                raise RuntimeError("Ultralytics is not installed in the current project environment, so real format conversion cannot run.")
            if normalized_format not in {"onnx", "engine", "tflite"}:
                raise RuntimeError(f"Unsupported export format: {normalized_format}")
            update_job(job_id, progress=35, message=f"Running real Ultralytics {normalized_format} conversion.")
            run_logged(job_id, [*command_base, "export", f"model={weights}", f"format={normalized_format}"])
            patterns = {
                "onnx": ["*.onnx"],
                "engine": ["*.engine"],
                "tflite": ["*.tflite"],
            }
            candidates = []
            for pattern in patterns[normalized_format]:
                candidates.extend(weights.parent.rglob(pattern))
            if not candidates:
                raise RuntimeError(f"Conversion finished, but no {normalized_format} file was generated.")
            source = max(candidates, key=lambda item: item.stat().st_mtime)
            target = export_dir / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        update_job(
            job_id,
            status="succeeded",
            progress=100,
            message="Real model export complete.",
            result={"export_path": str(target), "download_url": path_to_url(target), "format": normalized_format},
        )
    except Exception as exc:
        update_job(job_id, status="failed", message=f"Export failed: {exc}")


class Handler(BaseHTTPRequestHandler):
    server_version = "VisionSentinelBackend/0.1"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # 跨域隔离:ONNX Runtime 的 wasm 线程构建需要 SharedArrayBuffer,
        # 浏览器仅在 crossOriginIsolated(即同时带 COOP+COEP 头)时才开放。
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            return
        return self.static_response(path, send_body=False)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/health":
                return self.json_response(
                    {
                        "ok": True,
                        "service": "Vision Sentinel backend",
                        "time": now(),
                        "capabilities": {
                            "ffmpeg_processing": bool(FFMPEG and FFPROBE),
                            "ultralytics_training": find_yolo_command() is not None,
                            "browser_onnx_inference": all(path.exists() for path in [
                                ROOT / "Driver-Monitoring-System/public/static/models/soham_best.onnx",
                                ROOT / "Driver-Monitoring-System/public/static/models/chaitanya_best.onnx",
                                ROOT / "Driver-Monitoring-System/public/static/models/yolov8n_coco.onnx",
                            ]),
                            "sixdrepnet_head_pose": sixdrepnet_status(),
                        },
                    }
                )
            if path == "/api/datasets":
                with connect() as conn:
                    rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
                return self.json_response({"items": [row_to_dict(row) for row in rows]})
            if path.startswith("/api/datasets/") and path.endswith("/summary"):
                dataset_id = path.split("/")[3]
                return self.json_response(dataset_summary(dataset_id))
            if path.startswith("/api/datasets/") and path.endswith("/samples"):
                dataset_id = path.split("/")[3]
                with connect() as conn:
                    rows = conn.execute(
                        "SELECT * FROM samples WHERE dataset_id = ? ORDER BY created_at DESC LIMIT 200",
                        (dataset_id,),
                    ).fetchall()
                return self.json_response({"items": [row_to_dict(row) for row in rows]})
            if path.startswith("/api/samples/") and path.endswith("/file"):
                sample_id = path.split("/")[3]
                with connect() as conn:
                    sample = conn.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
                if not sample:
                    return self.error_response(HTTPStatus.NOT_FOUND, "sample not found")
                return self.file_response(Path(sample["path"]))
            if path.startswith("/api/samples/") and path.endswith("/annotation"):
                sample_id = path.split("/")[3]
                with connect() as conn:
                    annotation = conn.execute(
                        "SELECT * FROM annotations WHERE sample_id = ? ORDER BY created_at DESC LIMIT 1",
                        (sample_id,),
                    ).fetchone()
                return self.json_response({"annotation": row_to_dict(annotation) if annotation else None})
            if path == "/api/jobs":
                with connect() as conn:
                    rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50").fetchall()
                return self.json_response({"items": [row_to_dict(row) for row in rows]})
            if path.startswith("/api/jobs/"):
                job_id = path.split("/")[-1]
                with connect() as conn:
                    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if not row:
                    return self.error_response(HTTPStatus.NOT_FOUND, "job not found")
                payload = row_to_dict(row)
                log_path = JOBS / f"{job_id}.log"
                payload["log"] = log_path.read_text(encoding="utf-8")[-4000:] if log_path.exists() else ""
                return self.json_response(payload)
            if path == "/api/models":
                with connect() as conn:
                    rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
                return self.json_response({"items": [row_to_dict(row) for row in rows]})
            if path == "/api/annotation-tracks":
                return self.json_response({"items": list_annotation_tracks()})
            if path.startswith("/api/annotation-tracks/"):
                target = safe_annotation_track_path(unquote(path.split("/")[3]))
                if not target or not target.exists():
                    return self.error_response(HTTPStatus.NOT_FOUND, "annotation track not found")
                return self.json_response(json.loads(target.read_text(encoding="utf-8")))
            if path == "/api/manual-control":
                return self.json_response(manual_control_payload())
            if path == "/api/head-pose/6drepnet/status":
                return self.json_response(sixdrepnet_status())
            return self.static_response(path)
        except KeyError:
            return self.error_response(HTTPStatus.NOT_FOUND, "resource not found")
        except Exception as exc:
            return self.error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/manual-control":
                if not control_token_authorized(self.headers):
                    return self.error_response(HTTPStatus.UNAUTHORIZED, "manual control token is invalid or missing")
                return self.json_response(set_manual_control(self.read_json()))
            if path == "/api/annotation-tracks":
                payload = self.read_json()
                if not isinstance(payload.get("frames"), list) or not payload["frames"]:
                    return self.error_response(HTTPStatus.BAD_REQUEST, "annotation track requires frames")
                stem = time.strftime("annotation-track-%Y%m%d-%H%M%S")
                target = ANNOTATION_TRACKS / f"{stem}.json"
                suffix = 1
                while target.exists():
                    target = ANNOTATION_TRACKS / f"{stem}-{suffix}.json"
                    suffix += 1
                target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return self.json_response({"name": target.name, "size": target.stat().st_size}, HTTPStatus.CREATED)
            if path == "/api/datasets":
                payload = self.read_json()
                dataset_id = make_id("ds")
                with connect() as conn:
                    conn.execute(
                        "INSERT INTO datasets (id, name, source, description, created_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            dataset_id,
                            payload.get("name") or "Driver State Dataset",
                            payload.get("source") or "Cabin Camera",
                            payload.get("description") or "",
                            now(),
                        ),
                    )
                (DATASETS / dataset_id).mkdir(parents=True, exist_ok=True)
                return self.json_response({"id": dataset_id, "summary": dataset_summary(dataset_id)}, HTTPStatus.CREATED)
            if path.startswith("/api/datasets/") and path.endswith("/upload"):
                dataset_id = path.split("/")[3]
                return self.upload_samples(dataset_id)
            if path == "/api/annotations":
                payload = self.read_json()
                sample_id = payload.get("sample_id")
                if not sample_id:
                    return self.error_response(HTTPStatus.BAD_REQUEST, "sample_id is required")
                with connect() as conn:
                    sample = conn.execute(
                        "SELECT id FROM samples WHERE id = ? AND dataset_id = ?",
                        (sample_id, payload["dataset_id"]),
                    ).fetchone()
                if not sample:
                    return self.error_response(HTTPStatus.BAD_REQUEST, "sample does not belong to dataset")
                task_type = payload.get("task_type") or "detection"
                annotation_id = make_id("ann")
                boxes = []
                for raw_box in payload.get("boxes") or []:
                    box = {
                        "label": str(raw_box.get("label") or payload["label"]),
                        "x": min(max(float(raw_box.get("x", 0.5)), 0.0), 1.0),
                        "y": min(max(float(raw_box.get("y", 0.5)), 0.0), 1.0),
                        "w": min(max(float(raw_box.get("w", 0.0)), 0.001), 1.0),
                        "h": min(max(float(raw_box.get("h", 0.0)), 0.001), 1.0),
                    }
                    boxes.append(box)
                if task_type == "detection" and not boxes:
                    return self.error_response(HTTPStatus.BAD_REQUEST, "detection annotation requires at least one box")
                with connect() as conn:
                    conn.execute(
                        "DELETE FROM annotations WHERE sample_id = ? AND task_type = ?",
                        (sample_id, task_type),
                    )
                    conn.execute(
                        """
                        INSERT INTO annotations (id, dataset_id, sample_id, task_type, label, boxes_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            annotation_id,
                            payload["dataset_id"],
                            sample_id,
                            task_type,
                            payload["label"],
                            json.dumps(boxes, ensure_ascii=False),
                            now(),
                        ),
                    )
                    conn.execute(
                        "UPDATE samples SET status = 'annotated', label = ? WHERE id = ?",
                        (payload["label"], sample_id),
                    )
                return self.json_response({"id": annotation_id}, HTTPStatus.CREATED)
            if path == "/api/process-jobs":
                payload = self.read_json()
                job_id = create_job("process", dataset_id=payload["dataset_id"])
                threading.Thread(
                    target=process_dataset,
                    args=(job_id, payload["dataset_id"], payload.get("options") or {}),
                    daemon=True,
                ).start()
                return self.json_response({"job_id": job_id}, HTTPStatus.ACCEPTED)
            if path == "/api/train-jobs":
                payload = self.read_json()
                config = TrainConfig(
                    dataset_id=payload["dataset_id"],
                    model=payload.get("model", "yolov8n.pt"),
                    image_size=int(payload.get("image_size", 640)),
                    epochs=int(payload.get("epochs", 50)),
                    batch_size=int(payload.get("batch_size", 16)),
                    learning_rate=float(payload.get("learning_rate", 0.001)),
                    pretrained=bool(payload.get("pretrained", True)),
                )
                job_id = create_job("train", dataset_id=config.dataset_id)
                threading.Thread(target=train_yolo, args=(job_id, config), daemon=True).start()
                return self.json_response({"job_id": job_id}, HTTPStatus.ACCEPTED)
            if path == "/api/evaluate-jobs":
                payload = self.read_json()
                job_id = create_job("evaluate", dataset_id=payload.get("dataset_id"), model_id=payload["model_id"])
                threading.Thread(
                    target=evaluate_model,
                    args=(job_id, payload["model_id"], payload.get("dataset_id")),
                    daemon=True,
                ).start()
                return self.json_response({"job_id": job_id}, HTTPStatus.ACCEPTED)
            if path == "/api/export-jobs":
                payload = self.read_json()
                job_id = create_job("export", model_id=payload["model_id"])
                threading.Thread(
                    target=export_model,
                    args=(job_id, payload["model_id"], payload.get("format", "pt")),
                    daemon=True,
                ).start()
                return self.json_response({"job_id": job_id}, HTTPStatus.ACCEPTED)
            if path == "/api/head-pose/6drepnet":
                payload = self.read_json()
                image = str(payload.get("image") or "")
                if not image:
                    return self.error_response(HTTPStatus.BAD_REQUEST, "image is required")
                return self.json_response(predict_sixdrepnet_pose(image))
            return self.error_response(HTTPStatus.NOT_FOUND, "endpoint not found")
        except KeyError as exc:
            return self.error_response(HTTPStatus.BAD_REQUEST, f"missing field: {exc}")
        except Exception as exc:
            return self.error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def read_multipart_files(self, field_name: str) -> list[dict[str, object]]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("upload requires multipart/form-data")

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("utf-8") + b"\r\n"
            b"MIME-Version: 1.0\r\n\r\n"
            + raw_body
        )
        if not message.is_multipart():
            return []

        files: list[dict[str, object]] = []
        for part in message.iter_parts():
            disposition = part.get_content_disposition()
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            if disposition != "form-data" or name != field_name or not filename:
                continue
            files.append(
                {
                    "filename": Path(filename).name,
                    "media_type": part.get_content_type(),
                    "content": part.get_payload(decode=True) or b"",
                }
            )
        return files

    def upload_samples(self, dataset_id: str) -> None:
        with connect() as conn:
            exists = conn.execute("SELECT id FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if not exists:
            return self.error_response(HTTPStatus.NOT_FOUND, "dataset not found")

        saved = []
        target_dir = DATASETS / dataset_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in self.read_multipart_files("files"):
            sample_id = make_id("sample")
            filename = str(item["filename"])
            target = target_dir / f"{sample_id}_{filename}"
            with target.open("wb") as output:
                output.write(item["content"])
            media_type = str(item["media_type"] or mimetypes.guess_type(filename)[0] or "application/octet-stream")
            with connect() as conn:
                conn.execute(
                    """
                    INSERT INTO samples (id, dataset_id, filename, path, media_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (sample_id, dataset_id, filename, str(target), media_type, now()),
                )
            saved.append({"id": sample_id, "filename": filename, "media_type": media_type})
        return self.json_response({"saved": saved, "summary": dataset_summary(dataset_id)}, HTTPStatus.CREATED)

    def static_response(self, request_path: str, *, send_body: bool = True) -> None:
        if request_path == "/":
            request_path = "/index.html"
        relative = Path(unquote(request_path).lstrip("/"))
        target = (ROOT / relative).resolve()
        if not str(target).startswith(str(ROOT)) or not target.exists() or target.is_dir():
            return self.error_response(HTTPStatus.NOT_FOUND, "not found")
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".wasm":
            mime = "application/wasm"
        elif target.suffix == ".mjs":
            mime = "text/javascript"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Cache-Control", static_cache_control(target))
        self.end_headers()
        if send_body:
            with target.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)

    def file_response(self, target: Path) -> None:
        resolved = target.resolve()
        if not str(resolved).startswith(str(DATASETS.resolve())) or not resolved.exists() or resolved.is_dir():
            return self.error_response(HTTPStatus.NOT_FOUND, "sample file not found")
        body = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def error_response(self, status: HTTPStatus, message: str) -> None:
        self.json_response({"error": message}, status)


def main() -> None:
    init_db()
    host = os.environ.get("VISION_SENTINEL_HOST", "0.0.0.0")
    port = int(os.environ.get("VISION_SENTINEL_PORT", "8000"))
    cert_file = os.environ.get("VISION_SENTINEL_SSL_CERT")
    key_file = os.environ.get("VISION_SENTINEL_SSL_KEY")
    server = ThreadingHTTPServer((host, port), Handler)
    scheme = "http"
    if cert_file and key_file:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"Vision Sentinel backend running at {scheme}://{host}:{port}")
    print("Open the app at /index.html or call /api/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
