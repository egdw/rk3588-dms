#!/usr/bin/env python3
"""MQTT command bridge for Vision Sentinel manual control.

This bridge intentionally avoids Python MQTT dependencies. It shells out to
mosquitto_sub, which is already present on the target server, and forwards
recognized commands to the local Vision Sentinel HTTP API.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request


MQTT_HOST = os.environ.get("VISION_SENTINEL_MQTT_HOST", "192.168.2.13")
MQTT_PORT = int(os.environ.get("VISION_SENTINEL_MQTT_PORT", "1883"))
MQTT_TOPIC = os.environ.get("VISION_SENTINEL_MQTT_TOPIC", "vision-sentinel/control")
MQTT_USERNAME = os.environ.get("VISION_SENTINEL_MQTT_USERNAME", "")
MQTT_PASSWORD = os.environ.get("VISION_SENTINEL_MQTT_PASSWORD", "")
API_URL = os.environ.get("VISION_SENTINEL_CONTROL_API", "https://127.0.0.1:8443/api/manual-control")
CONTROL_TOKEN = os.environ.get("VISION_SENTINEL_CONTROL_TOKEN", "")
MOSQUITTO_SUB = os.environ.get("VISION_SENTINEL_MOSQUITTO_SUB", shutil.which("mosquitto_sub") or "mosquitto_sub")
RECONNECT_DELAY_SECONDS = float(os.environ.get("VISION_SENTINEL_MQTT_RECONNECT_SECONDS", "3"))

COMMANDS = {
    "manual_enter": {"mode": "manual", "key": ""},
    "enter": {"mode": "manual", "key": ""},
    "head_down": {"mode": "manual", "key": "head_down"},
    "head": {"mode": "manual", "key": "head_down"},
    "drowsy": {"mode": "manual", "key": "drowsy"},
    "eye_closure": {"mode": "manual", "key": "drowsy"},
    "phone": {"mode": "manual", "key": "phone"},
    "manual_exit": {"mode": "auto", "key": ""},
    "exit": {"mode": "auto", "key": ""},
}

RUNNING = True


def log(message: str) -> None:
    print(f"[mqtt-bridge] {message}", flush=True)


def stop(_signum, _frame) -> None:
    global RUNNING
    RUNNING = False


def decode_command(payload: str) -> dict | None:
    text = payload.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {"command": text}
    command = str(data.get("command") or data.get("key") or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapped = COMMANDS.get(command)
    if not mapped:
        log(f"ignored unknown command: {command or text}")
        return None
    return {
        **mapped,
        "source": "mqtt",
        "mqtt_command": command,
    }


def post_manual_control(payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if CONTROL_TOKEN:
        request.add_header("Authorization", f"Bearer {CONTROL_TOKEN}")
    context = ssl._create_unverified_context() if API_URL.startswith("https://") else None
    try:
        with urllib.request.urlopen(request, timeout=3, context=context) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    log(f"forwarded {payload['mqtt_command']} -> mode={result.get('mode')} key={result.get('key') or '-'}")


def mqtt_command() -> list[str]:
    command = [
        MOSQUITTO_SUB,
        "-h",
        MQTT_HOST,
        "-p",
        str(MQTT_PORT),
        "-t",
        MQTT_TOPIC,
        "-v",
    ]
    if MQTT_USERNAME:
        command.extend(["-u", MQTT_USERNAME])
    if MQTT_PASSWORD:
        command.extend(["-P", MQTT_PASSWORD])
    return command


def payload_from_line(line: str) -> str:
    line = line.rstrip("\n")
    prefix = f"{MQTT_TOPIC} "
    if line.startswith(prefix):
        return line[len(prefix) :]
    return line


def run_once() -> int:
    command = mqtt_command()
    log(f"subscribing to mqtt://{MQTT_HOST}:{MQTT_PORT}/{MQTT_TOPIC}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if not RUNNING:
                break
            payload = decode_command(payload_from_line(line))
            if not payload:
                continue
            try:
                post_manual_control(payload)
            except Exception as exc:
                log(f"forward failed: {exc}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
    return process.returncode or 0


def main() -> int:
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while RUNNING:
        code = run_once()
        if not RUNNING:
            break
        log(f"mosquitto_sub exited with {code}; reconnecting in {RECONNECT_DELAY_SECONDS:g}s")
        time.sleep(RECONNECT_DELAY_SECONDS)
    log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
