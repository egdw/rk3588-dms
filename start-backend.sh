#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_FILE="$PROJECT_ROOT/backend/server.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VISION_SENTINEL_HOST="${VISION_SENTINEL_HOST:-0.0.0.0}"
VISION_SENTINEL_HTTPS="${VISION_SENTINEL_HTTPS:-0}"
if [[ "$VISION_SENTINEL_HTTPS" == "1" ]]; then
  VISION_SENTINEL_SCHEME="https"
  DEFAULT_PORT="8443"
else
  VISION_SENTINEL_SCHEME="http"
  DEFAULT_PORT="8000"
fi
VISION_SENTINEL_PORT="${VISION_SENTINEL_PORT:-$DEFAULT_PORT}"
VISION_SENTINEL_AUTO_PORT="${VISION_SENTINEL_AUTO_PORT:-1}"
VISION_SENTINEL_OPEN_BROWSER="${VISION_SENTINEL_OPEN_BROWSER:-0}"
VISION_SENTINEL_CERT_DIR="${VISION_SENTINEL_CERT_DIR:-$PROJECT_ROOT/certs}"
VISION_SENTINEL_SSL_CERT="${VISION_SENTINEL_SSL_CERT:-$VISION_SENTINEL_CERT_DIR/vision-sentinel.local.crt}"
VISION_SENTINEL_SSL_KEY="${VISION_SENTINEL_SSL_KEY:-$VISION_SENTINEL_CERT_DIR/vision-sentinel.local.key}"

die() {
  echo "启动失败：$*" >&2
  exit 1
}

require_python() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "找不到 Python 命令：$PYTHON_BIN"
}

browser_host() {
  if [[ "$VISION_SENTINEL_HOST" == "0.0.0.0" ]] || [[ "$VISION_SENTINEL_HOST" == "::" ]]; then
    echo "127.0.0.1"
  else
    echo "$VISION_SENTINEL_HOST"
  fi
}

list_lan_urls() {
  "$PYTHON_BIN" - "$VISION_SENTINEL_SCHEME" "$VISION_SENTINEL_PORT" <<'PY'
import socket
import sys

scheme = sys.argv[1]
port = int(sys.argv[2])
addresses = set()

try:
    infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    for info in infos:
        ip = info[4][0]
        if not ip.startswith("127."):
            addresses.add(ip)
except OSError:
    pass

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    ip = sock.getsockname()[0]
    if not ip.startswith("127."):
        addresses.add(ip)
except OSError:
    pass
finally:
    try:
        sock.close()
    except Exception:
        pass

for ip in sorted(addresses):
    print(f"{scheme}://{ip}:{port}/index.html")
PY
}

primary_lan_ip() {
  "$PYTHON_BIN" - <<'PY'
import socket

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    ip = sock.getsockname()[0]
    if not ip.startswith("127."):
        print(ip)
except OSError:
    pass
finally:
    try:
        sock.close()
    except Exception:
        pass
PY
}

is_port_free() {
  "$PYTHON_BIN" - "$VISION_SENTINEL_HOST" "$1" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind((host, port))
except PermissionError:
    sys.exit(2)
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

is_vision_sentinel_running() {
  "$PYTHON_BIN" - "$VISION_SENTINEL_SCHEME" "$VISION_SENTINEL_HOST" "$1" <<'PY'
import json
import ssl
import sys
import urllib.request

scheme = sys.argv[1]
host = sys.argv[2]
port = int(sys.argv[3])
context = ssl._create_unverified_context() if scheme == "https" else None
try:
    with urllib.request.urlopen(f"{scheme}://{host}:{port}/api/health", timeout=0.7, context=context) as response:
        data = json.loads(response.read().decode("utf-8"))
except Exception:
    sys.exit(1)

service = str(data.get("service", ""))
sys.exit(0 if data.get("ok") and "Vision Sentinel" in service else 1)
PY
}

port_has_wildcard_listener() {
  command -v lsof >/dev/null 2>&1 || return 1
  lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | awk 'NR > 1 {print $NF}' | grep -Eq "(^|[[:space:]])(\\*|0\\.0\\.0\\.0):$1$"
}

ensure_https_certificate() {
  [[ "$VISION_SENTINEL_HTTPS" == "1" ]] || return 0
  command -v openssl >/dev/null 2>&1 || die "找不到 openssl，无法生成 HTTPS 自签证书。"
  if [[ -f "$VISION_SENTINEL_SSL_CERT" ]] && [[ -f "$VISION_SENTINEL_SSL_KEY" ]]; then
    return 0
  fi
  mkdir -p "$VISION_SENTINEL_CERT_DIR"
  LAN_IP="$(primary_lan_ip)"
  SAN="DNS:localhost,IP:127.0.0.1"
  if [[ -n "$LAN_IP" ]]; then
    SAN="${SAN},IP:${LAN_IP}"
  fi
  openssl req -x509 -newkey rsa:2048 -sha256 -nodes \
    -keyout "$VISION_SENTINEL_SSL_KEY" \
    -out "$VISION_SENTINEL_SSL_CERT" \
    -days 825 \
    -subj "/CN=Vision Sentinel Local" \
    -addext "subjectAltName=${SAN}" >/dev/null 2>&1
}

find_available_port() {
  "$PYTHON_BIN" - "$VISION_SENTINEL_HOST" "$VISION_SENTINEL_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
base = int(sys.argv[2])

for port in range(base, base + 50):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except PermissionError:
        sock.close()
        sys.exit(2)
    except OSError:
        sock.close()
        continue
    sock.close()
    print(port)
    sys.exit(0)

sys.exit(1)
PY
}

require_python

[[ -d "$PROJECT_ROOT" ]] || die "项目目录不存在：$PROJECT_ROOT"
[[ -f "$SERVER_FILE" ]] || die "后端入口不存在：$SERVER_FILE"

cd "$PROJECT_ROOT"

PYTHON_PATH="$(command -v "$PYTHON_BIN")"
PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
BROWSER_HOST="$(browser_host)"
ensure_https_certificate

if VISION_SENTINEL_HOST="$BROWSER_HOST" is_vision_sentinel_running "$VISION_SENTINEL_PORT"; then
  if [[ "$VISION_SENTINEL_HOST" == "0.0.0.0" ]] || [[ "$VISION_SENTINEL_HOST" == "::" ]]; then
    if ! port_has_wildcard_listener "$VISION_SENTINEL_PORT" && [[ "$VISION_SENTINEL_AUTO_PORT" == "1" ]]; then
      set +e
      NEXT_PORT="$(find_available_port)"
      FIND_STATUS=$?
      set -e
      [[ "$FIND_STATUS" == "0" ]] || die "端口 ${VISION_SENTINEL_PORT} 上已有本机服务，且未找到可用的局域网端口。"
      echo "端口 ${VISION_SENTINEL_PORT} 上已有仅本机可访问的 Vision Sentinel，自动切换到 ${NEXT_PORT} 启用局域网访问。"
      VISION_SENTINEL_PORT="$NEXT_PORT"
    else
      echo "Vision Sentinel 已在运行。"
      echo "项目目录：$PROJECT_ROOT"
      echo "本机访问：${VISION_SENTINEL_SCHEME}://${BROWSER_HOST}:${VISION_SENTINEL_PORT}/index.html"
      echo "局域网访问："
      list_lan_urls | sed 's/^/  /'
      echo "健康检查：${VISION_SENTINEL_SCHEME}://${BROWSER_HOST}:${VISION_SENTINEL_PORT}/api/health"
      exit 0
    fi
  else
    echo "Vision Sentinel 已在运行。"
    echo "项目目录：$PROJECT_ROOT"
    echo "本机访问：${VISION_SENTINEL_SCHEME}://${BROWSER_HOST}:${VISION_SENTINEL_PORT}/index.html"
    echo "健康检查：${VISION_SENTINEL_SCHEME}://${BROWSER_HOST}:${VISION_SENTINEL_PORT}/api/health"
    exit 0
  fi
fi

set +e
is_port_free "$VISION_SENTINEL_PORT"
PORT_STATUS=$?
set -e

if [[ "$PORT_STATUS" == "2" ]]; then
  echo "提示：当前环境不允许预检查端口，将直接尝试启动后端。"
elif [[ "$PORT_STATUS" != "0" ]]; then
  if [[ "$VISION_SENTINEL_AUTO_PORT" == "1" ]]; then
    set +e
    NEXT_PORT="$(find_available_port)"
    FIND_STATUS=$?
    set -e
    [[ "$FIND_STATUS" == "0" ]] || die "端口 ${VISION_SENTINEL_PORT} 被占用，且未找到可用端口。"
    echo "端口 ${VISION_SENTINEL_PORT} 已被其他程序占用，自动切换到 ${NEXT_PORT}。"
    VISION_SENTINEL_PORT="$NEXT_PORT"
  else
    die "端口 ${VISION_SENTINEL_PORT} 已被占用。可设置 VISION_SENTINEL_PORT=8010 后重试。"
  fi
fi

export VISION_SENTINEL_HOST
export VISION_SENTINEL_PORT
if [[ "$VISION_SENTINEL_HTTPS" == "1" ]]; then
  export VISION_SENTINEL_SSL_CERT
  export VISION_SENTINEL_SSL_KEY
fi

APP_URL="${VISION_SENTINEL_SCHEME}://${BROWSER_HOST}:${VISION_SENTINEL_PORT}/index.html"
HEALTH_URL="${VISION_SENTINEL_SCHEME}://${BROWSER_HOST}:${VISION_SENTINEL_PORT}/api/health"

echo "正在启动 Vision Sentinel 后端..."
echo "项目目录：$PROJECT_ROOT"
echo "Python：$PYTHON_PATH ($PYTHON_VERSION)"
echo "监听地址：${VISION_SENTINEL_HOST}:${VISION_SENTINEL_PORT}"
echo "访问协议：${VISION_SENTINEL_SCHEME}"
if [[ "$VISION_SENTINEL_HTTPS" == "1" ]]; then
  echo "HTTPS 证书：$VISION_SENTINEL_SSL_CERT"
fi
echo "本机访问：$APP_URL"
if [[ "$VISION_SENTINEL_HOST" == "0.0.0.0" ]] || [[ "$VISION_SENTINEL_HOST" == "::" ]]; then
  echo "局域网访问："
  list_lan_urls | sed 's/^/  /'
fi
echo "健康检查：$HEALTH_URL"
echo "停止服务：在此终端按 Ctrl+C"

if [[ "$VISION_SENTINEL_OPEN_BROWSER" == "1" ]] && command -v open >/dev/null 2>&1; then
  (sleep 1.2; open "$APP_URL") >/dev/null 2>&1 &
fi

exec "$PYTHON_BIN" "$SERVER_FILE"
