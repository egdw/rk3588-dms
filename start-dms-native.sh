#!/usr/bin/env bash
# 一键启动 RK3588 原生 DMS(网页后端 :8080 + NPU 推理服务 :8600)
#
# 用法(板上):
#   cd ~/code/rk3588-dms && ./start-dms-native.sh            # 启动并后台常驻
#   ./start-dms-native.sh stop                               # 停止两个服务
#   ./start-dms-native.sh status                             # 查看运行状态
#   ./start-dms-native.sh logs [backend|service]             # 跟踪日志
#
# 启动后浏览器打开(注意 http, 不要 https):
#   http://<板子IP>:8080/index.html?infer=native#/live-detection
# 回退纯浏览器推理: 去掉 ?infer=native

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv/bin/python"
[ -x "$VENV" ] || VENV="python3"   # backend 纯 stdlib, 无 venv 也能跑
LOG_DIR="$ROOT/logs/dms"
mkdir -p "$LOG_DIR"

BACKEND_PORT="${DMS_BACKEND_PORT:-8080}"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
SERVICE_PID_FILE="$LOG_DIR/service.pid"
BACKEND_LOG="$LOG_DIR/backend.log"
SERVICE_LOG="$LOG_DIR/service.log"

is_running() {  # $1: pid 文件
  [ -f "$1" ] || return 1
  local pid
  pid="$(cat "$1" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_one() {  # $1: 名称 $2: pid 文件 $3: 日志 $4: 启动命令(字符串, 经 bash -c exec)
  local name="$1" pidfile="$2" logfile="$3" cmd="$4"
  if is_running "$pidfile"; then
    echo "[SKIP] $name 已在运行 (pid $(cat "$pidfile"))"
    return 0
  fi
  rm -f "$pidfile"
  nohup bash -c "$cmd" > "$logfile" 2>&1 &
  echo $! > "$pidfile"
  echo "[OK]   $name 已启动 (pid $(cat "$pidfile")), 日志: ${logfile#$ROOT/}"
}

stop_one() {  # $1: 名称 $2: pid 文件
  if is_running "$2"; then
    local pid
    pid="$(cat "$2")"
    kill "$pid" 2>/dev/null
    # 服务有子进程(taskset 包装), 按进程组再补一刀
    pkill -P "$pid" 2>/dev/null
    sleep 1
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
    echo "[OK]   $1 已停止 (pid $pid)"
  else
    echo "[SKIP] $1 未在运行"
  fi
  rm -f "$2"
}

wait_health() {  # $1: url $2: 最长等待秒数
  local waited=0
  while [ "$waited" -lt "$2" ]; do
    curl -sf -o /dev/null --max-time 2 "$1" && return 0
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

case "${1:-start}" in
  start)
    echo "== 启动原生 DMS =="

    # 1/2 网页后端(静态页/手动控制/6DRepNet), 端口 ${BACKEND_PORT}(8000 让给 g29-service)
    start_one "backend(${BACKEND_PORT})" "$BACKEND_PID_FILE" "$BACKEND_LOG" \
      "cd '$ROOT' && VISION_SENTINEL_PORT=$BACKEND_PORT exec $VENV backend/server.py"

    # 2/2 原生 NPU 服务(MJPEG/三模型INT8三核推理/WS/报警音频), 端口 8600, 绑大核
    start_one "dms_service(8600)" "$SERVICE_PID_FILE" "$SERVICE_LOG" \
      "cd '$ROOT' && DMS_CAMERA=\${DMS_CAMERA:-/dev/video1} exec taskset -c 4-7 '$ROOT/.venv/bin/python' rk3588_dms/service/dms_service.py --camera \${DMS_CAMERA:-/dev/video1}"

    echo "== 等待就绪 =="
    if wait_health "http://127.0.0.1:${BACKEND_PORT}/api/health" 15; then
      echo "[OK]   backend  健康: http://127.0.0.1:${BACKEND_PORT}/api/health"
    else
      echo "[WARN] backend 15s 未就绪, 查看 logs/dms/backend.log"
    fi
    if wait_health "http://127.0.0.1:8600/health" 30; then
      echo "[OK]   service 健康: http://127.0.0.1:8600/health (含 NPU 模型加载)"
    else
      echo "[WARN] service 30s 未就绪, 查看 logs/dms/service.log"
    fi

    IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [ -n "$IP" ] || IP="<板子IP>"
    echo
    echo "=============================================================="
    echo " 浏览器打开(默认即原生 NPU 推理):"
    echo "   http://${IP}:${BACKEND_PORT}/dms"
    echo " 回退纯浏览器推理:"
    echo "   http://${IP}:${BACKEND_PORT}/index.html?infer=browser#/live-detection"
    echo " 停止: ./start-dms-native.sh stop   状态: status   日志: logs [service|backend]"
    echo "=============================================================="
    ;;

  stop)
    stop_one "dms_service" "$SERVICE_PID_FILE"
    stop_one "backend" "$BACKEND_PID_FILE"
    ;;

  restart)
    bash "$0" stop
    sleep 1
    bash "$0" start
    ;;

  status)
    for entry in "backend(${BACKEND_PORT}):$BACKEND_PID_FILE" "dms_service(8600):$SERVICE_PID_FILE"; do
      name="${entry%%:*}"
      pidfile="${entry#*:}"
      if is_running "$pidfile"; then
        echo "[RUN ] $name (pid $(cat "$pidfile"))"
      else
        echo "[DOWN] $name"
      fi
    done
    curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8600/health \
      && echo "       service /health 可达" || echo "       service /health 不可达"
    ;;

  logs)
    target="${2:-service}"
    case "$target" in
      backend) tail -f "$BACKEND_LOG" ;;
      service) tail -f "$SERVICE_LOG" ;;
      *) echo "用法: $0 logs [backend|service]"; exit 1 ;;
    esac
    ;;

  *)
    echo "用法: $0 {start|stop|restart|status|logs [backend|service]}"
    exit 1
    ;;
esac
