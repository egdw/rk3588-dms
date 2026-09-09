#!/usr/bin/env bash
# 从 Mac 一键同步项目到 Linux 服务器并重启服务。
# 用法: ./deploy.sh [用户@主机]  默认 ztl@192.168.2.8
set -euo pipefail

REMOTE="${1:-ztl@192.168.2.8}"
DEST="/home/ztl/code/vision-sentinel"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 安全同步:只覆盖/新增文件,不批量删除远程目录中额外存在的文件。
rsync -a \
  --exclude '.DS_Store' \
  --exclude '.git' \
  --exclude '__pycache__/' \
  --exclude 'node_modules/' \
  --exclude 'storage/' \
  --exclude 'certs/' \
  ./ "$REMOTE:$DEST/"

ssh -t "$REMOTE" "sudo systemctl restart vision-sentinel && systemctl is-active vision-sentinel"
echo "部署完成: https://$(echo "$REMOTE" | cut -d@ -f2):8443/index.html#live-detection"
echo "MQTT WebSocket 控制: 检测页会直接连接 wss://192.168.2.13:8084/mqtt, 不需要在 2.8 安装桥接服务。"
