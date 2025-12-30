#!/usr/bin/env bash

INTERVAL="${1:-3}"  # 기본 3초

while true; do
  clear
  date "+%Y-%m-%d %H:%M:%S"

  echo "=== LOAD ==="
  uptime
  echo

  echo "=== MEMORY ==="
  free -h
  echo

  echo "=== DISK ==="
  df -h
  echo

  sleep "$INTERVAL"
done
