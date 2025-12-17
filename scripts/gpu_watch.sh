#!/usr/bin/env bash

INTERVAL="${1:-1}"  # 기본 1초, 인자로 변경 가능

while true; do
  clear
  date "+%Y-%m-%d %H:%M:%S"
  nvidia-smi
  sleep "$INTERVAL"
done
