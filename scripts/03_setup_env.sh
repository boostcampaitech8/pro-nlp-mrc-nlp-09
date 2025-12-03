#!/bin/bash
# [3] 환경 변수 및 가상환경 자동 실행 설정
SHARED_ROOT="/data/ephemeral/home/shared"
SHARED_VENV="$SHARED_ROOT/.venv"
SHELL_CONFIG="$HOME/.bashrc"

echo ">>> [3/3] .bashrc 설정 (가상환경 자동 실행)"

# 1. 공용 가상환경 자동 Activate 등록
# 터미널 켤 때마다 자동으로 shared/.venv가 켜지도록 설정
VENV_ACTIVATE_CMD="source $SHARED_VENV/bin/activate"

if ! grep -q "$SHARED_VENV/bin/activate" "$SHELL_CONFIG"; then
    echo "" >> "$SHELL_CONFIG"
    echo "# [MRC Project] Auto-activate Shared Venv" >> "$SHELL_CONFIG"
    echo "$VENV_ACTIVATE_CMD" >> "$SHELL_CONFIG"
    echo "    ✅ 가상환경 자동 실행이 .bashrc에 등록되었습니다."
else
    echo "    ℹ️  가상환경 자동 실행이 이미 설정되어 있습니다."
fi