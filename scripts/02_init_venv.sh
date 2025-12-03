#!/bin/bash
# [3] uv 설치 및 공용 가상환경 생성 (Bashrc 등록 포함)
set -e

SHARED_ROOT="/data/ephemeral/home/shared"
SHARED_VENV="$SHARED_ROOT/.venv"
SHELL_CONFIG="$HOME/.bashrc"

echo ">>> [2/3] uv 설치 및 공용 가상환경 생성 (Python 3.11)"

# ------------------------------------------------------------------
# 1. uv 설치 및 .bashrc 등록
# ------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "    📦 uv 설치를 시작합니다..."
    
    # 설치 스크립트 실행
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # 설치된 uv 경로 찾기 (보통 ~/.local/bin 또는 ~/.cargo/bin)
    if [ -f "$HOME/.local/bin/env" ]; then
        UV_ENV_PATH="$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        UV_ENV_PATH="$HOME/.cargo/env"
    else
        # fallback: 직접 PATH 추가
        UV_ENV_PATH=""
    fi

    # 현재 세션에 즉시 적용
    if [ -n "$UV_ENV_PATH" ]; then
        source "$UV_ENV_PATH"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # .bashrc에 영구 등록 (이미 등록되어 있지 않다면 추가)
    if ! grep -q "uv PATH" "$SHELL_CONFIG"; then
        echo "" >> "$SHELL_CONFIG"
        echo "# [MRC Project] uv PATH" >> "$SHELL_CONFIG"
        
        if [ -n "$UV_ENV_PATH" ]; then
            echo "source \"$UV_ENV_PATH\"" >> "$SHELL_CONFIG"
        else
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
        fi
        echo "    ✅ .bashrc에 uv 경로가 등록되었습니다."
    fi
else
    echo "    ✅ uv가 이미 설치되어 있습니다."
fi

# ------------------------------------------------------------------
# 2. 공용 가상환경 생성 및 패키지 설치
# ------------------------------------------------------------------
if [ ! -d "$SHARED_VENV" ]; then
    echo "    📦 공용 가상환경 생성 중 (Python 3.11)..."
    echo "       경로: $SHARED_VENV"
    
    # uv를 사용하여 가상환경 생성
    uv venv "$SHARED_VENV" --python 3.11
    
    # 패키지 설치를 위해 가상환경 활성화
    source "$SHARED_VENV/bin/activate"
    
    if [ -f "requirements.txt" ]; then 
        echo "    📦 requirements.txt 설치 중..."
        uv pip install -r requirements.txt
    fi
    if [ -f "deps.txt" ]; then 
        echo "    📦 deps.txt 설치 중..."
        uv pip install -r deps.txt
    fi
    
    echo "    ✅ 가상환경 생성 및 라이브러리 설치 완료"
else
    echo "    ℹ️  공용 가상환경이 이미 존재합니다."
fi