#!/bin/bash
# 유용한 시스템 도구 설치 (ncdu, tree, htop, tmux 등)
set -e

echo ">>> [추가] 시스템 유틸리티 설치"

# 1. 패키지 목록 정의
PACKAGES=(
    "ncdu"      # 디스크 용량 분석 (강력 추천)
    "tree"      # 디렉토리 구조 시각화
    "htop"      # CPU/RAM 사용량 모니터링 (top보다 예쁨)
    "tmux"      # 터미널 세션 유지 (SSH 끊겨도 작업 유지)
    "curl"      # 파일 다운로드
    "vim"       # 에디터
)

# 2. apt 업데이트 및 설치
echo "    📦 패키지 목록 업데이트 중..."
apt-get update -qq  # -qq: 조용히 실행

echo "    📦 다음 패키지들을 설치합니다: ${PACKAGES[*]}"
# DEBIAN_FRONTEND=noninteractive: 설치 중 Yes/No 물어보지 않게 함
DEBIAN_FRONTEND=noninteractive apt-get install -y "${PACKAGES[@]}"

echo "    ✅ 시스템 도구 설치 완료!"