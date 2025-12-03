#!/bin/bash
# [1] 디렉토리 및 심볼릭 링크 생성
set -e

SHARED_ROOT="/data/ephemeral/home/shared"
HOME_ROOT="/data/ephemeral/home"

# 사용자 목록
USERS=("dahyeong" "minseok" "taewon" "seunghwan" "junbeom" "sehun")

echo ">>> [1/3] 공용 디렉토리 및 사용자별 공간 생성"

# 1. 공용 데이터셋 폴더 생성 (Embeddings 저장용)
mkdir -p "$SHARED_ROOT/datasets/embeddings"

# 2. 사용자별 디렉토리 일괄 생성
echo "    - 사용자별 폴더 확인 및 생성 중..."
for USER in "${USERS[@]}"; do
    # (1) 결과물 저장소 (Shared Outputs)
    mkdir -p "$SHARED_ROOT/outputs/$USER"
    # (2) 개인 작업 공간 (Home)
    mkdir -p "$HOME_ROOT/$USER"
done
echo "    ✅ 모든 유저(6명)의 디렉토리 생성이 완료되었습니다."

echo ""
echo ">>> [2/3] 현재 프로젝트 심릭 링크 연결"

# 3. 현재 사용자 선택
PS3="👉 현재 본인의 ID를 번호로 선택해주세요: "
select CURRENT_USER in "${USERS[@]}"; do
    if [ -n "$CURRENT_USER" ]; then
        echo "    ✅ 선택된 사용자: $CURRENT_USER"
        break
    else
        # [수정] 깨진 부분 복구 (echo 명령어 및 따옴표 추가)
        echo "    ❌ 잘못된 선택입니다. 목록에 있는 번호를 입력해주세요."
    fi
done

# 4. 심볼릭 링크 연결
# ./data -> shared/datasets
if [ -L "./data" ]; then rm ./data; elif [ -d "./data" ]; then mv ./data ./data_backup; fi
ln -sfn "$SHARED_ROOT/datasets" ./data
echo "    ✅ ./data -> $SHARED_ROOT/datasets"

# ./outputs -> shared/outputs/{USER}
if [ -L "./outputs" ]; then rm ./outputs; elif [ -d "./outputs" ]; then mv ./outputs ./outputs_backup; fi
ln -sfn "$SHARED_ROOT/outputs/$CURRENT_USER" ./outputs
echo "    ✅ ./outputs -> $SHARED_ROOT/outputs/$CURRENT_USER"