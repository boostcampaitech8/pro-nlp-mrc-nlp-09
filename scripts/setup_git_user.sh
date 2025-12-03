#!/bin/bash
# scripts/setup_git_user.sh
# 설명: 현재 프로젝트에 한정하여 Git 사용자 이름과 이메일을 설정합니다.

# 에러 발생 시 중단
set -e

echo "========================================================"
echo "   Git Local Configuration Setup"
echo "========================================================"

# 1. Git 저장소인지 확인
if [ ! -d ".git" ]; then
    echo "❌ 에러: 현재 디렉토리에 .git 폴더가 없습니다."
    echo "   프로젝트 루트(MRC_Project)에서 이 스크립트를 실행해주세요."
    exit 1
fi

echo "ℹ️  이 설정은 현재 프로젝트(--local)에만 적용됩니다."
echo "   서버의 다른 사용자에게 영향을 주지 않습니다."
echo ""

# 2. 사용자 입력 받기
while true; do
    read -p "👉 Git에 표시될 이름(User Name)을 입력하세요: " GIT_NAME
    if [ -z "$GIT_NAME" ]; then
        echo "   이름은 비어있을 수 없습니다."
        continue
    fi
    
    read -p "👉 Git에 표시될 이메일(User Email)을 입력하세요: " GIT_EMAIL
    if [ -z "$GIT_EMAIL" ]; then
        echo "   이메일은 비어있을 수 없습니다."
        continue
    fi

    echo ""
    echo "--------------------------------------------------------"
    echo "   이름  : $GIT_NAME"
    echo "   이메일: $GIT_EMAIL"
    echo "--------------------------------------------------------"
    read -p "✅ 위 정보로 설정하시겠습니까? (y/n): " CONFIRM

    if [[ "$CONFIRM" == "y" || "$CONFIRM" == "Y" ]]; then
        break
    else
        echo "   다시 입력합니다..."
        echo ""
    fi
done

# 3. Git Config 적용 (절대 경로 무관하게 현재 git context 기준 local 적용)
git config --local user.name "$GIT_NAME"
git config --local user.email "$GIT_EMAIL"

# 4. 결과 확인
echo ""
echo "✅ 설정이 완료되었습니다! (현재 설정 확인)"
git config --local --list | grep user
echo "========================================================"