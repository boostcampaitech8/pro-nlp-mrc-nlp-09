#!/bin/bash
# ============================================================
# Retrieval Cache 생성 (Train/Val/Test)
# ============================================================
# 실행: bash scripts/retrieval/02_build_cache.sh
# ============================================================

set -e

cd "$(dirname "$0")/../.."

echo "============================================================"
echo "  Step 2: Retrieval Cache 생성"
echo "============================================================"
echo ""
echo "설정:"
echo "  - Retrieval: Weighted Hybrid (BM25 + KURE)"
echo "  - Alpha: 0.7 (BM25:0.7, KURE:0.3)"
echo "  - Top-K: 50"
echo ""

# Train 캐시
echo "[1/3] Train cache 생성..."
python -m src.retrieval.build_retrieval_cache \
    --split train \
    --top_k 50 \
    --alpha 0.7

# Validation 캐시
echo ""
echo "[2/3] Validation cache 생성..."
python -m src.retrieval.build_retrieval_cache \
    --split validation \
    --top_k 50 \
    --alpha 0.7

# Test 캐시
echo ""
echo "[3/3] Test cache 생성..."
python -m src.retrieval.build_retrieval_cache \
    --split test \
    --top_k 50 \
    --alpha 0.7

echo ""
echo "✅ 모든 캐시 생성 완료!"
echo ""
python -m src.retrieval.paths --status
echo ""
echo "다음 단계: make train CONFIG=configs/active/your_config.yaml"
