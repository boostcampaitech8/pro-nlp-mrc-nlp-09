#!/bin/bash
# ============================================================
# 전체 KURE Pipeline 실행 (원클릭)
# ============================================================
# 실행: bash scripts/retrieval/run_full_pipeline.sh
# ============================================================

set -e

SCRIPT_DIR="$(dirname "$0")"

echo "============================================================"
echo "  🚀 KURE + BM25 Weighted Hybrid Pipeline"
echo "============================================================"
echo ""

# Step 1: KURE Corpus Embedding
bash "$SCRIPT_DIR/01_build_kure_corpus.sh"

echo ""
echo "============================================================"
echo ""

# Step 2: Retrieval Cache
bash "$SCRIPT_DIR/02_build_cache.sh"

echo ""
echo "============================================================"
echo "  ✅ Pipeline 완료!"
echo "============================================================"
echo ""
echo "이제 학습을 시작하세요:"
echo "  make train CONFIG=configs/active/HANTAEK_roberta_large_hybrid_top10_offline.yaml"
