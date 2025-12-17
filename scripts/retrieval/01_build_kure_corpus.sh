#!/bin/bash
# ============================================================
# KURE Corpus Embedding 생성
# ============================================================
# 실행: bash scripts/retrieval/01_build_kure_corpus.sh
# ============================================================

set -e  # 에러 시 중단

cd "$(dirname "$0")/../.."  # 프로젝트 루트로 이동

echo "============================================================"
echo "  Step 1: KURE Corpus Embedding 생성"
echo "============================================================"
echo ""
echo "설정:"
echo "  - Model: nlpai-lab/KURE-v1"
echo "  - Chunking: OFF (train은 gold context 사용)"
echo "  - Output: data/embeddings/kure_corpus_emb.npy"
echo "           data/embeddings/kure_passages_meta.jsonl"
echo ""

# 실행
python -m src.retrieval.embed.build_kure_corpus \
    --wiki_path ./data/wikipedia_documents.json \
    --output_emb_path ./data/embeddings/kure_corpus_emb.npy \
    --output_meta_path ./data/embeddings/kure_passages_meta.jsonl \
    --batch_size 64

echo ""
echo "✅ KURE corpus embedding 생성 완료!"
echo ""
echo "다음 단계: bash scripts/retrieval/02_build_cache.sh"
