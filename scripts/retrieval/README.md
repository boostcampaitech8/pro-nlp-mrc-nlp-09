# Retrieval Pipeline Scripts

KURE + BM25 Weighted Hybrid Retrieval 파이프라인 실행 스크립트.

## 실행 순서

### 전체 파이프라인 (원클릭)
```bash
bash scripts/retrieval/run_full_pipeline.sh
```

### 개별 단계 실행

1. **KURE Corpus Embedding 생성**
   ```bash
   bash scripts/retrieval/01_build_kure_corpus.sh
   ```
   - 소요 시간: ~10-15분 (GPU 기준)
   - 출력: `data/embeddings/kure_corpus_emb.npy`, `kure_passages_meta.jsonl`

2. **Retrieval Cache 생성**
   ```bash
   bash scripts/retrieval/02_build_cache.sh
   ```
   - 소요 시간: ~5-10분
   - 출력: `data/cache/retrieval/{train,val,test}_top50.jsonl`

## 상태 확인
```bash
python -m src.retrieval.paths --status
```

## 학습 실행
```bash
make train CONFIG=configs/active/HANTAEK_roberta_large_hybrid_top10_offline.yaml
```
