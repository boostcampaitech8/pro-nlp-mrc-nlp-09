# Tests & Sanity Check Scripts

> Retrieval 파이프라인 검증 및 실험을 위한 스크립트 모음

---

## 📁 디렉토리 구조

```
tests/
├── README.md                      # 이 파일
│
├── sanity_embedding.py            # ✅ 임베딩 파일 무결성 검증
├── sanity_cache.py                # ✅ Retrieval 캐시 무결성 검증
├── measure_recall.py              # ✅ Recall@k 측정 및 alpha 탐색
│
├── retrieval_sanity.py            # 기존 retrieval 통합 sanity check
├── test_bm25_hybrid.py            # BM25 + Hybrid 테스트
├── test_kure_pipeline_sanity.py   # KURE 파이프라인 sanity check
├── test_retrieval_recall.py       # Retrieval recall 테스트
└── verify_retrieval_consistency.py # Retrieval 일관성 검증
```

---

## 🔧 주요 스크립트 사용법

### 1. sanity_embedding.py - 임베딩 검증

KURE corpus embedding과 passage metadata의 무결성을 검증합니다.

```bash
# 기본 실행 (paths.py 경로 사용)
python -m tests.sanity_embedding

# 특정 파일 지정
python -m tests.sanity_embedding \
    --embedding_path data/embeddings/kure_corpus_emb.npy \
    --meta_path data/embeddings/kure_passages_meta.jsonl

# JSON 출력
python -m tests.sanity_embedding --json
```

**검증 항목:**
- Embedding shape 및 dtype
- L2 Norm (정규화 확인, 1.0이어야 함)
- Embedding 값 분포
- Passage metadata 필드 구조
- Embedding-Meta 개수 일치

---

### 2. sanity_cache.py - 캐시 검증

train/val/test retrieval 캐시 파일의 무결성을 검증합니다.

```bash
# 전체 split 검증
python -m tests.sanity_cache

# 특정 split만
python -m tests.sanity_cache --splits val test

# JSON 출력
python -m tests.sanity_cache --json
```

**검증 항목:**
- 캐시 파일 존재 및 로드 가능 여부
- 항목 구조 (id, question, retrieved)
- 후보 구조 (passage_id, doc_id, score_dense, score_bm25)
- Score 분포 통계
- Passage ID 범위 (embedding과 일치하는지)

---

### 3. measure_recall.py - Recall@k 측정

Validation set을 기준으로 다양한 alpha 값에서 Recall@k를 측정합니다.

```bash
# 기본 실행 (alpha 0.3~1.0)
python -m tests.measure_recall

# 특정 alpha 값들
python -m tests.measure_recall --alphas 0.4 0.5 0.6

# 결과 저장
python -m tests.measure_recall --save_results logs/recall_$(date +%Y%m%d).json

# 세밀한 탐색
python -m tests.measure_recall --alphas 0.35 0.40 0.45 0.50 0.55

# 다른 K 값들
python -m tests.measure_recall --k_list 1 3 5 10 20 30 50
```

**출력:**
- Alpha별 Recall@1, @5, @10, @20, @50
- Best alpha 추천 (R@10 기준)
- JSON 결과 파일 (선택)

---

## 📊 빠른 검증 체크리스트

새로운 임베딩/캐시를 생성한 후 다음 명령어로 빠르게 검증:

```bash
# 1. 임베딩 검증
python -m tests.sanity_embedding

# 2. 캐시 검증
python -m tests.sanity_cache

# 3. Recall 측정 (성능 확인)
python -m tests.measure_recall --alphas 0.4 0.5 0.6
```

모든 테스트가 ✅ PASS이면 Reader 학습 진행 가능.

---

## 📝 결과 해석 가이드

### Embedding 검증
- **L2 Norm ≈ 1.0**: 정상 (normalize_embeddings=True)
- **L2 Norm ≠ 1.0**: 재생성 필요

### Cache 검증
- **Passage ID ≤ max_passage_id**: 정상
- **Passage ID > max_passage_id**: 임베딩과 불일치, 재생성 필요

### Recall@k
- **R@10 > 90%**: 좋음
- **R@10 80-90%**: 보통
- **R@10 < 80%**: 개선 필요 (alpha 조정 또는 retriever 개선)

---

## 📂 관련 문서

- [EMBEDDING_STRATEGY.md](../docs/EMBEDDING_STRATEGY.md) - 임베딩 형식 및 Answer Offset 전략
- [RETRIEVAL_EXPERIMENTS.md](../docs/RETRIEVAL_EXPERIMENTS.md) - 실험 결과 상세 기록
- [KURE_HYBRID_PIPELINE.md](../docs/KURE_HYBRID_PIPELINE.md) - 전체 파이프라인 설계
