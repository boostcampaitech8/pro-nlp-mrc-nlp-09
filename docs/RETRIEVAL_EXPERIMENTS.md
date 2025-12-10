# Retrieval 실험 결과 기록

> **마지막 업데이트**: 2025-12-10  
> **목적**: KURE + BM25 Weighted Hybrid Retrieval 실험 결과 상세 문서화

---

## 실험 개요

### 목표
- KURE-v1 Dense Retrieval과 BM25 Sparse Retrieval의 최적 가중치(α) 탐색
- Validation set 기준 Recall@k 성능 측정
- Reader 학습을 위한 최적 설정 도출

### 실험 환경
- **GPU**: Tesla V100-SXM2-32GB
- **Dense Model**: nlpai-lab/KURE-v1 (embedding dim: 1024)
- **Sparse Model**: BM25 (Okt tokenizer)
- **Corpus**: Wikipedia 56,737 unique documents

---

## 1. Corpus Embedding 생성

### 1.1 설정

```yaml
Model: nlpai-lab/KURE-v1
max_seq_length: 1024  # 원본 8192에서 축소 (95.5% 문서 커버)
batch_size: 128
normalize_embeddings: true  # L2 정규화
use_title: true  # "passage: {title} {text}" 형식
chunking: false  # 비활성화
```

### 1.2 토큰 분포 분석 (사전 조사)

KURE-v1 토크나이저로 Wikipedia 문서 샘플(5,000개) 분석:

| Percentile | Token Count |
|------------|-------------|
| 50% (중앙값) | 334 tokens |
| 75% | 493 tokens |
| 90% | 741 tokens |
| 95% | 990 tokens |
| 99% | 1,795 tokens |
| 99.5% | 2,129 tokens |

**토큰/문자 비율**: 평균 0.576 (한국어 특성상 낮음)

### 1.3 max_seq_length 선택 근거

| max_seq_length | Truncation 비율 | 선택 |
|----------------|-----------------|------|
| 512 | 23.3% | ❌ 정보 손실 큼 |
| 768 | 9.1% | |
| **1024** | **4.5%** | ✅ 선택 |
| 1536 | 1.5% | |
| 2048 | 0.6% | |

**결론**: `max_seq_length=1024`로 95.5% 문서 완전 커버, 속도/메모리 최적화

### 1.4 생성 결과

```
파일: data/embeddings/kure_corpus_emb.npy
Shape: (56737, 1024)
Size: 221.6 MB
dtype: float32
L2 Norm: 1.000000 (정규화 완료)
소요 시간: 약 25분
```

---

## 2. Retrieval Cache 생성

### 2.1 설정

```yaml
K_LARGE: 50  # 질문당 상위 50개 후보 저장
alpha: 0.7  # 캐시 생성 시 정렬용 (raw score 저장)
batch_size: 64
```

### 2.2 캐시 구조

```jsonl
{
  "id": "mrc-1-000067",
  "question": "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?",
  "retrieved": [
    {
      "passage_id": 14489,
      "doc_id": 18293,
      "score_dense": 0.5349,
      "score_bm25": 14.9394
    },
    ...
  ]
}
```

### 2.3 생성 결과

| Split | Questions | Candidates/Q | File Size |
|-------|-----------|--------------|-----------|
| train | 3,952 | 50 | 21.8 MB |
| val | 240 | 50 | 1.3 MB |
| test | 600 | 50 | 3.3 MB |

### 2.4 Score 분포

| Score Type | Min | Max | Mean | Std |
|------------|-----|-----|------|-----|
| BM25 | 2.34 | 48.78 | 7.96 | 3.21 |
| Dense (cosine) | 0.20 | 0.81 | 0.45 | 0.09 |

---

## 3. Recall@k 실험

### 3.1 실험 설정

- **평가 데이터**: Validation set (240 questions)
- **평가 방식**: Gold document가 top-k에 포함되는지 확인
- **정규화**: Per-query min-max normalization
- **Hybrid Score 계산**: `α × BM25_norm + (1-α) × Dense_norm`

### 3.2 Alpha별 Recall@k 결과

#### 초기 탐색 (0.1 간격)

| Alpha | R@1 | R@5 | R@10 | R@20 | R@50 |
|-------|-----|-----|------|------|------|
| 0.3 | 60.0% | 87.9% | 92.5% | 95.8% | 97.5% |
| 0.4 | 60.8% | 88.3% | 92.9% | 96.2% | 97.5% |
| 0.5 | 61.3% | 88.8% | 93.3% | 96.2% | 97.5% |
| 0.6 | 60.8% | 88.8% | 92.1% | 95.4% | 97.5% |
| 0.7 | 59.2% | 86.7% | 91.2% | 95.8% | 97.5% |
| 0.8 | 56.2% | 85.0% | 91.7% | 94.2% | 97.5% |
| 0.9 | 55.0% | 81.7% | 90.8% | 93.3% | 97.5% |
| 1.0 (BM25 only) | 55.0% | 81.2% | 89.6% | 92.9% | 97.5% |

#### 세밀한 탐색 (0.05 간격, doc_id 매핑 수정 후)

> **Note**: 2025-12-10 doc_id 매핑 이슈 수정됨. Wikipedia에 3,801개의 중복 텍스트가 있어 
> KURE가 사용하는 first doc_id와 measure_recall이 사용하던 last doc_id가 불일치했음.
> 수정 후 모든 Recall 지표가 +0.4~0.5%p 향상.

| Alpha | R@1 | R@5 | R@10 | R@20 | R@50 |
|-------|-----|-----|------|------|------|
| 0.30 | 61.7% | 90.0% | 93.3% | 96.7% | 97.9% |
| **0.35** | **62.5%** | **90.4%** | **94.2%** | **96.7%** | 97.9% |
| 0.40 | 62.5% | 89.6% | 93.8% | 96.7% | 97.9% |
| 0.45 | 61.3% | 89.6% | 94.2% | 96.7% | 97.9% |
| 0.50 | 61.7% | 89.2% | 93.8% | 96.7% | 97.9% |
| 0.55 | 62.1% | 89.6% | 92.9% | 96.7% | 97.9% |
| 0.60 | 61.3% | 89.2% | 92.5% | 95.8% | 97.9% |
| 0.70 | 59.6% | 87.1% | 91.7% | 96.2% | 97.9% |

### 3.3 분석

1. **최적 Alpha**: **0.35** (BM25 35% + KURE 65%)
   - R@1: 62.5% (최고)
   - R@5: 90.4% (최고)
   - R@10: 94.2% (최고)
   - R@20: 96.7%
   - R@50: 97.9%

2. **Pure BM25 (α=1.0) vs Hybrid (α=0.35)**:
   - R@1: ~55% → 62.5% (+7.5%p)
   - R@10: ~89% → 94.2% (+5.2%p)

3. **Dense의 기여**:
   - α=0.35에서 가장 효과적 (KURE 65% 가중치)
   - KURE의 의미적 유사도가 BM25 키워드 매칭보다 조금 더 중요
   - α < 0.3 또는 α > 0.6에서는 성능 하락

4. **R@50 = 97.9%**:
   - 모든 α에서 동일
   - 캐시에 50개 후보를 저장하면 97.9% 문서 커버 가능
   - Reader가 top-50 중에서 정답을 찾을 수 있는 상한선

5. **doc_id 매핑 수정 영향**:
   - Wikipedia에 3,801개 duplicate text 존재 (56,737 unique / 60,613 total)
   - KURE는 first doc_id 사용, 측정 스크립트는 last doc_id 사용 → 불일치
   - 수정 후 모든 지표 +0.4~0.5%p 상승

### 3.4 실험 재현 방법

```bash
# Recall 측정
python -m tests.measure_recall --alphas 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

# 결과 저장
python -m tests.measure_recall --save_results logs/recall_experiment_$(date +%Y%m%d).json
```

---

## 4. 권장 설정

### 4.1 Reader 학습 시 설정

```yaml
# configs/active/your_config.yaml
retrieval:
  alpha: 0.35  # BM25 35% + KURE 65% (최적값)
  k_ret: 20    # Dynamic Hard Negative용 후보 수
  
dataset:
  use_retrieval_cache: true
  cache_path: data/cache/retrieval/train_top50.jsonl
```

### 4.2 캐시 재생성 없이 Alpha 변경

캐시에는 raw score만 저장되어 있으므로, Reader 학습 시 `alpha` 파라미터만 변경하면 됨:

```python
# src/datasets/mrc_with_retrieval.py
candidates = compute_hybrid_score_for_candidates(candidates, alpha=0.5)
```

---

## 5. 향후 실험 계획

### 5.1 추가 실험 후보

1. **더 세밀한 Alpha 탐색**: 0.45, 0.50, 0.55 범위
2. **K_RET 변화에 따른 성능**: k_ret = 10, 20, 30, 40
3. **Hard Negative 비율 조정**: hard/medium 비율 실험
4. **Train set Recall 분석**: Overfitting 여부 확인

### 5.2 실험 시 주의사항

- Recall 측정은 **validation set만** 사용 (test set 분석 금지)
- Alpha 변경 실험은 캐시 재생성 불필요
- 극단적인 α (0.0 또는 1.0)는 캐시 top-50 외의 문서가 필요할 수 있음

---

## 6. Sanity Check 실행 방법

### 6.1 임베딩 검증

```bash
python -m tests.sanity_embedding
```

### 6.2 캐시 검증

```bash
python -m tests.sanity_cache
python -m tests.sanity_cache --splits val  # 특정 split만
```

### 6.3 Recall 측정

```bash
python -m tests.measure_recall
python -m tests.measure_recall --alphas 0.5 0.7 --save_results logs/recall.json
```

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2025-12-10 | 초기 실험 완료, 세밀한 탐색으로 alpha=0.35 최적 확인 |
| 2025-12-10 | doc_id 매핑 이슈 수정 (first vs last), Recall +0.4%p 향상 |
