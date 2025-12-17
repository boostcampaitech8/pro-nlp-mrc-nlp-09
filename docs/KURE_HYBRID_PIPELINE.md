# KURE-v1 + BM25 Weighted Hybrid Retrieval Pipeline

이 문서는 KURE-v1 기반 Dense Retrieval과 BM25를 결합한 Weighted Hybrid Retrieval 파이프라인 사용법을 설명합니다.

## 목차

1. [개요](#개요)
2. [파일 구조](#파일-구조)
3. [실행 순서](#실행-순서)
4. [상세 설명](#상세-설명)
5. [설정 옵션](#설정-옵션)
6. [train.py 통합](#trainpy-통합)

---

## 개요

### 주요 특징

- **Dense 모델**: `nlpai-lab/KURE-v1` (SentenceTransformer)
- **Sparse 모델**: BM25 (bm25s 라이브러리)
- **Hybrid 방식**: Per-query min-max 정규화 + α 가중합
- **학습 전략**: Dynamic Hard Negative Sampling
- **캐시 기반**: 오프라인 retrieval → JSONL 캐시 → 학습/추론

### 핵심 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `K_LARGE` | 50 | 캐시에 저장할 최대 후보 수 |
| `K_RET` | 20 | train/val/test에서 사용할 top-k |
| `K_READ` | 3 | train에서 한 번에 선택되는 context 수 |
| `alpha` | 0.7 | BM25 가중치 (KURE는 1-alpha) |

---

## 파일 구조

```
src/
├── retrieval/
│   ├── kure.py                      # KureRetrieval 클래스
│   ├── weighted_hybrid.py           # WeightedHybridRetrieval 클래스
│   ├── build_retrieval_cache.py     # 캐시 생성 스크립트
│   ├── compare_retrieval_methods.py # BM25/KURE/Hybrid 비교
│   └── embed/
│       └── build_kure_corpus.py     # KURE corpus embedding 생성
├── datasets/
│   ├── __init__.py
│   └── mrc_with_retrieval.py        # MRCWithRetrievalDataset
└── ...

data/
├── kure_corpus_emb.npy              # KURE corpus embedding
├── kure_passages_meta.jsonl         # Passage 메타데이터
└── retrieval_cache/
    ├── train_top50.jsonl
    ├── val_top50.jsonl
    └── test_top50.jsonl

configs/
└── exp_kure_weighted_hybrid.yaml    # 실험 설정
```

---

## 실행 순서

### Step 1: KURE Corpus Embedding 생성

```bash
# 기본 (chunking 없이)
python -m src.retrieval.embed.build_kure_corpus

# Chunking 적용 (Priority B)
python -m src.retrieval.embed.build_kure_corpus --use_chunking
```

출력:
- `./data/kure_corpus_emb.npy` (shape: [num_passages, 1024])
- `./data/kure_passages_meta.jsonl`

### Step 2: BM25/KURE/Hybrid 비교 (Optional)

```bash
python -m src.retrieval.compare_retrieval_methods \
    --alphas 0.6 0.7 0.8 \
    --output_path ./logs/retrieval_comparison.json
```

최적의 α 값을 선택합니다 (MRR@20 기준).

### Step 3: Retrieval 캐시 생성

```bash
python -m src.retrieval.build_retrieval_cache \
    --splits train val test \
    --alpha 0.7 \
    --k_large 50
```

출력:
- `./data/retrieval_cache/train_top50.jsonl`
- `./data/retrieval_cache/val_top50.jsonl`
- `./data/retrieval_cache/test_top50.jsonl`

### Step 4: Reader 학습 (Dynamic Hard Negative)

```bash
python train.py configs/exp_kure_weighted_hybrid.yaml
```

### Step 5: 추론 & 제출

```bash
python inference.py configs/exp_kure_weighted_hybrid.yaml
```

---

## 상세 설명

### 1. Weighted Hybrid Retrieval

기존 RRF(Reciprocal Rank Fusion)과 달리, **점수 기반 가중합**을 사용합니다:

```python
# Per-query 정규화
bm25_norm = (bm25 - bm25.min()) / (bm25.max() - bm25.min() + eps)
dense_norm = (dense - dense.min()) / (dense.max() - dense.min() + eps)

# 가중합
hybrid = α * bm25_norm + (1-α) * dense_norm
```

**장점**:
- 점수 스케일 차이 해결
- α로 직관적인 비율 조절
- Tie-breaking: BM25 → doc_id 순서

### 2. Dynamic Hard Negative Sampling

Train 시 매 iteration마다:

1. Gold document의 passage를 **positive**로 사용
2. Retrieval top-k 중 gold가 아닌 것들을 **hard negative**로 분류
3. Hard(상위 5개) / Medium(나머지) 구분
4. Pool에서 무작위 선택 → **Dynamic**

```python
# 예시: k_ret=20, k_read=3
pos_ctx = random.choice(pos_list)         # 1개
neg_hard = random.sample(hard_neg, 1~2)   # 1~2개
neg_medium = random.sample(medium_neg, 0~1)  # 0~1개

# 최종 pool에서 하나 선택
chosen = random.choice([pos_ctx] + neg_hard + neg_medium)
```

### 3. Chunking (Priority B)

긴 문서(>2000자)를 sliding window로 분할:

```
CHUNK_SIZE = 2000
STRIDE = 500

text = "..."  # 5000자
chunks = [
    (text[0:2000], 0, 2000),
    (text[1500:3500], 1500, 3500),
    (text[3000:5000], 3000, 5000),
]
```

---

## 설정 옵션

### YAML 설정 예시

```yaml
# configs/exp_kure_weighted_hybrid.yaml

retrieval_type: weighted_hybrid  # 핵심!
top_k_retrieval: 20
train_retrieval: true

retrieval:
  weighted_hybrid_alpha: 0.7
  kure_corpus_emb_path: ./data/kure_corpus_emb.npy
  kure_passages_meta_path: ./data/kure_passages_meta.jsonl
  retrieval_cache_dir: ./data/retrieval_cache

dynamic_hard_negative:
  enabled: true      # ← 이 설정이 true면 캐시 기반 DHN 학습 사용
  k_ret: 20          # retrieval top-k
  k_read: 3          # train에서 선택되는 context 수
  alpha: 0.7         # hybrid score 계산용 BM25 가중치
  use_title: true    # title을 context에 포함
```

### Factory 함수 사용

```python
from src.retrieval import get_retriever

retriever = get_retriever(
    retrieval_type="weighted_hybrid",
    tokenize_fn=tokenizer.tokenize,
    data_path="./data",
    corpus_emb_path="./data/kure_corpus_emb.npy",
    passages_meta_path="./data/kure_passages_meta.jsonl",
    alpha=0.7,
)
retriever.build()
```

---

## train.py 통합

### 동작 방식

`train.py`는 다음 조건을 확인하여 학습 방식을 결정합니다:

1. **YAML의 `dynamic_hard_negative.enabled`가 `true`인가?**
2. **캐시 파일이 존재하는가?** (`./data/retrieval_cache/train_top50.jsonl` 등)

두 조건이 모두 충족되면 → **MRCWithRetrievalDataset** 사용 (Dynamic Hard Negative)
그렇지 않으면 → **기존 HF Dataset.map()** 방식 사용

### 코드 흐름

```python
# train.py 내부 (간략화)

if use_dynamic_hard_negative:
    # 캐시 로드
    train_cache = load_retrieval_cache("./data/retrieval_cache/train_top50.jsonl")
    passages_corpus = load_passages_corpus(...)
    
    # MRCWithRetrievalDataset 사용
    train_dataset = MRCWithRetrievalDataset(
        examples=datasets["train"],
        retrieval_cache=train_cache,
        passages_corpus=passages_corpus,
        tokenizer=tokenizer,
        mode="train",
        k_ret=20,
        k_read=3,
        alpha=0.7,
    )
else:
    # 기존 방식
    train_dataset = datasets["train"].map(prepare_train_features, ...)
```

### 변경 사항 요약

| 항목 | 변경량 | 설명 |
|------|-------|------|
| import 추가 | +7줄 | MRCWithRetrievalDataset 등 |
| 캐시 확인 로직 | +30줄 | YAML 파싱 + 파일 존재 확인 |
| Dataset 생성 분기 | +40줄 | DHN 모드 vs 기존 모드 |
| 기존 코드 삭제 | 0줄 | 기존 로직 100% 유지 |

**총 추가: ~77줄**, 기존 로직 변경 없음

---

## 문제 해결

### 1. CUDA OOM

```yaml
per_device_train_batch_size: 4  # 줄이기
gradient_accumulation_steps: 4  # 늘리기
```

### 2. Embedding 생성 시간 단축

```bash
python -m src.retrieval.embed.build_kure_corpus --batch_size 128
```

### 3. 캐시 재생성 없이 α 변경

캐시에는 raw score만 저장되므로, 추론 시 다른 α로 hybrid score 재계산 가능:

```python
from src.datasets.mrc_with_retrieval import compute_hybrid_score_for_candidates

candidates = cache[qid]["retrieved"]
sorted_candidates = compute_hybrid_score_for_candidates(candidates, alpha=0.8)
```

### 4. Dynamic Hard Negative가 활성화되지 않음

```bash
# 캐시 파일 존재 확인
ls -la ./data/retrieval_cache/

# YAML 설정 확인
grep -A5 "dynamic_hard_negative" configs/exp_kure_weighted_hybrid.yaml
```

---

## Quick Start (전체 파이프라인)

```bash
# 1. KURE embedding 생성 (~30분)
python -m src.retrieval.embed.build_kure_corpus

# 2. α 비교 (~10분)
python -m src.retrieval.compare_retrieval_methods --alphas 0.6 0.7 0.8

# 3. 캐시 생성 (~20분)
python -m src.retrieval.build_retrieval_cache --splits train val test

# 4. 학습 (~2시간)
python train.py configs/exp_kure_weighted_hybrid.yaml

# 5. 추론 (~10분)
python inference.py configs/exp_kure_weighted_hybrid.yaml
```

---

## 참고 자료

- [KURE-v1 논문/모델](https://huggingface.co/nlpai-lab/KURE-v1)
- [bm25s 라이브러리](https://github.com/xhluca/bm25s)
- [Hard Negative Mining in Dense Retrieval](https://arxiv.org/abs/2007.00808)
