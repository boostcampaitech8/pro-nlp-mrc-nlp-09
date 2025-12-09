# 임베딩 파일 형식 및 Answer Offset 전략

> **작성일**: 2024-12-10  
> **목적**: ODQA 파이프라인에서 임베딩 파일 관리 및 Answer Offset 처리 전략 문서화

---

## 1. 임베딩 파일 구조

### 1.1 파일 위치 (중앙 집중 관리)

모든 임베딩/캐시 경로는 `src/retrieval/paths.py`에서 관리:

```
data/
├── embeddings/                          # Dense 임베딩
│   ├── koe5_corpus_emb.npy             # KoE5 corpus embedding (222M)
│   ├── kure_corpus_emb.npy             # KURE-v1 corpus embedding (~220M)
│   └── kure_passages_meta.jsonl        # Passage 메타데이터 (chunking 시)
│
├── indices/                             # Sparse 인덱스
│   └── sparse/
│       ├── bm25_model_okt.bin          # BM25 모델 (Okt 토크나이저)
│       ├── sparse_embedding.bin        # TF-IDF embedding
│       └── tfidv.bin                   # TF-IDF vectorizer
│
├── cache/                               # Retrieval 캐시
│   └── retrieval/
│       ├── train_top50.jsonl           # Train set retrieval 결과
│       ├── val_top50.jsonl             # Validation set retrieval 결과
│       └── test_top50.jsonl            # Test set retrieval 결과
│
└── wikipedia_documents.json             # 원본 corpus (56,737 문서)
```

### 1.2 파일 형식 상세

#### Dense Embedding (`.npy`)

```python
# 저장
corpus_emb = model.encode(texts, normalize_embeddings=True)
np.save("kure_corpus_emb.npy", corpus_emb)

# 로드
corpus_emb = np.load("kure_corpus_emb.npy")
# shape: (num_passages, embedding_dim)
# KoE5: (56737, 768)
# KURE: (56737, 1024)
```

**중요**: 임베딩은 **L2 정규화**되어 있어서 내적(dot product) = 코사인 유사도

#### Passage Metadata (`.jsonl`)

```jsonl
{"passage_id": 0, "doc_id": 0, "title": "문서제목", "text": "...", "start_char": 0, "end_char": 500, "is_chunk": false}
{"passage_id": 1, "doc_id": 0, "title": "문서제목", "text": "...", "start_char": 500, "end_char": 1000, "is_chunk": true}
...
```

| 필드 | 설명 |
|------|------|
| `passage_id` | 유일한 passage 식별자 (0부터 순차) |
| `doc_id` | 원본 wiki document ID |
| `title` | 문서 제목 |
| `text` | passage 텍스트 |
| `start_char` | 원본 문서 내 시작 위치 |
| `end_char` | 원본 문서 내 끝 위치 |
| `is_chunk` | chunking 여부 |

#### Retrieval Cache (`.jsonl`)

```jsonl
{
  "id": "mrc-0-000001",
  "question": "질문 텍스트",
  "retrieved": [
    {"passage_id": 123, "doc_id": 456, "score_dense": 0.85, "score_bm25": 12.5},
    {"passage_id": 789, "doc_id": 101, "score_dense": 0.82, "score_bm25": 11.2},
    ...
  ]
}
```

---

## 2. Answer Offset 전략 (⚠️ 핵심)

### 2.1 문제 상황

Train 데이터의 `answer_start`는 **원본 gold context** 기준 인덱스:

```python
example = {
    "context": "원본 gold context 텍스트...",  # 원본 문서
    "answers": {
        "text": ["정답"],
        "answer_start": [150]  # ← 원본 context 기준!
    }
}
```

그런데 Retrieval passage는 **chunking된 다른 텍스트**:

```python
retrieval_passage = "chunked 또는 다른 문서의 텍스트..."  # answer_start=150이 무의미
```

**결과**: `answer_start=150`을 그대로 쓰면 엉뚱한 위치가 label이 됨  
→ 모델이 잘못된 gradient를 받음 → 학습 효과 감소

### 2.2 해결 전략

#### 옵션 A: Positive는 Gold Context만 사용 (✅ 현재 구현)

```python
def _get_train_item(self, example, qid, question):
    # Positive: 항상 원본 gold context 사용
    selected_contexts.append(("pos", None))  # None = gold context 사용
    
    # Negative만 retrieval passage 사용
    for neg in hard_negatives:
        selected_contexts.append(("neg", neg))
    
    label, chosen = random.choice(selected_contexts)
    
    if label == "pos":
        # ✅ 원본 context 사용 → answer_start가 정확함
        return self._tokenize_with_gold_context(example, question)
    else:
        # Negative: CLS token이 answer
        return self._tokenize_without_answer(question, retrieval_passage)
```

**장점**:
- Answer offset 문제 100% 해소
- 구현 단순
- Hard negative는 여전히 retrieval에서 가져옴 (학습 효과 유지)

#### 옵션 B: Retrieval Passage에서도 Positive 사용 (확장용)

```python
def _get_train_item(self, example, qid, question):
    # ...
    if label == "pos" and use_retrieval_positive:
        answer_text = example["answers"]["text"][0]
        local_start = retrieval_passage.find(answer_text)
        
        if local_start == -1:
            # 정답이 passage에 없으면 gold context로 fallback
            return self._tokenize_with_gold_context(example, question)
        
        # ✅ passage 기준으로 answer_start 재계산
        return self._tokenize_with_answer_in_passage(
            question, retrieval_passage, answer_text, local_start
        )
```

**주의사항**:
- `answer_text.find()`는 **첫 번째 매칭**만 반환
- 동일 텍스트가 여러 번 나오면 잘못된 위치 가능
- 따라서 옵션 A가 더 안전함

---

## 3. 임베딩 생성/로드 흐름

### 3.1 Corpus Embedding 생성

```bash
# KURE corpus embedding 생성
python -m src.retrieval.embed.build_kure_corpus

# 내부 동작:
# 1. wikipedia_documents.json 로드
# 2. 중복 제거 (56,737개 unique passages)
# 3. SentenceTransformer("nlpai-lab/KURE-v1") 로드
# 4. 배치 인코딩 (normalize=True)
# 5. data/embeddings/kure_corpus_emb.npy 저장
# 6. data/embeddings/kure_passages_meta.jsonl 저장
```

### 3.2 Retrieval Cache 생성

```bash
# Weighted Hybrid (BM25 + KURE) 캐시 생성
python -m src.retrieval.build_retrieval_cache \
    --split train \
    --top_k 50 \
    --alpha 0.7
```

### 3.3 Reader에서 사용

```python
# 1. 캐시 로드
cache = load_retrieval_cache("data/cache/retrieval/train_top50.jsonl")

# 2. Passage corpus 로드
passages = load_passages_corpus(
    passages_meta_path="data/embeddings/kure_passages_meta.jsonl"
)

# 3. Dataset 생성
dataset = MRCWithRetrievalDataset(
    examples=train_examples,
    retrieval_cache=cache,
    passages_corpus=passages,
    tokenizer=tokenizer,
    mode="train",
)
```

---

## 4. Chunking 전략

### 4.1 언제 Chunking 하나?

| 상황 | Chunking |
|------|----------|
| 문서 길이 ≤ max_length | 필요 없음 |
| 문서 길이 > max_length | 필요함 |
| Dense retrieval (KURE/KoE5) | 보통 필요 |
| BM25 retrieval | 보통 불필요 |

### 4.2 Chunking 파라미터

```python
# build_kure_corpus.py 기본값
CHUNK_SIZE = 400      # 각 chunk 최대 글자 수
CHUNK_OVERLAP = 50    # chunk 간 겹침
MIN_CHUNK_SIZE = 100  # 최소 chunk 크기 (너무 작으면 버림)
```

### 4.3 Chunking 후 메타데이터 관리

```python
# 원본 문서 → 여러 passage로 분할
doc_id = 123
passages = [
    {"passage_id": 0, "doc_id": 123, "start_char": 0, "end_char": 400, "is_chunk": True},
    {"passage_id": 1, "doc_id": 123, "start_char": 350, "end_char": 750, "is_chunk": True},
    ...
]
```

---

## 5. 경로 관리 규칙

### 5.1 중앙 집중 관리 원칙

**절대 하드코딩 금지**. 모든 경로는 `paths.py`에서 가져오기:

```python
# ❌ 잘못된 방법
corpus_emb_path = "./data/embeddings/kure_corpus_emb.npy"

# ✅ 올바른 방법
from src.retrieval.paths import get_path
corpus_emb_path = get_path("kure_corpus_emb")
```

### 5.2 경로 상태 확인

```bash
python -m src.retrieval.paths --status
```

출력 예시:
```
📂 Dense Embeddings
--------------------------------------------------
  ✅ koe5_corpus_emb: 221.6 MB
  ❌ kure_corpus_emb: NOT FOUND
  ❌ kure_passages_meta: NOT FOUND
...
```

---

## 6. 문제 해결 체크리스트

### Answer가 CLS로만 나오는 경우

1. **answer_start가 context 기준인지 확인**
   ```python
   # 디버깅
   print(f"context length: {len(context)}")
   print(f"answer_start: {answer_start}")
   print(f"answer at position: {context[answer_start:answer_start+20]}")
   ```

2. **Chunking 여부 확인**
   - Chunked passage를 positive로 사용하면서 원본 answer_start를 쓰면 문제

3. **해결책**
   - 옵션 A 적용 (positive는 gold context만 사용)
   - 또는 옵션 B 적용 (answer_text.find()로 재계산)

### 임베딩 파일 불일치

1. **passage_id 일관성 확인**
   ```python
   # corpus_emb[passage_id] ↔ passage_metas[passage_id] 일치해야 함
   assert len(corpus_emb) == len(passage_metas)
   ```

2. **캐시 재생성**
   ```bash
   rm data/cache/retrieval/*.jsonl
   python -m src.retrieval.build_retrieval_cache --split train
   ```

---

## 7. 요약

| 구분 | 설명 |
|------|------|
| **Positive context** | 항상 원본 gold context 사용 (answer_start 정확) |
| **Negative context** | Retrieval passage 사용 (BM25/KURE hard negatives) |
| **임베딩 형식** | `.npy` (L2 정규화됨) |
| **메타데이터** | `.jsonl` (passage_id, doc_id, text 등) |
| **경로 관리** | `src/retrieval/paths.py` 중앙 집중 |

**핵심 원칙**: Reader는 임베딩 로직을 몰라도 됨. **텍스트 + 정확한 answer_start만 있으면 OK.**
