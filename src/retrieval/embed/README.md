# KoE5 Dense Retrieval Implementation

<!-- TODO: 현재 koE5 쓰지 않음 이 문서는 삭제 예정 -->
## 📁 파일 구조

```
src/retrieval/
├── base.py                     # BaseRetrieval 클래스
├── sparse.py                   # SparseRetrieval (BM25)
├── koe5.py                     # ⭐ KoE5Retrieval (새로 추가)
├── dense_zeroshot.py           # DenseRetrieval (기존)
└── embed/
    ├── __init__.py
    ├── build_koe5_corpus.py    # ⭐ Corpus embedding 생성
    └── test_koe5.py            # ⭐ 빠른 테스트
```

---

## 🚀 Quick Start

### 1. 설치
```bash
pip install sentence-transformers
```

### 2. Corpus Embedding 생성 (첫 실행 시 한 번만)
```bash
python -m src.retrieval.embed.build_koe5_corpus
```

**예상 시간**: GPU 3~5분, CPU 20~30분  
**출력**: `./data/koe5_corpus_emb.npy` (약 230MB)

### 3. 테스트
```bash
python -m src.retrieval.embed.test_koe5
```

### 4. Inference에서 사용

#### 방법 A: 직접 객체 생성 (추천)
```python
from src.retrieval.koe5 import KoE5Retrieval

retriever = KoE5Retrieval(
    data_path="./data",
    context_path="wikipedia_documents.json",
    corpus_emb_path="./data/koe5_corpus_emb.npy",
)
retriever.build()

# 단일 쿼리
scores, contexts = retriever.retrieve("질문", topk=20)

# Dataset 배치 처리
df = retriever.retrieve(datasets["validation"], topk=20)
```

#### 방법 B: inference.py 수정
```python
# inference.py에서 (lines 118-125 근처)
if data_args.eval_retrieval:
    from src.retrieval.koe5 import KoE5Retrieval
    
    retriever = KoE5Retrieval(
        corpus_emb_path="./data/koe5_corpus_emb.npy"
    )
    retriever.build()
    
    datasets = retrieve_and_build_dataset(
        retriever=retriever,
        datasets=datasets,
        data_args=data_args,
        include_answers=(inference_split != "test"),
    )
```

---

## 📊 성능 비교 (예상)

| Retrieval | EM (validation) | 특징 |
|-----------|----------------|------|
| **Sparse (BM25)** | 60~65 (현재) | 키워드 매칭, 빠름 |
| **KoE5 Dense** | 65~70 (예상) | 의미 매칭, 더 정확 |
| **Hybrid** | 70~75 (목표) | 둘 다 사용, 최고 성능 |

---

## 🔧 커스텀 옵션

### Corpus embedding 생성 시
```bash
# Title 제외 (text만)
python -m src.retrieval.embed.build_koe5_corpus --no_title

# 배치 크기 조정 (메모리 부족 시)
python -m src.retrieval.embed.build_koe5_corpus --batch_size 32

# 다른 경로 지정
python -m src.retrieval.embed.build_koe5_corpus \
    --wiki_path ./data/wikipedia_documents.json \
    --output_path ./data/my_corpus_emb.npy
```

### 테스트 시
```bash
# 다른 질문으로 테스트
python -m src.retrieval.embed.test_koe5 --query "원하는 질문" --topk 10
```

---

## 🎯 다음 단계 (내일)

1. **Validation 성능 측정**
   ```bash
   # BM25 baseline
   python inference.py --inference_split validation --eval_retrieval
   
   # KoE5 비교 (inference.py 수정 후)
   python inference.py --inference_split validation --eval_retrieval
   ```

2. **Hybrid 구현**
   - BM25 top-50 + KoE5 top-50 → union
   - Dense score로 재정렬
   - alpha 하이퍼파라미터 튜닝

3. **Test 제출**
   - Best 설정으로 test inference
   - `predictions.json` 생성

---

## ⚠️ 주의사항

1. **Embedding shape mismatch 에러**
   - 원인: corpus가 바뀌었는데 embedding은 그대로
   - 해결: `python -m src.retrieval.embed.build_koe5_corpus` 재실행

2. **메모리 부족**
   - `batch_size` 줄이기: `--batch_size 32`
   - CPU 사용: GPU 없어도 작동함 (단, 느림)

3. **첫 실행 시 모델 다운로드**
   - KoE5-base 모델 자동 다운로드 (~1.5GB)
   - 인터넷 연결 필요

---

## 📝 코드 예시

### 예시 1: 단일 query 검색
```python
from src.retrieval.koe5 import KoE5Retrieval

retriever = KoE5Retrieval()
retriever.build()

query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
scores, contexts = retriever.retrieve(query, topk=5)

for i, (score, ctx) in enumerate(zip(scores, contexts), 1):
    print(f"Top-{i} (score={score:.4f}):")
    print(ctx[:200], "...\n")
```

### 예시 2: Dataset 배치 검색
```python
from datasets import load_from_disk
from src.retrieval.koe5 import KoE5Retrieval

# Dataset 로드
datasets = load_from_disk("./data/train_dataset")

# Retrieval
retriever = KoE5Retrieval()
retriever.build()
df = retriever.retrieve(datasets["validation"], topk=20)

# 결과 확인
print(df.head())
print(f"Columns: {df.columns.tolist()}")
# ['question', 'id', 'context', 'original_context', 'answers']
```

---

**구현 완료! 이제 corpus embedding 생성하고 테스트해보세요.**
