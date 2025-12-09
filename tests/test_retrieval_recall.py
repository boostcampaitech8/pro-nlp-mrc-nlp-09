"""
Retrieval Top-K Recall 테스트

KoE5 Dense Retrieval과 TF-IDF Sparse Retrieval 성능을 측정합니다.
- Top-K Recall: 정답 문서가 상위 K개 안에 포함되는 비율 (%)
- 실행 시간: validation 240개 기준 ~10-30초

📖 사용법:
    # KoE5 Dense Retrieval 테스트
    python tests/test_retrieval_recall.py --retriever koe5 --topk 1,5,10,20,50

    # TF-IDF Sparse Retrieval 테스트
    python tests/test_retrieval_recall.py --retriever tfidf --topk 1,5,10,20,50

    # validation 대신 train split 사용
    python tests/test_retrieval_recall.py --retriever koe5 --split train

    # 도움말
    python tests/test_retrieval_recall.py --help

📊 예상 결과:
    KoE5 (Dense):
        recall@1  : ~45-50%
        recall@10 : ~75-85%
        recall@50 : ~85-90%

    TF-IDF (Sparse):
        recall@1  : ~35-45%
        recall@10 : ~60-70%
        recall@50 : ~75-85%
"""

import argparse
import os
import sys
import time
from typing import List, Dict

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import SparseRetrieval, KoE5Retrieval


def initialize_retriever(retriever_type: str):
    """Retriever 초기화 (koe5 또는 tfidf)"""
    if retriever_type == "koe5":
        return KoE5Retrieval(
            data_path="./data",
            context_path="wikipedia_documents.json",
            corpus_emb_path="./data/koe5_corpus_emb.npy",
        )
    else:  # tfidf
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        return SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
        )


def analyze_full_dataset(
    retriever_type: str,
    dataset_path: str,
    topk_list: List[int] = [1, 5, 10, 20, 50, 100],
) -> None:
    """
    전체 train+validation 데이터셋으로 recall@k 분석 (retrieval_sanity.py 스타일)

    성능 최적화: k=max(topk_list) 한 번만 계산 후 슬라이싱
    """
    print("\n" + "=" * 80)
    print(f"📊 FULL DATASET ANALYSIS ({retriever_type.upper()})")
    print("=" * 80)

    # 전체 데이터셋 로드
    from datasets import concatenate_datasets

    ds = load_from_disk(dataset_path)
    train_ds = ds["train"].flatten_indices()
    valid_ds = ds["validation"].flatten_indices()

    print(f"📁 Train samples: {len(train_ds)}")
    print(f"📁 Valid samples: {len(valid_ds)}")

    # Retriever 초기화
    retriever = initialize_retriever(retriever_type)
    retriever.build()

    max_k = max(topk_list)

    # Train 분석
    print(f"\n⏳ Train 데이터셋 분석 중... (Top-{max_k} 계산)")
    _, train_doc_indices = retriever.get_relevant_doc_bulk(
        train_ds["question"], k=max_k
    )

    print("\n" + "=" * 80)
    print(f"📈 Train Dataset (n={len(train_ds)})")
    print("=" * 80)
    print(f"{'Top-K':<8} {'Recall@K':<12} {'Match/Total':<15}")
    print("-" * 80)

    # 정답 문서 추출 (한 번만)
    train_gold_contexts = [ex["context"] for ex in train_ds]

    for k in topk_list:
        # 각 K에 대해 recall 계산 (이미 계산된 indices 슬라이싱)
        hits = 0
        for gold_ctx, indices in zip(train_gold_contexts, train_doc_indices):
            topk_contexts = [retriever.contexts[idx] for idx in indices[:k]]
            if gold_ctx in topk_contexts:
                hits += 1

        recall = hits / len(train_ds)
        print(f"{k:<8} {recall:<12.1%} {hits}/{len(train_ds)}")

    # Validation 분석
    print(f"\n⏳ Validation 데이터셋 분석 중... (Top-{max_k} 계산)")
    _, valid_doc_indices = retriever.get_relevant_doc_bulk(
        valid_ds["question"], k=max_k
    )

    print("\n" + "=" * 80)
    print(f"📈 Validation Dataset (n={len(valid_ds)})")
    print("=" * 80)
    print(f"{'Top-K':<8} {'Recall@K':<12} {'Match/Total':<15}")
    print("-" * 80)

    # 정답 문서 추출 (한 번만)
    valid_gold_contexts = [ex["context"] for ex in valid_ds]

    for k in topk_list:
        hits = 0
        for gold_ctx, indices in zip(valid_gold_contexts, valid_doc_indices):
            topk_contexts = [retriever.contexts[idx] for idx in indices[:k]]
            if gold_ctx in topk_contexts:
                hits += 1

        recall = hits / len(valid_ds)
        print(f"{k:<8} {recall:<12.1%} {hits}/{len(valid_ds)}")

    # 전체 데이터 분석
    full_ds = concatenate_datasets([train_ds, valid_ds])
    print(f"\n⏳ Full 데이터셋 계산 중...")
    full_doc_indices = train_doc_indices + valid_doc_indices
    full_gold_contexts = train_gold_contexts + valid_gold_contexts

    print("\n" + "=" * 80)
    print(f"📈 Full Dataset - Train + Valid (n={len(full_ds)})")
    print("=" * 80)
    print(f"{'Top-K':<8} {'Recall@K':<12} {'Match/Total':<15}")
    print("-" * 80)

    for k in topk_list:
        hits = 0
        for gold_ctx, indices in zip(full_gold_contexts, full_doc_indices):
            topk_contexts = [retriever.contexts[idx] for idx in indices[:k]]
            if gold_ctx in topk_contexts:
                hits += 1

        recall = hits / len(full_ds)
        print(f"{k:<8} {recall:<12.1%} {hits}/{len(full_ds)}")

    print("=" * 80)
    print("\n💡 해석:")
    print("  - Recall@K: Question 던졌을 때 정답 document가 Top-K 안에 있는 비율")
    print("  - 높을수록 좋음 (Retriever가 정답 문서를 잘 찾음)")
    print("  - K가 클수록 Recall은 증가 (더 많은 문서를 검색하므로)")


def calculate_recall(
    dataset,
    retriever,
    topk_list: List[int] = [1, 5, 10, 20, 50],
) -> Dict[int, Dict[str, float]]:
    """
    Top-K Recall 계산

    Args:
        dataset: HF Dataset (question, context, id 포함)
        retriever: build()된 retrieval 객체
        topk_list: 측정할 K 값들

    Returns:
        {k: {"recall": 0.xx, "match": N, "total": M}, ...}
    """
    max_k = max(topk_list)

    # Retrieval 수행 (최대 K로 검색)
    print(f"\n[1/3] Retrieving top-{max_k} documents...")
    queries = dataset["question"]
    doc_scores, doc_indices = retriever.get_relevant_doc_bulk(queries, k=max_k)

    # 정답 문서 추출 (gold context)
    print(f"[2/3] Extracting gold contexts...")
    gold_contexts = [ex["context"] for ex in dataset]

    # Recall 계산
    print(f"[3/3] Calculating recall@K...")
    recalls = {k: [] for k in topk_list}

    for i, (gold_ctx, indices) in enumerate(
        tqdm(zip(gold_contexts, doc_indices), total=len(gold_contexts), disable=True)
    ):
        # 검색된 문서들
        retrieved_contexts = [retriever.contexts[idx] for idx in indices]

        # 각 K에 대해 정답이 포함되었는지 확인
        for k in topk_list:
            topk_contexts = retrieved_contexts[:k]
            # 정답 문서가 top-k 안에 있으면 1, 없으면 0
            hit = int(gold_ctx in topk_contexts)
            recalls[k].append(hit)

    # 결과 구조화
    results = {}
    total_samples = len(gold_contexts)
    for k in topk_list:
        match_count = int(np.sum(recalls[k]))
        recall = match_count / total_samples if total_samples > 0 else 0.0
        results[k] = {
            "recall": recall,
            "match": match_count,
            "total": total_samples,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval Top-K Recall 테스트 - KoE5 vs TF-IDF 성능 비교",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # KoE5 Dense Retrieval 테스트 (validation)
  python tests/test_retrieval_recall.py --retriever koe5 --topk 1,5,10,20,50
  
  # TF-IDF Sparse Retrieval 테스트 (validation)
  python tests/test_retrieval_recall.py --retriever tfidf --topk 1,5,10,20,50
  
  # Train split으로 테스트
  python tests/test_retrieval_recall.py --retriever koe5 --split train --topk 1,5,10,20,50,100
  
  # 전체 데이터셋 분석 (train + validation)
  python tests/test_retrieval_recall.py --retriever koe5 --analyze_full

출력 해석:
  - Recall@K: 정답 문서가 Top-K 안에 포함된 비율
  - Match/Total: 정답을 찾은 개수 / 전체 샘플 수
  - K가 클수록 Recall은 증가 (더 많은 문서를 검색)
        """,
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="koe5",
        choices=["koe5", "tfidf"],
        help="Retrieval method (koe5: dense KoE5, tfidf: sparse TF-IDF)",
    )
    parser.add_argument(
        "--topk",
        type=str,
        default="1,5,10,20,50",
        help="Top-K values to evaluate (comma-separated, e.g., '1,5,10,20,50')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/train_dataset",
        help="Dataset path (HF datasets format)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--analyze_full",
        action="store_true",
        help="전체 train+valid 데이터셋으로 recall@k 분석 수행 (retrieval_sanity.py 스타일)",
    )

    args = parser.parse_args()

    # Top-K 파싱
    if args.analyze_full:
        topk_list = [1, 5, 10, 20, 50, 100]
    else:
        topk_list = sorted([int(k.strip()) for k in args.topk.split(",")])

    # Config 출력
    print("=" * 40)
    print("Retrieval Recall Test Config")
    print("=" * 40)
    print(f"retriever       = {args.retriever}")
    print(f"dataset         = {args.dataset}")
    print(f"split           = {args.split}")
    print(f"topk            = {topk_list}")
    print(f"analyze_full    = {args.analyze_full}")
    print("=" * 40)
    print()

    # 전체 데이터셋 분석 모드
    if args.analyze_full:
        analyze_full_dataset(
            retriever_type=args.retriever,
            dataset_path=args.dataset,
            topk_list=topk_list,
        )
        return

    # 단일 split 분석 모드
    print("=" * 80)
    print(f"🔍 RETRIEVAL RECALL EVALUATION ({args.retriever.upper()})")
    print("=" * 80)
    print(f"Dataset  : {args.dataset} ({args.split} split)")
    print(f"Top-K    : {topk_list}")
    print("=" * 80)

    # 1. 데이터셋 로드
    print(f"\n[LOAD] Loading dataset...")
    datasets = load_from_disk(args.dataset)
    eval_dataset = datasets[args.split]
    print(f"   ✓ Loaded {len(eval_dataset)} examples")

    # 2. Retriever 초기화
    print(f"\n[BUILD] Initializing {args.retriever.upper()} retriever...")
    start_time = time.time()
    retriever = initialize_retriever(args.retriever)
    retriever.build()
    build_time = time.time() - start_time
    print(f"   ✓ Build completed in {build_time:.2f}s")

    # 3. Recall 계산
    print(f"\n[EVAL] Calculating recall...")
    start_time = time.time()
    results = calculate_recall(eval_dataset, retriever, topk_list)
    eval_time = time.time() - start_time

    # 4. 결과 출력 (retrieval_sanity.py 스타일)
    total_samples = results[topk_list[0]]["total"]

    print("\n" + "=" * 80)
    print(f"📊 {args.split.capitalize()} Dataset (n={total_samples})")
    print("=" * 80)
    print(f"{'Top-K':<8} {'Recall@K':<12} {'Match/Total':<15}")
    print("-" * 80)

    for k in topk_list:
        recall = results[k]["recall"]
        match = results[k]["match"]
        total = results[k]["total"]
        print(f"{k:<8} {recall:<12.1%} {match}/{total}")

    print("=" * 80)
    print(
        f"⏱️  Evaluation time: {eval_time:.2f}s ({eval_time / total_samples * 1000:.1f}ms per query)"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
