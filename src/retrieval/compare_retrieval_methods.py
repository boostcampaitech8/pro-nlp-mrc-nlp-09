"""
BM25 / KURE / Hybrid 비교 스크립트

Validation set에 대해 다양한 retrieval 모드의 성능을 비교합니다:
- BM25-only
- KURE-only
- Hybrid (α = 0.6, 0.7, 0.8)

측정 지표:
- Recall@k (k = 1, 5, 10, 20)
- MRR@k (Mean Reciprocal Rank)

Usage:
    python -m src.retrieval.compare_retrieval_methods

    # 특정 alpha 범위로
    python -m src.retrieval.compare_retrieval_methods --alphas 0.5 0.6 0.7 0.8 0.9
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from src.retrieval.weighted_hybrid import WeightedHybridRetrieval
from src.retrieval.bm25 import BM25Retrieval
from src.retrieval.kure import KureRetrieval


def compute_recall_at_k(
    retrieved_doc_ids: List[List[int]],
    gold_doc_ids: List[int],
    k: int,
) -> float:
    """
    Recall@k 계산.

    Args:
        retrieved_doc_ids: 각 query별 retrieved document ID 리스트
        gold_doc_ids: 각 query별 gold document ID
        k: top-k

    Returns:
        Recall@k 값 (0~1)
    """
    hits = 0
    for retrieved, gold in zip(retrieved_doc_ids, gold_doc_ids):
        top_k = retrieved[:k]
        if gold in top_k:
            hits += 1
    return hits / len(gold_doc_ids)


def compute_mrr_at_k(
    retrieved_doc_ids: List[List[int]],
    gold_doc_ids: List[int],
    k: int,
) -> float:
    """
    MRR@k (Mean Reciprocal Rank) 계산.

    Args:
        retrieved_doc_ids: 각 query별 retrieved document ID 리스트
        gold_doc_ids: 각 query별 gold document ID
        k: top-k

    Returns:
        MRR@k 값 (0~1)
    """
    rr_sum = 0.0
    for retrieved, gold in zip(retrieved_doc_ids, gold_doc_ids):
        top_k = retrieved[:k]
        if gold in top_k:
            rank = top_k.index(gold) + 1
            rr_sum += 1.0 / rank
    return rr_sum / len(gold_doc_ids)


def evaluate_retrieval(
    questions: List[str],
    gold_doc_ids: List[int],
    retriever,
    k_values: List[int] = [1, 5, 10, 20],
    use_doc_id_mapping: bool = True,
) -> Dict[str, float]:
    """
    Retriever 성능 평가.

    Args:
        questions: 질문 리스트
        gold_doc_ids: gold document ID 리스트
        retriever: build()된 retriever
        k_values: 평가할 k 값들
        use_doc_id_mapping: passage_id -> doc_id 변환 여부

    Returns:
        {"Recall@k": float, "MRR@k": float, ...}
    """
    # Retrieval 수행
    max_k = max(k_values)
    scores, indices = retriever.get_relevant_doc_bulk(questions, k=max_k)

    # passage_id -> doc_id 변환 (chunking 사용 시)
    if use_doc_id_mapping and hasattr(retriever, "kure_retriever"):
        # WeightedHybridRetrieval인 경우
        if retriever.use_passages_mode:
            retrieved_doc_ids = []
            for idx_list in indices:
                doc_ids = [
                    retriever.kure_retriever.passages_meta[pid]["doc_id"]
                    for pid in idx_list
                ]
                retrieved_doc_ids.append(doc_ids)
        else:
            retrieved_doc_ids = [
                [retriever.ids[idx] for idx in idx_list] for idx_list in indices
            ]
    elif hasattr(retriever, "passages_meta") and retriever.passages_meta:
        # KureRetrieval (passages mode)
        retrieved_doc_ids = [
            [retriever.passages_meta[pid]["doc_id"] for pid in idx_list]
            for idx_list in indices
        ]
    else:
        # BM25 또는 document mode
        retrieved_doc_ids = [
            [retriever.ids[idx] for idx in idx_list] for idx_list in indices
        ]

    # 메트릭 계산
    metrics = {}
    for k in k_values:
        recall = compute_recall_at_k(retrieved_doc_ids, gold_doc_ids, k)
        mrr = compute_mrr_at_k(retrieved_doc_ids, gold_doc_ids, k)
        metrics[f"Recall@{k}"] = recall
        metrics[f"MRR@{k}"] = mrr

    return metrics


def run_comparison(
    data_path: str = "./data",
    train_dataset_path: str = "./data/train_dataset",
    corpus_emb_path: str = "./data/kure_corpus_emb.npy",
    passages_meta_path: str = "./data/kure_passages_meta.jsonl",
    tokenizer_name: str = "klue/roberta-base",
    alphas: List[float] = [0.6, 0.7, 0.8],
    k_values: List[int] = [1, 5, 10, 20],
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    BM25 / KURE / Hybrid 비교 실행.

    Args:
        data_path: 데이터 디렉토리
        train_dataset_path: train dataset 경로 (validation split 포함)
        corpus_emb_path: KURE corpus embedding 경로
        passages_meta_path: passages metadata 경로
        tokenizer_name: tokenizer 이름
        alphas: 비교할 alpha 값들
        k_values: 평가할 k 값들
        output_path: 결과 저장 경로 (optional)

    Returns:
        {method_name: {metric: value, ...}, ...}
    """
    print("=" * 80)
    print("Retrieval Methods Comparison")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/5] Loading validation data...")
    dataset = load_from_disk(train_dataset_path)
    val_data = dataset["validation"]

    questions = val_data["question"]

    # Gold document ID 추출
    # document_id 필드가 있으면 사용, 없으면 context에서 추론
    if "document_id" in val_data.column_names:
        gold_doc_ids = val_data["document_id"]
    else:
        # context를 기반으로 wiki에서 doc_id 찾기
        print("   ⚠️ No document_id field, loading wiki to match contexts...")
        with open(f"{data_path}/wikipedia_documents.json", "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # context -> doc_id 매핑
        context_to_doc = {}
        for doc_id, doc_info in wiki.items():
            context_to_doc[doc_info["text"]] = int(doc_id)

        gold_doc_ids = []
        for ctx in val_data["context"]:
            gold_doc_ids.append(context_to_doc.get(ctx, -1))

    print(f"   ✓ Loaded {len(questions)} validation examples")

    # Tokenizer 로드
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    results = {}

    # === BM25-only ===
    print("\n[3/5] Evaluating BM25-only...")
    bm25_retriever = BM25Retrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=data_path,
    )
    bm25_retriever.build()

    bm25_metrics = evaluate_retrieval(
        questions, gold_doc_ids, bm25_retriever, k_values, use_doc_id_mapping=False
    )
    results["BM25-only"] = bm25_metrics
    print(f"   BM25-only: {bm25_metrics}")

    # === KURE-only ===
    print("\n[4/5] Evaluating KURE-only...")
    kure_retriever = KureRetrieval(
        data_path=data_path,
        corpus_emb_path=corpus_emb_path,
        passages_meta_path=passages_meta_path,
    )
    kure_retriever.build()

    kure_metrics = evaluate_retrieval(
        questions, gold_doc_ids, kure_retriever, k_values, use_doc_id_mapping=True
    )
    results["KURE-only"] = kure_metrics
    print(f"   KURE-only: {kure_metrics}")

    # === Hybrid (다양한 alpha) ===
    print("\n[5/5] Evaluating Hybrid with various alpha values...")
    for alpha in alphas:
        print(f"\n   Evaluating Hybrid(α={alpha})...")
        hybrid_retriever = WeightedHybridRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=data_path,
            corpus_emb_path=corpus_emb_path,
            passages_meta_path=passages_meta_path,
            alpha=alpha,
        )
        hybrid_retriever.build()

        hybrid_metrics = evaluate_retrieval(
            questions, gold_doc_ids, hybrid_retriever, k_values, use_doc_id_mapping=True
        )
        results[f"Hybrid(α={alpha})"] = hybrid_metrics
        print(f"   Hybrid(α={alpha}): {hybrid_metrics}")

    # 결과 요약
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)

    # 테이블 형식 출력
    header = (
        ["Method"] + [f"Recall@{k}" for k in k_values] + [f"MRR@{k}" for k in k_values]
    )
    print(f"{'Method':<20}" + "".join([f"{h:<12}" for h in header[1:]]))
    print("-" * (20 + 12 * len(header[1:])))

    for method, metrics in results.items():
        row = [method]
        for k in k_values:
            row.append(f"{metrics[f'Recall@{k}']:.4f}")
        for k in k_values:
            row.append(f"{metrics[f'MRR@{k}']:.4f}")
        print(f"{row[0]:<20}" + "".join([f"{v:<12}" for v in row[1:]]))

    # 최적 설정 찾기
    best_method = max(results.items(), key=lambda x: x[1].get("MRR@20", 0))
    print(
        f"\n✅ Best method by MRR@20: {best_method[0]} ({best_method[1]['MRR@20']:.4f})"
    )

    # 결과 저장
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n📁 Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval methods")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Data directory path",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="./data/train_dataset",
        help="Train dataset path",
    )
    parser.add_argument(
        "--corpus_emb_path",
        type=str,
        default="./data/kure_corpus_emb.npy",
        help="KURE corpus embedding path",
    )
    parser.add_argument(
        "--passages_meta_path",
        type=str,
        default="./data/kure_passages_meta.jsonl",
        help="Passages metadata path",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="klue/roberta-base",
        help="Tokenizer name",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.6, 0.7, 0.8],
        help="Alpha values to compare",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        help="K values for Recall@k and MRR@k",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./logs/retrieval_comparison.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    run_comparison(
        data_path=args.data_path,
        train_dataset_path=args.train_dataset_path,
        corpus_emb_path=args.corpus_emb_path,
        passages_meta_path=args.passages_meta_path,
        tokenizer_name=args.tokenizer_name,
        alphas=args.alphas,
        k_values=args.k_values,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
