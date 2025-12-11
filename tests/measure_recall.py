#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TODO: gold context 위치 순위, 각 샘플 뽑으면서 context bm25, dense 점수 표기, context 점수 순 나열 (retrieval가 가져가는 순서 등)
"""
Retrieval Recall@k 측정 스크립트

Validation set을 기준으로 다양한 alpha 값에서 Recall@k를 측정합니다.

Usage:
    python -m tests.measure_recall
    python -m tests.measure_recall --alphas 0.3 0.5 0.7
    python -m tests.measure_recall --save_results logs/recall_results.json

Output:
    - Recall@1, @5, @10, @20, @50 for each alpha
    - Best alpha recommendation
    - Detailed per-query analysis (optional)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_from_disk

# 프로젝트 루트 추가
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.retrieval.paths import get_path


def load_cache(cache_path: str) -> Dict[str, Dict]:
    """캐시 파일을 딕셔너리로 로드."""
    cache = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            cache[item["id"]] = item
    return cache


def build_text_to_doc_id_mapping(wiki_path: str) -> Dict[str, int]:
    """
    문서 텍스트 → doc_id 매핑 생성.

    ⚠️ 중요: 동일 텍스트가 여러 doc_id에 존재할 수 있음.
    KURE 임베딩은 "첫 번째" doc_id를 사용하므로, 여기서도 첫 번째를 사용.
    """
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)

    text_to_doc_id = {}
    for doc_id, doc in wiki.items():
        text = doc["text"]
        # 첫 번째 등장한 doc_id만 저장 (KURE 임베딩과 일관성 유지)
        if text not in text_to_doc_id:
            text_to_doc_id[text] = int(doc_id)

    return text_to_doc_id


def compute_hybrid_scores(
    candidates: List[Dict], alpha: float, eps: float = 1e-9
) -> List[Dict]:
    """
    Raw score로부터 hybrid score 계산 및 정렬.

    Args:
        candidates: retrieval 후보 리스트 (score_bm25, score_dense 포함)
        alpha: BM25 가중치 (0-1, 1이면 BM25만)
        eps: 0 나누기 방지

    Returns:
        hybrid_score가 추가되고 정렬된 후보 리스트
    """
    if not candidates:
        return candidates

    bm25_scores = np.array([c["score_bm25"] for c in candidates])
    dense_scores = np.array([c["score_dense"] for c in candidates])

    # Per-query min-max normalization
    bm25_n = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min() + eps
    )
    dense_n = (dense_scores - dense_scores.min()) / (
        dense_scores.max() - dense_scores.min() + eps
    )

    # Weighted combination
    hybrid_scores = alpha * bm25_n + (1 - alpha) * dense_n

    # 정렬 인덱스
    sorted_indices = np.argsort(-hybrid_scores)

    sorted_candidates = []
    for idx in sorted_indices:
        cand = candidates[idx].copy()
        cand["hybrid_score"] = float(hybrid_scores[idx])
        sorted_candidates.append(cand)

    return sorted_candidates


def compute_recall_at_k(
    val_data,
    cache: Dict[str, Dict],
    text_to_doc_id: Dict[str, int],
    alpha: float,
    k_list: List[int] = [1, 5, 10, 20, 50],
) -> Tuple[Dict[int, float], int, List[Dict]]:
    """
    Recall@k 계산.

    Args:
        val_data: validation dataset
        cache: retrieval 캐시 {id -> {question, retrieved}}
        text_to_doc_id: context -> doc_id 매핑
        alpha: BM25 가중치
        k_list: 측정할 k 값 목록

    Returns:
        (recall_dict, total_count, per_query_results)
    """
    hits = {k: 0 for k in k_list}
    total = 0
    per_query_results = []

    for example in val_data:
        qid = example["id"]
        gold_context = example["context"]
        question = example["question"]

        # Gold doc_id 찾기
        gold_doc_id = text_to_doc_id.get(gold_context)
        if gold_doc_id is None:
            continue

        if qid not in cache:
            continue

        candidates = cache[qid]["retrieved"]
        sorted_candidates = compute_hybrid_scores(candidates, alpha)

        total += 1

        # 각 k에서 hit 확인
        query_result = {
            "id": qid,
            "question": question[:100],
            "gold_doc_id": gold_doc_id,
            "hits": {},
            "top_candidates": [],
        }

        for k in k_list:
            top_k_doc_ids = [c["doc_id"] for c in sorted_candidates[:k]]
            if gold_doc_id in top_k_doc_ids:
                hits[k] += 1
                query_result["hits"][k] = True
            else:
                query_result["hits"][k] = False

        # Top-5 후보 저장
        for c in sorted_candidates[:5]:
            query_result["top_candidates"].append(
                {
                    "doc_id": c["doc_id"],
                    "passage_id": c["passage_id"],
                    "hybrid_score": c.get("hybrid_score", 0),
                    "is_gold": c["doc_id"] == gold_doc_id,
                }
            )

        per_query_results.append(query_result)

    recall = {k: hits[k] / total * 100 if total > 0 else 0 for k in k_list}

    return recall, total, per_query_results


def run_recall_measurement(
    alphas: List[float] = None,
    dataset_path: str = None,
    cache_path: str = None,
    wiki_path: str = None,
    k_list: List[int] = None,
    verbose: bool = True,
    save_per_query: bool = False,
) -> Dict:
    """
    여러 alpha에 대해 Recall@k 측정.

    Returns:
        종합 결과 딕셔너리
    """
    if alphas is None:
        alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if k_list is None:
        k_list = [1, 5, 10, 20, 50]

    # 경로 설정
    if dataset_path is None:
        dataset_path = get_path("train_dataset")
    if cache_path is None:
        cache_path = os.path.join(get_path("retrieval_cache_dir"), "val_top50.jsonl")
    if wiki_path is None:
        wiki_path = get_path("wiki_corpus")

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset_path": dataset_path,
            "cache_path": cache_path,
            "wiki_path": wiki_path,
            "alphas": alphas,
            "k_list": k_list,
        },
        "results": {},
        "best_alpha": None,
        "total_samples": 0,
    }

    if verbose:
        print("=" * 70)
        print("📊 Retrieval Recall@k Measurement")
        print("=" * 70)
        print(f"Dataset: {dataset_path}")
        print(f"Cache: {cache_path}")
        print(f"Alphas: {alphas}")
        print(f"K values: {k_list}")

    # 데이터 로드
    if verbose:
        print("\n[1/3] Loading data...")

    dataset = load_from_disk(dataset_path)
    val_data = dataset["validation"]

    cache = load_cache(cache_path)
    text_to_doc_id = build_text_to_doc_id_mapping(wiki_path)

    if verbose:
        print(f"      Validation samples: {len(val_data)}")
        print(f"      Cache entries: {len(cache)}")

    # 각 alpha에 대해 측정
    if verbose:
        print("\n[2/3] Measuring Recall@k...")
        print("-" * 70)
        header = "Alpha  |  " + "  ".join([f"R@{k:2d}" for k in k_list])
        print(header)
        print("-" * 70)

    best_recall_10 = -1
    best_alpha = None

    for alpha in alphas:
        recall, total, per_query = compute_recall_at_k(
            val_data, cache, text_to_doc_id, alpha, k_list
        )

        results["results"][alpha] = {
            "recall": recall,
            "total_samples": total,
        }

        if save_per_query:
            results["results"][alpha]["per_query"] = per_query

        results["total_samples"] = total

        # Best alpha 추적 (R@10 기준)
        if recall.get(10, 0) > best_recall_10:
            best_recall_10 = recall.get(10, 0)
            best_alpha = alpha

        if verbose:
            recall_str = "  ".join([f"{recall[k]:5.1f}%" for k in k_list])
            print(f" {alpha:.2f}  |  {recall_str}")

    results["best_alpha"] = {
        "alpha": best_alpha,
        "recall_at_10": best_recall_10,
    }

    if verbose:
        print("-" * 70)
        print(f"\n[3/3] Best Alpha: {best_alpha} (R@10 = {best_recall_10:.1f}%)")
        print("\n" + "=" * 70)
        print("✅ Recall Measurement Complete")
        print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Measure Retrieval Recall@k")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=None,
        help="Alpha values to test (default: 0.3 to 1.0)",
    )
    parser.add_argument(
        "--k_list",
        nargs="+",
        type=int,
        default=None,
        help="K values for Recall@k (default: 1 5 10 20 50)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Dataset path",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Cache file path",
    )
    parser.add_argument(
        "--wiki_path",
        type=str,
        default=None,
        help="Wikipedia JSON path",
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--save_per_query",
        action="store_true",
        help="Include per-query results in output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    results = run_recall_measurement(
        alphas=args.alphas,
        dataset_path=args.dataset_path,
        cache_path=args.cache_path,
        wiki_path=args.wiki_path,
        k_list=args.k_list,
        verbose=not args.quiet,
        save_per_query=args.save_per_query,
    )

    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results) or ".", exist_ok=True)
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.save_results}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
