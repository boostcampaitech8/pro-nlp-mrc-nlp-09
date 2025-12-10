#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retrieval 캐시 Sanity Check 스크립트

train/val/test retrieval 캐시 파일의 무결성을 검증합니다.

Usage:
    python -m tests.sanity_cache
    python -m tests.sanity_cache --split val
    python -m tests.sanity_cache --all

Checks:
    1. 캐시 파일 존재 및 로드 가능 여부
    2. 캐시 항목 구조 검증 (id, question, retrieved)
    3. Retrieved 후보 구조 검증 (passage_id, doc_id, score_dense, score_bm25)
    4. Score 분포 통계
    5. Passage ID 범위 검증 (embedding과 일치하는지)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 프로젝트 루트 추가
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.retrieval.paths import get_path


def load_cache(cache_path: str) -> List[Dict]:
    """캐시 파일 로드."""
    cache = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            cache.append(json.loads(line.strip()))
    return cache


def check_cache_file(
    cache_path: str, max_passage_id: Optional[int] = None
) -> Tuple[bool, Dict]:
    """
    캐시 파일 검증.

    Args:
        cache_path: 캐시 파일 경로
        max_passage_id: passage_id 최대값 (embedding 크기 - 1)

    Returns:
        (success, results_dict)
    """
    results = {
        "file_exists": False,
        "num_questions": 0,
        "candidates_per_question": None,
        "structure_valid": False,
        "score_stats": None,
        "passage_id_range": None,
        "passage_id_valid": None,
        "sample_entry": None,
    }

    required_entry_fields = ["id", "question", "retrieved"]
    required_candidate_fields = ["passage_id", "doc_id", "score_dense", "score_bm25"]

    if not os.path.exists(cache_path):
        return False, results

    results["file_exists"] = True

    try:
        cache = load_cache(cache_path)
    except Exception as e:
        results["error"] = str(e)
        return False, results

    results["num_questions"] = len(cache)

    if len(cache) == 0:
        return False, results

    # 구조 검증
    first = cache[0]
    entry_fields_ok = all(field in first for field in required_entry_fields)

    if entry_fields_ok and len(first["retrieved"]) > 0:
        cand_fields_ok = all(
            field in first["retrieved"][0] for field in required_candidate_fields
        )
    else:
        cand_fields_ok = False

    results["structure_valid"] = entry_fields_ok and cand_fields_ok

    # 후보 수 확인
    candidates_counts = [len(item["retrieved"]) for item in cache]
    results["candidates_per_question"] = {
        "min": min(candidates_counts),
        "max": max(candidates_counts),
        "mean": np.mean(candidates_counts),
    }

    # Score 통계
    all_bm25 = []
    all_dense = []
    all_passage_ids = []

    for item in cache:
        for cand in item["retrieved"]:
            all_bm25.append(cand["score_bm25"])
            all_dense.append(cand["score_dense"])
            all_passage_ids.append(cand["passage_id"])

    results["score_stats"] = {
        "bm25": {
            "min": float(np.min(all_bm25)),
            "max": float(np.max(all_bm25)),
            "mean": float(np.mean(all_bm25)),
            "std": float(np.std(all_bm25)),
        },
        "dense": {
            "min": float(np.min(all_dense)),
            "max": float(np.max(all_dense)),
            "mean": float(np.mean(all_dense)),
            "std": float(np.std(all_dense)),
        },
    }

    # Passage ID 범위
    results["passage_id_range"] = {
        "min": min(all_passage_ids),
        "max": max(all_passage_ids),
    }

    # Passage ID 유효성 (embedding 크기와 비교)
    if max_passage_id is not None:
        results["passage_id_valid"] = max(all_passage_ids) <= max_passage_id

    # 샘플 항목
    sample_entry = {
        "id": first["id"],
        "question": first["question"][:80] + "..."
        if len(first["question"]) > 80
        else first["question"],
        "num_retrieved": len(first["retrieved"]),
        "first_candidate": first["retrieved"][0] if first["retrieved"] else None,
    }
    results["sample_entry"] = sample_entry

    success = results["structure_valid"]
    if results["passage_id_valid"] is not None:
        success = success and results["passage_id_valid"]

    return success, results


def run_cache_sanity_check(
    splits: List[str] = None,
    cache_dir: str = None,
    embedding_path: str = None,
    verbose: bool = True,
) -> Dict:
    """
    캐시 sanity check 실행.

    Args:
        splits: 체크할 split 목록 (기본: ["train", "val", "test"])
        cache_dir: 캐시 디렉토리 경로
        embedding_path: 임베딩 파일 경로 (passage_id 검증용)
        verbose: 상세 출력 여부

    Returns:
        종합 결과 딕셔너리
    """
    if splits is None:
        splits = ["train", "val", "test"]

    if cache_dir is None:
        cache_dir = get_path("retrieval_cache_dir")

    # Max passage ID 확인
    max_passage_id = None
    if embedding_path is None:
        embedding_path = get_path("kure_corpus_emb")

    if os.path.exists(embedding_path):
        emb = np.load(embedding_path)
        max_passage_id = emb.shape[0] - 1

    results = {
        "cache_dir": cache_dir,
        "max_passage_id": max_passage_id,
        "splits": {},
        "overall_pass": True,
    }

    if verbose:
        print("=" * 70)
        print("📊 Retrieval Cache Sanity Check")
        print("=" * 70)
        if max_passage_id is not None:
            print(f"Max passage_id (from embedding): {max_passage_id}")

    for i, split in enumerate(splits):
        cache_path = os.path.join(cache_dir, f"{split}_top50.jsonl")

        if verbose:
            print(f"\n[{i + 1}/{len(splits)}] {split.upper()}: {cache_path}")
            print("-" * 50)

        success, split_results = check_cache_file(cache_path, max_passage_id)
        results["splits"][split] = split_results

        if not success:
            results["overall_pass"] = False

        if verbose:
            if split_results["file_exists"]:
                print(f"      Questions: {split_results['num_questions']}")
                if split_results["candidates_per_question"]:
                    cpc = split_results["candidates_per_question"]
                    print(
                        f"      Candidates/question: {cpc['min']}-{cpc['max']} (mean={cpc['mean']:.1f})"
                    )
                print(
                    f"      Structure valid: {'✅' if split_results['structure_valid'] else '❌'}"
                )

                if split_results["score_stats"]:
                    bm25 = split_results["score_stats"]["bm25"]
                    dense = split_results["score_stats"]["dense"]
                    print(
                        f"      BM25 scores: [{bm25['min']:.2f}, {bm25['max']:.2f}], mean={bm25['mean']:.2f}"
                    )
                    print(
                        f"      Dense scores: [{dense['min']:.3f}, {dense['max']:.3f}], mean={dense['mean']:.3f}"
                    )

                if split_results["passage_id_range"]:
                    pid = split_results["passage_id_range"]
                    print(f"      Passage ID range: [{pid['min']}, {pid['max']}]")

                if split_results["passage_id_valid"] is not None:
                    print(
                        f"      Passage ID valid: {'✅' if split_results['passage_id_valid'] else '❌'}"
                    )

                print(f"      → {'✅ PASS' if success else '❌ FAIL'}")
            else:
                print(f"      → ❌ FAIL (file not found)")

    if verbose:
        print("\n" + "=" * 70)
        if results["overall_pass"]:
            print("✅ All Cache Sanity Checks PASSED")
        else:
            print("❌ Some Cache Sanity Checks FAILED")
        print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieval Cache Sanity Check")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to check (default: train val test)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all splits",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory path",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default=None,
        help="Embedding file path for passage_id validation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    splits = args.splits
    if args.all or splits is None:
        splits = ["train", "val", "test"]

    results = run_cache_sanity_check(
        splits=splits,
        cache_dir=args.cache_dir,
        embedding_path=args.embedding_path,
        verbose=not args.quiet and not args.json,
    )

    if args.json:
        import json as json_module

        print(json_module.dumps(results, indent=2, ensure_ascii=False))

    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
