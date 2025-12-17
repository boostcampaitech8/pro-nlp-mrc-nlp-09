#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임베딩 파일 Sanity Check 스크립트

KURE corpus embedding과 passage metadata의 무결성을 검증합니다.

Usage:
    python -m tests.sanity_embedding
    python -m tests.sanity_embedding --embedding_path data/embeddings/kure_corpus_emb.npy

Checks:
    1. Embedding shape 및 dtype 검증
    2. L2 Norm 검증 (정규화 확인)
    3. Embedding 값 분포 검증
    4. Passage metadata 로드 및 일치 확인
    5. Meta 필드 구조 검증
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 프로젝트 루트 추가
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.retrieval.paths import get_path


def check_embedding_file(emb_path: str) -> Tuple[bool, Dict]:
    """
    임베딩 파일 검증.

    Returns:
        (success, results_dict)
    """
    results = {
        "file_exists": False,
        "shape": None,
        "dtype": None,
        "file_size_mb": None,
        "l2_norm_mean": None,
        "l2_norm_min": None,
        "l2_norm_max": None,
        "value_mean": None,
        "value_std": None,
        "value_min": None,
        "value_max": None,
    }

    if not os.path.exists(emb_path):
        return False, results

    results["file_exists"] = True
    results["file_size_mb"] = os.path.getsize(emb_path) / 1024 / 1024

    emb = np.load(emb_path)
    results["shape"] = emb.shape
    results["dtype"] = str(emb.dtype)

    # L2 Norm 체크
    norms = np.linalg.norm(emb, axis=1)
    results["l2_norm_mean"] = float(norms.mean())
    results["l2_norm_min"] = float(norms.min())
    results["l2_norm_max"] = float(norms.max())

    # 값 분포
    results["value_mean"] = float(emb.mean())
    results["value_std"] = float(emb.std())
    results["value_min"] = float(emb.min())
    results["value_max"] = float(emb.max())

    # 정규화 체크 (L2 norm이 1.0에 가까워야 함)
    is_normalized = abs(results["l2_norm_mean"] - 1.0) < 0.001

    return is_normalized, results


def check_passages_meta(meta_path: str) -> Tuple[bool, Dict]:
    """
    Passage metadata 파일 검증.

    Returns:
        (success, results_dict)
    """
    results = {
        "file_exists": False,
        "num_passages": 0,
        "required_fields_present": False,
        "sample_entry": None,
        "is_chunk_stats": None,
    }

    required_fields = [
        "passage_id",
        "doc_id",
        "title",
        "text",
        "start_char",
        "end_char",
        "is_chunk",
    ]

    if not os.path.exists(meta_path):
        return False, results

    results["file_exists"] = True

    passages = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line.strip()))

    results["num_passages"] = len(passages)

    if len(passages) == 0:
        return False, results

    # 필드 검증
    first = passages[0]
    results["required_fields_present"] = all(
        field in first for field in required_fields
    )

    # 샘플 저장 (text는 축약)
    sample = {
        k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
        for k, v in first.items()
    }
    results["sample_entry"] = sample

    # Chunking 통계
    is_chunk_count = sum(1 for p in passages if p.get("is_chunk", False))
    results["is_chunk_stats"] = {
        "chunked": is_chunk_count,
        "not_chunked": len(passages) - is_chunk_count,
    }

    return results["required_fields_present"], results


def check_consistency(emb_path: str, meta_path: str) -> Tuple[bool, str]:
    """
    임베딩과 메타데이터 일관성 검증.
    """
    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        return False, "파일이 존재하지 않습니다."

    emb = np.load(emb_path)

    num_passages = 0
    with open(meta_path, "r", encoding="utf-8") as f:
        for _ in f:
            num_passages += 1

    if emb.shape[0] == num_passages:
        return True, f"일치: {emb.shape[0]} passages"
    else:
        return False, f"불일치: embedding={emb.shape[0]}, meta={num_passages}"


def run_sanity_check(
    embedding_path: str = None,
    meta_path: str = None,
    verbose: bool = True,
) -> Dict:
    """
    전체 sanity check 실행.

    Returns:
        종합 결과 딕셔너리
    """
    # 기본 경로 설정
    if embedding_path is None:
        embedding_path = get_path("kure_corpus_emb")
    if meta_path is None:
        meta_path = get_path("kure_passages_meta")

    results = {
        "embedding_path": embedding_path,
        "meta_path": meta_path,
        "embedding_check": None,
        "meta_check": None,
        "consistency_check": None,
        "overall_pass": False,
    }

    if verbose:
        print("=" * 70)
        print("📊 KURE Corpus Embedding Sanity Check")
        print("=" * 70)

    # 1. Embedding 검증
    if verbose:
        print(f"\n[1/3] Embedding 검증: {embedding_path}")
    emb_ok, emb_results = check_embedding_file(embedding_path)
    results["embedding_check"] = emb_results

    if verbose:
        if emb_results["file_exists"]:
            print(f"      Shape: {emb_results['shape']}")
            print(f"      dtype: {emb_results['dtype']}")
            print(f"      Size: {emb_results['file_size_mb']:.1f} MB")
            print(
                f"      L2 Norm: mean={emb_results['l2_norm_mean']:.6f}, "
                f"min={emb_results['l2_norm_min']:.6f}, max={emb_results['l2_norm_max']:.6f}"
            )
            print(
                f"      Values: mean={emb_results['value_mean']:.6f}, std={emb_results['value_std']:.6f}"
            )
            print(f"      → {'✅ PASS' if emb_ok else '❌ FAIL (not normalized)'}")
        else:
            print(f"      → ❌ FAIL (file not found)")

    # 2. Meta 검증
    if verbose:
        print(f"\n[2/3] Passages Meta 검증: {meta_path}")
    meta_ok, meta_results = check_passages_meta(meta_path)
    results["meta_check"] = meta_results

    if verbose:
        if meta_results["file_exists"]:
            print(f"      Passages: {meta_results['num_passages']}")
            print(
                f"      Required fields: {'✅' if meta_results['required_fields_present'] else '❌'}"
            )
            if meta_results["is_chunk_stats"]:
                stats = meta_results["is_chunk_stats"]
                print(
                    f"      Chunking: {stats['chunked']} chunked, {stats['not_chunked']} not chunked"
                )
            print(f"      → {'✅ PASS' if meta_ok else '❌ FAIL'}")
        else:
            print(f"      → ❌ FAIL (file not found)")

    # 3. 일관성 검증
    if verbose:
        print(f"\n[3/3] 일관성 검증")
    consistency_ok, consistency_msg = check_consistency(embedding_path, meta_path)
    results["consistency_check"] = {"pass": consistency_ok, "message": consistency_msg}

    if verbose:
        print(f"      {consistency_msg}")
        print(f"      → {'✅ PASS' if consistency_ok else '❌ FAIL'}")

    # 종합 결과
    results["overall_pass"] = emb_ok and meta_ok and consistency_ok

    if verbose:
        print("\n" + "=" * 70)
        if results["overall_pass"]:
            print("✅ All Sanity Checks PASSED")
        else:
            print("❌ Some Sanity Checks FAILED")
        print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="KURE Embedding Sanity Check")
    parser.add_argument(
        "--embedding_path",
        type=str,
        default=None,
        help="Path to embedding .npy file",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="Path to passages meta .jsonl file",
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

    results = run_sanity_check(
        embedding_path=args.embedding_path,
        meta_path=args.meta_path,
        verbose=not args.quiet and not args.json,
    )

    if args.json:
        # numpy types를 JSON serializable하게 변환
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            return obj

        import json as json_module

        serializable_results = json_module.loads(
            json_module.dumps(results, default=convert_to_serializable)
        )
        print(json_module.dumps(serializable_results, indent=2, ensure_ascii=False))

    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
