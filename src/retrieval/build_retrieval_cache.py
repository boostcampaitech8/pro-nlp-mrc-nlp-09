"""
Retrieval 캐시 생성 스크립트

train/val/test 각 split에 대해 retrieval 결과를 미리 계산하여 캐시로 저장합니다.
캐시에는 raw score (score_bm25, score_dense)만 저장하여, 추후 alpha 조정이 가능합니다.

Usage:
    # 기본 (BM25 + KURE Hybrid)
    python -m src.retrieval.build_retrieval_cache

    # 특정 split만
    python -m src.retrieval.build_retrieval_cache --splits train val

    # 다른 alpha로 캐시 생성 (score 자체는 동일, 순위만 다름)
    python -m src.retrieval.build_retrieval_cache --alpha 0.6

Output:
    [paths.py에서 관리]
    - train_cache: data/cache/retrieval/train_top50.jsonl
    - val_cache: data/cache/retrieval/val_top50.jsonl
    - test_cache: data/cache/retrieval/test_top50.jsonl
"""

import argparse
import json
import os
from typing import Dict, List, Optional

from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from src.retrieval.weighted_hybrid import WeightedHybridRetrieval
from src.retrieval.bm25 import BM25Retrieval
from src.retrieval.kure import KureRetrieval
from src.retrieval.paths import get_path, ensure_dir


# === 기본 설정 ===
DEFAULT_K_LARGE = 50  # 캐시에 저장할 최대 개수
DEFAULT_ALPHA = 0.35  # BM25 가중치 (base.yaml과 일치)


def build_cache_for_split(
    retriever: WeightedHybridRetrieval,
    questions: List[str],
    question_ids: List[str],
    k_large: int = DEFAULT_K_LARGE,
    output_path: str = "./data/retrieval_cache/split_top50.jsonl",
    batch_size: int = 32,
) -> None:
    """
    특정 split에 대해 retrieval 캐시를 생성합니다.

    Args:
        retriever: WeightedHybridRetrieval 인스턴스 (build 완료 상태)
        questions: 질문 리스트
        question_ids: 질문 ID 리스트
        k_large: 캐시에 저장할 상위 개수
        output_path: 출력 JSONL 파일 경로
        batch_size: 배치 크기
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n[build_cache] Processing {len(questions)} questions...")
    print(f"  → Output: {output_path}")
    print(f"  → K_LARGE: {k_large}")

    # 배치 단위로 처리
    all_results = []
    for start_idx in tqdm(range(0, len(questions), batch_size), desc="Retrieval"):
        end_idx = min(start_idx + batch_size, len(questions))
        batch_questions = questions[start_idx:end_idx]
        batch_ids = question_ids[start_idx:end_idx]

        # Raw scores 계산 (hybrid score가 아닌 bm25/dense raw score)
        batch_candidates = retriever.get_scores_for_cache(batch_questions, k=k_large)

        for qid, q_text, candidates in zip(
            batch_ids, batch_questions, batch_candidates
        ):
            all_results.append(
                {
                    "id": qid,
                    "question": q_text,
                    "retrieved": candidates,
                }
            )

    # JSONL로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  ✓ Saved {len(all_results)} entries to {output_path}")


def load_cache(cache_path: str) -> Dict[str, Dict]:
    """
    캐시 파일을 로드하여 {id -> {question, retrieved}} 딕셔너리로 반환합니다.

    Args:
        cache_path: 캐시 JSONL 파일 경로

    Returns:
        캐시 딕셔너리
    """
    cache = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            cache[item["id"]] = {
                "question": item["question"],
                "retrieved": item["retrieved"],
            }
    return cache


def compute_hybrid_score(
    candidates: List[Dict],
    alpha: float,
    eps: float = 1e-9,
) -> List[Dict]:
    """
    캐시된 raw score로부터 hybrid score를 계산합니다.

    Args:
        candidates: retrieval 후보 리스트
        alpha: BM25 가중치
        eps: 정규화 시 0 나누기 방지

    Returns:
        hybrid_score가 추가된 후보 리스트 (hybrid_score 내림차순 정렬됨)
    """
    import numpy as np

    if not candidates:
        return candidates

    bm25_scores = np.array([c["score_bm25"] for c in candidates])
    dense_scores = np.array([c["score_dense"] for c in candidates])

    # Per-query min-max 정규화
    bm25_n = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min() + eps
    )
    dense_n = (dense_scores - dense_scores.min()) / (
        dense_scores.max() - dense_scores.min() + eps
    )

    # 가중합
    hybrid_scores = alpha * bm25_n + (1 - alpha) * dense_n

    # 결과에 hybrid_score 추가
    for c, h in zip(candidates, hybrid_scores):
        c["hybrid_score"] = float(h)

    # hybrid_score 기준 정렬 (내림차순)
    sorted_candidates = sorted(
        candidates,
        key=lambda x: (-x["hybrid_score"], -x["score_bm25"], x["passage_id"]),
    )

    return sorted_candidates


def main():
    parser = argparse.ArgumentParser(
        description="Build retrieval cache for train/val/test splits"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Data directory path",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default=None,  # None이면 paths.py 기본값 사용
        help="Train dataset path (contains train and validation splits)",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default=None,  # None이면 paths.py 기본값 사용
        help="Test dataset path",
    )
    parser.add_argument(
        "--corpus_emb_path",
        type=str,
        default=None,  # None이면 paths.py 기본값 사용
        help="KURE corpus embedding path",
    )
    parser.add_argument(
        "--passages_meta_path",
        type=str,
        default=None,  # None이면 paths.py 기본값 사용
        help="KURE passages metadata path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,  # None이면 paths.py 기본값 사용
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--k_large",
        type=int,
        default=DEFAULT_K_LARGE,
        help="Number of candidates to cache per question",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="BM25 weight for hybrid retrieval (for ranking during cache generation)",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="klue/roberta-base",
        help="Tokenizer name for BM25 tokenization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for retrieval",
    )

    args = parser.parse_args()

    # 기본 경로 설정 (paths.py에서 관리)
    if args.train_dataset_path is None:
        args.train_dataset_path = get_path("train_dataset")
    if args.test_dataset_path is None:
        args.test_dataset_path = get_path("test_dataset")
    if args.corpus_emb_path is None:
        args.corpus_emb_path = get_path("kure_corpus_emb")
    if args.passages_meta_path is None:
        args.passages_meta_path = get_path("kure_passages_meta")
    if args.output_dir is None:
        args.output_dir = get_path("retrieval_cache_dir")

    # 출력 디렉토리 생성
    ensure_dir("retrieval_cache_dir")

    print("=" * 80)
    print("Retrieval Cache Builder")
    print("=" * 80)
    print(f"  Splits: {args.splits}")
    print(f"  K_LARGE: {args.k_large}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)

    # Tokenizer 로드 (BM25용)
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Retriever 초기화
    print("\n[2/3] Building retriever...")
    retriever = WeightedHybridRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        corpus_emb_path=args.corpus_emb_path,
        passages_meta_path=args.passages_meta_path,
        alpha=args.alpha,
    )
    retriever.build()

    # 각 split에 대해 캐시 생성
    print("\n[3/3] Building caches...")

    for split in args.splits:
        print(f"\n{'=' * 40}")
        print(f"Processing split: {split}")
        print(f"{'=' * 40}")

        # 데이터셋 로드
        if split in ["train", "val", "validation"]:
            dataset_path = args.train_dataset_path
            split_key = "train" if split == "train" else "validation"
        elif split == "test":
            dataset_path = args.test_dataset_path
            split_key = "validation"  # test dataset은 validation key로 저장됨
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"Loading dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)

        if split_key in dataset:
            data = dataset[split_key]
        else:
            # test_dataset은 단일 split일 수 있음
            data = dataset

        questions = data["question"]
        question_ids = data["id"]

        print(f"  → Found {len(questions)} questions")

        # 출력 파일명
        output_filename = f"{split}_top{args.k_large}.jsonl"
        output_path = os.path.join(args.output_dir, output_filename)

        # 캐시 생성
        build_cache_for_split(
            retriever=retriever,
            questions=questions,
            question_ids=question_ids,
            k_large=args.k_large,
            output_path=output_path,
            batch_size=args.batch_size,
        )

    print("\n" + "=" * 80)
    print("✅ All caches built successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
