# tests/test_retrieval_sanity.py

"""
Retrieval 모듈(Sparse / Dense)에 대한 sanity check & 최소 기능 테스트 스크립트.

- SparseRetrieval:
  - build()로 TF-IDF embedding 로드/생성
  - train+valid에서 num_samples개 샘플 뽑아서 retrieve()
  - original_context가 top-k 안에 들어있는 비율(hit@k) 출력

- DenseRetrieval:
  - build()로 corpus dense embedding 로드/생성
  - 같은 방식으로 retrieve() + hit@k 출력
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk

# 프로젝트 루트(src 상위)를 PYTHONPATH에 추가
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.retrieval import SparseRetrieval, DenseRetrieval  # noqa
from transformers import AutoTokenizer  # noqa


def compute_hit_at_k_by_index(
    dataset: Dataset,
    retriever,
    doc_indices: List[List[int]],
) -> float:
    """
    원본 context가 retrieved top-k 문서 인덱스 안에 있는지 확인.
    문서 단위 정확한 매칭으로 hit@k 계산.

    Args:
        dataset: original context를 포함한 데이터셋
        retriever: contexts 리스트를 가진 retriever 인스턴스
        doc_indices: 각 query별 top-k 문서 인덱스 [[idx1, idx2, ...], ...]

    Returns:
        hit@k 비율 (0.0 ~ 1.0)
    """
    if "context" not in dataset.column_names:
        print("⚠️  context 컬럼이 없어 hit@k를 계산할 수 없습니다.")
        return float("nan")

    # context -> index 매핑 (contexts는 유니크하다는 전제)
    ctx2idx = {ctx: i for i, ctx in enumerate(retriever.contexts)}

    hits = 0
    total = len(dataset)

    for i, ex in enumerate(dataset):
        orig_ctx = ex["context"]
        orig_idx = ctx2idx.get(orig_ctx, None)

        if orig_idx is None:
            # 원본 context가 corpus에 없는 경우 (miss로 처리)
            continue

        if orig_idx in doc_indices[i]:
            hits += 1

    return hits / total if total > 0 else float("nan")


def compute_recall_precision_at_k(
    dataset: Dataset,
    retriever,
    doc_indices: List[List[int]],
    verbose: bool = False,
) -> dict:
    """
    Document ID 기반으로 Recall@k와 Precision@k 계산.

    목적: Question을 던졌을 때, 정답이 있는 Document를 Top-K 안에서 찾았는지 확인

    Args:
        dataset: document_id(정답 문서 ID)를 포함한 데이터셋
        retriever: contexts와 ids(각 문서의 document_id)를 가진 retriever
        doc_indices: 각 query별 top-k 문서 인덱스 [[idx1, idx2, ...], ...]
        verbose: 디버깅 정보 출력 여부

    Returns:
        {
            'recall_at_k': 정답 문서를 찾은 비율 (0.0~1.0),
            'exact_match_count': 정답 문서를 찾은 개수,
            'total_samples': 전체 샘플 수
        }
    """
    if "document_id" not in dataset.column_names:
        print("⚠️  document_id 컬럼이 없어 recall/precision을 계산할 수 없습니다.")
        return {
            "recall_at_k": float("nan"),
            "exact_match_count": 0,
            "total_samples": len(dataset),
        }

    total = len(dataset)
    exact_matches = 0

    # 디버깅용: 첫 3개 샘플만 상세 출력
    if verbose and total > 0:
        print("\n[검색 결과 예시] 상위 3개 샘플")
        print("=" * 100)

    for i, ex in enumerate(dataset):
        gold_doc_id = ex["document_id"]
        gold_title = ex.get("title", "")

        # retrieved top-k 문서들의 document_id 추출
        retrieved_doc_ids = [retriever.ids[idx] for idx in doc_indices[i]]

        # 정답 document_id가 retrieved 안에 있는지 확인
        is_match = gold_doc_id in retrieved_doc_ids
        if is_match:
            exact_matches += 1

        # 디버깅 출력 - title과 context preview 포함
        if verbose and i < 3:
            print(f"\n샘플 #{i + 1} {'✅ HIT' if is_match else '❌ MISS'}")
            print(f"Question: {ex['question'][:80]}...")
            print(f"정답 문서: [ID:{gold_doc_id}] {gold_title}")

            # Top-3 검색 결과 출력
            print(f"검색 결과 (Top-{min(3, len(doc_indices[i]))}):")
            for rank, idx in enumerate(doc_indices[i][:3], 1):
                doc_id = retriever.ids[idx]
                title = retriever.titles[idx] if hasattr(retriever, "titles") else ""
                # Context 원본 그대로 출력 (노이즈 확인 가능)
                context_preview = retriever.contexts[idx][:200].replace("\n", " ")
                match_mark = "⭐" if doc_id == gold_doc_id else "  "
                print(f"  {match_mark} {rank}. [ID:{doc_id}] {title}")
                print(f"      {context_preview}...")

    if verbose and total > 0:
        print("=" * 100)

    recall_at_k = exact_matches / total if total > 0 else 0.0

    return {
        "recall_at_k": recall_at_k,
        "exact_match_count": exact_matches,
        "total_samples": total,
    }


def load_small_eval_dataset(dataset_path: str, num_samples: int) -> Dataset:
    """
    train + validation 합쳐서 num_samples 개 샘플만 shuffle해서 사용.
    """
    ds = load_from_disk(dataset_path)
    full = concatenate_datasets(
        [
            ds["train"].flatten_indices(),
            ds["validation"].flatten_indices(),
        ]
    )
    full = full.shuffle(seed=2024)
    num_samples = min(num_samples, len(full))
    small = full.select(range(num_samples))
    return small


def test_sparse(
    dataset_path: str,
    data_path: str,
    context_path: str,
    num_samples: int,
    topk: int,
    use_faiss: bool,
    show_examples: int = 3,
) -> None:
    print("\n" + "=" * 80)
    print("🔎 SPARSE RETRIEVAL SANITY CHECK")
    print("=" * 80)

    small_ds = load_small_eval_dataset(dataset_path, num_samples)
    print(f"📊 평가 샘플: {len(small_ds)}개 (train+validation에서 랜덤 추출)")

    # tokenizer는 그냥 klue/bert-base 기준으로
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=data_path,
        context_path=context_path,
    )

    # Base 스타일 build() (내부에서 get_sparse_embedding 호출)
    print("📥 Retriever 초기화 중...")
    retriever.build()
    print(f"✅ Sparse p_embedding shape: {retriever.p_embedding.shape}")

    # Dataset 단위 retrieve
    df = retriever.retrieve(small_ds, topk=topk)
    print(f"✅ Retrieve 완료: {len(df)}개 샘플")

    # hit@k 계산 (Context 텍스트 매칭 기반)
    _, doc_indices = retriever.get_relevant_doc_bulk(small_ds["question"], k=topk)
    hit = compute_hit_at_k_by_index(small_ds, retriever, doc_indices)

    # Recall@k 계산 (Document ID 매칭 기반) - show_examples 개수만큼 출력
    metrics = compute_recall_precision_at_k(
        small_ds, retriever, doc_indices, verbose=(show_examples > 0)
    )

    print("\n" + "=" * 80)
    print("📊 RETRIEVAL 성능 메트릭")
    print("=" * 80)
    print(f"평가 샘플 수: {metrics['total_samples']}개")
    print(f"Top-K: {topk}")
    print()
    print(f"Context 텍스트 매칭:")
    print(
        f"  Hit@{topk}: {hit:.1%}  ({int(hit * metrics['total_samples'])}/{metrics['total_samples']})"
    )
    print(f"  → 정답 context가 검색된 문서 텍스트에 포함된 비율")
    print()
    print(f"Document ID 매칭:")
    print(
        f"  Recall@{topk}: {metrics['recall_at_k']:.1%}  ({metrics['exact_match_count']}/{metrics['total_samples']})"
    )
    print(f"  → 정답 document_id가 Top-{topk} 안에 있는 비율")
    print("=" * 80)


def test_dense(
    dataset_path: str,
    data_path: str,
    context_path: str,
    num_samples: int,
    topk: int,
    dense_model: str,
    dense_embedding_path: str,
) -> None:
    print("\n" + "=" * 80)
    print("🔎 DENSE RETRIEVAL SANITY CHECK")
    print("=" * 80)

    small_ds = load_small_eval_dataset(dataset_path, num_samples)

    retriever = DenseRetrieval(
        model_name_or_path=dense_model,
        data_path=data_path,
        context_path=context_path,
        embedding_path=dense_embedding_path,
        max_length=256,
        batch_size=16,
    )

    # 전체 corpus 기준 build() (embedding_path가 있으면 로드, 없으면 계산 후 저장)
    retriever.build()
    print(f"✅ Dense p_embedding shape: {retriever.p_embedding.shape}")

    # Dataset 단위 retrieve
    df = retriever.retrieve(small_ds, topk=topk)
    print("✅ Dense retrieve() 결과 DataFrame columns:", df.columns.tolist())
    print(df.head(3))

    # hit@k 계산 (문서 인덱스 기반)
    _, doc_indices = retriever.get_relevant_doc_bulk(small_ds["question"], k=topk)
    hit = compute_hit_at_k_by_index(small_ds, retriever, doc_indices)
    print(f"✅ Dense hit@{topk} on {len(df)} samples: {hit:.4f}")

    # 단일 query 테스트
    q = small_ds[0]["question"]
    print("\n[예시 쿼리(Dense)]", q)
    result = retriever.retrieve(q, topk=3)
    scores, contexts = result
    print(f"Top-1 score: {scores[0]:.4f}")
    print("Top-1 passage (앞 200자):")
    print(contexts[0][:200].replace("\n", " ") + "...")
    print("=" * 80)


def analyze_full_dataset(
    dataset_path: str,
    data_path: str,
    context_path: str,
    topk_list: List[int] = [1, 5, 10, 20, 50],
    save_log: bool = False,
) -> None:
    """
    전체 train+validation 데이터셋으로 Sparse Retrieval의 recall@k 분석.

    성능 최적화: k=max(topk_list) 한 번만 계산 후 슬라이싱
    """
    print("\n" + "=" * 80)
    print("📊 FULL DATASET ANALYSIS (Train + Validation)")
    print("=" * 80)

    # 전체 데이터셋 로드
    ds = load_from_disk(dataset_path)
    train_ds = ds["train"].flatten_indices()
    valid_ds = ds["validation"].flatten_indices()

    print(f"📁 Train samples: {len(train_ds)}")
    print(f"📁 Valid samples: {len(valid_ds)}")

    # Tokenizer & Retriever 초기화
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=data_path,
        context_path=context_path,
    )
    retriever.build()

    max_k = max(topk_list)

    # 로그 저장용
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("SPARSE RETRIEVAL ANALYSIS REPORT")
    log_lines.append("=" * 80)
    log_lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Max K: {max_k}")
    log_lines.append("")

    # Train 분석 (한 번만 계산)
    print(f"\n⏳ Train 데이터셋 분석 중... (Top-{max_k} 계산)")
    _, train_doc_indices = retriever.get_relevant_doc_bulk(
        train_ds["question"], k=max_k
    )

    print("\n" + "=" * 80)
    print("📈 Train Dataset (n={})".format(len(train_ds)))
    print("=" * 80)
    print(f"{'Top-K':<8} {'Recall@K':<12} {'Match/Total':<15}")
    print("-" * 80)

    log_lines.append(f"Train Dataset (n={len(train_ds)})")
    log_lines.append("-" * 80)

    for k in topk_list:
        # 이미 계산된 결과에서 슬라이싱
        sliced_indices = [indices[:k] for indices in train_doc_indices]
        metrics = compute_recall_precision_at_k(train_ds, retriever, sliced_indices)

        result_line = f"{k:<8} {metrics['recall_at_k']:<12.1%} {metrics['exact_match_count']}/{metrics['total_samples']}"
        print(result_line)
        log_lines.append(
            f"  Recall@{k:3d}: {metrics['recall_at_k']:.4f} ({metrics['exact_match_count']}/{metrics['total_samples']})"
        )

    log_lines.append("")

    # Validation 분석 (한 번만 계산)
    print(f"\n⏳ Validation 데이터셋 분석 중... (Top-{max_k} 계산)")
    _, valid_doc_indices = retriever.get_relevant_doc_bulk(
        valid_ds["question"], k=max_k
    )

    print("\n" + "=" * 80)
    print("📈 Validation Dataset (n={})".format(len(valid_ds)))
    print("=" * 80)
    print(f"{'Top-K':<8} {'Recall@K':<12} {'Match/Total':<15}")
    print("-" * 80)

    log_lines.append(f"Validation Dataset (n={len(valid_ds)})")
    log_lines.append("-" * 80)

    for k in topk_list:
        sliced_indices = [indices[:k] for indices in valid_doc_indices]
        metrics = compute_recall_precision_at_k(valid_ds, retriever, sliced_indices)

        result_line = f"{k:<8} {metrics['recall_at_k']:<12.1%} {metrics['exact_match_count']}/{metrics['total_samples']}"
        print(result_line)
        log_lines.append(
            f"  Recall@{k:3d}: {metrics['recall_at_k']:.4f} ({metrics['exact_match_count']}/{metrics['total_samples']})"
        )

    log_lines.append("")

    # 전체 데이터 분석 (한 번만 계산)
    full_ds = concatenate_datasets([train_ds, valid_ds])
    print(f"\n⏳ Full 데이터셋 분석 중... (Top-{max_k} 계산)")
    full_doc_indices = train_doc_indices + valid_doc_indices

    print("\n" + "=" * 80)
    print("📈 Full Dataset - Train + Valid (n={})".format(len(full_ds)))
    print("=" * 80)
    print(f"{'Top-K':<8} {'Recall@K':<12} {'Match/Total':<15}")
    print("-" * 80)

    log_lines.append(f"Full Dataset (n={len(full_ds)})")
    log_lines.append("-" * 80)

    for k in topk_list:
        sliced_indices = [indices[:k] for indices in full_doc_indices]
        metrics = compute_recall_precision_at_k(full_ds, retriever, sliced_indices)

        result_line = f"{k:<8} {metrics['recall_at_k']:<12.1%} {metrics['exact_match_count']}/{metrics['total_samples']}"
        print(result_line)
        log_lines.append(
            f"  Recall@{k:3d}: {metrics['recall_at_k']:.4f} ({metrics['exact_match_count']}/{metrics['total_samples']})"
        )

    print("=" * 80)
    print("\n💡 해석:")
    print("  - Recall@K: Question 던졌을 때 정답 document가 Top-K 안에 있는 비율")
    print("  - 높을수록 좋음 (Retriever가 정답 문서를 잘 찾음)")
    print("  - K가 클수록 Recall은 증가 (더 많은 문서를 검색하므로)")

    # 로그 저장
    if save_log:
        log_lines.append("")
        log_lines.append("=" * 80)
        log_file = f"logs/retrieval_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs("logs", exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"\n💾 로그 저장: {log_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval sanity check (Sparse / Dense)"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/train_dataset",
        help="HuggingFace load_from_disk로 저장된 train_dataset 경로",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="wikipedia_documents.json이 있는 경로",
    )
    parser.add_argument(
        "--context_path",
        type=str,
        default="wikipedia_documents.json",
        help="위키 코퍼스 파일명",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="sanity check용으로 사용할 샘플 수",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="retrieval 시 top-k passage 개수",
    )

    # Sparse / Dense on/off
    parser.add_argument(
        "--sparse",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Sparse retrieval 테스트 여부 (default: True)",
    )
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Sparse에서 faiss indexer 사용 여부",
    )
    parser.add_argument(
        "--dense",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Dense retrieval 테스트 여부 (default: False)",
    )
    parser.add_argument(
        "--analyze_full",
        action="store_true",
        help="전체 train+valid 데이터셋으로 recall@k 분석 수행",
    )
    parser.add_argument(
        "--save_log",
        action="store_true",
        help="분석 결과를 로그 파일로 저장",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=0,
        help="Sanity check 시 출력할 예시 개수 (기본: 0)",
    )

    # Dense 설정
    parser.add_argument(
        "--dense_model",
        type=str,
        default="upskyy/gte-base-korean",
        help="DenseRetrieval에 사용할 HF embedding 모델 이름",
    )
    parser.add_argument(
        "--dense_embedding_path",
        type=str,
        default=None,
        help="corpus dense embedding을 저장/로딩할 npy 경로 (None이면 자동 생성)",
    )

    args = parser.parse_args()

    # sparse/dense 둘 다 False면 강제로 sparse만 켜기
    if not args.sparse and not args.dense:
        print("⚠️ sparse/dense 모두 False라서, sparse만 True로 설정합니다.")
        args.sparse = True

    return args


def main() -> None:
    args = parse_args()

    # Dense embedding path 자동 생성 로직
    if args.dense and args.dense_embedding_path is None:
        model_slug = args.dense_model.replace("/", "_")
        args.dense_embedding_path = f"{args.data_path}/dense_embedding_{model_slug}.npy"
        print(f"📌 Auto-generated dense_embedding_path: {args.dense_embedding_path}")

    print("=== Retrieval Sanity Check Config ===")
    print(f"dataset_path        = {args.dataset_path}")
    print(f"data_path           = {args.data_path}")
    print(f"context_path        = {args.context_path}")
    print(f"num_samples         = {args.num_samples}")
    print(f"topk                = {args.topk}")
    print(f"sparse              = {args.sparse}")
    print(f"use_faiss           = {args.use_faiss}")
    print(f"dense               = {args.dense}")
    print(f"dense_model         = {args.dense_model}")
    print(f"dense_embedding_path= {args.dense_embedding_path}")
    print(f"analyze_full        = {args.analyze_full}")
    print(f"save_log            = {args.save_log}")
    print(f"show_examples       = {args.show_examples}")
    print("=====================================")

    # 전체 데이터셋 분석 모드
    if args.analyze_full:
        analyze_full_dataset(
            dataset_path=args.dataset_path,
            data_path=args.data_path,
            context_path=args.context_path,
            topk_list=[1, 5, 10, 20, 50, 100],
            save_log=args.save_log,
        )
        return
    # Sanity check 모드
    if args.sparse:
        test_sparse(
            dataset_path=args.dataset_path,
            data_path=args.data_path,
            context_path=args.context_path,
            num_samples=args.num_samples,
            topk=args.topk,
            use_faiss=args.use_faiss,
            show_examples=args.show_examples,
        )

    if args.dense:
        test_dense(
            dataset_path=args.dataset_path,
            data_path=args.data_path,
            context_path=args.context_path,
            num_samples=args.num_samples,
            topk=args.topk,
            dense_model=args.dense_model,
            dense_embedding_path=args.dense_embedding_path,
        )


if __name__ == "__main__":
    main()
