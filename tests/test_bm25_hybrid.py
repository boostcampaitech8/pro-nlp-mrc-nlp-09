"""
BM25 및 Hybrid Retrieval 빠른 테스트

사용법:
    # BM25만 테스트
    python tests/test_bm25_hybrid.py --method bm25

    # Hybrid (BM25 + KoE5) 테스트
    python tests/test_bm25_hybrid.py --method hybrid --alpha 0.5

    # 전체 비교 (TF-IDF, BM25, KoE5, Hybrid)
    python tests/test_bm25_hybrid.py --method all
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_from_disk
from transformers import AutoTokenizer

from src.retrieval import BM25Retrieval, KoE5Retrieval, HybridRetrieval, SparseRetrieval


def calculate_recall(dataset, retriever, k=10):
    """Recall@K 계산"""
    queries = dataset["question"]
    gold_contexts = [ex["context"] for ex in dataset]

    print(f"   Retrieving top-{k} documents...")
    _, doc_indices = retriever.get_relevant_doc_bulk(queries, k=k)

    hits = 0
    for gold_ctx, indices in zip(gold_contexts, doc_indices):
        retrieved_contexts = [retriever.contexts[idx] for idx in indices]
        if gold_ctx in retrieved_contexts:
            hits += 1

    recall = hits / len(dataset)
    return recall, hits, len(dataset)


def test_retriever(name, retriever, dataset, topk_list=[1, 5, 10, 20, 50]):
    """특정 retriever 테스트"""
    print("\n" + "=" * 80)
    print(f"🔍 {name}")
    print("=" * 80)

    start_time = time.time()
    retriever.build()
    build_time = time.time() - start_time

    print(f"Build time: {build_time:.1f}s")
    print()
    print(f"{'Top-K':<8} {'Recall@K':<12} {'Match/Total':<15}")
    print("-" * 80)

    # 모든 k 값에 대해 한 번에 계산 (progress bar 한 번만)
    max_k = max(topk_list)
    queries = dataset["question"]
    gold_contexts = [ex["context"] for ex in dataset]

    print(f"Retrieving top-{max_k}...", end=" ", flush=True)
    _, doc_indices = retriever.get_relevant_doc_bulk(queries, k=max_k)
    print("✓")

    # 각 k에 대해 recall 계산 (slicing으로 빠르게)
    for k in topk_list:
        hits = 0
        for gold_ctx, indices in zip(gold_contexts, doc_indices):
            topk_contexts = [retriever.contexts[idx] for idx in indices[:k]]
            if gold_ctx in topk_contexts:
                hits += 1

        recall = hits / len(dataset)
        print(f"{k:<8} {recall:<12.1%} {hits}/{len(dataset)}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="BM25 및 Hybrid Retrieval 테스트")
    parser.add_argument(
        "--method",
        type=str,
        default="bm25",
        choices=["bm25", "hybrid", "all"],
        help="테스트할 방법",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid alpha 값 (BM25 가중치, Dense는 1-alpha)",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="rrf",
        choices=["rrf", "score"],
        help="Hybrid fusion 방법",
    )
    parser.add_argument(
        "--dataset", type=str, default="./data/train_dataset", help="데이터셋 경로"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="데이터셋 split",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 BM25 & Hybrid Retrieval Test")
    print("=" * 80)
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset} ({args.split} split)")
    if args.method == "hybrid":
        print(f"Alpha: {args.alpha} (BM25:{args.alpha}, Dense:{1 - args.alpha})")
        print(f"Fusion: {args.fusion}")
    print("=" * 80)

    # 데이터셋 로드
    datasets = load_from_disk(args.dataset)
    eval_dataset = datasets[args.split]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    # Top-K 리스트
    topk_list = [1, 5, 10, 20, 50]

    if args.method == "bm25":
        retriever = BM25Retrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
        )
        test_retriever("BM25", retriever, eval_dataset, topk_list)

    elif args.method == "hybrid":
        retriever = HybridRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
            corpus_emb_path="./data/koe5_corpus_emb.npy",
            alpha=args.alpha,
            fusion_method=args.fusion,
        )
        test_retriever(
            f"Hybrid (alpha={args.alpha}, {args.fusion})",
            retriever,
            eval_dataset,
            topk_list,
        )

    else:  # all
        # TF-IDF
        tfidf_retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
        )
        test_retriever("TF-IDF", tfidf_retriever, eval_dataset, topk_list)

        # BM25
        bm25_retriever = BM25Retrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
        )
        test_retriever("BM25", bm25_retriever, eval_dataset, topk_list)

        # KoE5
        koe5_retriever = KoE5Retrieval(
            data_path="./data",
            context_path="wikipedia_documents.json",
            corpus_emb_path="./data/koe5_corpus_emb.npy",
        )
        test_retriever("KoE5", koe5_retriever, eval_dataset, topk_list)

        # Hybrid
        hybrid_retriever = HybridRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
            corpus_emb_path="./data/koe5_corpus_emb.npy",
            alpha=0.5,
            fusion_method="rrf",
        )
        test_retriever(
            "Hybrid (alpha=0.5, RRF)", hybrid_retriever, eval_dataset, topk_list
        )

    print("\n✅ 테스트 완료!")


if __name__ == "__main__":
    main()
