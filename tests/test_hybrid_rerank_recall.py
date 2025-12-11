"""
Hybrid Retrieval (BM25Plus+Kiwi + KoE5) + Reranker 성능 측정 스크립트.

이 스크립트는 다음 파이프라인의 Recall 성능을 측정합니다:
1. Retrieval: Hybrid (BM25Plus + KoE5)
2. Reranking: Cross-Encoder (BGE-M3 등)

Usage:
    python tests/test_hybrid_rerank_recall.py \
        --retrieval_type hybrid \
        --reranker_name BAAI/bge-reranker-v2-m3 \
        --bm25_impl rank_bm25 --retrieval_tokenizer_name kiwi
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
from datasets import load_from_disk

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import get_retriever, BaseRetrieval
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.tokenization import get_tokenizer
from transformers import AutoTokenizer

def setup_components(args):
    """Retrieval 및 Reranker 초기화"""
    
    # 1. Tokenizer (Kiwi or Auto)
    print(f"\n[INIT] Setting up tokenizer: {args.retrieval_tokenizer_name}")
    # KoE5 등에서 사용할 모델 토크나이저 (필요 시)
    model_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large") # Default fallback
    
    tokenize_fn = get_tokenizer(args.retrieval_tokenizer_name, model_tokenizer)
    
    # 2. Retriever (Hybrid)
    print(f"[INIT] Setting up Retriever: {args.retrieval_type}")
    print(f"       - BM25 Impl: {args.bm25_impl} (k1={args.bm25_k1}, b={args.bm25_b}, delta={args.bm25_delta})")
    print(f"       - Hybrid Alpha: {args.alpha}")
    
    # get_retriever를 통해 HybridRetrieval 생성
    # kwargs로 BM25Plus 파라미터 전달
    retriever = get_retriever(
        retrieval_type=args.retrieval_type,
        tokenize_fn=tokenize_fn,
        data_path=args.data_path,
        context_path="wikipedia_documents.json",
        # Hybrid Args
        alpha=args.alpha,
        fusion_method="rrf", # or score
        # BM25 Args
        impl=args.bm25_impl,
        k1=args.bm25_k1,
        b=args.bm25_b,
        delta=args.bm25_delta,
        # Dense Args (KoE5 defaults)
        dense_model_name="monologg/koelectra-base-v3-discriminator", # KoE5 기본값? 확인 필요하지만 일단 패스
    )
    
    print("[INIT] Building retriever index...")
    retriever.build()
    
    # 3. Reranker
    reranker = None
    if args.reranker_name:
        print(f"[INIT] Setting up Reranker: {args.reranker_name}")
        reranker = CrossEncoderReranker(model_name=args.reranker_name)
    else:
        print("[INIT] No Reranker selected.")
        
    return retriever, reranker

def calculate_recall(
    dataset,
    retriever: BaseRetrieval,
    reranker: Optional[CrossEncoderReranker],
    topk_list: List[int],
    rerank_topk: int = 100
):
    """Recall 계산 루프"""
    queries = dataset["question"]
    gold_contexts = [ex["context"] for ex in dataset]
    
    # 1. Initial Retrieval
    # Reranker가 있으면 더 많이(rerank_topk) 가져와서 정렬
    # 없으면 그냥 max(topk_list)만큼 가져옴
    initial_k = rerank_topk if reranker else max(topk_list)
    
    print(f"\n[EVAL] Retrieving top-{initial_k} candidates...")
    start_time = time.time()
    
    # Hybrid Retrieval
    doc_scores, doc_indices = retriever.get_relevant_doc_bulk(queries, k=initial_k)
    
    retrieval_time = time.time() - start_time
    print(f"       Done in {retrieval_time:.2f}s")
    
    # 2. Reranking & Recall Calculation
    print(f"[EVAL] Reranking and calculating Recall...")
    
    recalls = {k: 0 for k in topk_list}
    total = len(queries)
    
    for i in tqdm(range(total), desc="Evaluating"):
        query = queries[i]
        gold_ctx = gold_contexts[i]
        
        # Candidate Passages
        indices = doc_indices[i]
        passages = [retriever.contexts[idx] for idx in indices]
        
        final_passages = passages
        
        # Reranking
        if reranker:
            # rerank returns scores for the input passages list
            r_scores = reranker.rerank(query, passages)
            
            # Sort passages by new scores
            scored = sorted(zip(passages, r_scores), key=lambda x: x[1], reverse=True)
            final_passages = [p for p, s in scored]
            
        # Check Recall for each K
        for k in topk_list:
            # Top-K 안에 gold_ctx가 있는가?
            # 정확한 string match (공백 등은 전처리 되어있다고 가정하거나 단순 포함관계 확인)
            # 여기서는 list membership (exact match)
            if gold_ctx in final_passages[:k]:
                recalls[k] += 1
                
    # 결과 출력
    print("\n" + "="*60)
    print(f"📊 Evaluation Results (N={total})")
    print("="*60)
    print(f"{ 'Metric':<15} | {'Score':<10} | {'Count':<10}")
    print("-" * 60)
    
    for k in sorted(topk_list):
        score = recalls[k] / total * 100
        print(f"Recall@{k:<2}       | {score:6.2f}%    | {recalls[k]}/{total}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--dataset_path", default="./data/train_dataset", type=str)
    parser.add_argument("--split", default="validation", type=str)
    
    # Retrieval
    parser.add_argument("--retrieval_type", default="hybrid", type=str)
    parser.add_argument("--alpha", default=0.5, type=float, help="Hybrid weight for BM25 (0.0-1.0)")
    
    # BM25 Custom
    parser.add_argument("--bm25_impl", default="rank_bm25", type=str)
    parser.add_argument("--bm25_k1", default=1.2, type=float)
    parser.add_argument("--bm25_b", default=0.6, type=float)
    parser.add_argument("--bm25_delta", default=0.5, type=float)
    
    # Tokenizer
    parser.add_argument("--retrieval_tokenizer_name", default="kiwi", type=str)
    
    # Reranker
    parser.add_argument("--reranker_name", default="BAAI/bge-reranker-v2-m3", type=str)
    parser.add_argument("--rerank_topk", default=50, type=int, help="Number of candidates to rerank")
    
    args = parser.parse_args()
    
    # Load Dataset
    print(f"[LOAD] Loading dataset: {args.dataset_path} ({args.split})")
    ds = load_from_disk(args.dataset_path)
    eval_ds = ds[args.split]
    
    # Setup
    retriever, reranker = setup_components(args)
    
    # Run
    calculate_recall(
        eval_ds, 
        retriever, 
        reranker, 
        topk_list=[1, 5, 10, 20, 30],
        rerank_topk=args.rerank_topk
    )

if __name__ == "__main__":
    main()
