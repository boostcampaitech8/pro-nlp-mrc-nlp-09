"""
Re-ranker Inference
- BM25로 상위 N개 후보 추출
- Cross-encoder로 재정렬하여 최종 top-k 선택
"""

import os
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
import re

# ============================
# Config
# ============================
CROSS_ENCODER_PATH = "./outputs/reranker/cross_encoder"  # 학습된 모델
WIKI_PATH = "./data/wikipedia_documents.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Re-ranking 설정
CANDIDATE_K = 100  # 1단계에서 추출할 후보 개수
TOP_K = 5          # 최종 반환할 문서 개수

print(f"Device: {DEVICE}")
print(f"Model path: {CROSS_ENCODER_PATH}")

# ============================
# Utility Functions
# ============================
def split_numbers(tokens):
    """숫자 분리 (BM25 성능 향상)"""
    new_tokens = []
    for t in tokens:
        split_t = re.sub(r'([0-9]+)([가-힣A-Za-z])', r'\1 \2', t)
        split_t = re.sub(r'([가-힣A-Za-z])([0-9]+)', r'\1 \2', split_t)
        parts = split_t.split()
        new_tokens.append(t)
        new_tokens.extend(parts)
    return new_tokens

# ============================
# Load Wikipedia
# ============================
print("\n" + "="*60)
print("Loading Wikipedia documents...")
print("="*60)

wiki_cache_path = "./data/embeddings/wiki_texts_dedup.pkl"

if os.path.exists(wiki_cache_path):
    print("Loading from cache...")
    with open(wiki_cache_path, "rb") as f:
        cached = pickle.load(f)
        wiki_texts = cached["wiki_texts"]
        wiki_ids = cached["wiki_ids"]
else:
    print("Loading from JSON and deduplicating...")
    with open(WIKI_PATH, "r", encoding="utf-8") as f:
        raw_wiki = json.load(f)
    
    seen = set()
    wiki_texts = []
    wiki_ids = []
    
    for k, v in raw_wiki.items():
        text = v["text"].strip()
        sig = text[:200]
        if sig not in seen:
            seen.add(sig)
            wiki_texts.append(text)
            wiki_ids.append(k)
    
    os.makedirs(os.path.dirname(wiki_cache_path), exist_ok=True)
    with open(wiki_cache_path, "wb") as f:
        pickle.dump({"wiki_texts": wiki_texts, "wiki_ids": wiki_ids}, f)

print(f"Total documents: {len(wiki_texts)}")

# ============================
# BM25 Setup
# ============================
print("\n" + "="*60)
print("Setting up BM25...")
print("="*60)

okt = Okt()
tokens_cache_path = "./data/embeddings/wiki_corpus_okt_tokens.pkl"

if os.path.exists(tokens_cache_path):
    print("Loading cached tokens...")
    with open(tokens_cache_path, "rb") as f:
        wiki_corpus_tokens = pickle.load(f)
else:
    print("Tokenizing with Okt...")
    wiki_corpus_tokens = []
    for text in tqdm(wiki_texts):
        base_tokens = okt.morphs(text)
        tokens = split_numbers(base_tokens)
        wiki_corpus_tokens.append(tokens)
    
    os.makedirs(os.path.dirname(tokens_cache_path), exist_ok=True)
    with open(tokens_cache_path, "wb") as f:
        pickle.dump(wiki_corpus_tokens, f)

bm25 = BM25Okapi(wiki_corpus_tokens)
print("BM25 ready!")

# ============================
# Cross-Encoder Setup
# ============================
print("\n" + "="*60)
print("Loading Cross-Encoder model...")
print("="*60)

if not os.path.exists(CROSS_ENCODER_PATH):
    print(f"ERROR: Model not found at {CROSS_ENCODER_PATH}")
    print("Please train the model first using reranker_train.py")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_PATH)
model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_PATH).to(DEVICE)
model.eval()
print("Cross-Encoder loaded!")

# ============================
# Re-ranking Function
# ============================
def retrieve_and_rerank(query, candidate_k=CANDIDATE_K, top_k=TOP_K, batch_size=16):
    """
    2단계 검색 + 재정렬
    
    Args:
        query: 검색 질문
        candidate_k: 1단계에서 추출할 후보 개수
        top_k: 최종 반환할 문서 개수
        batch_size: Cross-encoder 배치 크기
    
    Returns:
        List[Tuple[float, int, str]]: (score, doc_idx, document)
    """
    
    # ==========================================
    # Stage 1: BM25 Retrieval
    # ==========================================
    print(f"\n[Stage 1] BM25 retrieval (top-{candidate_k})...")
    
    q_tokens = split_numbers(okt.morphs(query))
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_indices = np.argsort(bm25_scores)[-candidate_k:][::-1]
    
    candidates = [(i, wiki_texts[i]) for i in bm25_indices]
    
    print(f"Retrieved {len(candidates)} candidates")
    
    # ==========================================
    # Stage 2: Cross-Encoder Re-ranking
    # ==========================================
    print(f"\n[Stage 2] Cross-encoder re-ranking...")
    
    all_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(candidates), batch_size), desc="Re-ranking"):
            batch_candidates = candidates[i:i+batch_size]
            batch_texts = [doc for _, doc in batch_candidates]
            
            # Tokenize query-document pairs
            inputs = tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            ).to(DEVICE)
            
            # Get scores
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            
            all_scores.extend(scores)
    
    # ==========================================
    # Sort by cross-encoder score
    # ==========================================
    results = []
    for (doc_idx, doc_text), score in zip(candidates, all_scores):
        results.append((score, doc_idx, doc_text))
    
    results.sort(key=lambda x: x[0], reverse=True)
    
    return results[:top_k]

# ============================
# Test Queries
# ============================
test_queries = [
    "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?",
    "현대적 인사조직관리의 시발점이 된 책은?",
    "강희제가 1717년에 쓴 글은 누구를 위해 쓰여졌는가?",
    "11~12세기에 제작된 본존불은 보통 어떤 나라의 특징이 전파되었나요?",
    "명문이 적힌 유물을 구성하는 그릇의 총 개수는?",
]

print("\n" + "="*60)
print("Running Test Queries")
print("="*60)

for query in test_queries:
    print("\n" + "="*60)
    print(f"Query: {query}")
    print("="*60)
    
    results = retrieve_and_rerank(query, candidate_k=CANDIDATE_K, top_k=TOP_K)
    
    print(f"\n📊 Top-{TOP_K} Results:\n")
    for rank, (score, doc_idx, doc_text) in enumerate(results, 1):
        print(f"[Rank {rank}] Score: {score:.4f} | Doc ID: {doc_idx}")
        print(f"{doc_text[:300]}...")
        print()

# ============================
# Evaluation Mode
# ============================
def evaluate_on_dataset(dataset_path="./data/train_dataset", split="validation"):
    """
    전체 데이터셋에 대한 평가
    """
    from datasets import load_from_disk
    
    print("\n" + "="*60)
    print(f"Evaluating on {split} set")
    print("="*60)
    
    dataset = load_from_disk(dataset_path)[split]
    
    correct = 0
    mrr_sum = 0.0
    
    for example in tqdm(dataset, desc="Evaluating"):
        query = example["question"]
        gold_context = example["context"]
        
        results = retrieve_and_rerank(query, candidate_k=CANDIDATE_K, top_k=TOP_K)
        retrieved_docs = [doc for _, _, doc in results]
        
        # Top-k accuracy
        if gold_context in retrieved_docs:
            correct += 1
            
            # MRR
            rank = retrieved_docs.index(gold_context) + 1
            mrr_sum += 1.0 / rank
    
    accuracy = correct / len(dataset)
    mrr = mrr_sum / len(dataset)
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Top-{TOP_K} Accuracy: {accuracy:.4f} ({correct}/{len(dataset)})")
    print(f"MRR: {mrr:.4f}")
    
    return accuracy, mrr

# Uncomment to run evaluation
# evaluate_on_dataset()