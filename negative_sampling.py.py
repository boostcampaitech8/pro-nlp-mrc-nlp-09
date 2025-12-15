import os
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import matplotlib.pyplot as plt

data_path = "./data"
context_path = "./data/wikipedia_documents.json"
bi_encoder_model = "snumin44/biencoder-ko-bert-question"
cross_encoder_model = "Dongjin-kr/ko-reranker"
candidate = 200
hard_sample_k = 5
device = "cuda"
bi_topk = 30

# 1. Dedup Wikipedia 로드/생성
wiki_cache_path = os.path.join(data_path, "embeddings", "wiki_texts_dedup.pkl")
if os.path.isfile(wiki_cache_path):
    print("Loading cached deduplicated Wikipedia documents...")
    with open(wiki_cache_path, "rb") as f:
        cached = pickle.load(f)
        wiki_texts = cached["wiki_texts"]
        wiki_ids = cached["wiki_ids"]
else:
    print("Deduplicating Wikipedia documents...")
    with open(context_path, "r", encoding="utf-8") as f:
        raw_wiki = json.load(f)
    seen = set()
    wiki_texts, wiki_ids = [], []
    for k, v in raw_wiki.items():
        t = v["text"].strip()
        sig = t[:200]
        if sig not in seen:
            seen.add(sig)
            wiki_texts.append(t)
            wiki_ids.append(k)
    os.makedirs(os.path.dirname(wiki_cache_path), exist_ok=True)
    with open(wiki_cache_path, "wb") as f:
        pickle.dump({"wiki_ids": wiki_ids, "wiki_texts": wiki_texts}, f)
print("Deduped wiki docs:", len(wiki_texts))


# 2. BM25 후보 로드/생성
bm25_path = os.path.join(data_path, "embeddings", "bm25_candidates.pkl")
if os.path.isfile(bm25_path):
    print("Loading cached BM25 candidates...")
    with open(bm25_path, "rb") as f:
        bm25_candidates = pickle.load(f)
else:
    print("Generating BM25 candidates...")
    tokenized_corpus = [doc.split() for doc in wiki_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    train_dataset = load_from_disk(os.path.join(data_path, "train_dataset"))
    questions = [ex["question"] for ex in train_dataset["train"]]
    bm25_candidates = []
    for q in tqdm(questions):
        tokenized_q = q.split()
        topk_idx = bm25.get_top_n(tokenized_q, wiki_texts, n=candidate)
        bm25_candidates.append(topk_idx)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_candidates, f)
print("BM25 candidates loaded:", len(bm25_candidates))


# 3. 모델 로드
print("Loading DPR (bi-encoder) model...")
bi_model = SentenceTransformer(bi_encoder_model, device=device)
print("Loading Cross-Encoder model...")
cross_model = CrossEncoder(cross_encoder_model, device=device, max_length=512)


# 4. Train 데이터
train_dataset = load_from_disk(os.path.join(data_path, "train_dataset"))
train_examples = list(train_dataset["train"])
questions = [ex["question"] for ex in train_examples]


# 5. bi-encoder top-N 후보 선정
print("Embedding BM25 candidates with bi-encoder...")
negative_samples = []
all_sims = []

for i, question in enumerate(tqdm(questions)):
    candidates = bm25_candidates[i]
    candidate_embs = bi_model.encode(candidates, batch_size=len(candidates), normalize_embeddings=True)
    q_emb = bi_model.encode([question], normalize_embeddings=True)
    sims = (candidate_embs @ q_emb.T).squeeze()  # cosine similarity
    all_sims.extend(sims.tolist())

    # bi-encoder top-k
    top_idx = np.argsort(-sims)[:bi_topk]
    top_candidates = [candidates[j] for j in top_idx]

    # 6. Cross-Encoder rerank & hard negative
    cross_scores = []
    batch_size = 64
    for k in range(0, len(top_candidates), batch_size):
        batch = top_candidates[k:k+batch_size]
        pairs = [(question, c) for c in batch]
        with torch.no_grad():
            batch_scores = cross_model.predict(pairs)
        cross_scores.extend(batch_scores)

    sim_low, sim_high = 0.003, 0.08
    hard_idx = [j for j, s in enumerate(cross_scores) if sim_low <= s <= sim_high]
    chosen = sorted(hard_idx, key=lambda j: cross_scores[j], reverse=True)[:hard_sample_k]
    negatives = [top_candidates[j] for j in chosen]

    negative_samples.append({
        "question": question,
        "positive": train_examples[i].get("context", ""),
        "negatives": negatives
    })

# 7. 통계 및 저장
'''
all_sims = np.array(all_sims)
print("\n=== Bi/Cross-Encoder Similarity Stats ===")
print("count :", len(all_sims))
print("mean  :", np.mean(all_sims))
print("median:", np.median(all_sims))
print("std   :", np.std(all_sims))
print("min   :", np.min(all_sims))
print("max   :", np.max(all_sims))
print("Percentiles (50,75,90,95,99) :", np.percentile(all_sims, [50,75,90,95,99]))
'''

dataset_dict = {
    "question": [ex["question"] for ex in negative_samples],
    "positive": [ex["positive"] for ex in negative_samples],
    "negatives": [ex["negatives"] for ex in negative_samples],
}
dpr_dataset = Dataset.from_dict(dataset_dict)
save_path = os.path.join(data_path, "train_dataset", "negative")
dpr_dataset.save_to_disk(save_path)
print("\nSaved DPR + CrossEncoder negatives at:", save_path)
