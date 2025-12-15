# ========================================================
# DPR Wiki Embedding + BM25 Rerank + Dedup + Hard Negative Sampling (Clean Version)
# SentenceTransformer + Mecab + 숫자 분리 + Similarity 분포 그래프 저장 + BM25 캐시
# ========================================================

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import re
from mecab import MeCab
from sklearn.cluster import KMeans
import random

# ========================================================
# 숫자 분리 함수
# ========================================================
def split_numbers(tokens):
    new_tokens = []
    for t in tokens:
        split_t = re.sub(r'([0-9]+)([가-힣A-Za-z])', r'\1 \2', t)
        split_t = re.sub(r'([가-힣A-Za-z])([0-9]+)', r'\1 \2', split_t)
        parts = split_t.split()
        new_tokens.append(t)
        new_tokens.extend(parts)
    return new_tokens
#Mean Pooling을 직접이용하는 bi-encoder
def encode_mean_pool(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1)
    mean_vec = (last_hidden * mask).sum(1) / mask.sum(1)
    return F.normalize(mean_vec, dim=-1)

# ========================================================
# 1. 모델 로드
# ========================================================
MODEL_NAME = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
bi_model = SentenceTransformer(MODEL_NAME, device="cuda")

# ========================================================
# 2. Wikipedia 로드 + Dedup
# ========================================================
wiki_path = "./data/wikipedia_documents_normalized.json"
wiki_cache_path = "./data/embeddings/wiki_texts_dedup.pkl"

if os.path.isfile(wiki_cache_path):
    print("Loading cached deduplicated Wikipedia documents...")
    with open(wiki_cache_path, "rb") as f:
        cached = pickle.load(f)
        wiki_texts = cached["wiki_texts"]
        wiki_ids = cached["wiki_ids"]
else:
    print("Deduplicating Wikipedia documents...")
    with open(wiki_path, "r", encoding="utf-8") as f:
        raw_wiki = json.load(f)

    seen = set()
    wiki_texts = []
    wiki_ids = []

    for k, v in raw_wiki.items():
        t = v["text"].strip()
        sig = t[:200]
        if sig not in seen:
            seen.add(sig)
            wiki_texts.append(t)
            wiki_ids.append(k)

    print(f"[After Dedup] wiki passages: {len(wiki_texts)}")

    os.makedirs(os.path.dirname(wiki_cache_path), exist_ok=True)
    with open(wiki_cache_path, "wb") as f:
        pickle.dump({"wiki_ids": wiki_ids, "wiki_texts": wiki_texts}, f)
    print(f"Saved deduplicated wiki documents at {wiki_cache_path}")


# ========================================================
# 3. BM25 준비 (Mecab+ 숫자 분리)
# ========================================================
mecab = MeCab()
tokens_cache_path = "./data/embeddings/wiki_corpus_mecab_tokens.pkl"

if os.path.exists(tokens_cache_path):
    print("Loading cached Mecab tokens...")
    with open(tokens_cache_path, "rb") as f:
        wiki_corpus_tokens = pickle.load(f)
else:
    print("Tokenizing Wikipedia texts with Mecab + number split...")
    wiki_corpus_tokens = []
    for text in tqdm(wiki_texts):
        base_tokens = mecab.morphs(text)
        tokens = split_numbers(base_tokens)
        wiki_corpus_tokens.append(tokens)
    os.makedirs(os.path.dirname(tokens_cache_path), exist_ok=True)
    with open(tokens_cache_path, "wb") as f:
        pickle.dump(wiki_corpus_tokens, f)

bm25 = BM25Okapi(wiki_corpus_tokens)
#bm25 = BM25Okapi(wiki_corpus_tokens, k1=1.5, b=0.7) #한국어 최적화 파라미터

# ========================================================
# 4. SentenceTransformer Embedding
# ========================================================
embed_path = "./data/embeddings/wiki_emb_sentence_mecab.bin"

if os.path.isfile(embed_path):
    print("Loading wiki embeddings...")
    with open(embed_path, "rb") as f:
        wiki_emb = pickle.load(f)

else:
    print("Generating wiki embeddings with SentenceTransformer (encode)...")
    batch_size = 128
    all_embs = []

    for i in tqdm(range(0, len(wiki_texts), batch_size)):
        batch = wiki_texts[i:i+batch_size]
        emb = bi_model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embs.append(emb)

    wiki_emb = np.vstack(all_embs)

    os.makedirs(os.path.dirname(embed_path), exist_ok=True)
    with open(embed_path, "wb") as f:
        pickle.dump(wiki_emb, f)

print("wiki_emb shape:", wiki_emb.shape)

# ========================================================
# 5. Train 데이터 로드
# ========================================================
train_dataset = load_from_disk("./data/train_dataset_clean")
train_examples = list(train_dataset["train"])

# ========================================================
# 6. BM25 후보 캐시 저장/불러오기
# ========================================================
bm25_candidates_path = "./data/embeddings/bm25_candidates_mecab.pkl"
bm25_expand = 200

if os.path.exists(bm25_candidates_path):
    print("Loading cached BM25 candidates...")
    with open(bm25_candidates_path, "rb") as f:
        bm25_candidates_all = pickle.load(f)
else:
    print("Generating BM25 candidates...")
    bm25_candidates_all = []
    for ex in tqdm(train_examples, desc="Collect BM25 Candidates"):
        question = ex["question"]
        positive = ex["context"]
        q_tokens = split_numbers(mecab.morphs(question))
        scores = bm25.get_scores(q_tokens)
        bm25_idx = np.argsort(scores)[-bm25_expand:][::-1]
        doc_id_pos = ex["document_id"]
        filtered_ids = [
                j for j in bm25_idx 
                if wiki_ids[j] != doc_id_pos
            ]
        if len(filtered_ids) == 0:
            filtered_ids = [bm25_idx[0]]
        candidates = [wiki_texts[i] for i in filtered_ids]
        bm25_candidates_all.append({
            "question": question,
            "bm25_candidates": candidates,
            "filtered_ids": filtered_ids  # index 정보도 저장
        })
    os.makedirs(os.path.dirname(bm25_candidates_path), exist_ok=True)
    with open(bm25_candidates_path, "wb") as f:
        pickle.dump(bm25_candidates_all, f)
    print(f"Saved BM25 candidates to {bm25_candidates_path}")

# ========================================================
# 7. Hard Negative Sampling

NUM_EASY = 1
NUM_MID = 1
NUM_HARD = 3   # clustering으로 뽑을 hard 개수
TOP_HARD_POOL = 30   # hard 후보군 size

negative_samples = []
all_sims = []
cached_q_emb = {}

def get_q_emb(q):
    if q in cached_q_emb:
        return cached_q_emb[q]
    emb = bi_model.encode(q, normalize_embeddings=True)
    cached_q_emb[q] = emb
    return emb


# 1) 전체 similarity 분포 수집
print("Collecting similarity distribution for auto-threshold...")
for i, ex in enumerate(train_examples):
    question = ex["question"]
    bm25_entry = bm25_candidates_all[i]
    filtered_ids = bm25_entry["filtered_ids"]

    q_emb = get_q_emb(question)
    cand_emb = wiki_emb[filtered_ids]
    sims = cand_emb @ q_emb
    all_sims.extend(sims.tolist())

all_sims = np.array(all_sims)

# 2) threshold 자동 결정
sim_low = np.quantile(all_sims, 0.20)    # mid threshold low
sim_mid = np.quantile(all_sims, 0.40)    # mid threshold high
# hard는 top-K pool에서 clustering

print(f"[Auto Threshold] mid_low={sim_low:.4f}, mid_high={sim_mid:.4f}")

# ========================================================
# 3) Negative Sampling 시작
# ========================================================
negative_samples = []

for idx, ex in tqdm(enumerate(train_examples), desc="Sampling negatives", total=len(train_examples)):
    question = ex["question"]
    positive = ex["context"]

    bm25_entry = bm25_candidates_all[idx]
    cand_texts = bm25_entry["bm25_candidates"]
    filtered_ids = bm25_entry["filtered_ids"]

    q_emb = get_q_emb(question)
    cand_emb = wiki_emb[filtered_ids]
    sims = cand_emb @ q_emb

    # ------------------------------------------------------------------
    # (1) EASY NEGATIVE: BM25 하위 30% 중 랜덤 1개
    # ------------------------------------------------------------------
    n = len(filtered_ids)
    bottom_30_start = int(n * 0.70)
    easy_pool_ids = filtered_ids[bottom_30_start:]
    easy_pool_texts = cand_texts[bottom_30_start:]

    if len(easy_pool_texts) == 0:
        easy = random.choice(cand_texts)  # fallback
    else:
        easy = random.choice(easy_pool_texts)

    # ------------------------------------------------------------------
    # (2) MID NEGATIVE: similarity 중간구간(sim_low ~ sim_mid)
    # ------------------------------------------------------------------
    mid_idx = [j for j, s in enumerate(sims) if sim_low <= s <= sim_mid]
    if len(mid_idx) == 0:
        # fallback: similarity median 근처 문서 선택
        mid_idx = [int(np.argsort(np.abs(sims - np.median(sims)))[0])]
    mid = cand_texts[mid_idx[0]]

    # ------------------------------------------------------------------
    # (3) HARD NEGATIVE: similarity top-30 → clustering → 3개 선택
    # ------------------------------------------------------------------
    # top-30 후보 pool
    top_pool_idx = np.argsort(sims)[-TOP_HARD_POOL:][::-1]
    top_pool_emb = cand_emb[top_pool_idx]

    if len(top_pool_emb) < NUM_HARD:
        hard_docs = [cand_texts[j] for j in top_pool_idx[:NUM_HARD]]
    else:
        # KMeans clustering
        kmeans = KMeans(n_clusters=NUM_HARD, random_state=42)
        kmeans.fit(top_pool_emb)

        hard_docs = []
        for c in range(NUM_HARD):
            # cluster center와 가장 가까운 문서 선택
            cluster_indices = np.where(kmeans.labels_ == c)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_emb = top_pool_emb[cluster_indices]
            center = kmeans.cluster_centers_[c]

            # center와 가장 가까운 index
            closest_idx = cluster_indices[np.argmin(np.linalg.norm(cluster_emb - center, axis=1))]

            doc = cand_texts[top_pool_idx[closest_idx]]
            hard_docs.append(doc)

    # 부족하면 fallback
    while len(hard_docs) < NUM_HARD:
        hard_docs.append(random.choice(cand_texts))

    # ------------------------------------------------------------------
    # 최종 negative 세트 저장
    # ------------------------------------------------------------------
    negative_samples.append({
        "question": question,
        "positive": positive,
        "negatives": {
            "easy": easy,
            "mid": mid,
            "hard": hard_docs
        }
    })

# ========================================================
# 7-1. 유사도 통계 + 그래프 저장
# ========================================================
os.makedirs("./data/plots", exist_ok=True)
all_sims = np.array(all_sims)
print("\n=== DPR Similarity Stats ===")
print("count :", len(all_sims))
print("mean  :", np.mean(all_sims))
print("median:", np.median(all_sims))
print("std   :", np.std(all_sims))
print("min   :", np.min(all_sims))
print("max   :", np.max(all_sims))

plt.hist(all_sims, bins=50)
plt.xlabel("DPR Similarity")
plt.ylabel("Frequency")
plt.title("Distribution of DPR Similarity (SentenceTransformer)")
plt.savefig("./data/plots/sim_distribution.png")
plt.close()

# ========================================================
# 8. Arrow Dataset 저장
# ========================================================
dataset_dict = {
    "question": [ex["question"] for ex in negative_samples],
    "positive": [ex["positive"] for ex in negative_samples],
    "negatives": [ex["negatives"] for ex in negative_samples],
}

dpr_dataset = Dataset.from_dict(dataset_dict)
save_path = "./data/train_dataset/negative"
dpr_dataset.save_to_disk(save_path)

print("\nSaved hardest negatives at:", save_path)
