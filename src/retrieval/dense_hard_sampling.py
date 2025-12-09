import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import re

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

def get_hard_sample(
        data_path,
        context_path,
        reranker_model,
        bm25_candidate,
        hard_sample_k
    ):
    # ========================================================
    # 1. 모델 로드
    # ========================================================
    MODEL_NAME = reranker_model
    bi_model = SentenceTransformer(MODEL_NAME, device="cuda")

    # ========================================================
    # 2. Wikipedia 로드 + Dedup
    # ========================================================
    wiki_path = context_path
    wiki_cache_path = os.path.join(data_path, "embeddings", "wiki_texts_dedup.pkl")

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
    # 3. BM25 준비 (Okt + 숫자 분리)
    # ========================================================
    okt = Okt()
    tokens_cache_path = os.path.join(data_path, "embeddings", "wiki_corpus_okt_tokens.pkl")

    if os.path.exists(tokens_cache_path):
        print("Loading cached Okt tokens...")
        with open(tokens_cache_path, "rb") as f:
            wiki_corpus_tokens = pickle.load(f)
    else:
        print("Tokenizing Wikipedia texts with Okt + number split...")
        wiki_corpus_tokens = []
        for text in tqdm(wiki_texts):
            base_tokens = okt.morphs(text)
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
    embed_path = os.path.join(data_path, "embeddings", "wiki_emb_sentence.bin")

    if os.path.isfile(embed_path):
        print("Loading wiki embeddings...")
        with open(embed_path, "rb") as f:
            wiki_emb = pickle.load(f)
    else:
        print("Generating wiki embeddings with SentenceTransformer...")
        batch_size = 128
        all_embs = []
        for i in tqdm(range(0, len(wiki_texts), batch_size)):
            batch = wiki_texts[i:i+batch_size]
            emb = bi_model.encode(batch, batch_size=batch_size, normalize_embeddings=True)
            all_embs.append(emb)
        wiki_emb = np.vstack(all_embs)
        os.makedirs(os.path.dirname(embed_path), exist_ok=True)
        with open(embed_path, "wb") as f:
            pickle.dump(wiki_emb, f)

    print("wiki_emb shape:", wiki_emb.shape)

    # ========================================================
    # 5. Train 데이터 로드
    # ========================================================
    train_dataset = load_from_disk(data_path+"/train_dataset")
    train_examples = list(train_dataset["train"])

    # ========================================================
    # 6. BM25 후보 캐시 저장/불러오기
    # ========================================================
    bm25_candidates_path = os.path.join(data_path, "embeddings", "bm25_candidates.pkl")
    bm25_expand = bm25_candidate

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
            q_tokens = split_numbers(okt.morphs(question))
            scores = bm25.get_scores(q_tokens)
            bm25_idx = np.argsort(scores)[-bm25_expand:][::-1]
            filtered_ids = [i for i in bm25_idx if wiki_texts[i] != positive]
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
    # ========================================================
    top_k = hard_sample_k
    sim_low = 0.28
    sim_high = 0.45
    negative_samples = []
    all_sims = []

    for i, ex in tqdm(enumerate(train_examples), desc="Hard Negative Sampling", total=len(train_examples)):
        question = ex["question"]
        positive = ex["context"]

        # BM25 후보 불러오기
        bm25_entry = bm25_candidates_all[i]
        cand_texts = bm25_entry["bm25_candidates"]
        filtered_ids = bm25_entry["filtered_ids"]

        # DPR Similarity
        q_emb = bi_model.encode(question, normalize_embeddings=True)
        cand_emb = wiki_emb[filtered_ids]
        sims = cand_emb @ q_emb
        all_sims.extend(sims.tolist())

        hard_idx = [i for i, s in enumerate(sims) if sim_low <= s <= sim_high]
        if len(hard_idx) < top_k:
            extra = np.argsort(sims)[-top_k:][::-1]
            extra = [i for i in extra if i not in hard_idx]
            hard_idx += extra[: top_k - len(hard_idx)]

        chosen = sorted(hard_idx, key=lambda i: sims[i], reverse=True)[:top_k]
        negatives = [cand_texts[i] for i in chosen]

        negative_samples.append({
            "question": question,
            "positive": positive,
            "negatives": negatives
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
    save_path = os.path.join(data_path, "train_dataset", "negative")
    dpr_dataset.save_to_disk(save_path)

    print("\nSaved hardest negatives at:", save_path)
    pass