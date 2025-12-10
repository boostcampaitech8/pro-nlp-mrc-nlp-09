import os
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
import matplotlib.pyplot as plt

def get_hard_sample(
        data_path,
        context_path,
        bi_encoder_model,
        cross_encoder_model,
        candidate,
        hard_sample_k=5
    ):
    device = "cuda"

    # ========================================================
    # 1. 모델 로드 (float32)
    # ========================================================
    print("Loading DPR model...")
    bi_model = SentenceTransformer(bi_encoder_model, device=device)

    print("Loading Cross-Encoder model...")
    cross_model = CrossEncoder(cross_encoder_model, device=device, max_length=512)

    # ========================================================
    # 2. Wikipedia 로드 + Dedup
    # ========================================================
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

    # ========================================================
    # 3. DPR Wikipedia Embedding
    # ========================================================
    embed_path = os.path.join(data_path, "embeddings", "wiki_emb_sentence.bin")
    if os.path.isfile(embed_path):
        print("Loading wiki embeddings...")
        with open(embed_path, "rb") as f:
            wiki_emb = pickle.load(f)
    else:
        print("Generating wiki embeddings...")
        batch_size = 128
        all_embs = []
        for i in tqdm(range(0, len(wiki_texts), batch_size)):
            batch = wiki_texts[i:i+batch_size]
            emb = bi_model.encode(batch, batch_size=len(batch), normalize_embeddings=True)
            all_embs.append(emb)
        wiki_emb = np.vstack(all_embs)
        os.makedirs(os.path.dirname(embed_path), exist_ok=True)
        with open(embed_path, "wb") as f:
            pickle.dump(wiki_emb, f)
    print("wiki_emb shape:", wiki_emb.shape)

    # ========================================================
    # 4. Train 데이터 로드
    # ========================================================
    train_dataset = load_from_disk(os.path.join(data_path, "train_dataset"))
    train_examples = list(train_dataset["train"])
    questions = [ex["question"] for ex in train_examples]

    # ========================================================
    # 5. DPR 후보 추출 (GPU, float32)
    # ========================================================
    print("Retrieving candidates with DPR (GPU, full wiki)...")
    wiki_emb_t = torch.tensor(wiki_emb, device=device, dtype=torch.float32)
    dpr_batch_size = 64

    retrieved_candidates = []
    for i in tqdm(range(0, len(questions), dpr_batch_size), desc="Question Embedding"):
        batch_q = questions[i:i+dpr_batch_size]
        q_embs = bi_model.encode(batch_q, batch_size=len(batch_q), normalize_embeddings=True)
        q_embs_t = torch.tensor(q_embs, device=device, dtype=torch.float32)
        sims = torch.matmul(q_embs_t, wiki_emb_t.T)
        topk_vals, topk_idx = torch.topk(sims, candidate, dim=1)
        topk_idx = topk_idx.cpu().numpy()

        for j, idxs in enumerate(topk_idx):
            candidates = [wiki_texts[k] for k in idxs]
            retrieved_candidates.append({
                "question": batch_q[j],
                "positive": train_examples[i+j].get("context", ""),
                "candidates": candidates
            })

    # ========================================================
    # 6. Cross-Encoder Rerank (GPU float32)
    # ========================================================
    print("Reranking candidates with Cross-Encoder...")
    negative_samples = []
    all_sims = []
    cross_batch_size = 512

    for item in tqdm(retrieved_candidates, desc="Cross-Encoder Rerank"):
        question = item["question"]
        positive = item["positive"]
        candidates = [str(c) for c in item["candidates"]]

        scores = []
        for k in range(0, len(candidates), cross_batch_size):
            batch = candidates[k:k+cross_batch_size]
            pairs = [(question, c) for c in batch]
            with torch.no_grad():
                batch_scores = cross_model.predict(pairs)  # float32
            scores.extend(batch_scores)

        all_sims.extend(scores)

        sim_low, sim_high = 0.003, 0.08
        hard_idx = [i for i, s in enumerate(scores) if sim_low <= s <= sim_high]
        chosen = sorted(hard_idx, key=lambda i: scores[i], reverse=True)[:hard_sample_k]
        negatives = [candidates[i] for i in chosen]

        negative_samples.append({
            "question": question,
            "positive": positive,
            "negatives": negatives
        })
    

    # ========================================================
    # 7. 통계 및 저장
    # ========================================================
    os.makedirs("./data/plots", exist_ok=True)
    all_sims = np.array(all_sims)

    # 기본 통계 출력
    print("\n=== Cross-Encoder Similarity Stats ===")
    print("count :", len(all_sims))
    print("mean  :", np.mean(all_sims))
    print("median:", np.median(all_sims))
    print("std   :", np.std(all_sims))
    print("min   :", np.min(all_sims))
    print("max   :", np.max(all_sims))
    print("Percentiles (50,75,90,95,99) :", np.percentile(all_sims, [50,75,90,95,99]))

    # ========================================================
    # 1) 전체 0~1 히스토그램 (참고용)
    plt.figure(figsize=(8,4))
    plt.hist(all_sims, bins=50, range=(0,1))
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.title("Cross-Encoder Similarity Distribution (0~1)")
    plt.savefig("./data/plots/sim_distribution_full.png")
    plt.close()

    # ========================================================
    # 2) 0~0.1 확대 히스토그램 (hard negative 확인용)
    plt.figure(figsize=(8,4))
    plt.hist(all_sims, bins=50, range=(0,0.1))
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.title("Cross-Encoder Similarity Distribution (0~0.3 zoom)")
    plt.savefig("./data/plots/sim_distribution_zoom.png")
    plt.close()

    # ========================================================
    # 3) 로그 스케일 히스토그램 (0 근처 후보 확인용)
    plt.figure(figsize=(8,4))
    plt.hist(all_sims+1e-6, bins=50, log=True)  # log scale, 0 방지 위해 +1e-6
    plt.xlabel("Similarity")
    plt.ylabel("Frequency (log scale)")
    plt.title("Cross-Encoder Similarity Distribution (log scale)")
    plt.savefig("./data/plots/sim_distribution_log.png")
    plt.close()

    neg_counts = [len(ex["negatives"]) for ex in negative_samples]
    print("\n=== Negative Sample Stats per Question ===")
    print("min negatives per question :", np.min(neg_counts))
    print("max negatives per question :", np.max(neg_counts))
    print("avg negatives per question :", np.mean(neg_counts))

    dataset_dict = {
        "question": [ex["question"] for ex in negative_samples],
        "positive": [ex["positive"] for ex in negative_samples],
        "negatives": [ex["negatives"] for ex in negative_samples],
    }
    dpr_dataset = Dataset.from_dict(dataset_dict)
    save_path = os.path.join(data_path, "train_dataset", "negative")
    dpr_dataset.save_to_disk(save_path)
    print("\nSaved DPR + CrossEncoder negatives at:", save_path)
