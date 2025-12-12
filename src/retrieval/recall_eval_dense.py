import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# Config
# ============================
QUESTION_ENCODER_PATH = "./outputs/sehun/question_encoder"
CONTEXT_EMB_PATH = "./data/embeddings/context_embeddings.npy"
CHUNK_META_PATH = "./data/embeddings/chunk_metadata.npy"
DEDUP_WIKI_PATH = "./data/embeddings/wiki_texts_dedup.pkl"
TRAIN_DATA_PATH = "./data/train_dataset_clean"
TOP_K_LIST = [1, 5, 10, 20, 50, 100]

# ============================
# Load models / data
# ============================
print("Loading question encoder...")
q_tokenizer = AutoTokenizer.from_pretrained(QUESTION_ENCODER_PATH)
q_encoder = AutoModel.from_pretrained(QUESTION_ENCODER_PATH).to(DEVICE)
q_encoder.eval()

print("Loading context embeddings...")
context_emb = torch.tensor(np.load(CONTEXT_EMB_PATH), device=DEVICE)
chunk_meta = np.load(CHUNK_META_PATH)

print("Loading deduplicated wiki texts...")
with open(DEDUP_WIKI_PATH, "rb") as f:
    wiki_texts = pickle.load(f)["wiki_texts"]

print(f"Dedup Wiki passages loaded: {len(wiki_texts)}")


# ============================
# Mean Pooling (DPR와 동일)
# ============================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (B, L, H)
    mask = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * mask).sum(1)
    sum_mask = mask.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


# ============================
# Encode Question (mean pooling)
# ============================
def encode_question(text):
    inputs = q_tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = q_encoder(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
    return emb


# ============================
# Evaluation Function
# ============================
def evaluate_split(name, dataset):
    print(f"\n\n📊 {name} Dataset (n={len(dataset)})")
    print("=" * 70)

    total = len(dataset)
    match_counts = {k: 0 for k in TOP_K_LIST}

    for item in tqdm(dataset):
        question = item["question"]
        gold_ctx = item["context"].strip()

        # Encode query
        q_emb = encode_question(question)  # (1, dim)

        # Dot product similarity
        sims = (context_emb @ q_emb.T).squeeze()  # [num_chunks]

        # Sorted chunk indices
        sorted_idx = torch.argsort(sims, descending=True)

        # Convert chunk → doc id (uniq doc)
        ranked_passages = []
        seen_docs = set()

        for idx in sorted_idx.tolist():
            doc_id = chunk_meta[idx]
            if doc_id not in seen_docs:
                ranked_passages.append(wiki_texts[doc_id])
                seen_docs.add(doc_id)
            if len(ranked_passages) >= max(TOP_K_LIST):
                break

        # ========== GOLD MATCHING ==========  
        # substring 기반 (완전 일치보다 훨씬 정확)
        for k in TOP_K_LIST:
            top_k_passages = ranked_passages[:k]
            if any(gold_ctx[:150] in p for p in top_k_passages):
                match_counts[k] += 1

    # Result summary
    print("Top-K    Recall@K    Match/Total")
    print("-" * 70)
    for k in TOP_K_LIST:
        recall = match_counts[k] / total * 100
        print(f"{k:<8} {recall:5.1f}%     {match_counts[k]}/{total}")


# ============================
# Load Datasets
# ============================
ds = load_from_disk(TRAIN_DATA_PATH)
train_ds = ds["train"]
valid_ds = ds["validation"]

# ============================
# Run Evaluation
# ============================
evaluate_split("Train", train_ds)
evaluate_split("Validation", valid_ds)

full = list(train_ds) + list(valid_ds)
evaluate_split("Full Dataset (Train + Valid)", full)
