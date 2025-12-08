import os
import json
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# ======================================================
# 설정
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./outputs/minseok/context_encoder"
EMB_CACHE_PATH = "./data/embeddings/context_embeddings.npy"
CHUNK_META_PATH = "./data/embeddings/chunk_metadata.npy"
WIKI_PATH = "./data/wikipedia_documents.json"

MAX_LENGTH = 512
STRIDE = 256

# ======================================================
# 모델/토크나이저 로드
# ======================================================
context_encoder = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
context_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ======================================================
# 위키 문서 불러오기 + Dedup
# ======================================================
with open(WIKI_PATH, "r", encoding="utf-8") as f:
    raw_wiki = json.load(f)

seen = set()
wiki_texts = []
for k, v in raw_wiki.items():
    text = v["text"].strip()
    sig = text[:200]  # 상위 200자 기준 중복 제거
    if sig not in seen:
        seen.add(sig)
        wiki_texts.append(text)

print(f"[After dedup] wiki passages: {len(wiki_texts)}")

# ======================================================
# 캐시 로드 또는 생성
# ======================================================
if os.path.exists(EMB_CACHE_PATH) and os.path.exists(CHUNK_META_PATH):
    print("Loading cached context embeddings and chunk metadata...")
    context_embeddings = torch.tensor(np.load(EMB_CACHE_PATH), device=DEVICE)
    chunk_metadata = np.load(CHUNK_META_PATH)
else:
    print("Embedding all Wikipedia passages...")

    all_chunk_embeddings = []
    chunk_metadata = []

    for doc_idx, passage in enumerate(tqdm(wiki_texts, desc="Encoding passages")):
        # 토크나이저: chunk 분할
        inputs = context_tokenizer(
            passage,
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            padding=True,
            return_overflowing_tokens=True,
            return_tensors="pt"
        )

        input_ids_chunks = inputs["input_ids"].to(DEVICE)
        attention_mask_chunks = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = context_encoder(
                input_ids=input_ids_chunks,
                attention_mask=attention_mask_chunks
            )
            emb = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]
            emb = F.normalize(emb, p=2, dim=1)
            all_chunk_embeddings.append(emb.cpu().numpy())

        # 각 chunk마다 원문 doc_idx 반복
        chunk_metadata.extend([doc_idx] * input_ids_chunks.size(0))

        torch.cuda.empty_cache()

    # ======================================================
    # 전체 chunk embedding 합치기
    # ======================================================
    context_embeddings = np.vstack(all_chunk_embeddings).astype(np.float32)
    chunk_metadata = np.array(chunk_metadata, dtype=np.int32)

    # 저장
    os.makedirs(os.path.dirname(EMB_CACHE_PATH), exist_ok=True)
    np.save(EMB_CACHE_PATH, context_embeddings)
    np.save(CHUNK_META_PATH, chunk_metadata)
    print(f"Saved context embeddings to {EMB_CACHE_PATH}")
    print(f"Saved chunk metadata to {CHUNK_META_PATH}")

print(f"Context embeddings shape: {context_embeddings.shape}")
print(f"Chunk metadata shape: {chunk_metadata.shape}")
