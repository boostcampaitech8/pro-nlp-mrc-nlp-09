import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

def dense_embedding(model_path, wiki_path, embeddings_output_path, metadata_output_path):
    # ======================================================
    # 설정
    # ======================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.join(model_path, "context_encoder")
    EMB_CACHE_PATH = embeddings_output_path
    CHUNK_META_PATH = metadata_output_path
    WIKI_PATH = wiki_path

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
    if os.path.exists(WIKI_PATH):
        print("Loading deduplicated wiki texts from hardsampling cache...")
        with open(WIKI_PATH, "rb") as f:
            cached = pickle.load(f)
            wiki_texts = cached["wiki_texts"]
            wiki_ids = cached["wiki_ids"]  # 추가: metadata용
        print(f"Loaded {len(wiki_texts)} deduplicated passages from cache")
    else:
        raise FileNotFoundError(
            f"hardsampling dedup cache not found: {WIKI_PATH}\n"
            f"Run dense_hard_sampling.py first!"
        )
    
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
    pass