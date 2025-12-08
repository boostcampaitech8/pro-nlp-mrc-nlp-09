import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ==============================
# 설정
# ==============================
question_encoder_path = "./outputs/minseok/question_encoder"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_cache_path = "./data/embeddings/context_embeddings.npy"
chunk_metadata_path = "./data/embeddings/chunk_metadata.npy"
dedup_wiki_path = "./data/embeddings/wiki_texts_dedup.pkl"
top_k = 5
candidate_k = 10  # 중복 제거 후에도 top_k를 채우기 위해 충분히 크게

# ==============================
# 모델 로드
# ==============================
question_tokenizer = AutoTokenizer.from_pretrained(question_encoder_path)
question_encoder = AutoModel.from_pretrained(question_encoder_path).to(device)
question_encoder.eval()

# ==============================
# 중복 제거된 wiki 문서 로드
# ==============================
if os.path.exists(dedup_wiki_path):
    print("Loading cached deduplicated wiki texts...")
    with open(dedup_wiki_path, "rb") as f:
        cached = pickle.load(f)  # dict!
        wiki_texts = cached["wiki_texts"]  # list 추출!
else:
    raise FileNotFoundError(f"{dedup_wiki_path} not found.")

# ==============================
# chunk embedding 및 metadata 로드
# ==============================
context_embeddings = torch.tensor(np.load(embedding_cache_path), device=device)
context_embeddings = F.normalize(context_embeddings, p=2, dim=1)
chunk_metadata = np.load(chunk_metadata_path)

# ==============================
# 쿼리 임베딩 (다중 query 지원)
# ==============================
queries = [
    "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?",
    "현대적 인사조직관리의 시발점이 된 책은?",
    "강희제가 1717년에 쓴 글은 누구를 위해 쓰여졌는가?",
    "11~12세기에 제작된 본존불은 보통 어떤 나라의 특징이 전파되었나요?",
    "명문이 적힌 유물을 구성하는 그릇의 총 개수는?",
    "카드모스의 부하들이 간 곳에는 무엇이 있었는가?",
    "관우를 불태워 죽이려한 사람 누구인가?",
    "참호 속에 무기와 장비를 버리고 도주한 집단은?",
    "제2차 세계 대전에 참전하여 사망한 자식은?",
    "고려 공민왕이 처가 식구들과 아내와 함께 피신처로 삼은 마을은?"
]

q_inputs = question_tokenizer(
    queries,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)
q_inputs = {k: v.to(device) for k, v in q_inputs.items()}

with torch.no_grad():
    q_outputs = question_encoder(**q_inputs)
    q_emb = q_outputs.pooler_output if hasattr(q_outputs, "pooler_output") else q_outputs.last_hidden_state[:,0]
    q_emb = F.normalize(q_emb, p=2, dim=1)  # [batch_size, hidden_dim]

# ==============================
# dot product로 top-k 후보 검색 (중복 제거 전 충분히 많이)
# ==============================
scores = q_emb @ context_embeddings.T  # [batch_size, num_chunks]
topk_scores, topk_indices = torch.topk(scores, k=candidate_k, dim=1)

topk_scores = topk_scores.cpu().numpy()
topk_indices = topk_indices.cpu().numpy()

# ==============================
# top-k 출력 (중복 문서 제거, top_k 보장)
# ==============================
for q_idx, query in enumerate(queries):
    print(f"\nQuery: {query}\n{'='*40}")
    seen_docs = set()
    count = 0
    for idx, score in zip(topk_indices[q_idx], topk_scores[q_idx]):
        doc_idx = chunk_metadata[idx]  # chunk -> passage 매핑
        if doc_idx in seen_docs:
            continue
        seen_docs.add(doc_idx)
        print(f"Score: {score:.4f}")
        print(wiki_texts[doc_idx][:400], "...\n")
        count += 1
        if count >= top_k:
            break
