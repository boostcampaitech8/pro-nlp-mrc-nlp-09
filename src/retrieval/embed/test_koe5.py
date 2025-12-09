"""
KoE5 embedding 빠른 테스트

Usage:
    python -m src.retrieval.embed.test_koe5
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer


def test_koe5_retrieval(
    query: str = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?",
    corpus_emb_path: str = "./data/koe5_corpus_emb.npy",
    wiki_path: str = "./data/wikipedia_documents.json",
    topk: int = 5,
):
    """KoE5 retrieval 간단 테스트"""

    print("=" * 80)
    print("KoE5 Retrieval Test")
    print("=" * 80)

    # 1. 모델 & corpus embedding 로드
    print("\n[1/4] Loading model and corpus embeddings...")
    model = SentenceTransformer("nlpai-lab/KoE5")
    corpus_emb = np.load(corpus_emb_path)
    print(f"   ✓ Corpus embeddings: {corpus_emb.shape}")

    # 2. Wiki contexts 로드
    print(f"\n[2/4] Loading Wikipedia contexts...")
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)

    unique_texts = {}
    for doc_id, doc_info in wiki.items():
        text = doc_info["text"]
        if text not in unique_texts:
            unique_texts[text] = doc_info.get("title", "")

    contexts = list(unique_texts.keys())
    titles = [unique_texts[text] for text in contexts]
    print(f"   ✓ Total contexts: {len(contexts)}")

    # 3. Query embedding
    print(f"\n[3/4] Encoding query...")
    print(f"   Query: {query}")
    query_emb = model.encode(
        [f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True
    )[0]

    # 4. Similarity search
    print(f"\n[4/4] Searching top-{topk} documents...")
    scores = corpus_emb @ query_emb
    top_indices = np.argsort(-scores)[:topk]

    print("\n" + "=" * 80)
    print(f"Top-{topk} Results:")
    print("=" * 80)

    for rank, idx in enumerate(top_indices, 1):
        print(f"\n[Rank {rank}] Score: {scores[idx]:.4f}")
        print(f"Title: {titles[idx]}")
        print(f"Text: {contexts[idx][:200]}...")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?",
    )
    parser.add_argument(
        "--corpus_emb_path", type=str, default="./data/koe5_corpus_emb.npy"
    )
    parser.add_argument(
        "--wiki_path", type=str, default="./data/wikipedia_documents.json"
    )
    parser.add_argument("--topk", type=int, default=5)

    args = parser.parse_args()

    test_koe5_retrieval(
        query=args.query,
        corpus_emb_path=args.corpus_emb_path,
        wiki_path=args.wiki_path,
        topk=args.topk,
    )
