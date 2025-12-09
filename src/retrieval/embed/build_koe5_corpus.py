"""
KoE5 모델로 Wikipedia corpus embedding 생성

Usage:
    python -m src.retrieval.embed.build_koe5_corpus

Output:
    ./data/koe5_corpus_emb.npy (shape: [num_docs, 1024])
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def build_koe5_embeddings(
    wiki_path: str = "./data/wikipedia_documents.json",
    output_path: str = "./data/koe5_corpus_emb.npy",
    model_name: str = "nlpai-lab/KoE5",
    batch_size: int = 128,
    use_title: bool = True,
):
    """
    KoE5로 Wikipedia corpus embedding 생성

    Args:
        wiki_path: Wikipedia JSON 파일 경로
        output_path: 출력 .npy 파일 경로
        model_name: SentenceTransformer 모델명
        batch_size: 인코딩 배치 크기
        use_title: title + text 조합 여부
    """

    print("=" * 80)
    print("KoE5 Corpus Embedding Builder")
    print("=" * 80)

    # 1. Wiki 로드 (중복 제거)
    print(f"\n[1/4] Loading Wikipedia from {wiki_path}...")
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)

    unique_texts = {}
    for doc_id, doc_info in wiki.items():
        text = doc_info["text"]
        if text not in unique_texts:
            unique_texts[text] = {"title": doc_info.get("title", ""), "doc_id": doc_id}

    contexts = list(unique_texts.keys())
    titles = [unique_texts[text]["title"] for text in contexts]

    print(f"   ✓ Total unique contexts: {len(contexts)}")

    # 2. 모델 로드
    print(f"\n[2/4] Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(
        f"   ✓ Model loaded (embedding dim: {model.get_sentence_embedding_dimension()})"
    )

    # 3. Title + Text 조합 (KoE5는 passage: prefix 필수)
    print(f"\n[3/4] Preparing corpus texts (use_title={use_title})...")
    if use_title:
        corpus_texts = [
            f"passage: {title} {text}" for title, text in zip(titles, contexts)
        ]
    else:
        corpus_texts = [f"passage: {text}" for text in contexts]

    print(f"   ✓ Example: {corpus_texts[0][:100]}...")

    # 4. Encoding (배치로 처리)
    print(f"\n[4/4] Encoding corpus (batch_size={batch_size})...")
    embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # 중요: L2 normalize for cosine similarity
        convert_to_numpy=True,
    )

    # 5. 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

    print("\n" + "=" * 80)
    print(f"✅ Saved to {output_path}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Size: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build KoE5 corpus embeddings")
    parser.add_argument(
        "--wiki_path",
        type=str,
        default="./data/wikipedia_documents.json",
        help="Wikipedia JSON file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/koe5_corpus_emb.npy",
        help="Output .npy file path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nlpai-lab/KoE5",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Encoding batch size"
    )
    parser.add_argument(
        "--no_title", action="store_true", help="Don't use title (only text)"
    )

    args = parser.parse_args()

    build_koe5_embeddings(
        wiki_path=args.wiki_path,
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        use_title=not args.no_title,
    )
