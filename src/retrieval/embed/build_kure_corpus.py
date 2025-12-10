"""
KURE-v1 모델로 Wikipedia corpus embedding 생성 (with optional chunking)

Usage:
    # 기본 (chunking 없이)
    python -m src.retrieval.embed.build_kure_corpus

    # Chunking 적용
    python -m src.retrieval.embed.build_kure_corpus --use_chunking

Output:
    [paths.py에서 관리]
    - kure_corpus_emb: data/embeddings/kure_corpus_emb.npy
    - kure_passages_meta: data/embeddings/kure_passages_meta.jsonl
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 프로젝트 루트를 path에 추가 (상대 import 문제 해결)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.paths import get_path, ensure_dir

# === 기본 설정 ===
DEFAULT_CHUNK_SIZE = 2000  # 문자 수 기준
DEFAULT_STRIDE = 500
MIN_CHUNK_LENGTH = 300  # 이 길이 미만의 chunk는 버림
MAX_LENGTH_NO_CHUNK = 2000  # 이 길이 이하의 문서는 chunking 안함


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride: int = DEFAULT_STRIDE,
    min_length: int = MIN_CHUNK_LENGTH,
) -> List[Tuple[str, int, int]]:
    """
    긴 텍스트를 sliding window 방식으로 chunking.

    Args:
        text: 원본 텍스트
        chunk_size: 각 chunk의 최대 문자 수
        stride: chunk 간 이동 거리 (chunk_size - overlap)
        min_length: 이 길이 미만의 chunk는 버림

    Returns:
        List of (chunk_text, start_char, end_char)
    """
    if len(text) <= chunk_size:
        # 짧은 문서는 그대로 반환
        return [(text, 0, len(text))]

    chunks = []
    start = 0
    step = chunk_size - stride  # 겹치는 부분을 고려한 이동 거리

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # 너무 짧은 chunk는 버림 (마지막 chunk 제외)
        if len(chunk) >= min_length or start == 0:
            chunks.append((chunk, start, end))

        # 이미 끝에 도달했으면 종료
        if end >= len(text):
            break

        start += step

    return chunks


def build_kure_embeddings(
    wiki_path: Optional[str] = None,  # None이면 paths.py 기본값 사용
    output_emb_path: Optional[str] = None,  # None이면 paths.py 기본값 사용
    output_meta_path: Optional[str] = None,  # None이면 paths.py 기본값 사용
    model_name: str = "nlpai-lab/KURE-v1",
    batch_size: int = 512,
    use_title: bool = True,
    use_chunking: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    stride: int = DEFAULT_STRIDE,
    min_chunk_length: int = MIN_CHUNK_LENGTH,
    max_length_no_chunk: int = MAX_LENGTH_NO_CHUNK,
):
    """
    KURE-v1로 Wikipedia corpus embedding 생성

    Args:
        wiki_path: Wikipedia JSON 파일 경로 (None이면 paths.py 기본값)
        output_emb_path: 출력 embedding .npy 파일 경로 (None이면 paths.py 기본값)
        output_meta_path: 출력 metadata .jsonl 파일 경로 (None이면 paths.py 기본값)
        model_name: SentenceTransformer 모델명
        batch_size: 인코딩 배치 크기
        use_title: title + text 조합 여부
        use_chunking: 길이 기반 chunking 적용 여부
        chunk_size: chunk 최대 문자 수
        stride: chunk 간 이동 거리
        min_chunk_length: 최소 chunk 길이
        max_length_no_chunk: 이 길이 이하의 문서는 chunking 안함
    """
    # 기본 경로 설정 (paths.py에서 관리)
    wiki_path = wiki_path or get_path("wiki_corpus")
    output_emb_path = output_emb_path or get_path("kure_corpus_emb")
    output_meta_path = output_meta_path or get_path("kure_passages_meta")

    # 출력 디렉토리 생성
    ensure_dir("kure_corpus_emb")

    print("=" * 80)
    print("KURE-v1 Corpus Embedding Builder")
    print(f"  Model: {model_name}")
    print(f"  Wiki: {wiki_path}")
    print(f"  Output EMB: {output_emb_path}")
    print(f"  Output META: {output_meta_path}")
    print(f"  Chunking: {use_chunking}")
    if use_chunking:
        print(f"    - chunk_size: {chunk_size}")
        print(f"    - stride: {stride}")
        print(f"    - min_chunk_length: {min_chunk_length}")
        print(f"    - max_length_no_chunk: {max_length_no_chunk}")
    print("=" * 80)

    # 1. Wiki 로드 (중복 제거)
    print(f"\n[1/5] Loading Wikipedia from {wiki_path}...")
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)

    # 중복 제거 (text 기준)
    unique_docs: Dict[str, Dict] = {}
    for doc_id, doc_info in wiki.items():
        text = doc_info["text"]
        if text not in unique_docs:
            unique_docs[text] = {
                "doc_id": int(doc_id),
                "title": doc_info.get("title", ""),
                "text": text,
            }

    print(f"   ✓ Total unique documents: {len(unique_docs)}")

    # 2. Passage 생성 (chunking 여부에 따라)
    print(f"\n[2/5] Building passages (chunking={use_chunking})...")
    passages = []  # List of dict
    passage_id = 0

    for text, doc_info in tqdm(unique_docs.items(), desc="Processing docs"):
        doc_id = doc_info["doc_id"]
        title = doc_info["title"]

        if use_chunking and len(text) > max_length_no_chunk:
            # Chunking 적용
            chunks = chunk_text(text, chunk_size, stride, min_chunk_length)
            for chunk_text_str, start_char, end_char in chunks:
                passages.append(
                    {
                        "passage_id": passage_id,
                        "doc_id": doc_id,
                        "title": title,
                        "text": chunk_text_str,
                        "start_char": start_char,
                        "end_char": end_char,
                        "is_chunk": True,
                    }
                )
                passage_id += 1
        else:
            # 원본 그대로 (짧은 문서 또는 chunking 미사용)
            passages.append(
                {
                    "passage_id": passage_id,
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "start_char": 0,
                    "end_char": len(text),
                    "is_chunk": False,
                }
            )
            passage_id += 1

    print(f"   ✓ Total passages: {len(passages)}")

    # Chunking 통계
    if use_chunking:
        chunked = sum(1 for p in passages if p["is_chunk"])
        non_chunked = len(passages) - chunked
        print(f"   - Chunked passages: {chunked}")
        print(f"   - Non-chunked passages: {non_chunked}")

    # 3. 모델 로드
    print(f"\n[3/5] Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # KURE-v1의 기본 max_seq_length는 8192이지만,
    # 실제 Wikipedia 토큰 분포 분석 결과:
    #   - 평균: 435 tokens, 토큰/문자 비율: 0.576
    #   - 95%ile: 990 tokens, 99%ile: 1795 tokens
    # max_seq_length=1024로 설정 시 95.5% 문서 완전 커버 (4.5%만 truncation)
    # 이로써 속도/메모리 크게 개선
    original_max_len = model.max_seq_length
    model.max_seq_length = 1024
    print(
        f"   ⚡ max_seq_length: {original_max_len} → {model.max_seq_length} (95.5% coverage)"
    )

    embed_dim = model.get_sentence_embedding_dimension()
    print(f"   ✓ Model loaded (embedding dim: {embed_dim})")

    # 4. Passage 텍스트 준비 (KURE는 query:/passage: prefix 사용)
    # KURE-v1의 경우 instruction을 사용하는 것이 권장되지만,
    # corpus embedding에는 passage prefix만 사용
    print(f"\n[4/5] Preparing passage texts (use_title={use_title})...")
    if use_title:
        corpus_texts = [f"passage: {p['title']} {p['text']}" for p in passages]
    else:
        corpus_texts = [f"passage: {p['text']}" for p in passages]

    print(f"   ✓ Example: {corpus_texts[0][:150]}...")

    # 5. Encoding (배치로 처리)
    print(f"\n[5/5] Encoding passages (batch_size={batch_size})...")
    embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
        convert_to_numpy=True,
    )

    # 6. 저장
    os.makedirs(os.path.dirname(output_emb_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_meta_path) or ".", exist_ok=True)

    # Embedding 저장
    np.save(output_emb_path, embeddings)

    # Metadata 저장 (JSONL 형식)
    with open(output_meta_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print(f"✅ Embeddings saved to {output_emb_path}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Size: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    print(f"✅ Metadata saved to {output_meta_path}")
    print(f"   Passages: {len(passages)}")
    print("=" * 80)

    return embeddings, passages


def load_passages_meta(
    meta_path: str = "./data/kure_passages_meta.jsonl",
) -> List[Dict]:
    """
    저장된 passage metadata를 로드합니다.

    Args:
        meta_path: metadata JSONL 파일 경로

    Returns:
        List of passage dictionaries
    """
    passages = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line.strip()))
    return passages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build KURE-v1 corpus embeddings")
    parser.add_argument(
        "--wiki_path",
        type=str,
        default="./data/wikipedia_documents.json",
        help="Wikipedia JSON file path",
    )
    parser.add_argument(
        "--output_emb_path",
        type=str,
        default="./data/kure_corpus_emb.npy",
        help="Output embedding .npy file path",
    )
    parser.add_argument(
        "--output_meta_path",
        type=str,
        default="./data/kure_passages_meta.jsonl",
        help="Output metadata .jsonl file path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nlpai-lab/KURE-v1",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Encoding batch size (default 16 for V100 32GB with long docs)",
    )
    parser.add_argument(
        "--no_title",
        action="store_true",
        help="Don't use title (only text)",
    )
    parser.add_argument(
        "--use_chunking",
        action="store_true",
        help="Apply length-based chunking",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum chunk size in characters",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Stride between chunks",
    )
    parser.add_argument(
        "--min_chunk_length",
        type=int,
        default=MIN_CHUNK_LENGTH,
        help="Minimum chunk length to keep",
    )
    parser.add_argument(
        "--max_length_no_chunk",
        type=int,
        default=MAX_LENGTH_NO_CHUNK,
        help="Documents shorter than this won't be chunked",
    )

    args = parser.parse_args()

    build_kure_embeddings(
        wiki_path=args.wiki_path,
        output_emb_path=args.output_emb_path,
        output_meta_path=args.output_meta_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        use_title=not args.no_title,
        use_chunking=args.use_chunking,
        chunk_size=args.chunk_size,
        stride=args.stride,
        min_chunk_length=args.min_chunk_length,
        max_length_no_chunk=args.max_length_no_chunk,
    )
