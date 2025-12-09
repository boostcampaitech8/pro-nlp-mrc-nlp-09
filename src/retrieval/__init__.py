# src/retrieval/__init__.py

"""
Retrieval 모듈 - Sparse, Dense, BM25, Hybrid 등을 통합 관리

Usage:
    from src.retrieval import get_retriever

    # BM25 단독
    retriever = get_retriever(
        retrieval_type="bm25",
        tokenize_fn=tokenizer.tokenize,
        data_path="./data",
        context_path="wikipedia_documents.json",
    )

    # Hybrid (BM25 + KoE5)
    retriever = get_retriever(
        retrieval_type="hybrid",
        tokenize_fn=tokenizer.tokenize,
        data_path="./data",
        context_path="wikipedia_documents.json",
        corpus_emb_path="./data/koe5_corpus_emb.npy",
        alpha=0.5,  # BM25:Dense = 0.5:0.5
        fusion_method="rrf",  # or "score"
    )

    retriever.build()
    df = retriever.retrieve(dataset, topk=5)
"""

from typing import Callable, Optional

from .base import BaseRetrieval
from .sparse import SparseRetrieval
from .dense_zeroshot import DenseRetrieval
from .koe5 import KoE5Retrieval
from .bm25 import BM25Retrieval
from .hybrid import HybridRetrieval


# 나중에 시간나면 factorcy 패턴 적용
def get_retriever(
    retrieval_type: str = "sparse",
    tokenize_fn: Optional[Callable] = None,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
    # Sparse 옵션
    # (tokenize_fn만 있으면 됨)
    # Dense 옵션
    dense_model_name: Optional[str] = None,
    embedding_path: Optional[str] = None,
    corpus_emb_path: Optional[str] = None,  # KoE5/Hybrid용
    max_length: int = 256,
    batch_size: int = 64,
    # BM25 옵션
    k1: float = 1.5,
    b: float = 0.75,
    # Hybrid 옵션
    alpha: float = 0.5,
    fusion_method: str = "rrf",
    # 향후 확장: DPR 옵션
    # dpr_question_encoder: Optional[str] = None,
    # dpr_context_encoder: Optional[str] = None,
) -> BaseRetrieval:
    """
    retrieval_type에 따라 적절한 Retriever 인스턴스를 반환하는 Factory 함수.

    Args:
        retrieval_type: "sparse", "dense", "koe5", "bm25", "hybrid" 중 하나
        tokenize_fn: Sparse/BM25용 tokenizer 함수 (필수: sparse, bm25, hybrid일 때)
        data_path: 데이터 경로
        context_path: wiki corpus 파일명
        dense_model_name: Dense용 sentence encoder 모델명
        embedding_path: Dense embedding 캐시 경로 (None이면 자동 생성)
        corpus_emb_path: KoE5/Hybrid corpus embedding 경로
        max_length: Dense tokenizer max length
        batch_size: Dense embedding batch size
        k1: BM25 파라미터 (term frequency saturation)
        b: BM25 파라미터 (length normalization)
        alpha: Hybrid 가중치 (BM25 weight, Dense는 1-alpha)
        fusion_method: Hybrid 결합 방법 ("rrf" or "score")

    Returns:
        BaseRetrieval을 상속한 구체적인 retriever 인스턴스

    Raises:
        ValueError: 지원하지 않는 retrieval_type
        ValueError: 필수 파라미터 누락 시
    """
    retrieval_type = retrieval_type.lower()

    if retrieval_type == "sparse":
        if tokenize_fn is None:
            raise ValueError(
                "❌ retrieval_type='sparse' requires tokenize_fn parameter.\n"
                "Example: tokenize_fn=tokenizer.tokenize"
            )
        return SparseRetrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
        )

    elif retrieval_type == "dense":
        if dense_model_name is None:
            raise ValueError(
                "❌ retrieval_type='dense' requires dense_model_name parameter.\n"
                "Example: dense_model_name='klue/roberta-base'"
            )

        # embedding_path가 없으면 자동 생성
        if embedding_path is None:
            model_slug = dense_model_name.replace("/", "_")
            embedding_path = f"{data_path}/dense_embedding_{model_slug}.npy"
            print(f"[Factory] Auto-generated embedding_path: {embedding_path}")

        return DenseRetrieval(
            model_name_or_path=dense_model_name,
            data_path=data_path,
            context_path=context_path,
            embedding_path=embedding_path,
            max_length=max_length,
            batch_size=batch_size,
        )

    elif retrieval_type == "koe5":
        if corpus_emb_path is None:
            corpus_emb_path = f"{data_path}/koe5_corpus_emb.npy"
            print(f"[Factory] Auto-generated corpus_emb_path: {corpus_emb_path}")

        return KoE5Retrieval(
            data_path=data_path,
            context_path=context_path,
            corpus_emb_path=corpus_emb_path,
        )

    elif retrieval_type == "bm25":
        if tokenize_fn is None:
            raise ValueError(
                "❌ retrieval_type='bm25' requires tokenize_fn parameter.\n"
                "Example: tokenize_fn=tokenizer.tokenize"
            )
        return BM25Retrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
            k1=k1,
            b=b,
        )

    elif retrieval_type == "hybrid":
        if tokenize_fn is None:
            raise ValueError(
                "❌ retrieval_type='hybrid' requires tokenize_fn parameter.\n"
                "Example: tokenize_fn=tokenizer.tokenize"
            )
        if corpus_emb_path is None:
            corpus_emb_path = f"{data_path}/koe5_corpus_emb.npy"
            print(f"[Factory] Auto-generated corpus_emb_path: {corpus_emb_path}")

        return HybridRetrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
            corpus_emb_path=corpus_emb_path,
            alpha=alpha,
            fusion_method=fusion_method,
        )

    elif retrieval_type == "dpr":
        raise NotImplementedError(
            "❌ DPR is not implemented yet.\nComing soon in the next iteration!"
        )

    else:
        raise ValueError(
            f"❌ Unsupported retrieval_type: '{retrieval_type}'.\n"
            f"Supported types: 'sparse', 'dense', 'koe5', 'bm25', 'hybrid'"
        )


# Public API
__all__ = [
    "BaseRetrieval",
    "SparseRetrieval",
    "DenseRetrieval",
    "KoE5Retrieval",
    "BM25Retrieval",
    "HybridRetrieval",
    "get_retriever",
]
