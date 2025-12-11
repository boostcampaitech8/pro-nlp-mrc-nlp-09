# src/retrieval/__init__.py

"""
Retrieval 모듈 - Sparse, Dense, BM25, Hybrid, KURE 등을 통합 관리

Usage:
    from src.retrieval import get_retriever
    from src.retrieval.paths import get_path  # 경로는 paths.py에서 중앙 관리

    # BM25 단독
    retriever = get_retriever(
        retrieval_type="bm25",
        tokenize_fn=tokenizer.tokenize,
        data_path="./data",
        context_path="wikipedia_documents.json",
    )

    # KoE5 (경로는 자동으로 paths.py에서 가져옴)
    retriever = get_retriever(
        retrieval_type="koe5",
        data_path="./data",
    )

    # WeightedHybrid (BM25 + KURE) - per-query 정규화 + 가중합
    retriever = get_retriever(
        retrieval_type="weighted_hybrid",
        tokenize_fn=tokenizer.tokenize,
        data_path="./data",
        alpha=0.35,  # BM25 가중치 (base.yaml과 일치)
    )

    retriever.build()
    df = retriever.retrieve(dataset, topk=5)
"""

from typing import Callable, Optional

from .base import BaseRetrieval
from .sparse import SparseRetrieval
from .dense_zeroshot import DenseRetrieval
from .koe5 import KoE5Retrieval
from .kure import KureRetrieval
from .bm25 import BM25Retrieval
from .hybrid import HybridRetrieval
from .weighted_hybrid import WeightedHybridRetrieval
from .paths import get_path, PATHS


# 나중에 시간나면 factorcy 패턴 적용
def get_retriever(
    retrieval_type: str = "sparse",
    tokenize_fn: Optional[Callable] = None,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
    config_path: Optional[str] = None,
    # Sparse 옵션
    # (tokenize_fn만 있으면 됨)
    # Dense 옵션
    dense_model_name: Optional[str] = None,
    embedding_path: Optional[str] = None,
    corpus_emb_path: Optional[str] = None,  # KoE5/Hybrid용
    passages_meta_path: Optional[str] = None,  # KURE/WeightedHybrid용
    max_length: int = 256,
    batch_size: int = 64,
    # BM25 옵션
    k1: float = 1.5,
    b: float = 0.75,
    # Hybrid 옵션
    alpha: float = 0.35,  # BM25 가중치 (base.yaml과 일치)
    fusion_method: str = "rrf",
    # 향후 확장: DPR 옵션
    # dpr_question_encoder: Optional[str] = None,
    # dpr_context_encoder: Optional[str] = None,
    **kwargs,
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
            config_path=config_path,
            **kwargs,
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
            config_path=config_path,
            **kwargs,
        )

    elif retrieval_type == "koe5":
        # corpus_emb_path가 None이면 KoE5Retrieval 내부에서 paths.py 기본값 사용
        return KoE5Retrieval(
            data_path=data_path,
            context_path=context_path,
            corpus_emb_path=corpus_emb_path,  # None이면 기본 경로 사용
            config_path=config_path,
            **kwargs,
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
            config_path=config_path,
            **kwargs,
        )

    elif retrieval_type == "hybrid":
        if tokenize_fn is None:
            raise ValueError(
                "❌ retrieval_type='hybrid' requires tokenize_fn parameter.\n"
                "Example: tokenize_fn=tokenizer.tokenize"
            )
        # corpus_emb_path가 None이면 paths.py 기본값 사용
        if corpus_emb_path is None:
            corpus_emb_path = get_path("koe5_corpus_emb")
            print(f"[Factory] Using default corpus_emb_path: {corpus_emb_path}")

        return HybridRetrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
            corpus_emb_path=corpus_emb_path,
            alpha=alpha,
            fusion_method=fusion_method,
            config_path=config_path,
            **kwargs,
        )

    elif retrieval_type == "kure":
        # KURE-v1 단독 Dense Retrieval
        # corpus_emb_path가 None이면 KureRetrieval 내부에서 paths.py 기본값 사용
        return KureRetrieval(
            data_path=data_path,
            context_path=context_path,
            corpus_emb_path=corpus_emb_path,  # None이면 기본 경로 사용
            passages_meta_path=passages_meta_path,  # None이면 기본 경로 사용
            config_path=config_path,
            **kwargs,
        )

    elif retrieval_type == "weighted_hybrid":
        # BM25 + KURE Weighted Hybrid (per-query 정규화 + 가중합)
        if tokenize_fn is None:
            raise ValueError(
                "❌ retrieval_type='weighted_hybrid' requires tokenize_fn parameter.\n"
                "Example: tokenize_fn=tokenizer.tokenize"
            )
        # corpus_emb_path가 None이면 paths.py 기본값 사용
        if corpus_emb_path is None:
            corpus_emb_path = get_path("kure_corpus_emb")
            print(f"[Factory] Using default corpus_emb_path: {corpus_emb_path}")
        if passages_meta_path is None:
            passages_meta_path = get_path("kure_passages_meta")
            print(f"[Factory] Using default passages_meta_path: {passages_meta_path}")

        # alpha 직접 사용 (kwargs에 weighted_alpha가 있으면 우선 사용)
        effective_alpha = kwargs.pop("weighted_alpha", alpha)
        print(f"[Factory] WeightedHybridRetrieval alpha={effective_alpha}")

        return WeightedHybridRetrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
            corpus_emb_path=corpus_emb_path,
            passages_meta_path=passages_meta_path,
            alpha=effective_alpha,
            config_path=config_path,
            **kwargs,
        )

    elif retrieval_type == "dpr":
        raise NotImplementedError(
            "❌ DPR is not implemented yet.\nComing soon in the next iteration!"
        )

    else:
        raise ValueError(
            f"❌ Unsupported retrieval_type: '{retrieval_type}'.\n"
            f"Supported types: 'sparse', 'dense', 'koe5', 'kure', 'bm25', 'hybrid', 'weighted_hybrid'"
        )


# Public API
__all__ = [
    "BaseRetrieval",
    "SparseRetrieval",
    "DenseRetrieval",
    "KoE5Retrieval",
    "KureRetrieval",
    "BM25Retrieval",
    "HybridRetrieval",
    "WeightedHybridRetrieval",
    "get_retriever",
]
