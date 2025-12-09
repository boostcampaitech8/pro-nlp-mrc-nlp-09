"""
KoE5 기반 Dense Retrieval

Usage:
    from src.retrieval.koe5 import KoE5Retrieval

    retriever = KoE5Retrieval(
        corpus_emb_path="./data/koe5_corpus_emb.npy"
    )
    retriever.build()
    df = retriever.retrieve(dataset, topk=20)
"""

import os
from typing import List, NoReturn, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseRetrieval, timer


class KoE5Retrieval(BaseRetrieval):
    """
    KoE5 모델 기반 Dense Retrieval (SentenceTransformer).

    - BaseRetrieval 상속으로 contexts/titles 자동 로딩
    - SentenceTransformer 사용으로 간단한 인코딩
    - query: prefix / passage: prefix 자동 처리
    """

    def __init__(
        self,
        data_path: str = "./data",
        context_path: str = "wikipedia_documents.json",
        corpus_emb_path: str = "./data/koe5_corpus_emb.npy",
        model_name: str = "nlpai-lab/KoE5",
        batch_size: int = 128,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> NoReturn:
        """
        Args:
            data_path: 데이터 경로
            context_path: Wikipedia JSON 파일명
            corpus_emb_path: 사전 계산된 corpus embedding .npy 경로
            model_name: SentenceTransformer 모델명
            batch_size: Query encoding 배치 크기
            config_path: YAML config 경로 (optional)
        """
        super().__init__(
            config_path=config_path,
            data_path=data_path,
            context_path=context_path,
            **kwargs,
        )

        self.corpus_emb_path = corpus_emb_path
        self.model_name = model_name
        self.batch_size = batch_size

        # SentenceTransformer 모델 로드
        print(f"[KoE5Retrieval] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

        self.corpus_emb: Optional[np.ndarray] = None  # (num_docs, dim)

    def build(self) -> NoReturn:
        """
        Corpus embedding 로드.
        """
        if not os.path.exists(self.corpus_emb_path):
            raise FileNotFoundError(
                f"Corpus embedding not found: {self.corpus_emb_path}\n"
                f"Run: python -m src.retrieval.embed.build_koe5_corpus"
            )

        print(f"[KoE5Retrieval] Loading corpus embeddings from {self.corpus_emb_path}")
        self.corpus_emb = np.load(self.corpus_emb_path)

        # Shape 검증
        if self.corpus_emb.shape[0] != len(self.contexts):
            raise ValueError(
                f"Embedding shape mismatch!\n"
                f"  Corpus embeddings: {self.corpus_emb.shape[0]}\n"
                f"  Loaded contexts: {len(self.contexts)}\n"
                f"  → Re-run: python -m src.retrieval.embed.build_koe5_corpus"
            )

        print(f"[KoE5Retrieval] Corpus embeddings loaded: {self.corpus_emb.shape}")

    def get_relevant_doc_bulk(
        self, queries: List[str], k: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        여러 query에 대해 top-k 문서 검색.

        Args:
            queries: 검색 쿼리 리스트
            k: 반환할 문서 개수

        Returns:
            doc_scores: 각 query별 top-k 점수 [[score1, ...], ...]
            doc_indices: 각 query별 top-k 인덱스 [[idx1, ...], ...]
        """
        assert self.corpus_emb is not None, "build()를 먼저 호출해야 합니다."

        # KoE5는 query: prefix 필수
        query_texts = [f"query: {q}" for q in queries]

        with timer("encode queries (KoE5)"):
            query_emb = self.model.encode(
                query_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        # Cosine similarity (이미 normalized라 내적만 하면 됨)
        with timer("compute similarity"):
            scores = query_emb @ self.corpus_emb.T  # (B, num_docs)

        # Top-k 추출
        num_docs = scores.shape[1]
        k = min(k, num_docs)

        # argsort로 top-k 인덱스 추출
        topk_indices = np.argsort(-scores, axis=1)[:, :k]  # (B, k)
        topk_scores = np.take_along_axis(scores, topk_indices, axis=1)  # (B, k)

        doc_scores: List[List[float]] = topk_scores.tolist()
        doc_indices: List[List[int]] = topk_indices.tolist()

        return doc_scores, doc_indices
