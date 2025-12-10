"""KURE-v1 기반 Dense Retrieval

KoE5Retrieval과 유사하지만 KURE-v1 모델을 사용하며,
passage 단위의 메타데이터를 지원합니다.

Usage:
    from src.retrieval.kure import KureRetrieval
    from src.retrieval.paths import get_path

    retriever = KureRetrieval(
        corpus_emb_path=get_path("kure_corpus_emb"),
        passages_meta_path=get_path("kure_passages_meta"),
    )
    retriever.build()
    df = retriever.retrieve(dataset, topk=20)
"""

import json
import os
from typing import Dict, List, NoReturn, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseRetrieval, timer
from .paths import get_path, exists

# 기본 경로 상수 (paths.py에서 관리)
DEFAULT_CORPUS_EMB_PATH = get_path("kure_corpus_emb")
DEFAULT_PASSAGES_META_PATH = get_path("kure_passages_meta")


class KureRetrieval(BaseRetrieval):
    """
    KURE-v1 모델 기반 Dense Retrieval (SentenceTransformer).

    - BaseRetrieval 상속으로 contexts/ids 자동 로딩
    - Passage 메타데이터 지원 (chunking 시 필요)
    - query: prefix / passage: prefix 자동 처리
    """

    def __init__(
        self,
        data_path: str = "./data",
        context_path: str = "wikipedia_documents.json",
        corpus_emb_path: Optional[str] = None,  # None이면 기본 경로 사용
        passages_meta_path: Optional[str] = None,  # None이면 기본 경로 사용
        model_name: str = "nlpai-lab/KURE-v1",
        batch_size: int = 128,
        config_path: Optional[str] = None,
        use_passages_mode: bool = False,  # True: passages_meta 기반, False: wiki contexts 기반
        **kwargs,
    ) -> NoReturn:
        """
        Args:
            data_path: 데이터 경로
            context_path: Wikipedia JSON 파일명
            corpus_emb_path: 사전 계산된 corpus embedding .npy 경로
            passages_meta_path: passage 메타데이터 JSONL 경로 (chunking 사용 시)
            model_name: SentenceTransformer 모델명
            batch_size: Query encoding 배치 크기
            config_path: YAML config 경로 (optional)
            use_passages_mode: True면 passages_meta 기반, False면 wiki contexts 기반
        """
        super().__init__(
            config_path=config_path,
            data_path=data_path,
            context_path=context_path,
            **kwargs,
        )

        # 경로 설정 (None이면 기본 경로 사용)
        self.corpus_emb_path = corpus_emb_path or DEFAULT_CORPUS_EMB_PATH
        self.passages_meta_path = passages_meta_path or DEFAULT_PASSAGES_META_PATH
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_passages_mode = use_passages_mode

        # SentenceTransformer 모델 로드
        print(f"[KureRetrieval] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

        self.corpus_emb: Optional[np.ndarray] = None  # (num_passages, dim)
        self.passages_meta: Optional[List[Dict]] = None  # passage 메타데이터

        # passage_id -> context_text 매핑 (캐시용)
        self._passage_texts: Optional[List[str]] = None

    def build(self) -> NoReturn:
        """
        Corpus embedding 및 passage 메타데이터 로드.
        """
        if not os.path.exists(self.corpus_emb_path):
            raise FileNotFoundError(
                f"Corpus embedding not found: {self.corpus_emb_path}\n"
                f"Run: python -m src.retrieval.embed.build_kure_corpus"
            )

        print(f"[KureRetrieval] Loading corpus embeddings from {self.corpus_emb_path}")
        self.corpus_emb = np.load(self.corpus_emb_path)
        print(f"[KureRetrieval] Corpus embeddings loaded: {self.corpus_emb.shape}")

        # Passages meta 로드 (존재하는 경우)
        if self.passages_meta_path and os.path.exists(self.passages_meta_path):
            print(
                f"[KureRetrieval] Loading passages meta from {self.passages_meta_path}"
            )
            self.passages_meta = []
            with open(self.passages_meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.passages_meta.append(json.loads(line.strip()))
            print(
                f"[KureRetrieval] Passages meta loaded: {len(self.passages_meta)} passages"
            )

            # Shape 검증
            if self.corpus_emb.shape[0] != len(self.passages_meta):
                raise ValueError(
                    f"Embedding shape mismatch!\n"
                    f"  Corpus embeddings: {self.corpus_emb.shape[0]}\n"
                    f"  Passages meta: {len(self.passages_meta)}\n"
                    f"  → Re-run: python -m src.retrieval.embed.build_kure_corpus"
                )

            # passage_id -> text 캐시 구성
            self._passage_texts = [p["text"] for p in self.passages_meta]

            # use_passages_mode 자동 설정
            self.use_passages_mode = True
        else:
            # passages_meta 없으면 기존 wiki contexts 사용
            print(
                f"[KureRetrieval] No passages meta found, using wiki contexts directly"
            )

            # Shape 검증 (wiki contexts 기준)
            if self.corpus_emb.shape[0] != len(self.contexts):
                raise ValueError(
                    f"Embedding shape mismatch!\n"
                    f"  Corpus embeddings: {self.corpus_emb.shape[0]}\n"
                    f"  Loaded contexts: {len(self.contexts)}\n"
                    f"  → Re-run: python -m src.retrieval.embed.build_kure_corpus"
                )

            self._passage_texts = self.contexts
            self.use_passages_mode = False

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
                        (use_passages_mode=True: passage_id, False: context_idx)
        """
        assert self.corpus_emb is not None, "build()를 먼저 호출해야 합니다."

        # KURE-v1은 query: prefix 사용
        query_texts = [f"query: {q}" for q in queries]

        with timer("encode queries (KURE)"):
            query_emb = self.model.encode(
                query_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        # Cosine similarity (이미 normalized라 내적만 하면 됨)
        with timer("compute similarity"):
            scores = query_emb @ self.corpus_emb.T  # (B, num_passages)

        # Top-k 추출
        num_passages = scores.shape[1]
        k = min(k, num_passages)

        # argsort로 top-k 인덱스 추출
        topk_indices = np.argsort(-scores, axis=1)[:, :k]  # (B, k)
        topk_scores = np.take_along_axis(scores, topk_indices, axis=1)  # (B, k)

        doc_scores: List[List[float]] = topk_scores.tolist()
        doc_indices: List[List[int]] = topk_indices.tolist()

        return doc_scores, doc_indices

    def get_dense_scores_all(self, queries: List[str]) -> np.ndarray:
        """
        모든 query에 대해 전체 corpus와의 dense score를 반환.

        Hybrid retrieval에서 사용.

        Args:
            queries: 검색 쿼리 리스트

        Returns:
            scores: (num_queries, num_passages) shape의 score 배열
        """
        assert self.corpus_emb is not None, "build()를 먼저 호출해야 합니다."

        query_texts = [f"query: {q}" for q in queries]

        query_emb = self.model.encode(
            query_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(queries) > 100,
        )

        # Cosine similarity
        scores = query_emb @ self.corpus_emb.T  # (B, num_passages)
        return scores

    def get_passage_text(self, passage_id: int) -> str:
        """
        passage_id에 해당하는 텍스트를 반환.

        Args:
            passage_id: passage 인덱스

        Returns:
            passage text
        """
        if self._passage_texts is None:
            raise ValueError("build()를 먼저 호출해야 합니다.")
        return self._passage_texts[passage_id]

    def get_passage_meta(self, passage_id: int) -> Optional[Dict]:
        """
        passage_id에 해당하는 메타데이터를 반환.

        Args:
            passage_id: passage 인덱스

        Returns:
            passage metadata dict or None
        """
        if self.passages_meta is None:
            return None
        return self.passages_meta[passage_id]

    def get_doc_id_from_passage(self, passage_id: int) -> int:
        """
        passage_id에 해당하는 원본 doc_id를 반환.

        Args:
            passage_id: passage 인덱스

        Returns:
            원본 document ID
        """
        if self.passages_meta is not None:
            return self.passages_meta[passage_id]["doc_id"]
        else:
            # passages_meta 없으면 self.ids 사용
            return self.ids[passage_id]
