"""
Hybrid Retrieval: BM25 + Dense(KoE5) 결합.

두 가지 retrieval 방식의 장점을 결합:
- BM25: Lexical matching (정확한 단어 매칭)
- Dense: Semantic matching (의미적 유사도)

결합 방식:
1. Reciprocal Rank Fusion (RRF): 순위 기반 결합
2. Score Fusion: 점수 정규화 후 가중합
"""
from typing import List, NoReturn, Optional, Tuple, Callable

import numpy as np
from tqdm.auto import tqdm

from .koe5 import KoE5Retrieval
from .kure import KureRetrieval # KureRetrieval import 추가
from .bm25 import BM25Retrieval
from .base import BaseRetrieval


class HybridRetrieval(BaseRetrieval):
    """
    BM25 + Dense Hybrid Retrieval. Dense는 KoE5 또는 Kure 선택 가능.

    사용법:
        retriever = HybridRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
            dense_retriever_type="koe5", # 또는 "kure"
            alpha=0.5,  # BM25:Dense = 0.5:0.5
            fusion_method="rrf",  # or "score"
        )

        retriever.build()
        scores, indices = retriever.get_relevant_doc_bulk(queries, k=10)

    Args:
        alpha: BM25 가중치 (0~1). Dense 가중치는 (1-alpha)
        fusion_method: "rrf" (Reciprocal Rank Fusion) or "score" (Score Fusion)
        rrf_k: RRF 파라미터 (기본값 60)
        dense_retriever_type: 사용할 Dense Retrieval 타입 ("koe5" or "kure")
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]],
        config_path: Optional[str] = None,
        data_path: Optional[str] = None,
        context_path: Optional[str] = None,
        corpus_emb_path: Optional[str] = None,
        passages_meta_path: Optional[str] = None, # KURE를 위해 추가
        alpha: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        dense_retriever_type: str = "koe5", # 새로 추가된 인자
        **kwargs,
    ) -> NoReturn:
        super().__init__(
            config_path=config_path,
            data_path=data_path,
            context_path=context_path,
            **kwargs,
        )

        # Config에서 hybrid 설정 추출
        retrieval_config = self.config.get("retrieval", {})
        self.alpha = retrieval_config.get("hybrid_alpha", alpha)
        self.fusion_method = retrieval_config.get("fusion_method", fusion_method)
        self.rrf_k = retrieval_config.get("rrf_k", rrf_k)
        self.dense_retriever_type = retrieval_config.get("dense_retriever_type", dense_retriever_type) # config에서 가져오기

        # corpus_emb_path가 None이면 config에서 가져오기 (각 dense type에 맞게)
        if corpus_emb_path is None:
            if self.dense_retriever_type == "koe5":
                corpus_emb_path = retrieval_config.get("koe5_corpus_emb_path") # koe5 전용 config 키
            elif self.dense_retriever_type == "kure":
                corpus_emb_path = retrieval_config.get("kure_corpus_emb_path") # kure 전용 config 키
            
        if passages_meta_path is None and self.dense_retriever_type == "kure":
            passages_meta_path = retrieval_config.get("kure_passages_meta_path")

        # BM25 retriever
        self.bm25_retriever = BM25Retrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
            **kwargs,
        )

        # Dense retriever (KoE5 또는 Kure)
        if self.dense_retriever_type == "koe5":
            print("[HybridRetrieval] Using KoE5 as dense retriever")
            self.dense_retriever = KoE5Retrieval(
                data_path=data_path,
                context_path=context_path,
                corpus_emb_path=corpus_emb_path,
                **kwargs,
            )
        elif self.dense_retriever_type == "kure":
            print("[HybridRetrieval] Using Kure as dense retriever")
            self.dense_retriever = KureRetrieval(
                data_path=data_path,
                context_path=context_path,
                corpus_emb_path=corpus_emb_path,
                passages_meta_path=passages_meta_path, # Kure에만 필요
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported dense_retriever_type: {self.dense_retriever_type}")


        print(f"[HybridRetrieval] alpha={self.alpha}, method={self.fusion_method}")

    def build(self) -> NoReturn:
        """두 retriever 모두 build"""
        print("[HybridRetrieval] Building BM25...")
        self.bm25_retriever.build()

        print("[HybridRetrieval] Building Dense (KoE5)...")
        self.dense_retriever.build()

        # contexts는 동일한 것을 사용
        self.contexts = self.bm25_retriever.contexts
        self.ids = self.bm25_retriever.ids

    def get_relevant_doc_bulk(
        self,
        queries: List[str],
        k: int = 1,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Hybrid retrieval: BM25 + Dense 결합.

        Args:
            queries: 검색 쿼리 리스트
            k: 반환할 문서 개수

        Returns:
            doc_scores: 결합된 점수
            doc_indices: 결합된 문서 인덱스
        """
        # 두 retriever에서 더 많은 후보 검색 (k*2)
        retrieve_k = min(k * 3, len(self.contexts))

        # BM25 검색
        bm25_scores, bm25_indices = self.bm25_retriever.get_relevant_doc_bulk(
            queries, k=retrieve_k
        )

        # Dense 검색
        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc_bulk(
            queries, k=retrieve_k
        )

        # Fusion
        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(bm25_indices, dense_indices, k)
        else:  # score fusion
            return self._score_fusion(
                bm25_scores, bm25_indices, dense_scores, dense_indices, k
            )

    def _reciprocal_rank_fusion(
        self,
        bm25_indices: List[List[int]],
        dense_indices: List[List[int]],
        k: int,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Reciprocal Rank Fusion (RRF).

        RRF(d) = Σ 1 / (k + rank(d))

        - 순위 기반 결합으로 점수 스케일 차이 영향 없음
        - 검증된 방법 (Cormack et al. 2009)
        """
        final_scores = []
        final_indices = []

        for bm25_idx, dense_idx in zip(bm25_indices, dense_indices):
            # 각 문서의 RRF 점수 계산
            rrf_scores = {}

            # BM25 contribution
            for rank, doc_id in enumerate(bm25_idx, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.alpha / (
                    self.rrf_k + rank
                )

            # Dense contribution
            for rank, doc_id in enumerate(dense_idx, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - self.alpha) / (
                    self.rrf_k + rank
                )

            # Top-k 추출
            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
                :k
            ]

            final_indices.append([doc_id for doc_id, _ in sorted_docs])
            final_scores.append([score for _, score in sorted_docs])

        return final_scores, final_indices

    def _score_fusion(
        self,
        bm25_scores: List[List[float]],
        bm25_indices: List[List[int]],
        dense_scores: List[List[float]],
        dense_indices: List[List[int]],
        k: int,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Score Fusion: 점수 정규화 후 가중합.

        normalized_score = (score - min) / (max - min)
        final_score = alpha * bm25_norm + (1-alpha) * dense_norm
        """
        final_scores = []
        final_indices = []

        for bm25_s, bm25_i, dense_s, dense_i in zip(
            bm25_scores, bm25_indices, dense_scores, dense_indices
        ):
            # 점수 정규화
            bm25_arr = np.array(bm25_s)
            dense_arr = np.array(dense_s)

            bm25_norm = (bm25_arr - bm25_arr.min()) / (
                bm25_arr.max() - bm25_arr.min() + 1e-8
            )
            dense_norm = (dense_arr - dense_arr.min()) / (
                dense_arr.max() - dense_arr.min() + 1e-8
            )

            # 문서별 점수 집계
            doc_scores = {}
            for i, doc_id in enumerate(bm25_i):
                doc_scores[doc_id] = self.alpha * bm25_norm[i]

            for i, doc_id in enumerate(dense_i):
                if doc_id in doc_scores:
                    doc_scores[doc_id] += (1 - self.alpha) * dense_norm[i]
                else:
                    doc_scores[doc_id] = (1 - self.alpha) * dense_norm[i]

            # Top-k 추출
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[
                :k
            ]

            final_indices.append([doc_id for doc_id, _ in sorted_docs])
            final_scores.append([score for _, score in sorted_docs])

        return final_scores, final_indices
