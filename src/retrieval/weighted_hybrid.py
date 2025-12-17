"""
Weighted Hybrid Retrieval: BM25 + KURE 가중합 (per-query 정규화)

기존 HybridRetrieval(RRF)과 달리:
- BM25와 Dense(KURE) 점수를 per-query로 min-max 정규화
- 가중합으로 최종 점수 계산
- α (BM25 가중치)를 조절하여 두 retriever의 비율 제어

Usage:
    from src.retrieval.weighted_hybrid import WeightedHybridRetrieval

    retriever = WeightedHybridRetrieval(
        tokenize_fn=tokenizer.tokenize,
        corpus_emb_path="./data/kure_corpus_emb.npy",
        passages_meta_path="./data/kure_passages_meta.jsonl",
        alpha=0.7,  # BM25 가중치
    )
    retriever.build()
    df = retriever.retrieve(dataset, topk=20)
"""

import json
from typing import Callable, Dict, List, NoReturn, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from .base import BaseRetrieval, timer
from .bm25 import BM25Retrieval
from .kure import KureRetrieval


class WeightedHybridRetrieval(BaseRetrieval):
    """
    BM25 + KURE Weighted Hybrid Retrieval (per-query 정규화 + 가중합).

    특징:
        - Per-query min-max 정규화로 스케일 불일치 해결
        - α 가중치로 BM25 vs Dense 비율 조절
        - Tie-breaking: BM25 score → doc_id/passage_id 순

    사용법:
        retriever = WeightedHybridRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
            corpus_emb_path="./data/kure_corpus_emb.npy",
            passages_meta_path="./data/kure_passages_meta.jsonl",
            alpha=0.7,  # BM25:KURE = 0.7:0.3
        )
        retriever.build()
        scores, indices = retriever.get_relevant_doc_bulk(queries, k=20)

    Args:
        alpha: BM25 가중치 (0~1). KURE 가중치는 (1-alpha)
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]],
        config_path: Optional[str] = None,
        data_path: Optional[str] = None,
        context_path: Optional[str] = None,
        corpus_emb_path: Optional[str] = None,
        passages_meta_path: Optional[str] = None,
        alpha: float = 0.7,
        eps: float = 1e-9,  # 정규화 시 0으로 나누기 방지
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
        self.alpha = retrieval_config.get("weighted_hybrid_alpha", alpha)
        self.eps = eps

        # 기본 경로 설정
        if corpus_emb_path is None:
            corpus_emb_path = retrieval_config.get(
                "kure_corpus_emb_path", f"{self.data_path}/kure_corpus_emb.npy"
            )
        if passages_meta_path is None:
            passages_meta_path = retrieval_config.get(
                "kure_passages_meta_path", f"{self.data_path}/kure_passages_meta.jsonl"
            )

        self.corpus_emb_path = corpus_emb_path
        self.passages_meta_path = passages_meta_path

        # BM25 retriever (wiki contexts 기준)
        self.bm25_retriever = BM25Retrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
            **kwargs,
        )

        # KURE retriever
        self.kure_retriever = KureRetrieval(
            data_path=data_path,
            context_path=context_path,
            corpus_emb_path=corpus_emb_path,
            passages_meta_path=passages_meta_path,
            **kwargs,
        )

        print(
            f"[WeightedHybridRetrieval] alpha={self.alpha} (BM25:{self.alpha:.1f}, KURE:{1 - self.alpha:.1f})"
        )

    def build(self) -> NoReturn:
        """두 retriever 모두 build"""
        print("[WeightedHybridRetrieval] Building BM25...")
        self.bm25_retriever.build()

        print("[WeightedHybridRetrieval] Building KURE...")
        self.kure_retriever.build()

        # contexts, ids는 BM25 기준 (wiki contexts)
        self.contexts = self.bm25_retriever.contexts
        self.ids = self.bm25_retriever.ids
        self.titles = self.bm25_retriever.titles

        # KURE가 passages_meta 모드인지 확인
        self.use_passages_mode = self.kure_retriever.use_passages_mode
        if self.use_passages_mode:
            print(f"[WeightedHybridRetrieval] Using passages mode (chunked corpus)")
            print(f"   - BM25 contexts: {len(self.contexts)}")
            print(f"   - KURE passages: {len(self.kure_retriever.passages_meta)}")
        else:
            print(f"[WeightedHybridRetrieval] Using document mode (no chunking)")
            print(f"   - Total documents: {len(self.contexts)}")

    def get_relevant_doc_bulk(
        self,
        queries: List[str],
        k: int = 1,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Weighted Hybrid retrieval: BM25 + KURE 점수 정규화 후 가중합.

        Args:
            queries: 검색 쿼리 리스트
            k: 반환할 문서 개수

        Returns:
            doc_scores: 결합된 점수
            doc_indices: 결합된 문서/패시지 인덱스
        """
        if self.use_passages_mode:
            return self._retrieve_with_passages(queries, k)
        else:
            return self._retrieve_with_documents(queries, k)

    def _retrieve_with_documents(
        self,
        queries: List[str],
        k: int,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        문서 단위 retrieval (chunking 없는 경우).

        BM25와 KURE가 같은 인덱스 공간을 사용.
        """
        num_docs = len(self.contexts)

        # BM25: 전체 문서에 대한 점수 계산
        with timer("BM25 retrieval"):
            bm25_scores_list, bm25_indices_list = (
                self.bm25_retriever.get_relevant_doc_bulk(
                    queries,
                    k=num_docs,  # 전체 문서에 대해 점수 계산
                )
            )

        # KURE: 전체 문서에 대한 점수 계산
        with timer("KURE dense scores"):
            dense_scores = self.kure_retriever.get_dense_scores_all(
                queries
            )  # (B, num_docs)

        final_scores = []
        final_indices = []

        for q_idx in range(len(queries)):
            # BM25 점수를 전체 문서 배열로 변환
            bm25_full = np.zeros(num_docs)
            for idx, score in zip(bm25_indices_list[q_idx], bm25_scores_list[q_idx]):
                bm25_full[idx] = score

            dense_full = dense_scores[q_idx]  # (num_docs,)

            # Per-query min-max 정규화
            bm25_n = self._min_max_normalize(bm25_full)
            dense_n = self._min_max_normalize(dense_full)

            # 가중합
            hybrid = self.alpha * bm25_n + (1 - self.alpha) * dense_n

            # Top-k 추출 (tie-breaking: hybrid → bm25 → doc_idx)
            # argsort로 정렬 후 안정 정렬을 위해 tie-breaking 처리
            sorted_indices = self._stable_argsort(hybrid, bm25_full, k)

            final_indices.append(sorted_indices.tolist())
            final_scores.append([hybrid[i] for i in sorted_indices])

        return final_scores, final_indices

    def _retrieve_with_passages(
        self,
        queries: List[str],
        k: int,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Passage 단위 retrieval (chunking 적용된 경우).

        BM25는 doc 단위, KURE는 passage 단위.
        doc_id를 통해 매핑.
        """
        num_passages = len(self.kure_retriever.passages_meta)
        num_docs = len(self.contexts)

        # doc_id -> context_idx 매핑 구성
        doc_id_to_ctx_idx = {doc_id: idx for idx, doc_id in enumerate(self.ids)}

        # passage_id -> doc_id 매핑
        passage_to_doc = [p["doc_id"] for p in self.kure_retriever.passages_meta]

        # BM25: 문서 단위 점수
        with timer("BM25 retrieval"):
            bm25_scores_list, bm25_indices_list = (
                self.bm25_retriever.get_relevant_doc_bulk(queries, k=num_docs)
            )

        # KURE: passage 단위 점수
        with timer("KURE dense scores"):
            dense_scores = self.kure_retriever.get_dense_scores_all(
                queries
            )  # (B, num_passages)

        final_scores = []
        final_indices = []

        for q_idx in range(len(queries)):
            # BM25 점수를 doc_id 기준 dict로 변환
            bm25_by_doc = {}
            for idx, score in zip(bm25_indices_list[q_idx], bm25_scores_list[q_idx]):
                doc_id = self.ids[idx]
                bm25_by_doc[doc_id] = score

            # 각 passage에 대해 해당 doc의 BM25 점수 할당
            bm25_full = np.array(
                [
                    bm25_by_doc.get(passage_to_doc[pid], 0.0)
                    for pid in range(num_passages)
                ]
            )

            dense_full = dense_scores[q_idx]  # (num_passages,)

            # Per-query min-max 정규화
            bm25_n = self._min_max_normalize(bm25_full)
            dense_n = self._min_max_normalize(dense_full)

            # 가중합
            hybrid = self.alpha * bm25_n + (1 - self.alpha) * dense_n

            # Top-k 추출
            sorted_indices = self._stable_argsort(hybrid, bm25_full, k)

            # Passage 인덱스를 문서 인덱스로 변환
            doc_indices = []
            for passage_idx in sorted_indices:
                doc_id = passage_to_doc[passage_idx]
                ctx_idx = doc_id_to_ctx_idx.get(doc_id)
                if ctx_idx is not None:
                    doc_indices.append(ctx_idx)
                else:
                    # 매핑이 없는 경우 경고 (이론적으로는 발생하지 않아야 함)
                    print(f"Warning: doc_id {doc_id} not found in contexts")
            
            final_indices.append(doc_indices)
            final_scores.append([hybrid[i] for i in sorted_indices])

        return final_scores, final_indices

    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Per-query min-max 정규화.

        Args:
            scores: (num_docs,) 또는 (num_passages,)

        Returns:
            정규화된 점수 (0~1 범위)
        """
        min_val = scores.min()
        max_val = scores.max()
        return (scores - min_val) / (max_val - min_val + self.eps)

    def _stable_argsort(
        self, primary: np.ndarray, secondary: np.ndarray, k: int
    ) -> np.ndarray:
        """
        Stable sorting with tie-breaking.

        정렬 순서:
            1. primary (내림차순)
            2. secondary (내림차순) - tie-break
            3. index (오름차순) - deterministic tie-break

        Args:
            primary: 주 정렬 기준 (hybrid score)
            secondary: 보조 정렬 기준 (BM25 score)
            k: 반환할 개수

        Returns:
            정렬된 인덱스 배열 (상위 k개)
        """
        n = len(primary)
        indices = np.arange(n)

        # numpy의 lexsort는 마지막 키가 primary
        # 내림차순을 위해 음수 사용
        sorted_idx = np.lexsort((indices, -secondary, -primary))

        return sorted_idx[:k]

    def get_scores_for_cache(
        self,
        queries: List[str],
        k: int = 50,
    ) -> List[List[Dict]]:
        """
        캐시 생성용: 각 query에 대해 top-k 후보의 raw score들을 반환.

        Args:
            queries: 검색 쿼리 리스트
            k: 반환할 후보 개수

        Returns:
            List of list of dicts, 각 dict는:
            {
                "passage_id": int (또는 doc_idx),
                "doc_id": int,
                "score_dense": float,
                "score_bm25": float,
            }
        """
        if self.use_passages_mode:
            return self._get_scores_with_passages(queries, k)
        else:
            return self._get_scores_with_documents(queries, k)

    def _get_scores_with_documents(
        self,
        queries: List[str],
        k: int,
    ) -> List[List[Dict]]:
        """문서 단위 캐시 생성"""
        num_docs = len(self.contexts)

        # BM25 점수
        bm25_scores_list, bm25_indices_list = self.bm25_retriever.get_relevant_doc_bulk(
            queries, k=num_docs
        )

        # KURE 점수
        dense_scores = self.kure_retriever.get_dense_scores_all(queries)

        results = []
        for q_idx in range(len(queries)):
            # BM25 점수 full array
            bm25_full = np.zeros(num_docs)
            for idx, score in zip(bm25_indices_list[q_idx], bm25_scores_list[q_idx]):
                bm25_full[idx] = score

            dense_full = dense_scores[q_idx]

            # 정규화 후 hybrid로 top-k 선정
            bm25_n = self._min_max_normalize(bm25_full)
            dense_n = self._min_max_normalize(dense_full)
            hybrid = self.alpha * bm25_n + (1 - self.alpha) * dense_n

            sorted_indices = self._stable_argsort(hybrid, bm25_full, k)

            candidates = []
            for idx in sorted_indices:
                candidates.append(
                    {
                        "passage_id": int(idx),  # 문서 모드에서는 doc_idx
                        "doc_id": int(self.ids[idx]),
                        "score_dense": float(dense_full[idx]),
                        "score_bm25": float(bm25_full[idx]),
                    }
                )
            results.append(candidates)

        return results

    def _get_scores_with_passages(
        self,
        queries: List[str],
        k: int,
    ) -> List[List[Dict]]:
        """Passage 단위 캐시 생성"""
        num_passages = len(self.kure_retriever.passages_meta)
        num_docs = len(self.contexts)

        # passage_id -> doc_id
        passage_to_doc = [p["doc_id"] for p in self.kure_retriever.passages_meta]

        # BM25 점수 (문서 단위)
        bm25_scores_list, bm25_indices_list = self.bm25_retriever.get_relevant_doc_bulk(
            queries, k=num_docs
        )

        # KURE 점수 (passage 단위)
        dense_scores = self.kure_retriever.get_dense_scores_all(queries)

        results = []
        for q_idx in range(len(queries)):
            # BM25 by doc_id
            bm25_by_doc = {}
            for idx, score in zip(bm25_indices_list[q_idx], bm25_scores_list[q_idx]):
                doc_id = self.ids[idx]
                bm25_by_doc[doc_id] = score

            # Passage별 BM25 점수
            bm25_full = np.array(
                [
                    bm25_by_doc.get(passage_to_doc[pid], 0.0)
                    for pid in range(num_passages)
                ]
            )

            dense_full = dense_scores[q_idx]

            # 정규화 후 hybrid로 top-k 선정
            bm25_n = self._min_max_normalize(bm25_full)
            dense_n = self._min_max_normalize(dense_full)
            hybrid = self.alpha * bm25_n + (1 - self.alpha) * dense_n

            sorted_indices = self._stable_argsort(hybrid, bm25_full, k)

            candidates = []
            for idx in sorted_indices:
                candidates.append(
                    {
                        "passage_id": int(idx),
                        "doc_id": int(passage_to_doc[idx]),
                        "score_dense": float(dense_full[idx]),
                        "score_bm25": float(bm25_full[idx]),
                    }
                )
            results.append(candidates)

        return results

    def get_passage_text(self, passage_id: int) -> str:
        """
        passage_id에 해당하는 텍스트를 반환.

        Args:
            passage_id: passage/document 인덱스

        Returns:
            텍스트
        """
        if self.use_passages_mode:
            return self.kure_retriever.get_passage_text(passage_id)
        else:
            return self.contexts[passage_id]

    def get_title(self, passage_id: int) -> str:
        """
        passage_id에 해당하는 title을 반환.

        Args:
            passage_id: passage/document 인덱스

        Returns:
            title
        """
        if self.use_passages_mode:
            return self.kure_retriever.passages_meta[passage_id]["title"]
        else:
            return self.titles[passage_id]
