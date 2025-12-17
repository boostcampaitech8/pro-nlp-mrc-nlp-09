"""
BM25 기반 Sparse Retrieval (Fast Implementation).

TF-IDF 대신 BM25 알고리즘을 사용한 retrieval.
bm25s 라이브러리 사용 (NumPy 기반 vectorized, rank-bm25보다 10-100배 빠름).

설치:
    pip install bm25s

Reference:
    https://github.com/xhluca/bm25s
"""

import os
import pickle
import random
from typing import Callable, List, NoReturn, Optional, Tuple

import numpy as np
from datasets import Dataset
from tqdm.auto import tqdm

try:
    import bm25s
except ImportError:
    raise ImportError(
        "bm25s is required for fast BM25 retrieval.\nInstall it with: pip install bm25s"
    )

from .base import BaseRetrieval, timer

seed = 2024
random.seed(seed)
np.random.seed(seed)


class BM25Retrieval(BaseRetrieval):
    """
    BM25 기반 Sparse Retrieval.

    사용법:
        # 1) YAML 기반
        retriever = BM25Retrieval(
            tokenize_fn=tokenizer.tokenize,
            config_path="configs/exp/xxx.yaml",
        )

        # 2) 직접 파라미터
        retriever = BM25Retrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
        )

    BM25 vs TF-IDF:
        - BM25: 문서 길이 정규화, 포화(saturation) 효과 고려
        - TF-IDF: 단순 term frequency × inverse document frequency
        - BM25가 일반적으로 더 좋은 성능 (특히 긴 문서)
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]],
        config_path: Optional[str] = None,
        data_path: Optional[str] = None,
        context_path: Optional[str] = None,
        k1: float = 1.5,  # BM25 파라미터: term frequency saturation
        b: float = 0.75,  # BM25 파라미터: length normalization
        **kwargs,
    ) -> NoReturn:
        super().__init__(
            config_path=config_path,
            data_path=data_path,
            context_path=context_path,
            **kwargs,
        )
        self.tokenize_fn = tokenize_fn

        # BM25 파라미터
        retrieval_config = self.config.get("retrieval", {})
        self.k1 = retrieval_config.get("bm25_k1", k1)
        self.b = retrieval_config.get("bm25_b", b)

        self.bm25 = None  # build()에서 생성
        self.tokenized_corpus = None

    def build(self) -> NoReturn:
        """
        BM25 인덱스 생성/로딩 (bm25s 사용).

        - 기존 인덱스가 있으면 로드
        - 없으면 corpus tokenize 후 bm25s 생성
        """
        index_dir = self.data_path

        # bm25s는 디렉토리에 여러 파일로 저장
        index_file = os.path.join(index_dir, "bm25_index.pkl")

        if os.path.isfile(index_file):
            print(f"[BM25Retrieval] Loading BM25 index from {index_dir}/")
            try:
                self.bm25 = bm25s.BM25.load(index_dir, load_corpus=True)
                self.tokenized_corpus = self.bm25.corpus

                # 크기 검증
                if len(self.tokenized_corpus) != len(self.contexts):
                    print(f"⚠️  WARNING: BM25 index 크기 불일치!")
                    print(f"   - Loaded corpus size: {len(self.tokenized_corpus)}")
                    print(f"   - Current contexts: {len(self.contexts)}")
                    print(f"   → BM25 index 재생성...")
                    self._build_bm25_index()
                    self._save_bm25_index(index_dir)
                else:
                    print(
                        f"[BM25Retrieval] Index loaded: {len(self.tokenized_corpus)} documents"
                    )
            except Exception as e:
                print(f"⚠️  Loading failed: {e}")
                print(f"   → BM25 index 재생성...")
                self._build_bm25_index()
                self._save_bm25_index(index_dir)
        else:
            print(f"[BM25Retrieval] Building BM25 index...")
            self._build_bm25_index()
            self._save_bm25_index(index_dir)
            print(f"[BM25Retrieval] Index saved to {index_dir}/")

    def _build_bm25_index(self) -> NoReturn:
        """Corpus tokenize 후 BM25 인덱스 생성 (bm25s 사용)"""
        print(f"[BM25Retrieval] Tokenizing {len(self.contexts)} documents...")
        self.tokenized_corpus = [
            self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing")
        ]

        print(f"[BM25Retrieval] Creating BM25 index (k1={self.k1}, b={self.b})...")
        # bm25s는 stemmer 없이 사용 가능 (이미 tokenize됨)
        self.bm25 = bm25s.BM25()
        self.bm25.index(self.tokenized_corpus, show_progress=False)

        # 파라미터 설정 (bm25s는 internal params 사용)
        # k1, b는 생성 시 설정 불가하지만 기본값이 동일 (1.5, 0.75)

    def _save_bm25_index(self, pickle_path: str) -> NoReturn:
        """BM25 인덱스 저장 (bm25s.save_index 사용)"""
        # bm25s는 자체 save/load 메서드 제공
        save_dir = os.path.dirname(pickle_path)
        self.bm25.save(save_dir, corpus=self.tokenized_corpus)

    def get_relevant_doc_bulk(
        self,
        queries: List[str],
        k: int = 1,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        여러 query에 대해 BM25 score 기반 top-k 문서 검색.

        bm25s는 vectorized 연산으로 TF-IDF만큼 빠름.

        Args:
            queries: 검색 쿼리 리스트
            k: 반환할 문서 개수

        Returns:
            doc_scores: 각 query별 top-k 문서 점수 [[score1, score2, ...], ...]
            doc_indices: 각 query별 top-k 문서 인덱스 [[idx1, idx2, ...], ...]
        """
        # Tokenize queries
        tokenized_queries = [self.tokenize_fn(q) for q in queries]

        # BM25 검색 (vectorized, 매우 빠름)
        results = self.bm25.retrieve(
            tokenized_queries,
            k=k,
            show_progress=False,
            n_threads=1,  # 단일 스레드 (멀티 스레드는 오버헤드 있을 수 있음)
        )

        # 결과 변환 (bm25s는 (indices, scores) 튜플 반환)
        doc_indices = results.documents.tolist()
        doc_scores = results.scores.tolist()

        return doc_scores, doc_indices
