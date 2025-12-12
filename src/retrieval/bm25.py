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
        delta: float = 0.5, # BM25Plus parameter
        impl: str = "bm25s", # "bm25s" or "rank_bm25"
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
        self.delta = retrieval_config.get("bm25_delta", delta)
        self.impl = retrieval_config.get("bm25_impl", impl)

        self.bm25 = None  # build()에서 생성
        self.tokenized_corpus = None

    def build(self) -> NoReturn:
        """
        BM25 인덱스 생성/로딩.
        """
        index_dir = self.data_path
        
        # 인덱스 파일명에 구현체와 파라미터 정보 포함 (충돌 방지)
        impl_suffix = f"_{self.impl}"
        if self.impl == "rank_bm25":
            impl_suffix += f"_plus_k{self.k1}_b{self.b}_d{self.delta}"
        else:
            impl_suffix += f"_k{self.k1}_b{self.b}" # bm25s defaults usually
            
        index_name = f"bm25_index{impl_suffix}.pkl"
        index_file = os.path.join(index_dir, index_name)

        if os.path.isfile(index_file):
            print(f"[BM25Retrieval] Loading BM25 index from {index_file}")
            try:
                with open(index_file, "rb") as f:
                    self.bm25 = pickle.load(f)
                
                # bm25s는 별도 corpus 로딩 필요할 수 있음 (save 방식에 따라 다름)
                # 여기서는 pickle 통째로 저장/로드 가정 for rank_bm25
                # bm25s는 save/load 메서드 사용해야 함
                if self.impl == "bm25s":
                     # bm25s uses directory-based save usually, but let's check my previous code
                     # I used self.bm25.load. 
                     # For consistency with previous code structure:
                     self.bm25 = bm25s.BM25.load(index_dir, load_corpus=True)
                     self.tokenized_corpus = self.bm25.corpus
                else:
                    # rank_bm25 stores corpus inside the object usually? No, it doesn't store text.
                    # It relies on build-time corpus. But pickle saves the object state.
                    pass

                print(f"[BM25Retrieval] Index loaded.")
            except Exception as e:
                print(f"⚠️  Loading failed: {e}")
                print(f"   → BM25 index 재생성...")
                self._build_bm25_index()
                self._save_bm25_index(index_file)
        else:
            print(f"[BM25Retrieval] Building BM25 index ({self.impl})...")
            self._build_bm25_index()
            self._save_bm25_index(index_file)
            print(f"[BM25Retrieval] Index saved to {index_file}")

    def _build_bm25_index(self) -> NoReturn:
        """Corpus tokenize 후 BM25 인덱스 생성"""
        print(f"[BM25Retrieval] Tokenizing {len(self.contexts)} documents...")
        self.tokenized_corpus = [
            self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing")
        ]

        if self.impl == "rank_bm25":
            try:
                from rank_bm25 import BM25Plus
            except ImportError:
                 raise ImportError("rank_bm25 is required for BM25Plus. `pip install rank_bm25`")
            
            print(f"[BM25Retrieval] Creating BM25Plus index (k1={self.k1}, b={self.b}, delta={self.delta})...")
            self.bm25 = BM25Plus(self.tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)
            
        else:
            # bm25s (Default)
            print(f"[BM25Retrieval] Creating bm25s index (k1={self.k1}, b={self.b})...")
            self.bm25 = bm25s.BM25() # bm25s doesn't support params in constructor easily in v0.1.4?
            # bm25s.index() actually builds it.
            # Note: bm25s might not support custom k1/b easily in index() in all versions, 
            # but usually it's fine.
            self.bm25.index(self.tokenized_corpus, show_progress=False)

    def _save_bm25_index(self, pickle_path: str) -> NoReturn:
        """BM25 인덱스 저장"""
        if self.impl == "bm25s":
            save_dir = os.path.dirname(pickle_path)
            # bm25s specific save (saves multiple files in dir)
            self.bm25.save(save_dir, corpus=self.tokenized_corpus)
            # Also touch the pickle_path so next time we find it
            with open(pickle_path, 'wb') as f:
                pickle.dump("dummy marker for bm25s", f)
        else:
            # rank_bm25: pickle the object
            with open(pickle_path, "wb") as f:
                pickle.dump(self.bm25, f)

    def get_relevant_doc_bulk(
        self,
        queries: List[str],
        k: int = 1,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        여러 query에 대해 BM25 score 기반 top-k 문서 검색.
        """
        tokenized_queries = [self.tokenize_fn(q) for q in queries]

        if self.impl == "bm25s":
            results = self.bm25.retrieve(
                tokenized_queries,
                k=k,
                show_progress=False,
                n_threads=1,
            )
            doc_indices = results.documents.tolist()
            doc_scores = results.scores.tolist()
            return doc_scores, doc_indices
        
        else:
            # rank_bm25 (Iterative, slower)
            doc_scores = []
            doc_indices = []
            
            for query in tqdm(tokenized_queries, desc="BM25Plus Search"):
                scores = self.bm25.get_scores(query)
                top_k_idx = np.argsort(scores)[::-1][:k]
                top_k_scores = [scores[i] for i in top_k_idx]
                
                doc_scores.append(top_k_scores)
                doc_indices.append(top_k_idx.tolist())
                
            return doc_scores, doc_indices
