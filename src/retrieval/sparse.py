import os
import pickle
import random
from typing import Callable, List, NoReturn, Optional, Tuple, Union

# import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.auto import tqdm

from .base import BaseRetrieval, timer

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


class SparseRetrieval(BaseRetrieval):
    """
    TF-IDF 기반 Sparse Retrieval.
    생성 규약 (권장):
        # 1) YAML 기반 생성 (실전: train/inference)
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            config_path="configs/exp/xxx.yaml",  # retrieval 섹션 포함
        )
        # 2) 테스트/실험용: 직접 파라미터 지정
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
            use_faiss=True,
            num_clusters=64,
        )
    우선순위:
        1) 생성자 인자(use_faiss, num_clusters, ngram_range, max_features ...)
        2) YAML config 의 retrieval 섹션
        3) 코드 기본값
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]],
        config_path: Optional[str] = None,
        data_path: Optional[str] = None,
        context_path: Optional[str] = None,
        use_faiss: Optional[bool] = None,
        num_clusters: Optional[int] = None,
        ngram_range: Optional[Tuple[int, int]] = None,
        max_features: Optional[int] = None,
        **kwargs,
    ) -> NoReturn:
        super().__init__(
            config_path=config_path,
            data_path=data_path,
            context_path=context_path,
            **kwargs,
        )
        self.tokenize_fn = tokenize_fn

        # Config에서 retrieval 설정 추출 (있으면)
        retrieval_config = self.config.get("retrieval", {})

        self.use_faiss: bool = (
            use_faiss
            if use_faiss is not None
            else retrieval_config.get("use_faiss", False)
        )

        self.num_clusters: int = (
            num_clusters
            if num_clusters is not None
            else retrieval_config.get("num_clusters", 64)
        )

        # TF-IDF 세부 설정 (ngram, max_features)
        cfg_ngram = retrieval_config.get("ngram_range", [1, 2])
        cfg_max_features = retrieval_config.get("max_features", 50000)

        if ngram_range is None:
            ngram_range = tuple(cfg_ngram)  # e.g. [1,2] -> (1,2)
        if max_features is None:
            max_features = cfg_max_features

        self.ngram_range = ngram_range
        self.max_features = max_features

        # 3. TfidfVectorizer 초기화
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
        )

        self.p_embedding = None  # build()에서 생성
        self.indexer = None  # use_faiss=True일 때 build()에서 생성

    # === BaseRetrieval 인터페이스 구현 ===

    def build(self) -> NoReturn:
        """
        공통 빌드 진입점.

        1) TF-IDF sparse embedding (self.p_embedding) 생성/로딩
        2) self.use_faiss=True이면 FAISS indexer 또한 생성/로딩
        """
        self.get_sparse_embedding()

        if self.use_faiss:
            self._build_faiss_indexer()

    # === 기존 sparse 전용 로직들 ===

    def get_sparse_embedding(self) -> NoReturn:
        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = "sparse_embedding.bin"
        tfidfv_name = "tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)

            # 크기 검증: contexts 개수와 embedding 개수가 일치하는지 확인
            if self.p_embedding.shape[0] != len(self.contexts):
                print(f"⚠️  WARNING: Embedding 크기 불일치 감지!")
                print(f"   - Loaded embedding shape: {self.p_embedding.shape}")
                print(f"   - Current contexts count: {len(self.contexts)}")
                print(f"   - 원인: 중복 제거 로직 변경 또는 corpus 변경")
                print(f"   → Embedding을 재생성합니다...")

                # 재생성
                self.p_embedding = self.tfidfv.fit_transform(self.contexts)
                print(f"   - New embedding shape: {self.p_embedding.shape}")

                with open(emd_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
                with open(tfidfv_path, "wb") as file:
                    pickle.dump(self.tfidfv, file)
                print(f"✅ 새 embedding 저장 완료")
            else:
                print(
                    f"[SparseRetrieval] Embedding pickle load. Shape: {self.p_embedding.shape}"
                )
        else:
            print("[SparseRetrieval] Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("[SparseRetrieval] Embedding pickle saved.")

    def _build_faiss_indexer(self) -> NoReturn:
        """
        내부 메서드: TF-IDF embedding을 FAISS indexer로 변환.
        build()에서 use_faiss=True일 때 자동 호출됨.
        """
        import faiss  # 필요한 경우에만 import

        assert self.p_embedding is not None, (
            "get_sparse_embedding()을 먼저 호출해야 합니다."
        )

        indexer_name = f"faiss_clusters{self.num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("[SparseRetrieval] Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, self.num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("[SparseRetrieval] Faiss Indexer Saved.")

    # BaseRetrieval.get_relevant_doc() 는 get_relevant_doc_bulk()를 이용하므로,
    # 여기서는 bulk 버전만 정확히 구현해두면 된다.

    def get_relevant_doc(self, query: str, k: int = 1) -> Tuple[List[float], List[int]]:
        """
        단일 query 버전도 기존 형태를 유지하고 싶다면 오버라이드.
        (Base의 구현을 그대로 사용해도 무방하지만, 여기서는 기존 로직을 살림)
        """
        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert np.sum(query_vec) != 0, (
            "오류가 발생했습니다. 이 오류는 보통 query에 "
            "vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        )

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List[str], k: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        여러 query에 대해 유사도를 계산.
        use_faiss=True면 FAISS 검색, False면 TF-IDF exhaustive search.
        """
        if self.use_faiss and self.indexer is not None:
            return self._get_relevant_doc_bulk_faiss(queries, k)
        else:
            return self._get_relevant_doc_bulk_tfidf(queries, k)

    def _get_relevant_doc_bulk_tfidf(
        self, queries: List[str], k: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        TF-IDF exhaustive search.
        """
        query_vec = self.tfidfv.transform(queries)
        assert np.sum(query_vec) != 0, (
            "오류가 발생했습니다. 이 오류는 보통 query에 "
            "vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        )

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        doc_scores: List[List[float]] = []
        doc_indices: List[List[int]] = []

        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices

    # === FAISS 전용 내부 메서드 ===

    def _get_relevant_doc_bulk_faiss(
        self, queries: List[str], k: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        FAISS indexer를 사용한 근사 검색.
        """
        import faiss  # 사용 시점에만 import

        assert self.indexer is not None, "FAISS indexer가 빌드되지 않았습니다."

        query_vecs = self.tfidfv.transform(queries)
        assert np.sum(query_vecs) != 0, (
            "오류가 발생했습니다. 이 오류는 보통 query에 "
            "vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        )

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--use_faiss", action="store_true", help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:
        retriever.build()

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by faiss"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        retriever.build()

        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, contexts = retriever.retrieve(query)  # 명확성: indices -> contexts
