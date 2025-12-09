import os
from typing import List, NoReturn, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import BaseRetrieval, timer


class DenseRetrieval(BaseRetrieval):
    """
    Hugging Face sentence embedding 모델을 활용한 Dense Retrieval.

    - BaseRetrieval을 상속받아 contexts 로딩/공통 retrieve 로직은 그대로 사용
    - build():
        - embedding_path가 있으면 npy 로드
        - 없으면 corpus 전체 임베딩 계산 후 저장(optional)
    - get_relevant_doc_bulk():
        - query embedding과 corpus embedding cosine similarity로 top-k 계산
    """

    def __init__(
        self,
        model_name_or_path: str,
        data_path: str = "./data",
        context_path: str = "wikipedia_documents.json",
        embedding_path: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 64,
    ) -> NoReturn:
        super().__init__(data_path=data_path, context_path=context_path)

        self.model_name_or_path = model_name_or_path
        self.embedding_path = embedding_path
        self.max_length = max_length
        self.batch_size = batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

        self.p_embedding: Optional[np.ndarray] = None  # (num_docs, dim)

    # === BaseRetrieval 인터페이스 구현 ===

    def build(self) -> NoReturn:
        """
        corpus dense embedding 로드/계산.
        """
        if self.embedding_path is not None and os.path.isfile(self.embedding_path):
            self.p_embedding = np.load(self.embedding_path)
            print(
                f"[DenseRetrieval] Loaded corpus embeddings from {self.embedding_path} "
                f"with shape {self.p_embedding.shape}"
            )
        else:
            print("[DenseRetrieval] Building corpus embeddings...")
            with timer("build corpus dense embedding"):
                self._build_corpus_embedding()

            if self.embedding_path is not None:
                dirpath = os.path.dirname(self.embedding_path)
                if dirpath:  # 디렉토리 경로가 있을 때만 생성
                    os.makedirs(dirpath, exist_ok=True)
                np.save(self.embedding_path, self.p_embedding)
                print(
                    f"[DenseRetrieval] Saved corpus embeddings to {self.embedding_path}"
                )

    # === 내부 유틸 ===

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 리스트를 dense embedding (B, dim)으로 인코딩.
        - mean pooling + L2 normalize
        """
        all_embeddings: List[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state  # (B, L, H)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)

                # mean pooling
                summed = (last_hidden * attention_mask).sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                emb = summed / counts

                # L2 normalize
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

                all_embeddings.append(emb.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def _build_corpus_embedding(self) -> NoReturn:
        self.p_embedding = self._encode_texts(self.contexts)
        print(f"[DenseRetrieval] Corpus embeddings shape: {self.p_embedding.shape}")

    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        return self._encode_texts(queries)

    # === 핵심 검색 로직 ===

    def get_relevant_doc_bulk(
        self, queries: List[str], k: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        queries: 문자열 리스트
        k: top-k 문서 개수

        반환:
            doc_scores: 각 query별 top-k 점수 리스트
            doc_indices: 각 query별 top-k 문서 인덱스 리스트
        """
        assert self.p_embedding is not None, (
            "Corpus embeddings (p_embedding)가 없습니다. build()를 먼저 호출하세요."
        )

        # (B, dim)
        with timer("encode queries (dense)"):
            q_emb = self._encode_queries(queries)

        corpus_emb = self.p_embedding  # (N, dim)

        # (B, dim) @ (dim, N) = (B, N)
        with timer("matrix multiply (queries x corpus)"):
            scores = np.matmul(q_emb, corpus_emb.T)  # cosine similarity (L2-normalized)

        num_docs = scores.shape[1]
        k = min(k, num_docs)

        # argpartition으로 top-k 후보 뽑은 뒤, 그 안에서 sort
        topk_indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        topk_scores = np.take_along_axis(scores, topk_indices, axis=1)

        order = np.argsort(-topk_scores, axis=1)
        topk_indices = np.take_along_axis(topk_indices, order, axis=1)
        topk_scores = np.take_along_axis(scores, topk_indices, axis=1)

        doc_scores: List[List[float]] = topk_scores.tolist()
        doc_indices: List[List[int]] = topk_indices.tolist()
        return doc_scores, doc_indices
