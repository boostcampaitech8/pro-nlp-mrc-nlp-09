import json
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm


@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BaseRetrieval(ABC):
    """
    공통 Retrieval 베이스 클래스.

    - 위키 코퍼스 로딩 (contexts, ids)
    - 문자열/데이터셋 입력을 공통으로 처리하는 retrieve()
    - 서브클래스는 build(), get_relevant_doc_bulk()만 구현하면 됨.
    """

    def __init__(
        self,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        self.data_path = data_path
        self.context_path = context_path

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # context와 document_id, title을 함께 저장
        # set 은 매번 순서가 바뀌므로 dict 로 유니크한 context만 추출
        # document_id는 int로 변환 (데이터셋의 document_id가 int 타입)
        unique_contexts = {}
        for doc_id, doc_info in wiki.items():
            text = doc_info["text"]
            if text not in unique_contexts:
                unique_contexts[text] = {
                    "doc_id": int(doc_id),
                    "title": doc_info.get("title", ""),
                }

        self.contexts = list(unique_contexts.keys())
        self.ids = [v["doc_id"] for v in unique_contexts.values()]
        # ⚠️ NOTE: self.ids는 wikipedia_documents.json의 실제 doc_id (int)
        #    단순 인덱스 [0,1,2,...]가 아님! contexts와 1:1 매핑됨
        self.titles = [v["title"] for v in unique_contexts.values()]
        print(
            f"[BaseRetrieval] Lengths of unique contexts : {len(self.contexts)}"
        )  # === 서브클래스에서 구현해야 하는 부분 ===

    @abstractmethod
    def build(self) -> NoReturn:
        """
        서브클래스에서 embedding / index 등을 준비하는 함수.

        예)
        - SparseRetrieval: TF-IDF fit + passage embedding
        - DenseRetrieval: corpus dense embedding 계산/로딩
        """
        raise NotImplementedError

    @abstractmethod
    def get_relevant_doc_bulk(
        self, queries: List[str], k: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        여러 query에 대해 상위 k개 점수+인덱스를 반환하는 core 함수.

        반환:
            doc_scores: [[score_1, ..., score_k], ...]  (len = num_queries)
            doc_indices: [[idx_1, ..., idx_k], ...]
        """
        raise NotImplementedError

    # === 공통 유틸 ===

    def get_relevant_doc(self, query: str, k: int = 1) -> Tuple[List[float], List[int]]:
        """
        단일 query 버전. 내부적으로 get_relevant_doc_bulk를 호출한다.
        """
        scores, indices = self.get_relevant_doc_bulk([query], k=k)
        return scores[0], indices[0]

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: int = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        공통 retrieval 인터페이스:

        - query_or_dataset: str 또는 HF Dataset
        - str:
            (scores, [context1, context2, ...]) 반환 + 상위 문서 출력
        - Dataset:
            question / id / context (+ original_context / answers)로 DataFrame 생성
        """

        assert hasattr(self, "contexts"), (
            "contexts가 초기화되지 않았습니다. BaseRetrieval.__init__가 호출되었는지 확인하세요."
        )

        # 1) 단일 문자열 쿼리
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                [query_or_dataset], k=topk
            )
            doc_scores = doc_scores[0]
            doc_indices = doc_indices[0]

            print("[Search query]\n", query_or_dataset, "\n")
            for i in range(topk):
                print(f"Top-{i + 1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return doc_scores, [self.contexts[pid] for pid in doc_indices]

        # 2) HF Dataset (train / validation / test)
        elif isinstance(query_or_dataset, Dataset):
            total = []
            queries = query_or_dataset["question"]

            with timer(f"{self.__class__.__name__} query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(queries, k=topk)

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc=f"{self.__class__.__name__} retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                # validation/train처럼 정답이 있는 경우
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

        else:
            raise TypeError(
                f"query_or_dataset 타입이 잘못되었습니다: {type(query_or_dataset)}"
            )
