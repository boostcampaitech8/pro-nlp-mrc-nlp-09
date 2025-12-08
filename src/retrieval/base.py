import json
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm


@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드 (HfArgumentParser 방식 재사용).

    Args:
        config_path: .yaml 파일 경로

    Returns:
        설정 딕셔너리

    Note:
        transformers.HfArgumentParser가 내부적으로 사용하는 방식과 동일.
        pyyaml 의존성이 없으면 자동으로 설치 유도.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config file support. "
            "Install it with: pip install pyyaml"
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BaseRetrieval(ABC):
    """
    공통 Retrieval 베이스 클래스.

    - 위키 코퍼스 로딩 (contexts, ids, titles)
    - 문자열/데이터셋 입력을 공통으로 처리하는 retrieve()
    - 서브클래스는 build(), get_relevant_doc_bulk()만 구현하면 됨.

    생성 규약:
        BaseRetrieval(config_path=None, data_path=None, context_path=None, **kwargs)

    우선순위:
        1) 생성자 인자(data_path, context_path 등 kwargs)
        2) YAML config 내 retrieval 섹션 (config['retrieval'])
        3) 코드 내부 default 값
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        data_path: Optional[str] = None,
        context_path: Optional[str] = None,
        **kwargs,
    ) -> NoReturn:
        # 1. Config 로딩 (있으면)
        if config_path:
            self.config = load_yaml_config(config_path)
            retrieval_config = self.config.get("retrieval", {})
        else:
            self.config = {}
            retrieval_config = {}

        # 2. 우선순위 적용: kwargs > config > 기본값
        #    (Sparse/DenseRetrieval에서 data_path/context_path를 넘겨도 여기서 처리됨)
        self.data_path = (
            data_path
            or kwargs.get("data_path")
            or retrieval_config.get("data_path", "./data")
        )
        self.context_path = (
            context_path
            or kwargs.get("context_path")
            or retrieval_config.get("context_path", "wikipedia_documents.json")
        )

        # 3. Wikipedia corpus 로드
        wiki_path = os.path.join(self.data_path, self.context_path)
        if not os.path.exists(wiki_path):
            raise FileNotFoundError(f"Wikipedia corpus not found: {wiki_path}")

        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # text 기준 유니크 context 추출 + doc_id(int), title 함께 저장
        unique_contexts: Dict[str, Dict[str, Any]] = {}
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
        # TODO: title을 인덱싱과 MRC 추론에 활용할 수 있을까?
        self.titles = [v["title"] for v in unique_contexts.values()]
        print(f"[BaseRetrieval] Lengths of unique contexts : {len(self.contexts)}")

    # === 서브클래스에서 구현해야 하는 부분 (오버라이딩)===

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
