import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

# import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
# from sklearn.feature_extraction.text import TfidfVectorizer # TfidfVectorizer 제거
from rank_bm25 import BM25Okapi # BM25Okapi 추가
from tqdm.auto import tqdm

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 BM25Okapi를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        self.raw_context_names = list(
            dict.fromkeys([v["title"] for v in wiki.values()])
        ) # Passage 제목 추가
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.tokenize_fn = tokenize_fn # tokenize_fn 저장

        # BM25Okapi 인스턴스를 생성하도록 변경
        self.bm25 = None  # get_sparse_embedding()로 생성합니다.

        self.p_embedding = None  # BM25에서는 토크나이징된 corpus 자체를 의미합니다.
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> NoReturn:
        """
        Summary:
            Passage Embedding을 만들고
            BM25 인스턴스를 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"bm25_sparse_embedding.bin"
        bm25_name = f"bm25_model.bin" # BM25 모델 자체 저장 파일명
        emd_path = os.path.join(self.data_path, pickle_name)
        bm25_model_path = os.path.join(self.data_path, bm25_name)

        if os.path.isfile(bm25_model_path): # BM25 모델이 저장된 파일 확인
            with open(bm25_model_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("BM25 model pickle load.")
        else:
            print("Build passage embedding (BM25)")
            tokenized_corpus = [self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing BM25 corpus")]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.p_embedding = tokenized_corpus # BM25에서는 토크나이징된 corpus 자체가 임베딩
            
            with open(bm25_model_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("BM25 model pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:
        """
        Summary:
            Faiss는 Dense Embedding에 더 적합합니다. Sparse Embedding (BM25) 에서는 일반적으로 사용되지 않습니다.
            이 프로젝트의 기존 Faiss 코드는 TfidfVectorizer의 dense array를 인덱싱했습니다.
            BM25의 경우 직접적인 Faiss 인덱싱은 비효율적이므로, 여기서는 해당 기능을 비활성화하거나 수정해야 합니다.
            일단 BM25의 score를 인덱싱하는 방식으로는 Faiss가 크게 유리하지 않아 BM25는 Faiss 없이 사용합니다.
        """
        print("BM25 is generally not used with Faiss indexer in this manner.")
        print("You can build a Faiss index for Dense Embeddings instead.")
        self.indexer = None # Faiss 인덱서는 BM25에 직접 사용되지 않음을 명시적으로 표시

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.
        """

        assert self.bm25 is not None, (
            "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."
        )

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i + 1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query bm25 search"): # exhaustive search 대신 bm25 search로 변경
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="BM25 retrieval: ") # Sparse -> BM25
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """

        tokenized_query = self.tokenize_fn(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        # 점수가 0인 경우를 방지하기 위해 정렬 시 안정성 추가
        # np.argsort는 기본적으로 오름차순이므로 [::-1]로 내림차순 변경
        sorted_result_indices = np.argsort(doc_scores)[::-1] 
        
        doc_score_list = doc_scores[sorted_result_indices].tolist()[:k]
        doc_indices_list = sorted_result_indices.tolist()[:k]
        return doc_score_list, doc_indices_list

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나의 Query 리스트를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """

        doc_scores_list = []
        doc_indices_list = []

        for query in queries:
            tokenized_query = self.tokenize_fn(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            sorted_result_indices = np.argsort(doc_scores)[::-1]

            doc_scores_list.append(doc_scores[sorted_result_indices].tolist()[:k])
            doc_indices_list.append(sorted_result_indices.tolist()[:k])
            
        return doc_scores_list, doc_indices_list

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            BM25는 Faiss 인덱서와 직접적으로 작동하지 않습니다. 이 함수는 비활성화됩니다.
        """
        print("BM25 is generally not used with Faiss indexer in this manner.")
        print("This function is disabled for BM25 retrieval.")
        return self.retrieve(query_or_dataset, topk) # 일반 retrieve 함수로 대체


    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        raise NotImplementedError("BM25는 Faiss 인덱서와 직접적으로 작동하지 않습니다.")

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        raise NotImplementedError("BM25는 Faiss 인덱서와 직접적으로 작동하지 않습니다.")


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
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", action="store_true", help="Use Faiss for retrieval (Note: Faiss is not directly applicable to BM25 in this setup)")

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
    
    # tokenizer.tokenize 대신 lambda x: tokenizer.tokenize(x)로 감싸서 사용하도록 변경 (BM25Okapi 요구사항)
    retriever = SparseRetrieval(
        tokenize_fn=lambda x: tokenizer.tokenize(x),
        data_path=args.data_path,
        context_path=args.context_path,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    # BM25는 Faiss를 직접 사용하지 않으므로 use_faiss는 무시됩니다.
    # if args.use_faiss:
    #     print("Faiss is not directly applicable to BM25. Running normal BM25 retrieval.")
    #     retriever.get_sparse_embedding() # BM25 모델 로드/빌드
    #     # test single query
    #     with timer("single query by bm25"):
    #         scores, indices = retriever.retrieve(query)
    #     # test bulk
    #     with timer("bulk query by bm25"):
    #         df = retriever.retrieve(full_ds)
    #         df["correct"] = df["original_context"] == df["context"]
    #         print("correct retrieval result by bm25", df["correct"].sum() / len(df))
    # else:
    retriever.get_sparse_embedding() # BM25 모델 로드/빌드
    with timer("bulk query by bm25"):
        df = retriever.retrieve(full_ds)
        df["correct"] = df["original_context"] == df["context"]
        print(
            "correct retrieval result by bm25",
            df["correct"].sum() / len(df),
        )

    with timer("single query by bm25"):
        scores, indices = retriever.retrieve(query)
