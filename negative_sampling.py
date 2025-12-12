# src/data/negative_sampling.py

from datasets import Dataset
from typing import List, Dict, Any

class NegativeSampler:
    """
    Train Negative Sampling Handler
    - Retriever(BM25/DPR/Hybrid)에 독립적
    - train dataset을 받아서 augmented dataset 생성
    """

    def __init__(self, retriever, num_negative_samples: int = 3):
        """
        retriever: BM25 또는 Dense retriever 객체
        num_negative_samples: 각 질문당 negative context 개수
        """
        self.retriever = retriever
        self.num_negative = num_negative_samples

    def get_negative_contexts(self, question: str, positive_context: str) -> List[str]:
        """
        BM25/DPR retriever를 사용하여 gold(positive) context를 제외한 상위 negative 문장 k개 추출
        """
        scores, contexts = self.retriever.retrieve(question, topk=20)
        
        negatives = []
        for ctx in contexts:
            if ctx != positive_context:
                negatives.append(ctx)
            if len(negatives) >= self.num_negative:
                break
    
        return negatives
    
    def augment_train_dataset(self, train_dataset: Dataset) -> Dataset:

        new_rows = []
    
        for example in train_dataset:
            q = example["question"]
            pos_ctx = example["context"]
            ans = example["answers"]
    
            # positive (원래 데이터)
            new_rows.append(example)
    
            # negative contexts 추가
            neg_contexts = self.get_negative_contexts(q, pos_ctx)
    
            for i, neg in enumerate(neg_contexts):
                new_rows.append({
                    "id": f"{example['id']}_neg{i}",
                    "question": q,
                    "context": neg,
                    "answers": {"text": [""], "answer_start": [-1]}
                })
    
        # huggingface Dataset으로 변환
        return Dataset.from_list(new_rows)
