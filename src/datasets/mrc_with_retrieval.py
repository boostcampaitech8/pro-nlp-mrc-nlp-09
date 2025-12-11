"""
MRC Dataset with Retrieval Cache & Dynamic Hard Negative Sampling

Retrieval 캐시를 사용하여 MRC 학습용 데이터셋을 구성합니다.
- Train: Dynamic Hard Negative Sampling
- Val/Test: Top-k retrieval 결과 사용

Usage:
    from src.datasets.mrc_with_retrieval import MRCWithRetrievalDataset

    dataset = MRCWithRetrievalDataset(
        examples=train_examples,
        retrieval_cache=cache_dict,
        passages_corpus=passages_list,
        tokenizer=tokenizer,
        mode="train",
        k_ret=20,
    )
"""

import json
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# === 기본 설정 ===
DEFAULT_K_RET = 20  # retrieval top-k
DEFAULT_K_READ = 3  # train에서 사용할 context 수 (1 pos + n neg)
DEFAULT_ALPHA = 0.35  # hybrid score 계산용 BM25 가중치 (base.yaml과 일치)


def load_retrieval_cache(cache_path: str) -> Dict[str, Dict]:
    """
    Retrieval 캐시 파일을 로드합니다.

    Args:
        cache_path: 캐시 JSONL 파일 경로

    Returns:
        {question_id -> {"question": str, "retrieved": List[Dict]}} 딕셔너리
    """
    cache = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            cache[item["id"]] = {
                "question": item["question"],
                "retrieved": item["retrieved"],
            }
    return cache


def load_passages_corpus(
    passages_meta_path: Optional[str] = None,
    wiki_path: Optional[str] = None,
) -> Tuple[List[str], List[Dict]]:
    """
    Passage corpus를 로드합니다.

    passages_meta가 있으면 chunked passage 사용,
    없으면 wiki documents 사용.

    Args:
        passages_meta_path: passages metadata JSONL 경로
        wiki_path: wikipedia_documents.json 경로

    Returns:
        (passage_texts, passage_metas) 튜플
        - passage_texts: passage_id 순서대로 텍스트 리스트
        - passage_metas: passage_id 순서대로 메타데이터 리스트
    """
    if passages_meta_path:
        # Chunked passages 사용
        passage_texts = []
        passage_metas = []
        with open(passages_meta_path, "r", encoding="utf-8") as f:
            for line in f:
                meta = json.loads(line.strip())
                passage_texts.append(meta["text"])
                passage_metas.append(meta)
        return passage_texts, passage_metas

    elif wiki_path:
        # Wiki documents 사용
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # 중복 제거 후 순서 유지
        unique_texts = {}
        for doc_id, doc_info in wiki.items():
            text = doc_info["text"]
            if text not in unique_texts:
                unique_texts[text] = {
                    "doc_id": int(doc_id),
                    "title": doc_info.get("title", ""),
                    "text": text,
                }

        passage_texts = list(unique_texts.keys())
        passage_metas = [
            {
                "passage_id": idx,
                "doc_id": unique_texts[text]["doc_id"],
                "title": unique_texts[text]["title"],
                "text": text,
                "start_char": 0,
                "end_char": len(text),
                "is_chunk": False,
            }
            for idx, text in enumerate(passage_texts)
        ]
        return passage_texts, passage_metas

    else:
        raise ValueError("Either passages_meta_path or wiki_path must be provided")


def compute_hybrid_score_for_candidates(
    candidates: List[Dict],
    alpha: float = DEFAULT_ALPHA,
    eps: float = 1e-9,
) -> List[Dict]:
    """
    캐시된 candidates에 hybrid_score를 계산하여 추가합니다.

    Args:
        candidates: [{"passage_id", "doc_id", "score_dense", "score_bm25"}, ...]
        alpha: BM25 가중치
        eps: 정규화 시 0 나누기 방지

    Returns:
        hybrid_score가 추가된 candidates (정렬 안 함)
    """
    if not candidates:
        return candidates

    bm25_scores = np.array([c["score_bm25"] for c in candidates])
    dense_scores = np.array([c["score_dense"] for c in candidates])

    # Per-query min-max 정규화
    bm25_n = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min() + eps
    )
    dense_n = (dense_scores - dense_scores.min()) / (
        dense_scores.max() - dense_scores.min() + eps
    )

    # 가중합
    hybrid_scores = alpha * bm25_n + (1 - alpha) * dense_n

    for c, h in zip(candidates, hybrid_scores):
        c["hybrid_score"] = float(h)

    return candidates


class MRCWithRetrievalDataset(Dataset):
    """
    Retrieval 캐시 기반 MRC Dataset (Dynamic Hard Negative Sampling 지원).

    Train 모드에서는:
        - Gold document를 positive로 사용
        - Retrieval 결과 중 gold가 아닌 것들을 hard negative로 샘플링
        - 매 epoch마다 다른 negative가 선택됨 (Dynamic)

    Val/Test 모드에서는:
        - 기존 MRC 파이프라인처럼 top-k retrieval 결과를 concatenate

    Args:
        examples: HF Dataset 또는 dict list (id, question, context, answers, document_id)
        retrieval_cache: {id -> {question, retrieved}} 캐시 딕셔너리
        passages_corpus: (passage_texts, passage_metas) 튜플
        tokenizer: HuggingFace tokenizer
        mode: "train", "val", "test"
        k_ret: retrieval top-k (default: 20)
        k_read: train에서 사용할 context 수 (default: 3)
        max_seq_length: 최대 시퀀스 길이
        doc_stride: sliding window stride
        alpha: hybrid score 계산용 BM25 가중치
        return_token_type_ids: token_type_ids 반환 여부 (RoBERTa는 False)
    """

    def __init__(
        self,
        examples: Union[List[Dict], Any],  # HF Dataset 또는 List[Dict]
        retrieval_cache: Dict[str, Dict],
        passages_corpus: Tuple[List[str], List[Dict]],
        tokenizer: PreTrainedTokenizer,
        mode: str = "train",
        k_ret: int = DEFAULT_K_RET,
        k_read: int = DEFAULT_K_READ,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        alpha: float = DEFAULT_ALPHA,
        return_token_type_ids: bool = True,
        use_title: bool = True,
    ):
        self.examples = examples
        self.retrieval_cache = retrieval_cache
        self.passage_texts, self.passage_metas = passages_corpus
        self.tokenizer = tokenizer
        self.mode = mode.lower()
        self.k_ret = k_ret
        self.k_read = k_read
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.alpha = alpha
        self.return_token_type_ids = return_token_type_ids
        self.use_title = use_title

        # doc_id -> passage_ids 매핑 구성
        self.doc_to_passages: Dict[int, List[int]] = {}
        for meta in self.passage_metas:
            doc_id = meta["doc_id"]
            passage_id = meta["passage_id"]
            if doc_id not in self.doc_to_passages:
                self.doc_to_passages[doc_id] = []
            self.doc_to_passages[doc_id].append(passage_id)

        # passage_id -> doc_id 매핑
        self.passage_to_doc = {
            meta["passage_id"]: meta["doc_id"] for meta in self.passage_metas
        }

        print(f"[MRCWithRetrievalDataset] mode={mode}, k_ret={k_ret}, k_read={k_read}")
        print(f"  - Examples: {len(examples)}")
        print(f"  - Passages: {len(self.passage_texts)}")
        print(f"  - Cache entries: {len(retrieval_cache)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Train: Dynamic Hard Negative Sampling으로 context 선택 후 tokenization
        Val/Test: Top-k concatenated context로 tokenization
        """
        example = self.examples[idx]
        qid = example["id"]
        question = example["question"]

        if self.mode == "train":
            return self._get_train_item(example, qid, question)
        else:
            return self._get_eval_item(example, qid, question)

    def _get_train_item(
        self, example: Dict, qid: str, question: str
    ) -> Dict[str, torch.Tensor]:
        """
        Train 모드: Dynamic Hard Negative Sampling.

        ⚠️ 핵심 설계 원칙:
        1. Positive는 항상 **원본 gold context** 사용 (answer_start offset 문제 방지)
        2. Retrieval 결과는 **negative에만** 사용
        3. Positive로 retrieval passage 사용 시 answer_text.find()로 local_start 재계산

        이렇게 하면:
        - answer_start가 항상 context_text 기준으로 정확함
        - Hard negative는 retrieval에서 뽑으므로 KURE/BM25의 헷갈리는 오답들 학습 가능
        """
        # Gold document ID
        gold_doc_id = example.get("document_id")
        if gold_doc_id is None:
            gold_doc_id = self._infer_gold_doc_id(example)

        # Retrieval 캐시에서 후보 가져오기
        cache_entry = self.retrieval_cache.get(qid)
        if cache_entry is None:
            # 캐시에 없으면 gold context만 사용
            return self._tokenize_with_gold_context(example, question)

        candidates = cache_entry["retrieved"][: self.k_ret]
        candidates = compute_hybrid_score_for_candidates(candidates, self.alpha)

        # Negative만 분리 (gold_doc_id와 다른 것들)
        neg_candidates = [c for c in candidates if c["doc_id"] != gold_doc_id]

        # Hard/Medium negative 분할 (상위 5개 = hard, 나머지 = medium)
        hard_neg = neg_candidates[:5] if len(neg_candidates) >= 5 else neg_candidates
        medium_neg = neg_candidates[5:] if len(neg_candidates) > 5 else []

        # ========================================
        # Context pool 구성: pos 1개 + neg 여러 개
        # ========================================
        selected_contexts = []

        # ✅ Positive: 항상 원본 gold context 사용 (answer offset 문제 해결)
        # "pos" 라벨과 함께 None을 넣어서 gold context 사용 표시
        selected_contexts.append(("pos", None))

        # Hard negative 선택 (1~2개)
        if hard_neg:
            num_hard = min(random.randint(1, 2), len(hard_neg))
            for neg in random.sample(hard_neg, num_hard):
                selected_contexts.append(("neg", neg))

        # Medium negative 선택 (0~1개, k_read까지 채우기)
        remaining = self.k_read - len(selected_contexts)
        if remaining > 0 and medium_neg:
            num_medium = min(remaining, len(medium_neg))
            for neg in random.sample(medium_neg, num_medium):
                selected_contexts.append(("neg", neg))

        # ========================================
        # Dynamic Hard Negative: pool에서 하나 선택
        # ========================================
        label, chosen = random.choice(selected_contexts)

        if label == "pos":
            # ✅ Positive: 원본 gold context 사용 (answer_start가 정확함)
            return self._tokenize_with_gold_context(example, question)
        else:
            # Negative: retrieval passage 사용 (answer = CLS)
            passage_id = chosen["passage_id"]
            context_text = self.passage_texts[passage_id]
            title = self.passage_metas[passage_id].get("title", "")

            if self.use_title and title:
                context_text = f"{title} {self.tokenizer.sep_token} {context_text}"

            return self._tokenize_without_answer(question, context_text)

    def _get_eval_item(
        self, example: Dict, qid: str, question: str
    ) -> Dict[str, torch.Tensor]:
        """
        Val/Test 모드: Top-k retrieval 결과를 concatenate.
        """
        cache_entry = self.retrieval_cache.get(qid)
        if cache_entry is None:
            # 캐시에 없으면 원본 context 사용 (validation)
            if "context" in example:
                return self._tokenize_for_eval(question, example["context"], qid)
            else:
                raise ValueError(f"No cache entry for question {qid}")

        candidates = cache_entry["retrieved"][: self.k_ret]
        candidates = compute_hybrid_score_for_candidates(candidates, self.alpha)

        # Hybrid score 기준 정렬
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                -x.get("hybrid_score", 0),
                -x["score_bm25"],
                x["passage_id"],
            ),
        )

        # Top-k context concatenate
        context_parts = []
        for c in sorted_candidates:
            passage_id = c["passage_id"]
            text = self.passage_texts[passage_id]
            title = self.passage_metas[passage_id].get("title", "")

            if self.use_title and title:
                context_parts.append(f"{title} {self.tokenizer.sep_token} {text}")
            else:
                context_parts.append(text)

        context_text = " ".join(context_parts)

        return self._tokenize_for_eval(question, context_text, qid)

    def _tokenize_with_gold_context(
        self, example: Dict, question: str
    ) -> Dict[str, torch.Tensor]:
        """
        Gold context를 사용하여 tokenization (fallback).

        ✅ 이 메서드가 answer_start offset 문제를 해결하는 핵심:
        - example["context"]와 example["answers"]["answer_start"]는 같은 기준
        - 따라서 offset이 항상 정확함
        """
        context = example.get("context", "")
        answers = example.get("answers", {"text": [], "answer_start": []})
        return self._tokenize_with_answer(question, context, answers)

    def _tokenize_with_answer_in_passage(
        self,
        question: str,
        context: str,
        answer_text: str,
        local_start: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieval passage에서 answer 위치를 재계산하여 tokenization.

        ⚠️ 이 메서드는 retrieval passage를 positive로 사용할 때 호출.
        answer_text.find()로 찾은 local_start를 사용하여 정확한 label 생성.

        Args:
            question: 질문 텍스트
            context: retrieval passage 텍스트 (chunk)
            answer_text: 정답 텍스트
            local_start: context 내에서 answer의 시작 위치 (find()로 계산됨)

        Returns:
            tokenized dict with start_positions, end_positions
        """
        tokenized = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            return_token_type_ids=self.return_token_type_ids,
            padding="max_length",
        )

        if not self.return_token_type_ids and "token_type_ids" in tokenized:
            del tokenized["token_type_ids"]

        offset_mapping = tokenized.pop("offset_mapping")
        input_ids = tokenized["input_ids"]
        cls_index = input_ids.index(self.tokenizer.cls_token_id)

        # local_start 기준으로 token 위치 찾기
        answer_end = local_start + len(answer_text)
        start_position, end_position = self._find_token_positions(
            offset_mapping, local_start, answer_end, cls_index
        )

        tokenized["start_positions"] = start_position
        tokenized["end_positions"] = end_position

        return {k: torch.tensor(v) for k, v in tokenized.items()}

    def _tokenize_with_answer(
        self, question: str, context: str, answers: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Positive context에 대해 tokenization + answer label 계산.
        """
        tokenized = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=False,  # 단일 chunk만
            return_offsets_mapping=True,
            return_token_type_ids=self.return_token_type_ids,
            padding="max_length",
        )

        # token_type_ids 제거 (RoBERTa 등)
        if not self.return_token_type_ids and "token_type_ids" in tokenized:
            del tokenized["token_type_ids"]

        offset_mapping = tokenized.pop("offset_mapping")
        input_ids = tokenized["input_ids"]

        # CLS 위치
        cls_index = input_ids.index(self.tokenizer.cls_token_id)

        # Answer position 계산
        if len(answers.get("answer_start", [])) == 0:
            start_position = cls_index
            end_position = cls_index
        else:
            answer_start = answers["answer_start"][0]
            answer_text = answers["text"][0]
            answer_end = answer_start + len(answer_text)

            # Token 위치 찾기
            start_position, end_position = self._find_token_positions(
                offset_mapping, answer_start, answer_end, cls_index
            )

        tokenized["start_positions"] = start_position
        tokenized["end_positions"] = end_position

        # Tensor 변환
        return {k: torch.tensor(v) for k, v in tokenized.items()}

    def _tokenize_without_answer(
        self, question: str, context: str
    ) -> Dict[str, torch.Tensor]:
        """
        Negative context에 대해 tokenization (answer = CLS).
        """
        tokenized = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            return_token_type_ids=self.return_token_type_ids,
            padding="max_length",
        )

        if not self.return_token_type_ids and "token_type_ids" in tokenized:
            del tokenized["token_type_ids"]

        cls_index = tokenized["input_ids"].index(self.tokenizer.cls_token_id)
        tokenized["start_positions"] = cls_index
        tokenized["end_positions"] = cls_index

        return {k: torch.tensor(v) for k, v in tokenized.items()}

    def _tokenize_for_eval(
        self, question: str, context: str, example_id: str
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluation용 tokenization.
        """
        tokenized = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            return_token_type_ids=self.return_token_type_ids,
            padding="max_length",
        )

        if not self.return_token_type_ids and "token_type_ids" in tokenized:
            del tokenized["token_type_ids"]

        # eval에서는 example_id 필요
        tokenized["example_id"] = example_id

        # offset_mapping은 context 부분만 유지
        sequence_ids = (
            tokenized.sequence_ids(0) if hasattr(tokenized, "sequence_ids") else None
        )
        if sequence_ids:
            # context는 sequence_id == 1
            tokenized["offset_mapping"] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized["offset_mapping"])
            ]

        return {
            k: torch.tensor(v) if k != "example_id" else v for k, v in tokenized.items()
        }

    def _find_token_positions(
        self,
        offset_mapping: List[Tuple[int, int]],
        answer_start: int,
        answer_end: int,
        cls_index: int,
    ) -> Tuple[int, int]:
        """
        Offset mapping에서 answer의 시작/끝 token 위치를 찾습니다.
        """
        start_position = cls_index
        end_position = cls_index

        for idx, (start, end) in enumerate(offset_mapping):
            if start is None or end is None:
                continue
            if start <= answer_start < end:
                start_position = idx
            if start < answer_end <= end:
                end_position = idx
                break

        # 유효성 검증
        if start_position > end_position:
            start_position = cls_index
            end_position = cls_index

        return start_position, end_position

    def _infer_gold_doc_id(self, example: Dict) -> Optional[int]:
        """
        document_id가 없는 경우, 답이 포함된 문서를 추론합니다.
        """
        # 일단 None 반환 (이 경우 gold context fallback 사용)
        return None
