import logging
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from src.arguments import DataTrainingArguments
from src.retrieval.base import BaseRetrieval

logger = logging.getLogger(__name__)


def realign_answers_in_retrieved_context(example):
    """
    검색된 context에서 정답(answers['text'])을 찾아 answer_start를 갱신합니다.
    """
    retrieved_context = example["context"]
    original_answers = example["answers"]

    new_text = []
    new_answer_start = []

    # 원본 정답 텍스트들이 검색된 context에 존재하는지 확인
    for answer_text in original_answers["text"]:
        # 검색된 context에서 정답 텍스트의 시작 위치 찾기
        # 주의: 여러 번 등장할 수 있지만, 여기서는 첫 번째 등장 위치만 찾거나
        # 모든 등장 위치를 찾을 수도 있음. 일반적으로 첫 번째를 사용하거나,
        # 문맥상 가장 적절한 것을 찾아야 하지만, 단순 매칭으로는 첫 번째를 사용함.
        start_idx = retrieved_context.find(answer_text)

        # 정답을 찾은 경우에만 리스트에 추가
        if start_idx != -1:
            new_text.append(answer_text)
            new_answer_start.append(start_idx)

    # 갱신된 answers로 교체
    # 찾지 못한 경우 빈 리스트가 됨 -> 이후 필터링 대상
    example["answers"] = {"text": new_text, "answer_start": new_answer_start}
    return example


def retrieve_and_build_dataset(
    retriever: BaseRetrieval,
    dataset: Dataset,
    data_args: DataTrainingArguments,
    split_name: str = "validation",
    is_train: bool = False,
    tokenizer=None,
) -> Dataset:
    """
    Retriever를 사용해 question에 맞는 context를 검색하고 MRC용 데이터셋 생성.

    Args:
        retriever: 이미 build()된 retrieval 객체
        dataset: 원본 데이터셋 (단일 split)
        data_args: top_k_retrieval 등 설정
        split_name: 로그용 split 이름
        is_train: True일 경우 정답 재정렬(realignment) 및 필터링 수행
        tokenizer: Tokenizer 객체 (Title 포함 Context 생성을 위해 필요)

    Returns:
        Retrieved context가 포함된 Dataset
    """
    logger.info(
        f"🔍 Running retrieval on {split_name} split (top_k={data_args.top_k_retrieval})..."
    )

    if tokenizer:
        logger.info(
            f"🧩 Tokenizer detected. Title will be prepended to context using separator: '{tokenizer.sep_token}'"
        )
    else:
        logger.warning(
            "⚠️ Tokenizer not provided. Title will NOT be included in context."
        )

    # 1. Retrieval 수행
    # retriever.retrieve는 DataFrame을 반환함
    df = retriever.retrieve(
        dataset, topk=data_args.top_k_retrieval, tokenizer=tokenizer
    )

    # 2. 실제 DataFrame에 answers 컬럼이 있는지 확인
    has_answers = "answers" in df.columns

    # 3. HF Features 정의
    if has_answers:
        used_columns = ["id", "question", "context", "answers"]
        features = Features(
            {
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                "context": Value(dtype="string", id=None),
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
            }
        )
    else:
        used_columns = ["id", "question", "context"]
        features = Features(
            {
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                "context": Value(dtype="string", id=None),
            }
        )

    # 4. 필요한 컬럼만 남기고 HF Dataset으로 변환
    df = df[used_columns].reset_index(drop=True)
    new_dataset = Dataset.from_pandas(df, features=features)

    # --- RERANKING LOGIC ---
    if kwargs.get("reranker") and has_answers: # Reranking mostly useful when context is retrieved. 
        pass
    reranker = kwargs.get("reranker")
    if reranker:
        logger.info(f"🔄 Reranking retrieved passages using {reranker.model_name}...")

        initial_k = data_args.top_k_retrieval
        
        doc_scores, doc_indices = retriever.get_relevant_doc_bulk(
            dataset["question"], k=initial_k
        )
        
        final_contexts = []
        
        for idx, (scores, indices) in enumerate(zip(doc_scores, doc_indices)):
            question = dataset[idx]["question"]
            passages = [retriever.contexts[i] for i in indices]
            
            # Rerank
            rerank_scores = reranker.rerank(question, passages)
            
            scored_passages = list(zip(rerank_scores, passages))
            scored_passages.sort(key=lambda x: x[0], reverse=True)
            
            # Take top-k (or all of them sorted)
            sorted_passages = [p for _, p in scored_passages]
            
            # Join for MRC context
            final_contexts.append(" ".join(sorted_passages))
            
        # Update DataFrame with new contexts
        df["context"] = final_contexts
        
        # Re-create dataset
        new_dataset = Dataset.from_pandas(df[used_columns], features=features)

    # 5. Training일 경우 Answer Realignment 수행
    if is_train and has_answers:
        logger.info("🔄 Realigning answers in retrieved contexts for training...")
        original_len = len(new_dataset)

        # Answer 위치 재계산
        new_dataset = new_dataset.map(realign_answers_in_retrieved_context)

        # 정답을 찾지 못한 데이터(빈 리스트) 필터링
        # answers['text']가 비어있으면 정답이 없는 것
        def filter_valid_answers(example):
            return len(example["answers"]["text"]) > 0

        new_dataset = new_dataset.filter(filter_valid_answers)

        filtered_len = len(new_dataset)
        lost_count = original_len - filtered_len
        lost_ratio = (lost_count / original_len) * 100 if original_len > 0 else 0

        logger.warning(
            f"📉 Retrieval-Augmented Training Stats:\n"
            f"   - Original: {original_len}\n"
            f"   - Filtered (Answer Found): {filtered_len}\n"
            f"   - Lost: {lost_count} ({lost_ratio:.2f}%)\n"
            f"   * Lost examples means the correct answer was NOT found in top-{data_args.top_k_retrieval} retrieved passages."
        )

    return new_dataset
