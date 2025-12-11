import logging
from typing import Optional
from datasets import Dataset, Features, Sequence, Value
from src.arguments import DataTrainingArguments
from src.retrieval.base import BaseRetrieval

logger = logging.getLogger(__name__)


def realign_answers_in_retrieved_context(
    example,
    sep_token: Optional[str] = None,
    use_title: bool = False,
):
    """
    검색된 context에서 정답(answers['text'])을 찾아 answer_start를 갱신합니다.

    Args:
        example: HF Dataset example (context, answers 포함)
        sep_token: Title과 본문을 구분하는 separator (예: "[SEP]", "</s>")
        use_title: True이면 context에 title이 포함되어 있다고 가정하고,
                   title 영역을 건너뛰고 본문에서만 정답을 찾습니다.

    Returns:
        answer_start가 갱신된 example

    Note:
        - use_title=True이고 sep_token이 주어지면:
          context = "제목 [SEP] 본문" 형태에서 본문 부분만 검색
        - 정답이 title에만 있고 본문에 없으면 빈 리스트 반환 (이후 필터링)
    """
    retrieved_context = example["context"]
    original_answers = example["answers"]

    new_text = []
    new_answer_start = []

    # Title 영역 건너뛰기 위한 offset 계산
    search_start_offset = 0
    if use_title and sep_token:
        # sep_token 위치 찾기 (첫 번째 passage의 separator)
        sep_pos = retrieved_context.find(sep_token)
        if sep_pos != -1:
            # separator 뒤부터 검색 시작 (sep_token 길이 + 공백 고려)
            search_start_offset = sep_pos + len(sep_token)
            # separator 뒤의 공백 건너뛰기
            while (
                search_start_offset < len(retrieved_context)
                and retrieved_context[search_start_offset] == " "
            ):
                search_start_offset += 1

    # 원본 정답 텍스트들이 검색된 context에 존재하는지 확인
    for answer_text in original_answers["text"]:
        # Title 영역을 건너뛴 위치에서부터 검색
        start_idx = retrieved_context.find(answer_text, search_start_offset)

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
    top_k_override: int = None,
    use_title_override: Optional[bool] = None,
) -> Dataset:
    """
    Retriever를 사용해 question에 맞는 context를 검색하고 MRC용 데이터셋 생성.

    Args:
        retriever: 이미 build()된 retrieval 객체
        dataset: 원본 데이터셋 (단일 split)
        data_args: top_k_retrieval, use_title 등 설정
        split_name: 로그용 split 이름
        is_train: True일 경우 정답 재정렬(realignment) 및 필터링 수행
        tokenizer: Tokenizer 객체 (use_title=True일 때 sep_token 사용)
        top_k_override: top_k를 강제 지정 (None이면 data_args에서 결정)
        use_title_override: use_title 강제 지정 (None이면 data_args.use_title 사용)

    Returns:
        Retrieved context가 포함된 Dataset
    """
    # === use_title 결정 로직 ===
    use_title = (
        use_title_override
        if use_title_override is not None
        else getattr(data_args, "use_title", True)
    )

    # === Top-k 결정 로직 ===
    # 1) top_k_override가 지정되면 최우선
    # 2) is_train=True이면 train_top_k_retrieval (없으면 top_k_retrieval)
    # 3) is_train=False이면 infer_top_k_retrieval (없으면 top_k_retrieval)
    if top_k_override is not None:
        effective_top_k = top_k_override
        logger.info(f"🔧 top_k_override specified: {effective_top_k}")
    elif is_train:
        effective_top_k = (
            getattr(data_args, "train_top_k_retrieval", None)
            or data_args.top_k_retrieval
        )
        if getattr(data_args, "train_top_k_retrieval", None):
            logger.info(f"📚 Using train_top_k_retrieval: {effective_top_k}")
    else:
        effective_top_k = (
            getattr(data_args, "infer_top_k_retrieval", None)
            or data_args.top_k_retrieval
        )
        if getattr(data_args, "infer_top_k_retrieval", None):
            logger.info(f"🔍 Using infer_top_k_retrieval: {effective_top_k}")

    logger.info(
        f"🔍 Running retrieval on {split_name} split "
        f"(effective_top_k={effective_top_k}, use_title={use_title})..."
    )

    # use_title=True이면 tokenizer 필수
    if use_title and tokenizer is None:
        logger.warning(
            "⚠️ use_title=True but tokenizer not provided. "
            "Title will NOT be included in context. Pass tokenizer to enable title."
        )
        use_title = False

    if use_title and tokenizer:
        logger.info(
            f"🧩 Title enabled. Format: '{{title}} {tokenizer.sep_token} {{passage}}'"
        )
    else:
        logger.info("📄 Title disabled. Using passage text only.")

    # 1. Retrieval 수행
    # retriever.retrieve는 DataFrame을 반환함
    # tokenizer를 넘기면 title이 포함됨 (BaseRetrieval.retrieve 내부 로직)
    df = retriever.retrieve(
        dataset, topk=effective_top_k, tokenizer=tokenizer if use_title else None
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

    # 5. Training일 경우 Answer Realignment 수행
    if is_train and has_answers:
        logger.info("🔄 Realigning answers in retrieved contexts for training...")
        original_len = len(new_dataset)

        # sep_token 결정 (use_title이면 tokenizer.sep_token 사용)
        sep_token = tokenizer.sep_token if (use_title and tokenizer) else None

        # Answer 위치 재계산 (title-aware)
        def realign_fn(example):
            return realign_answers_in_retrieved_context(
                example, sep_token=sep_token, use_title=use_title
            )

        new_dataset = new_dataset.map(realign_fn)

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
            f"   * Lost examples means the correct answer was NOT found in top-{effective_top_k} retrieved passages."
        )

        # === Opus 피드백: lost_ratio 높을 때 경고 ===
        if lost_ratio > 25.0:
            logger.warning(
                f"⚠️ HIGH LOST RATIO ALERT ({lost_ratio:.1f}% > 25%)\n"
                f"   Consider increasing train_top_k_retrieval to 3 or higher.\n"
                f"   Current: top_k={effective_top_k}"
            )

    return new_dataset
