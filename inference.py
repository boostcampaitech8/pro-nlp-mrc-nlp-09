"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import os
import sys
import logging
from typing import Callable, Dict, List, NoReturn, Optional, Tuple

import evaluate
import numpy as np
from src.arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from src.retrieval import get_retriever, BaseRetrieval
from src.trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from src.utils import (
    check_no_error,
    postprocess_qa_predictions,
    wait_for_gpu_availability,
    get_config,
    get_logger,
    get_model_path,
    load_inference_dataset,
)
from src.utils.retrieval_utils import retrieve_and_build_dataset
from src.retrieval.paths import get_path

logger = get_logger(__name__, logging.INFO)


def load_retrieval_from_cache(
    cache_path: str,
    dataset: Dataset,
    data_args: DataTrainingArguments,
    alpha: float = 0.35,
) -> Dataset:
    """
    캐시된 retrieval 결과를 로드하여 Dataset을 구성합니다.

    Args:
        cache_path: retrieval cache JSONL 경로
        dataset: 원본 dataset (question, answers 등 포함)
        data_args: DataTrainingArguments
        alpha: hybrid score 계산용 BM25 가중치

    Returns:
        context가 retrieval 결과로 대체된 Dataset
    """
    import json
    import numpy as np

    # 캐시 로드
    cache = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            cache[item["id"]] = item

    # Passages corpus 로드 (캐시된 passage_id로 텍스트 조회)
    passages_meta_path = get_path("kure_passages_meta")
    wiki_path = get_path("wiki_corpus")

    if os.path.exists(passages_meta_path):
        passage_texts = []
        with open(passages_meta_path, "r", encoding="utf-8") as f:
            for line in f:
                meta = json.loads(line.strip())
                passage_texts.append(meta["text"])
    else:
        # Fallback: wiki corpus 사용
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        # 중복 제거 후 순서 유지
        unique_texts = {}
        for doc_id, doc_info in wiki.items():
            text = doc_info["text"]
            if text not in unique_texts:
                unique_texts[text] = text
        passage_texts = list(unique_texts.keys())

    # 결과 구성
    result_data = {
        "id": [],
        "question": [],
        "context": [],
        "answers": [] if "answers" in dataset.column_names else None,
    }

    top_k = data_args.top_k_retrieval

    for example in dataset:
        qid = example["id"]
        cache_entry = cache.get(qid)

        if cache_entry is None:
            logger.warning(f"⚠️  Cache miss for question {qid}, using empty context")
            context = ""
        else:
            # Hybrid score 계산 및 정렬
            candidates = cache_entry["retrieved"]

            if candidates:
                bm25_scores = np.array([c["score_bm25"] for c in candidates])
                dense_scores = np.array([c["score_dense"] for c in candidates])

                # Per-query min-max 정규화
                eps = 1e-9
                bm25_n = (bm25_scores - bm25_scores.min()) / (
                    bm25_scores.max() - bm25_scores.min() + eps
                )
                dense_n = (dense_scores - dense_scores.min()) / (
                    dense_scores.max() - dense_scores.min() + eps
                )

                # Hybrid score
                hybrid_scores = alpha * bm25_n + (1 - alpha) * dense_n

                # 정렬 및 top-k 선택
                sorted_indices = np.argsort(hybrid_scores)[::-1][:top_k]

                # Context 구성 (top-k passage concatenation)
                contexts = []
                for idx in sorted_indices:
                    passage_id = candidates[idx]["passage_id"]
                    if passage_id < len(passage_texts):
                        contexts.append(passage_texts[passage_id])

                context = " ".join(contexts)
            else:
                context = ""

        result_data["id"].append(qid)
        result_data["question"].append(example["question"])
        result_data["context"].append(context)
        if result_data["answers"] is not None:
            result_data["answers"].append(
                example.get("answers", {"text": [], "answer_start": []})
            )

    # Dataset 생성
    if result_data["answers"] is not None:
        features = Features(
            {
                "id": Value(dtype="string"),
                "question": Value(dtype="string"),
                "context": Value(dtype="string"),
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string"),
                        "answer_start": Value(dtype="int32"),
                    }
                ),
            }
        )
    else:
        features = Features(
            {
                "id": Value(dtype="string"),
                "question": Value(dtype="string"),
                "context": Value(dtype="string"),
            }
        )
        del result_data["answers"]

    return Dataset.from_dict(result_data, features=features)


# TODO: 현재 제출 파일 생성과 관련된 버그 존재함 (오류)
def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # gpu 사용 가능한지 체크
    wait_for_gpu_availability()

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = get_config(parser)

    # inference_split에 따라 do_eval/do_predict 자동 설정
    # 정책: validation/train은 do_eval만, test는 do_predict만
    inference_split = data_args.inference_split
    if inference_split == "test":
        # test: 정답이 없으므로 predict만 (메트릭 계산 불가)
        # test는 gold context가 없으므로 retrieval 필수
        if not data_args.eval_retrieval:
            raise ValueError(
                "❌ test split에는 gold context가 없으므로 eval_retrieval=True가 필수입니다.\n"
                "💡 config에서 eval_retrieval: true 설정 후 다시 실행하세요."
            )
        training_args.do_eval = False
        training_args.do_predict = True
        logger.info("🎯 Inference mode: TEST (do_predict only, no metrics)")
    else:
        # train/validation: 정답이 있으므로 do_eval만 수행 (메트릭 계산 + predictions 저장)
        training_args.do_eval = True
        training_args.do_predict = False
        logger.info(
            f"🎯 Inference mode: {inference_split.upper()} (do_eval only, with metrics)"
        )

    # 모델 경로 자동 결정 (use_trained_model=True이면 best checkpoint 자동 탐색)
    model_path = get_model_path(model_args, training_args, for_inference=True)
    logger.info(f"📦 Model path: {model_path}")

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    # inference_split에 맞는 데이터셋 로드
    datasets = load_inference_dataset(data_args, inference_split)
    logger.info(f"📊 Dataset loaded: {datasets}")

    # --- TOKENIZER SETUP (Retrieval specific) ---
    from src.utils.tokenization import get_tokenizer
    # model_args.tokenizer_name might be None, so use tokenizer (from AutoTokenizer) as fallback
    retrieval_tokenize_fn = get_tokenizer(
        data_args.retrieval_tokenizer_name, 
        model_tokenizer=AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True) # Re-instantiate or assume `tokenizer` variable is available later?
        # `tokenizer` is instantiated later in this script. Let's move this block AFTER tokenizer instantiation?
        # Or just use the one we are about to create.
    )

    # Validation split일 경우 eval_labels.json 생성 (실험용)
    if inference_split == "validation":
        import json

        labels_path = os.path.join(training_args.output_dir, "eval_labels.json")
        if not os.path.exists(labels_path):
            logger.info("📝 Creating eval_labels.json for validation experiments...")
            labels = {}
            for ex in datasets["validation"]:
                qid = ex["id"]
                answers = ex["answers"]["text"]  # list of answers
                labels[qid] = answers
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ eval_labels.json saved: {labels_path}")

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_path,
        use_fast=True,
    )
    
    # Refresh retrieval tokenizer with the correct model tokenizer if needed
    if data_args.retrieval_tokenizer_name == "auto":
        retrieval_tokenize_fn = tokenizer.tokenize

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
    )
    
    # --- RERANKER SETUP ---
    reranker = None
    if data_args.reranker_name:
        from src.retrieval.reranker import CrossEncoderReranker
        logger.info(f"🚀 Initializing Reranker: {data_args.reranker_name}")
        reranker = CrossEncoderReranker(model_name=data_args.reranker_name)

    # Config 경로 추출 (YAML 사용 시)
    config_path = (
        sys.argv[1] if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml") else None
    )

    # YAML에서 retrieval alpha 가져오기 (캐시 기반 retrieval용)
    retrieval_alpha = 0.35  # 기본값
    if config_path:
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
            retrieval_config = yaml_config.get("retrieval", {})
            retrieval_alpha = retrieval_config.get("alpha", 0.35)
        except Exception:
            pass

    # =========================================================================
    # Test/Non-test 분기: 명확한 정책 분리
    # 캐시가 있으면 캐시 사용, 없으면 실시간 retrieval로 fallback
    # =========================================================================
    if inference_split == "test":
        # TEST 분기: retrieval 필수, compare 불가
        logger.info("📍 TEST branch: retrieval required, no gold context")

        # 캐시 확인
        test_cache_path = get_path("test_cache")
        if os.path.exists(test_cache_path):
            logger.info(f"📦 Using cached retrieval from {test_cache_path}")
            new_test_dataset = load_retrieval_from_cache(
                cache_path=test_cache_path,
                dataset=datasets["validation"],
                data_args=data_args,
                alpha=retrieval_alpha,
            )
            retriever = None
        else:
            logger.info(
                f"⚠️  Cache not found, running live retrieval ({data_args.retrieval_type})"
            )
            retriever = get_retriever(
                retrieval_type=data_args.retrieval_type,
                tokenize_fn=retrieval_tokenize_fn,
                config_path=config_path,
                # Pass BM25 specific args from data_args if they are not in config (but config usually has them)
                # But get_retriever reads from config_path inside if provided.
                # data_args overrides?
                # Actually BM25Retrieval reads from config_path.
                # We should pass parameters directly to ensure CLI args work if used.
                impl=data_args.bm25_impl,
                delta=data_args.bm25_delta,
            )
            retriever.build()

            # Use shared utility for retrieval
            new_test_dataset = retrieve_and_build_dataset(
                retriever=retriever,
                dataset=datasets["validation"],
                data_args=data_args,
                split_name="test",
                is_train=False,
                tokenizer=tokenizer,
                reranker=reranker, # Pass reranker
            )

        datasets = DatasetDict({"validation": new_test_dataset})

        run_mrc(
            data_args=data_args,
            training_args=training_args,
            model_args=model_args,
            datasets=datasets,
            tokenizer=tokenizer,
            model=model,
            inference_split=inference_split,
            retriever=None,
            original_datasets=None,
        )

    else:
        # VALIDATION/TRAIN 분기: retrieval 선택적, compare 가능
        logger.info(
            f"📍 {inference_split.upper()} branch: retrieval optional, gold context available"
        )
        original_datasets = datasets  # compare용 백업 (gold context 보존)
        retriever = None

        if data_args.eval_retrieval:
            # 캐시 경로 결정 (validation/train)
            cache_path = (
                get_path("val_cache")
                if inference_split == "validation"
                else get_path("train_cache")
            )

            if os.path.exists(cache_path) and not reranker: # Skip cache if reranker is used (need raw passages)
                logger.info(f"📦 Using cached retrieval from {cache_path}")
                new_validation_dataset = load_retrieval_from_cache(
                    cache_path=cache_path,
                    dataset=datasets["validation"],
                    data_args=data_args,
                    alpha=retrieval_alpha,
                )
                datasets = DatasetDict({"validation": new_validation_dataset})
            else:
                if reranker:
                    logger.info("⚠️  Reranker enabled: Skipping cache to perform dynamic reranking.")
                else:
                    logger.info(
                        f"⚠️  Cache not found at {cache_path}, running live retrieval ({data_args.retrieval_type})"
                    )
                
                retriever = get_retriever(
                    retrieval_type=data_args.retrieval_type,
                    tokenize_fn=retrieval_tokenize_fn,
                    config_path=config_path,
                    impl=data_args.bm25_impl,
                    delta=data_args.bm25_delta,
                )
                retriever.build()

                # Use shared utility for retrieval
                new_validation_dataset = retrieve_and_build_dataset(
                    retriever=retriever,
                    dataset=datasets["validation"],
                    data_args=data_args,
                    split_name="validation",
                    is_train=False,
                    tokenizer=tokenizer,
                    reranker=reranker, # Pass reranker
                )
                datasets = DatasetDict({"validation": new_validation_dataset})
        else:
            logger.info("📄 eval_retrieval=False: using gold context")

        run_mrc(
            data_args=data_args,
            training_args=training_args,
            model_args=model_args,
            datasets=datasets,
            tokenizer=tokenizer,
            model=model,
            inference_split=inference_split,
            retriever=retriever,
            original_datasets=original_datasets,
        )


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    inference_split: str,
    retriever: Optional[BaseRetrieval] = None,
    original_datasets: Optional[DatasetDict] = None,
) -> NoReturn:
    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 모델 타입에 따라 token_type_ids 지원 여부 자동 판별
    # 핵심: tokenizer가 만들 수 있는가가 아니라, 모델이 받을 수 있는가가 중요
    model_type = getattr(model.config, "model_type", "").lower()
    tokenizer_says_it_can = "token_type_ids" in getattr(
        tokenizer, "model_input_names", []
    )
    type_vocab_size = getattr(model.config, "type_vocab_size", 0)

    # RoBERTa/XLM-R은 type_vocab_size=1 이라 token_type_ids 넣으면 인덱스 에러 발생
    use_token_type_ids = bool(tokenizer_says_it_can and type_vocab_size > 1)

    print(
        f"model_type={model_type} | tokenizer_has_token_type_ids={tokenizer_says_it_can} "
        f"| type_vocab_size={type_vocab_size} | use_token_type_ids={use_token_type_ids}"
    )

    # 오류가 있는지 확인합니다.
    _, max_seq_length = check_no_error(data_args, training_args, datasets, tokenizer)

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples, _use_token_type_ids=use_token_type_ids):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=_use_token_type_ids,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 안전장치: 혹시 token_type_ids가 남아있으면 제거
        if not _use_token_type_ids and "token_type_ids" in tokenized_examples:
            tokenized_examples.pop("token_type_ids")

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    logger.info(f"📊 Validation examples: {len(datasets['validation'])} questions")
    logger.info(
        f"📊 Evaluation spans after tokenization: {len(eval_dataset)} spans (with doc_stride={data_args.doc_stride})"
    )
    logger.info(
        f"📊 Average spans per question: {len(eval_dataset) / len(datasets['validation']):.1f}"
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.

        # inference_split에 따라 prefix 동적 설정
        prefix_map = {"train": "train", "validation": "val", "test": "test"}
        prefix = prefix_map.get(inference_split, "test")

        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
            prefix=prefix,  # train/val/test에 따라 동적 설정
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        # do_eval이 True면 항상 references 포함 (metric 계산 위해)
        if training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )
        elif training_args.do_predict:
            return formatted_predictions

    metric = evaluate.load("squad")

    def compute_metrics(p) -> Dict:
        # post_processing_function이 반환하는 타입에 따라 처리
        if isinstance(p, EvalPrediction):
            # do_eval 모드: EvalPrediction 객체
            predictions = p.predictions
            references = p.label_ids
            return metric.compute(predictions=predictions, references=references)
        else:
            # do_predict 모드: 이미 formatted list (metric 계산 불필요)
            return {}

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    # do_eval과 do_predict 실행 (상호 배타적)
    # - do_eval: trainer.evaluate() → 메트릭 계산 + predictions 저장 (validation/train용)
    # - do_predict: trainer.predict() → predictions만 저장, 메트릭 없음 (test용)

    if training_args.do_eval:
        # Evaluation 실행 (메트릭 계산됨)
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        # 동적 prefix 사용 (train/val/test에 따라 다른 파일명)
        prefix_map = {"train": "train", "validation": "val", "test": "test"}
        eval_prefix = prefix_map.get(inference_split, "test")

        trainer.log_metrics(eval_prefix, metrics)
        trainer.save_metrics(eval_prefix, metrics)

        logger.info(f"📊 Evaluation metrics saved: {eval_prefix}_results.json")
        logger.info("=" * 80)
        logger.info("✅ EVALUATION COMPLETED - Results saved:")
        logger.info(
            f"   📄 predictions_{eval_prefix}.json: {training_args.output_dir}/predictions_{eval_prefix}.json"
        )
        logger.info(
            f"   📄 nbest_predictions_{eval_prefix}.json: {training_args.output_dir}/nbest_predictions_{eval_prefix}.json"
        )
        logger.info(
            f"   📊 {eval_prefix}_pred.csv: {training_args.output_dir}/{eval_prefix}_pred.csv"
        )

        # Validation/Train일 경우 정답 비교 파일 생성
        if inference_split in ["validation", "train"]:
            import json
            import pandas as pd

            # predictions 로드
            pred_path = os.path.join(
                training_args.output_dir, f"predictions_{eval_prefix}.json"
            )
            with open(pred_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)

            # 정답과 예측 비교 데이터 생성
            comparison_data = []
            for ex in datasets["validation"]:
                qid = ex["id"]
                question = ex["question"]
                gold_answers = ex["answers"]["text"]
                pred_answer = predictions.get(qid, "")

                # EM 체크
                is_correct = pred_answer in gold_answers

                comparison_data.append(
                    {
                        "id": qid,
                        "question": question,
                        "gold_answers": " | ".join(gold_answers),  # 여러 정답은 | 구분
                        "prediction": pred_answer,
                        "correct": "✓" if is_correct else "✗",
                    }
                )

            # CSV 저장
            df = pd.DataFrame(comparison_data)
            comparison_csv = os.path.join(
                training_args.output_dir, f"{eval_prefix}_comparison.csv"
            )
            df.to_csv(comparison_csv, index=False, encoding="utf-8-sig")

            logger.info(
                f"   📊 {eval_prefix}_comparison.csv: {comparison_csv} (with gold answers)"
            )

        logger.info("=" * 80)

    elif training_args.do_predict:
        # Prediction만 실행 (메트릭 계산 안 됨, test용)
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json은 postprocess_qa_predictions()에서 이미 저장됨
        # prefix에 따라 파일명 동적 생성
        prefix_map = {"train": "train", "validation": "val", "test": "test"}
        prefix = prefix_map.get(inference_split, "test")

        logger.info("=" * 80)
        logger.info("✅ INFERENCE COMPLETED - Results saved:")
        logger.info(
            f"   📄 predictions_{prefix}.json: {training_args.output_dir}/predictions_{prefix}.json"
        )
        logger.info(
            f"   📄 nbest_predictions_{prefix}.json: {training_args.output_dir}/nbest_predictions_{prefix}.json"
        )
        logger.info(
            f"   📊 {prefix}_pred.csv: {training_args.output_dir}/{prefix}_pred.csv"
        )
        if inference_split == "test":
            logger.info(f"      👉 Use this CSV file for test submission!")
        logger.info("=" * 80)

        # Validation set에서 gold vs retrieval 비교 (옵션)
        if (
            inference_split == "validation"
            and hasattr(data_args, "compare_retrieval")
            and data_args.compare_retrieval
        ):
            compare_gold_vs_retrieval(
                original_datasets=original_datasets,
                retriever=retriever,
                trainer=trainer,
                tokenizer=tokenizer,
                data_args=data_args,
                training_args=training_args,
                prepare_validation_features=prepare_validation_features,
                column_names=column_names,
                predictions=predictions,
            )
        else:
            print(
                "No metric can be presented because there is no correct answer given. Job done!"
            )


def compare_gold_vs_retrieval(
    original_datasets: DatasetDict,
    retriever: Optional[BaseRetrieval],
    trainer: QuestionAnsweringTrainer,
    tokenizer,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    prepare_validation_features: Callable,
    column_names: List[str],
    predictions,
) -> NoReturn:
    """
    Validation set에서 gold context vs retrieval context 성능 비교.

    Args:
        original_datasets: Gold context가 있는 원본 데이터셋 (retrieval 적용 전)
        retriever: Retrieval 객체 (있으면 재사용, 없으면 새로 생성)
        trainer: QuestionAnsweringTrainer 인스턴스
        tokenizer: Tokenizer
        data_args: DataTrainingArguments
        training_args: TrainingArguments
        prepare_validation_features: Feature 전처리 함수
        column_names: 데이터셋 컬럼명 리스트
        predictions: Gold context로 이미 수행된 예측 결과
    """
    import json
    import csv
    from src.retrieval.sparse import SparseRetrieval

    logger.info("")
    logger.info("=" * 80)
    logger.info("🔍 RETRIEVAL COMPARISON MODE")
    logger.info("=" * 80)

    # 1. Gold context 예측 (이미 완료)
    logger.info("1️⃣  Gold context predictions (already done)")
    gold_pred_dict = {
        pred["id"]: pred["prediction_text"] for pred in predictions.predictions
    }

    # 정답 레이블 저장 (original_datasets 사용)
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    val_ref_dict = {
        ex["id"]: ex[answer_column_name] for ex in original_datasets["validation"]
    }

    eval_labels_path = os.path.join(training_args.output_dir, "eval_labels.json")
    with open(eval_labels_path, "w", encoding="utf-8") as f:
        json.dump(val_ref_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"   ✅ Labels saved: {eval_labels_path}")

    # Gold predictions CSV 저장
    eval_pred_gold_path = os.path.join(training_args.output_dir, "eval_pred_gold.csv")
    with open(eval_pred_gold_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for key, value in gold_pred_dict.items():
            writer.writerow([key, value])
    logger.info(f"   ✅ Gold predictions: {eval_pred_gold_path}")

    # 2. Retrieval context 예측
    logger.info("")
    logger.info("2️⃣  Running retrieval for validation set...")

    # Retrieval 객체: 전달받았으면 재사용, 없으면 새로 생성
    if retriever is None:
        logger.info("   Creating new retriever for comparison...")
        config_path = (
            sys.argv[1]
            if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml")
            else None
        )
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            config_path=config_path,
            use_faiss=data_args.use_faiss,
            num_clusters=data_args.num_clusters,
        )
        retriever.build()
    else:
        logger.info("   Reusing existing retriever...")

    # Retrieval 수행 (original_datasets 사용 - gold context 보존된 원본)
    val_questions = original_datasets["validation"]["question"]
    df_retrieved = retriever.retrieve(
        original_datasets["validation"], topk=data_args.top_k_retrieval
    )

    # Retrieved context로 새로운 dataset 생성
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
    val_with_retrieval = Dataset.from_pandas(
        df_retrieved[["id", "question", "context", "answers"]].reset_index(drop=True),
        features=features,
    )

    # Feature 생성
    val_retrieval_dataset = val_with_retrieval.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=["id", "question", "context", "answers"],
        load_from_cache_file=False,
        desc="Preparing validation features with retrieval",
    )

    # Retrieval 예측
    logger.info("   Running predictions with retrieved contexts...")
    val_retrieval_predictions = trainer.predict(
        test_dataset=val_retrieval_dataset, test_examples=val_with_retrieval
    )
    val_retrieval_pred_dict = {
        pred["id"]: pred["prediction_text"]
        for pred in val_retrieval_predictions.predictions
    }

    # Retrieval predictions CSV 저장
    eval_pred_retrieval_path = os.path.join(
        training_args.output_dir, "eval_pred_retrieval.csv"
    )
    with open(eval_pred_retrieval_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for key, value in val_retrieval_pred_dict.items():
            writer.writerow([key, value])
    logger.info(f"   ✅ Retrieval predictions: {eval_pred_retrieval_path}")

    # 3. 자동 비교 실행
    logger.info("")
    logger.info("3️⃣  Comparing gold vs retrieval performance...")

    import subprocess

    comparison_script = "scripts/compare_retrieval.py"
    if os.path.exists(comparison_script):
        result = subprocess.run(
            [sys.executable, comparison_script, training_args.output_dir],
            capture_output=False,
        )
        if result.returncode == 0:
            logger.info("   ✅ Comparison completed successfully!")
        else:
            logger.warning(f"   ⚠️  Comparison failed with code {result.returncode}")
    else:
        logger.warning(f"   ⚠️  Comparison script not found: {comparison_script}")
        logger.info(
            f"   💡 Run manually: python {comparison_script} {training_args.output_dir}"
        )

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
