import logging
import os
import sys
import numpy as np
import torch
import evaluate
from typing import NoReturn
from collections import defaultdict

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk
from trainer_qa import QuestionAnsweringTrainer
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
from utils_qa import check_no_error, postprocess_qa_predictions


logger = logging.getLogger(__name__)


# ============================================================================
# EDA 기반 추가 함수들
# ============================================================================

def remove_duplicates(dataset):
    """
    EDA 결과: 832개 중복 인덱스 발견
    → 학습 전에 중복 제거
    """
    if '__index_level_0__' not in dataset.column_names:
        logger.info("⚠️  __index_level_0__ 컬럼이 없습니다. 중복 제거를 건너뜁니다.")
        return dataset
    
    index_to_samples = defaultdict(list)
    
    for i, idx in enumerate(dataset['__index_level_0__']):
        index_to_samples[idx].append(i)
    
    # 중복된 경우 첫 번째만 유지
    unique_indices = []
    duplicates_removed = 0
    
    for idx, sample_indices in index_to_samples.items():
        unique_indices.append(sample_indices[0])
        if len(sample_indices) > 1:
            duplicates_removed += len(sample_indices) - 1
    
    logger.info(f"✅ 중복 제거 완료: {duplicates_removed}개 제거됨")
    logger.info(f"   원본: {len(dataset)}개 → 정제: {len(unique_indices)}개")
    
    return dataset.select(sorted(unique_indices))


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # dataset을 전처리합니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # ============================================================================
    # EDA 기반 개선된 Train preprocessing
    # ============================================================================
    def prepare_train_features(examples):
        """
        EDA 결과 기반 개선된 전처리:
        - 39.09%가 512 토큰 초과 → stride 최적화 (권장: 128)
        - 평균 답변 길이 6.28자 → 답변 길이 검증 추가
        - 답변 위치: 앞부분 46%, 중간 30%, 뒷부분 23% → 가중치 적용
        """
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "end position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        # ⭐ 추가: 샘플 가중치 저장
        tokenized_examples["sample_weights"] = []
        
        # 통계 수집용
        long_answer_count = 0
        out_of_span_count = 0
        total_processed = 0

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            
            # ⭐ 추가: 기본 가중치 설정
            weight = 1.0
            total_processed += 1

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["sample_weights"].append(weight)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # ⭐ 추가: 답변 길이 검증 (EDA: 평균 6.28자, 최대 83자)
                answer_length = len(answers["text"][0])
                if answer_length > 100:
                    long_answer_count += 1
                    if long_answer_count <= 3:  # 처음 3개만 로그
                        logger.warning(f"⚠️  비정상적으로 긴 답변 발견 (길이: {answer_length})")
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["sample_weights"].append(weight)
                    continue
                
                # ⭐ 추가: 답변 위치 기반 가중치 조정
                # EDA: 앞부분 46%, 중간 30%, 뒷부분 23%
                context_length = len(examples[context_column_name][sample_index])
                relative_position = start_char / context_length if context_length > 0 else 0
                
                if relative_position > 0.66:  # 뒷부분 - 더 어려움
                    weight *= 1.3
                elif relative_position < 0.33:  # 앞부분 - 상대적으로 쉬움
                    weight *= 0.9
                
                # ⭐ 추가: 답변 타입 기반 가중치 조정
                answer_text = answers["text"][0]
                # 인명, 기관/조직 (희귀 타입) - 가중치 증가
                if any(word in answer_text for word in ['대통령', '장관', '총리', '의원', '왕']):
                    weight *= 1.5
                elif any(word in answer_text for word in ['회사', '기업', '대학', '학교', '정부', '위원회']):
                    weight *= 1.5
                # 숫자 (17.28%)
                elif any(char.isdigit() for char in answer_text):
                    weight *= 1.1

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    out_of_span_count += 1
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["sample_weights"].append(weight)
                else:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    
                    # ⭐ 추가: 토큰 단위 답변 길이 검증 및 경고
                    predicted_answer_length = (tokenized_examples["end_positions"][-1] - 
                                              tokenized_examples["start_positions"][-1])
                    
                    if predicted_answer_length > 30:  # EDA: 평균 6.28자 → 토큰으로 약 10개 이하
                        if long_answer_count <= 3:  # 처음 3개만 로그
                            logger.warning(f"⚠️  긴 토큰 답변 발견 (토큰 수: {predicted_answer_length})")
                        long_answer_count += 1
                    
                    tokenized_examples["sample_weights"].append(weight)
        
        # ⭐ 추가: 전처리 통계 로깅
        if total_processed > 0:
            if long_answer_count > 0:
                logger.info(f"📊 전처리 통계: 비정상적으로 긴 답변 {long_answer_count}개 발견 "
                           f"({long_answer_count/total_processed*100:.2f}%)")
            if out_of_span_count > 0:
                logger.info(f"📊 전처리 통계: Span 벗어난 답변 {out_of_span_count}개 발견 "
                           f"({out_of_span_count/total_processed*100:.2f}%)")

        return tokenized_examples

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        
        # ⭐ 추가: 중복 제거
        logger.info("="*80)
        logger.info("🔄 데이터 전처리 시작")
        logger.info("="*80)
        logger.info(f"원본 Train 데이터: {len(train_dataset)}개")
        
        train_dataset = remove_duplicates(train_dataset)
        logger.info(f"중복 제거 후: {len(train_dataset)}개")

        # dataset에서 train feature를 생성합니다.
        logger.info("🔄 토큰화 진행 중...")
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Train tokenization",
        )
        
        logger.info(f"✅ 토큰화 완료: {len(train_dataset)}개 샘플")
        
        # ⭐ 추가: 샘플 가중치 통계 출력
        if 'sample_weights' in train_dataset.column_names:
            weights = np.array(train_dataset['sample_weights'])
            logger.info("="*80)
            logger.info("📊 샘플 가중치 통계")
            logger.info("="*80)
            logger.info(f"평균: {weights.mean():.3f}")
            logger.info(f"최소: {weights.min():.3f}")
            logger.info(f"최대: {weights.max():.3f}")
            logger.info(f"표준편차: {weights.std():.3f}")
            logger.info("="*80)

    # Validation preprocessing
    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Validation tokenization",
        )

    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    
    
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        
        logger.info("="*80)
        logger.info("🚀 모델 학습 시작")
        logger.info("="*80)
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()