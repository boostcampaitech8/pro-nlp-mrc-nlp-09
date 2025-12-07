import logging
import shutil
import os
import sys
import random
import numpy as np
import torch
import evaluate
from typing import NoReturn

from src.arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk
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
    to_serializable,
    print_section,
    get_logger,
)
from src.utils.metrics_tracker import MetricsTracker
from src.utils.evaluator import (
    FinalEvaluator,
    save_predictions,
    save_detailed_results,
)
from src.utils.analysis import save_prediction_analysis

seed = 2024
deterministic = False

random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정
torch.manual_seed(seed)  # torch random seed 고정
torch.cuda.manual_seed_all(seed)
if deterministic:  # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logger = get_logger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # gpu 사용 가능한지 체크
    wait_for_gpu_availability()

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = get_config(parser)

    #
    training_args.do_train = True
    training_args.do_eval = True

    # train.py는 "학습 전용" 스크립트로 사용
    if not training_args.do_train:
        raise ValueError(
            "train.py는 학습 전용 스크립트입니다. "
            "TrainingArguments.do_train=True로 설정한 YAML을 사용하세요."
        )

    logger.info("model is from: %s", model_args.model_name_or_path)
    logger.info("data is from: %s", data_args.train_dataset_name)
    logger.info("output_dir is: %s", training_args.output_dir)

    # 현재 사용 중인 arguments를 한 번에 로그로 남겨두기
    print_section("Model Arguments", model_args)
    print_section("Data Training Arguments", data_args)
    print("Trainging Arguments:")
    print(f"output_dir: {training_args.output_dir})")
    print(f"num_train_epochs: {training_args.num_train_epochs}")
    print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print(f"per_device_eval_batch_size: {training_args.per_device_eval_batch_size}")
    print(f"learning_rate: {training_args.learning_rate}")
    print(f"warmup_ratio: {training_args.warmup_ratio}")
    print(f"weight_decay: {training_args.weight_decay}")
    print(f"logging_steps: {training_args.logging_steps}")
    print(f"logging_first_step: {training_args.logging_first_step}")
    print(f"evaluation_strategy: {training_args.evaluation_strategy}")
    print(f"save_strategy: {training_args.save_strategy}")
    print(f"save_total_limit: {training_args.save_total_limit}")
    print(f"load_best_model_at_end: {training_args.load_best_model_at_end}")
    print(f"metric_for_best_model: {training_args.metric_for_best_model}")
    print(f"greater_is_better: {training_args.greater_is_better}")
    print(f"fp16: {training_args.fp16}")
    print(f"dataloader_num_workers: {training_args.dataloader_num_workers}")
    print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    print(f"report_to: {training_args.report_to}")

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.train_dataset_name)
    logger.info("load datasets: \n", datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    logger.info(
        f"training_args type: {type(training_args)}, "
        f"model_args type: {type(model_args)}, "
        f"datasets type: {type(datasets)}, "
        f"tokenizer type: {type(tokenizer)}, "
        f"model type: {type(model)}"
    )

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
    use_return_token_type_ids = bool(tokenizer_says_it_can and type_vocab_size > 1)

    print(
        f"model_type={model_type} | tokenizer_has_token_type_ids={tokenizer_says_it_can} "
        f"| type_vocab_size={type_vocab_size} | use_return_token_type_ids={use_return_token_type_ids}"
    )

    # 오류가 있는지 확인합니다. (checkpoint는 무시, max_seq_length만 사용)
    _, max_seq_length = check_no_error(data_args, training_args, datasets, tokenizer)

    # Train preprocessing / 전처리를 진행합니다.

    def prepare_train_features(examples, _use_token_type_ids=use_return_token_type_ids):
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
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation preprocessing
    def prepare_validation_features(
        examples, _use_token_type_ids=use_return_token_type_ids
    ):
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

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
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

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        # 항상 EvalPrediction 반환
        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]}
            for ex in datasets["validation"]
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad")
    logger.info("---- metric loaded: %s ----", metric)

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Metrics Tracker 초기화
    metrics_tracker = MetricsTracker(output_dir=training_args.output_dir)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        callbacks=[metrics_tracker],  # Metrics Tracker 추가
    )

    # Training (fresh run 방식으로 수정; 필요하면 YAML에 resume_from_checkpoint 명시)
    logger.info(
        "Starting training: model=%s, output_dir=%s",
        model_args.model_name_or_path,
        training_args.output_dir,
    )

    train_result = trainer.train(
        resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None)
    )

    logger.info("Training completed.")
    logger.info("Saving model to %s", training_args.output_dir)
    logger.info(f"최종 훈련 결과: {train_result.metrics}")

    # 모델 저장 (safetensors는 자동으로 처리됨)
    trainer.save_model()  # tokenizer까지 함께 저장
    trainer.save_state()

    # ✅ Best checkpoint 경로 명시적으로 저장 (inference에서 사용)
    if trainer.state.best_model_checkpoint:
        best_checkpoint_path = os.path.join(
            training_args.output_dir, "best_checkpoint_path.txt"
        )
        with open(best_checkpoint_path, "w") as f:
            f.write(trainer.state.best_model_checkpoint)
        logger.info(f"✅ Best checkpoint saved: {trainer.state.best_model_checkpoint}")
        logger.info(
            f"   Best metric ({training_args.metric_for_best_model}): {trainer.state.best_metric}"
        )
    else:
        logger.warning(
            "⚠️  No best checkpoint found (load_best_model_at_end might be False)"
        )

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

    # State 저장
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )

    # Evaluation - try-except로 감싸서 평가 실패해도 학습 결과는 보존
    try:
        logger.info(
            "Running final evaluation on validation set (%d examples)",
            len(eval_dataset),
        )
        logger.info(f"Best metric: {trainer.state.best_metric}")
        logger.info(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")

        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    except Exception as e:
        logger.warning(f"⚠️  Final evaluation failed: {e}")
        logger.warning("   Best checkpoint is already saved in output directory")

    # 학습 요약 출력
    metrics_tracker.print_summary()

    # 최종 성능 평가 (train + validation)
    # 주의: 이 부분이 실패해도 위에서 이미 best checkpoint는 저장됨
    try:
        logger.info("=" * 80)
        logger.info("Running final performance evaluation on all splits...")
        logger.info("=" * 80)

        final_evaluator = FinalEvaluator(output_dir=training_args.output_dir)

        # 1. Train set 평가 (validation 형식으로 변환 필요)
        logger.info("Evaluating on TRAIN set...")
        train_dataset_for_eval = datasets["train"].map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,  # 캐시 충돌 방지: 여러 모델 연달아 실행 시 필수
            desc="Preparing train features for evaluation",
        )
        train_predictions = trainer.predict(
            test_dataset=train_dataset_for_eval, test_examples=datasets["train"]
        )
        # predictions는 리스트 형태 [{"id": ..., "prediction_text": ...}, ...]
        train_pred_dict = {
            pred["id"]: pred["prediction_text"]
            for pred in train_predictions.predictions
        }
        train_ref_dict = {ex["id"]: ex[answer_column_name] for ex in datasets["train"]}

        # train에서는 retrieval 사용 안함
        final_evaluator.evaluate_split(
            predictions=train_pred_dict,
            references=train_ref_dict,
            split_name="train",
            with_retrieval=False,
        )
        save_predictions(train_pred_dict, training_args.output_dir, "train")
        # 사후 분석을 위한 confidence 정보 저장 (detailed_results보다 먼저 실행)
        save_prediction_analysis(
            train_predictions,
            datasets["train"],
            training_args.output_dir,
            "train",
            answer_column_name,
        )
        save_detailed_results(
            train_pred_dict, datasets["train"], training_args.output_dir, "train"
        )

        # 2. Validation set 평가 (이미 평가됨, 결과 저장만)
        logger.info("Evaluating on VALIDATION set (gold context)...")
        val_predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )
        # predictions는 리스트 형태 [{"id": ..., "prediction_text": ...}, ...]
        val_pred_dict = {
            pred["id"]: pred["prediction_text"] for pred in val_predictions.predictions
        }
        val_ref_dict = {
            ex["id"]: ex[answer_column_name] for ex in datasets["validation"]
        }

        final_evaluator.evaluate_split(
            predictions=val_pred_dict,
            references=val_ref_dict,
            split_name="validation",
            with_retrieval=False,
        )
        save_predictions(val_pred_dict, training_args.output_dir, "val")
        # 사후 분석을 위한 confidence 정보 저장 (detailed_results보다 먼저 실행)
        save_prediction_analysis(
            val_predictions,
            datasets["validation"],
            training_args.output_dir,
            "val",
            answer_column_name,
        )
        save_detailed_results(
            val_pred_dict, datasets["validation"], training_args.output_dir, "val"
        )

        # eval_pred_gold.csv 저장 (gold context 사용한 validation 예측)
        import csv

        eval_pred_gold_path = os.path.join(
            training_args.output_dir, "eval_pred_gold.csv"
        )
        with open(eval_pred_gold_path, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for key, value in val_pred_dict.items():
                writer.writerow([key, value])
        logger.info(
            f"✅ Validation predictions (gold context) saved to {eval_pred_gold_path}"
        )

        # 정답 레이블 저장 (스코어링용)
        import json

        eval_labels_path = os.path.join(training_args.output_dir, "eval_labels.json")
        with open(eval_labels_path, "w", encoding="utf-8") as f:
            json.dump(val_ref_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Validation labels saved to {eval_labels_path}")

        # 3. 최종 summary 저장 및 출력
        final_evaluator.save_summary()
        final_evaluator.print_summary()
        logger.info("✅ Final performance evaluation completed successfully")

    except Exception as e:
        logger.warning(f"⚠️  Final evaluation failed (but model is already saved): {e}")
        logger.warning(
            "   Best checkpoint and metrics are preserved in output directory"
        )

    # Best checkpoint 경로를 파일로 저장 (inference에서 자동 로드용) - 이미 위에서 저장했으므로 제거 가능
    # (중복 방지: 이미 trainer.save_state() 직후에 저장됨)

    # 학습에 사용된 yaml config 파일을 output_dir에 복사
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        config_path = sys.argv[1]
        if os.path.exists(config_path):
            os.makedirs(training_args.output_dir, exist_ok=True)
            shutil.copy2(
                config_path, os.path.join(training_args.output_dir, "config_used.yaml")
            )
        else:
            logger.warning(f"⚠️  Config file not found: {config_path}")


if __name__ == "__main__":
    main()
