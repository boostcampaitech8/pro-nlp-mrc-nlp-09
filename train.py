import logging
import shutil
import os
import sys
import random
import numpy as np
import torch
import evaluate
from typing import NoReturn
from train_process import normalize_text, safe_normalize, apply_clean
from src.arguments import DataTrainingArguments, ModelArguments, CustomTrainingArguments
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
    get_config, to_serializable, print_section,
    get_logger
)

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

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
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

    # gpu 사용 가능한지 체크
    wait_for_gpu_availability()

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

    run_mrc(data_args, training_args, model_args, datasets, tokenizer, model,config)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    config,
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

    # 모델 타입에 따라 token_type_ids 사용 여부 결정
    # RoBERTa, DeBERTa, ELECTRA 등은 token_type_ids를 사용하지 않음
    # 저장된 모델의 경우 config에서 model_type을 확인
    model_type = getattr(config, 'model_type', '').lower()
    model_name_lower = model_args.model_name_or_path.lower()
    use_token_type_ids = not any(
        mt in model_name_lower or mt in model_type
        for mt in ['roberta', 'deberta', 'electra', 'xlm']
    )
    print(f"Model type: {model_type}, use_token_type_ids: {use_token_type_ids}")

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Train preprocessing / 전처리를 진행합니다.

    def prepare_train_features(examples):
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
            return_token_type_ids=use_token_type_ids,  # BERT: True, RoBERTa/DeBERTa/ELECTRA: False
            padding="max_length" if data_args.pad_to_max_length else False,
        )

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
    
        # ==============================================
        # ⭐ Cleaning 적용 (context/question 정규화)
        # ==============================================
        if data_args.apply_cleaning:
            logger.info("🔧 Applying text normalization to TRAIN dataset...")
            train_dataset = train_dataset.map(
                lambda x: {
                    **x,
                    "context": safe_normalize(x["context"]),
                    "question": safe_normalize(x["question"]),
                },
                num_proc=data_args.preprocessing_num_workers,
                desc="Cleaning train dataset"
            )

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation preprocessing
    def prepare_validation_features(examples):
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
            return_token_type_ids=use_token_type_ids,  # BERT: True, RoBERTa/DeBERTa/ELECTRA: False
            padding="max_length" if data_args.pad_to_max_length else False,
        )

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
    if data_args.apply_cleaning:
        logger.info("🔧 Applying text normalization to VALIDATION dataset...")
        eval_dataset = eval_dataset.map(
            lambda x: {
                **x,
                "context": safe_normalize(x["context"]),
                "question": safe_normalize(x["question"]),
            },
            num_proc=data_args.preprocessing_num_workers,
            desc="Cleaning validation dataset"
        )
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

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
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

    trainer.save_model()  # tokenizer까지 함께 저장
    trainer.save_state()

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

    # Evaluation
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

    # 학습에 사용된 yaml config 파일을 output_dir에 복사
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        os.makedirs(training_args.output_dir, exist_ok=True)
        shutil.copy2(sys.argv[1],
                     os.path.join(training_args.output_dir, "config_used.yaml"))


if __name__ == "__main__":
    main()
