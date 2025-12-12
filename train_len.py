import logging
import os
import sys
import random
import numpy as np
import torch
import evaluate
from typing import NoReturn

import wandb  # ← 추가

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
    wait_for_gpu_availability
)


seed = 2024
deterministic = False

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


logger = logging.getLogger(__name__)


def load_mrc_dataset(base_dir: str) -> DatasetDict:
    real_base = os.path.join(base_dir, "data")

    train_dataset_path = os.path.join(real_base, "train_dataset")
    test_dataset_path = os.path.join(real_base, "test_dataset")

    print("\n📂 Loading dataset:")
    print(f" - Train dataset directory: {train_dataset_path}")
    print(f" - Test dataset directory: {test_dataset_path}")

    train_valid = load_from_disk(train_dataset_path)
    test = load_from_disk(test_dataset_path)

    datasets = DatasetDict({
        "train": train_valid["train"],
        "validation": train_valid["validation"],
        "test": test["validation"],
    })

    print("📌 Loaded Keys:", datasets.keys())
    return datasets


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    wait_for_gpu_availability()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    datasets = load_mrc_dataset(data_args.dataset_name)

    # ---------- wandb 설정 ----------
    training_args.report_to = ["wandb"]
    training_args.logging_steps = 100
    training_args.evaluation_strategy = "epoch"   # ← eval 중간 호출 방지

    # ---------- 실험용 max_length 강제설정 ----------
    if data_args.max_seq_length is None or data_args.max_seq_length < 512:
        data_args.max_seq_length = 512
    print(f"🔧 Using max_seq_length = {data_args.max_seq_length}")
    # --------------------------------------

    # 모델 로드
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args,
    training_args,
    model_args,
    datasets,
    tokenizer,
    model,
):

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    pad_on_right = tokenizer.padding_side == "right"

    # ---------- check_no_error가 max_length를 읽어감 ----------
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
    # ----------------------------------------------------------

    # --- Train preprocessing ---
    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name] if pad_on_right else examples[context_column_name],
            examples[context_column_name] if pad_on_right else examples[question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=datasets["train"].column_names,
        )

    # --- Validation preprocessing ---
    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name],
            examples[context_column_name],
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
                offset if sequence_ids[k] == context_index else None
                for k, offset in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if training_args.do_eval:
        eval_dataset = datasets["validation"].map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=datasets["validation"].column_names,
        )

    data_collator = DataCollatorWithPadding(tokenizer)

    def post_processing_function(examples, features, predictions, training_args):
        predictions = postprocess_qa_predictions(
            examples, features, predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        if training_args.do_eval:
            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

        return formatted_predictions

    metric = evaluate.load("squad")

    def compute_metrics(p):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

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

    # ---------- Training ----------
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    # ---------- Evaluation ----------
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
