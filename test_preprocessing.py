import logging
import shutil
import os
import sys
import random
import numpy as np
import torch
import evaluate
from typing import NoReturn
from train_process import run_mrc

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
 #---- 여기: safe_normalize 함수 그대로 복붙 ----
def safe_normalize(text: str):
    if not isinstance(text, str):
        return text

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[●★■◆▼▲…]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---- normalize 전체 example 적용 ----
def apply_clean(example):
    example["context"] = safe_normalize(example["context"])
    example["question"] = safe_normalize(example["question"])
    return example

data_args = DataTrainingArguments(
    train_dataset_name="./data/data/train_dataset",
    preprocessing_num_workers=2,
    doc_stride=128
)

model_args = ModelArguments(
    model_name_or_path="klue/bert-base"
)

training_args = CustomTrainingArguments(
    output_dir="./test_output",
    do_train=False,
    do_eval=False
)

datasets = load_from_disk("./data/data/train_dataset")

config = AutoConfig.from_pretrained(model_args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path)

run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)
