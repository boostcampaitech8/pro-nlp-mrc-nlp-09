import sys
import os
import logging
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, HfArgumentParser

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from src.retrieval import get_retriever
from src.utils.retrieval_utils import retrieve_and_build_dataset
from src.utils import get_config, get_logger

logger = get_logger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = get_config(parser)

    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Retrieval Type: {data_args.retrieval_type}")
    logger.info(f"Top K: {data_args.top_k_retrieval}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        use_fast=True,
    )

    # Load Datasets
    logger.info(f"Loading dataset from {data_args.train_dataset_name}")
    datasets = load_from_disk(data_args.train_dataset_name)

    # Initialize Retriever
    config_path = (
        sys.argv[1] if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml") else None
    )
    retriever = get_retriever(
        retrieval_type=data_args.retrieval_type,
        tokenize_fn=tokenizer.tokenize,
        config_path=config_path,
    )
    retriever.build()

    new_datasets = DatasetDict()

    # Process Train
    if "train" in datasets:
        logger.info("Processing TRAIN split...")
        new_train = retrieve_and_build_dataset(
            retriever=retriever,
            dataset=datasets["train"],
            data_args=data_args,
            split_name="train",
            is_train=True,
            tokenizer=tokenizer,
        )
        new_datasets["train"] = new_train

    # Process Validation
    if "validation" in datasets:
        logger.info("Processing VALIDATION split...")
        new_val = retrieve_and_build_dataset(
            retriever=retriever,
            dataset=datasets["validation"],
            data_args=data_args,
            split_name="validation",
            is_train=False,
            tokenizer=tokenizer,
        )
        new_datasets["validation"] = new_val

    # Save
    output_path = f"{data_args.train_dataset_name}_retrieved_{data_args.retrieval_type}_top{data_args.top_k_retrieval}"

    logger.info(f"Saving processed dataset to {output_path}")
    new_datasets.save_to_disk(output_path)
    logger.info("✅ Done! You can now use this dataset path in your config.")
    logger.info(f"   train_dataset_name: {output_path}")
    logger.info(f"   train_retrieval: false")
    logger.info(f"   eval_retrieval: false")


if __name__ == "__main__":
    main()
