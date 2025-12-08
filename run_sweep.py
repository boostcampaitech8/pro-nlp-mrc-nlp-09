#!/usr/bin/env python3
"""
WandB Sweep을 활용한 하이퍼파라미터 튜닝 스크립트

사용법:
    # 1. Sweep 생성 (처음 한 번만)
    python run_sweep.py --create --config configs/sweep_config.yaml --project mrc-sweep --entity your-entity

    # 2. Agent 실행 (여러 터미널에서 병렬 실행 가능)
    python run_sweep.py --sweep_id <sweep_id>

    # 또는 한 번에 생성 + 실행
    python run_sweep.py --create --run --config configs/sweep_config.yaml --count 10
"""

import argparse
import os
import sys
import yaml
import wandb
import logging
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed,
)
import evaluate

from src.arguments import DataTrainingArguments, ModelArguments
from src.trainer_qa import QuestionAnsweringTrainer
from src.utils import check_no_error, postprocess_qa_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_config():
    """
    WandB Sweep에서 호출되는 학습 함수
    wandb.config에서 하이퍼파라미터를 받아 학습 진행
    """
    # wandb 초기화 (sweep agent가 자동으로 config 주입)
    with wandb.init() as run:
        config = wandb.config
        
        # 출력 디렉토리 설정 (sweep_id와 run_id 포함)
        sweep_id = os.environ.get("WANDB_SWEEP_ID", "unknown")
        output_dir = f"./outputs/sweep/{sweep_id}/{run.id}"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"🚀 Starting training with config: {dict(config)}")
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Seed 고정
        seed = config.get("seed", 42)
        set_seed(seed)
        
        # 데이터셋 로드
        data_path = config.get("train_dataset_name", "./data/train_dataset")
        datasets = load_from_disk(data_path)
        logger.info(f"📊 Loaded datasets: {datasets}")
        
        # 모델 및 토크나이저 로드
        model_name = config.get("model_name_or_path", "klue/bert-base")
        logger.info(f"🤖 Loading model: {model_name}")
        
        model_config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            config=model_config,
        )
        
        # TrainingArguments 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Sweep에서 탐색하는 하이퍼파라미터
            num_train_epochs=config.get("num_train_epochs", 3),
            learning_rate=config.get("learning_rate", 3e-5),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 32),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            weight_decay=config.get("weight_decay", 0.01),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            
            # 고정 설정
            do_train=True,
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="no",              # 💾 체크포인트 저장 안함 (용량 절약)
            save_total_limit=None,
            load_best_model_at_end=False,    # 저장 안하므로 False
            metric_for_best_model="exact_match",
            greater_is_better=True,
            save_only_model=True,            # 💾 optimizer 제외
            
            # 로깅 설정 (WandB)
            report_to="wandb",
            logging_steps=50,
            logging_first_step=True,
            
            # 성능 최적화
            fp16=True,
            dataloader_num_workers=2,
            seed=seed,
        )
        
        # Data Arguments
        max_seq_length = config.get("max_seq_length", 384)
        doc_stride = config.get("doc_stride", 128)
        
        # 데이터 전처리
        column_names = datasets["train"].column_names
        question_column_name = "question"
        context_column_name = "context"
        answer_column_name = "answers"
        
        pad_on_right = tokenizer.padding_side == "right"
        
        # token_type_ids 사용 여부 결정
        model_type = getattr(model.config, "model_type", "").lower()
        tokenizer_says_it_can = "token_type_ids" in getattr(tokenizer, "model_input_names", [])
        type_vocab_size = getattr(model.config, "type_vocab_size", 0)
        use_token_type_ids = bool(tokenizer_says_it_can and type_vocab_size > 1)
        
        logger.info(f"model_type={model_type}, use_token_type_ids={use_token_type_ids}")
        
        def prepare_train_features(examples):
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=use_token_type_ids,
                padding="max_length",
            )
            
            if not use_token_type_ids and "token_type_ids" in tokenized_examples:
                tokenized_examples.pop("token_type_ids")
            
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
        
        def prepare_validation_features(examples):
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=use_token_type_ids,
                padding="max_length",
            )
            
            if not use_token_type_ids and "token_type_ids" in tokenized_examples:
                tokenized_examples.pop("token_type_ids")
            
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
        
        # 데이터셋 전처리
        train_dataset = datasets["train"].map(
            prepare_train_features,
            batched=True,
            remove_columns=column_names,
            num_proc=4,
        )
        
        eval_dataset = datasets["validation"].map(
            prepare_validation_features,
            batched=True,
            remove_columns=column_names,
            num_proc=4,
        )
        
        # Data Collator
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
        
        # Metric
        metric = evaluate.load("squad")
        
        def post_processing_function(examples, features, predictions, training_args):
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                max_answer_length=30,
                output_dir=training_args.output_dir,
            )
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)
        
        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=p.predictions, references=p.label_ids)
        
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
        )
        
        # 학습
        logger.info("🏃 Starting training...")
        train_result = trainer.train()
        
        # 평가
        logger.info("📊 Running evaluation...")
        metrics = trainer.evaluate()
        
        # WandB에 최종 메트릭 로깅
        wandb.log({
            "final_exact_match": metrics.get("eval_exact_match", 0),
            "final_f1": metrics.get("eval_f1", 0),
            "train_loss": train_result.metrics.get("train_loss", 0),
        })
        
        logger.info(f"✅ Training completed! Metrics: {metrics}")
        
        return metrics


def create_sweep(config_path: str, project: str, entity: str = None) -> str:
    """Sweep 생성"""
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)
    
    # 프로젝트/엔티티 설정
    sweep_config["project"] = project
    if entity:
        sweep_config["entity"] = entity
    
    # Sweep 생성
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    
    print(f"\n{'='*60}")
    print(f"✅ Sweep created successfully!")
    print(f"   Sweep ID: {sweep_id}")
    print(f"   Project: {project}")
    print(f"   Entity: {entity or 'default'}")
    print(f"\n🚀 To start an agent, run:")
    print(f"   python run_sweep.py --sweep_id {sweep_id}")
    print(f"{'='*60}\n")
    
    return sweep_id


def run_agent(sweep_id: str, count: int = None, project: str = None, entity: str = None):
    """Sweep Agent 실행"""
    print(f"\n🤖 Starting sweep agent for: {sweep_id}")
    print(f"   Count: {count if count else 'unlimited'}")
    
    wandb.agent(
        sweep_id,
        function=train_with_config,
        count=count,
        project=project,
        entity=entity,
    )


def main():
    parser = argparse.ArgumentParser(description="WandB Sweep for Hyperparameter Tuning")
    
    # Sweep 생성 옵션
    parser.add_argument("--create", action="store_true", help="Create a new sweep")
    parser.add_argument("--config", type=str, default="configs/sweep_config.yaml",
                        help="Path to sweep config YAML")
    
    # Agent 실행 옵션
    parser.add_argument("--sweep_id", type=str, help="Sweep ID to run agent for")
    parser.add_argument("--run", action="store_true", help="Run agent after creating sweep")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of runs (None for unlimited)")
    
    # WandB 설정
    parser.add_argument("--project", type=str, default="mrc-sweep",
                        help="WandB project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="WandB entity (team or username)")
    
    args = parser.parse_args()
    
    # Sweep 생성
    if args.create:
        sweep_id = create_sweep(args.config, args.project, args.entity)
        
        # 생성 후 바로 실행
        if args.run:
            run_agent(sweep_id, args.count, args.project, args.entity)
    
    # 기존 Sweep에 Agent 실행
    elif args.sweep_id:
        run_agent(args.sweep_id, args.count, args.project, args.entity)
    
    else:
        parser.print_help()
        print("\n💡 Examples:")
        print("  # Create a new sweep")
        print("  python run_sweep.py --create --config configs/sweep_config.yaml --project mrc-sweep")
        print("")
        print("  # Create and run immediately")
        print("  python run_sweep.py --create --run --count 10 --project mrc-sweep")
        print("")
        print("  # Run agent for existing sweep")
        print("  python run_sweep.py --sweep_id abc123 --count 5")


if __name__ == "__main__":
    main()

