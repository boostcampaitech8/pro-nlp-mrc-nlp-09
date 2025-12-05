"""
TAPT (Task-Adaptive Pre-Training) for MRC
Wikipedia 문서나 train 데이터의 context를 사용하여 MLM으로 추가 사전학습

Usage:
    python run_tapt.py \
        --model_name_or_path klue/roberta-base \
        --output_dir ./outputs/tapt_model \
        --num_train_epochs 3
"""

import os
import json
import argparse
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)


def load_wikipedia_corpus(data_path: str = "./data", context_path: str = "wikipedia_documents.json"):
    """Wikipedia 문서에서 텍스트 추출"""
    wiki_path = os.path.join(data_path, context_path)
    
    if not os.path.exists(wiki_path):
        raise FileNotFoundError(f"Wikipedia 문서를 찾을 수 없습니다: {wiki_path}")
    
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    
    # 중복 제거된 텍스트 추출
    texts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print(f"Wikipedia에서 {len(texts)}개의 고유 문서를 로드했습니다.")
    
    return texts


def load_train_corpus(dataset_path: str = "./data/train_dataset"):
    """Train 데이터셋에서 context 추출"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Train 데이터셋을 찾을 수 없습니다: {dataset_path}")
    
    datasets = load_from_disk(dataset_path)
    
    texts = []
    # train과 validation의 context 모두 사용
    for split in ["train", "validation"]:
        if split in datasets:
            texts.extend(datasets[split]["context"])
    
    # 중복 제거
    texts = list(set(texts))
    print(f"Train 데이터셋에서 {len(texts)}개의 고유 context를 로드했습니다.")
    
    return texts


def create_tapt_corpus(
    use_wikipedia: bool = True,
    use_train_data: bool = True,
    data_path: str = "./data",
    dataset_path: str = "./data/train_dataset",
    output_file: str = "./data/tapt_corpus.txt"
):
    """TAPT용 코퍼스 생성"""
    texts = []
    
    if use_wikipedia:
        try:
            wiki_texts = load_wikipedia_corpus(data_path)
            texts.extend(wiki_texts)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    
    if use_train_data:
        try:
            train_texts = load_train_corpus(dataset_path)
            texts.extend(train_texts)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    
    # 중복 제거
    texts = list(set(texts))
    print(f"총 {len(texts)}개의 고유 텍스트를 수집했습니다.")
    
    # 파일로 저장
    with open(output_file, "w", encoding="utf-8") as f:
        for text in texts:
            # 줄바꿈 제거하고 한 줄로
            clean_text = text.replace("\n", " ").strip()
            if clean_text:
                f.write(clean_text + "\n")
    
    print(f"TAPT 코퍼스가 {output_file}에 저장되었습니다.")
    return output_file


def tokenize_function(examples, tokenizer, max_length=512):
    """텍스트 토큰화"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_special_tokens_mask=True,
    )


def main():
    parser = argparse.ArgumentParser(description="TAPT for MRC")
    
    # 모델 관련
    parser.add_argument("--model_name_or_path", type=str, default="klue/roberta-base",
                        help="사전학습할 기본 모델")
    parser.add_argument("--output_dir", type=str, default="./outputs/tapt_model",
                        help="TAPT 모델 저장 경로")
    
    # 데이터 관련
    parser.add_argument("--data_path", type=str, default="./data",
                        help="데이터 경로")
    parser.add_argument("--dataset_path", type=str, default="./data/train_dataset",
                        help="Train 데이터셋 경로")
    parser.add_argument("--use_wikipedia", action="store_true", default=True,
                        help="Wikipedia 문서 사용")
    parser.add_argument("--use_train_data", action="store_true", default=True,
                        help="Train 데이터 context 사용")
    parser.add_argument("--corpus_file", type=str, default="./data/tapt_corpus.txt",
                        help="TAPT 코퍼스 파일 경로")
    
    # 학습 관련
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="학습 에폭 수")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                        help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="학습률")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="최대 시퀀스 길이")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="MLM 마스킹 확률")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="워밍업 비율")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="FP16 사용")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="저장 주기")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="최대 체크포인트 수")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")
    
    args = parser.parse_args()
    
    # 시드 고정
    set_seed(args.seed)
    
    print("=" * 50)
    print("TAPT (Task-Adaptive Pre-Training) 시작")
    print("=" * 50)
    print(f"기본 모델: {args.model_name_or_path}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"에폭 수: {args.num_train_epochs}")
    print("=" * 50)
    
    # 1. 코퍼스 생성
    print("\n[1/4] TAPT 코퍼스 생성 중...")
    corpus_file = create_tapt_corpus(
        use_wikipedia=args.use_wikipedia,
        use_train_data=args.use_train_data,
        data_path=args.data_path,
        dataset_path=args.dataset_path,
        output_file=args.corpus_file,
    )
    
    # 2. 데이터셋 로드
    print("\n[2/4] 데이터셋 로드 중...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    dataset = Dataset.from_dict({"text": texts})
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 3. 토크나이저 및 모델 로드
    print("\n[3/4] 모델 및 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    
    # 토큰화
    print("텍스트 토큰화 중...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )
    
    # Data Collator (MLM용)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    
    # 4. 학습
    print("\n[4/4] TAPT 학습 시작...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        fp16=args.fp16,
        logging_steps=100,
        save_strategy="no",  # 중간 체크포인트 저장 안 함 (디스크 절약)
        prediction_loss_only=True,
        seed=args.seed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 학습 실행
    trainer.train()
    
    # 모델 저장
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 50)
    print("TAPT 완료!")
    print(f"모델이 {args.output_dir}에 저장되었습니다.")
    print("=" * 50)
    print("\n다음 명령어로 MRC 학습에 사용하세요:")
    print(f"python train.py --model_name_or_path {args.output_dir} --output_dir ./outputs/mrc_tapt --do_train --do_eval")


if __name__ == "__main__":
    main()

