"""
사용 예시:
    # Test 데이터셋 사용 (기본)
    python ensemble.py --output_dir ./outputs/ensemble/test
    
    # Train 데이터셋의 validation 셋 사용
    python ensemble.py --use_train_validation --train_dataset ./data/train_dataset --output_dir ./outputs/ensemble/validation --no_retrieval
    
    # 커맨드라인에서 모델 경로 지정
    python ensemble.py --model_paths ./outputs/model1 ./outputs/model2 --weights 0.6 0.4
"""

import os
import csv
import argparse
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict, Dataset, Features, Value, Sequence
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.retrieval.weighted_hybrid import WeightedHybridRetrieval
from src.utils.qa import postprocess_qa_predictions


ENSEMBLE_MODELS = [
    ("./outputs/taewon/oceann2", 1.0),
    ("./outputs/taewon/roberta2", 1.0),
    ("./outputs/taewon/hanteck2", 1.0),
    ("./outputs/taewon/uomnf2", 1.0),
]



@dataclass
class EnsembleConfig:
    """앙상블 설정"""
    model_paths: Optional[List[str]] = None  # 모델 경로 리스트 
    weights: Optional[List[float]] = None  # 모델별 가중치
    output_dir: str = "./outputs/taewon/ensemble/3_3_3_1"  # 결과 저장 경로
    test_dataset_path: str = "./data/test_dataset"  # 테스트 데이터셋 경로
    train_dataset_path: Optional[str] = None  # 학습 데이터셋 경로 (validation 셋 사용 시)
    use_train_validation: bool = False  # train_dataset의 validation 셋 사용 여부
    max_seq_length: int = 384
    doc_stride: int = 128
    max_answer_length: int = 30
    top_k_retrieval: int = 10
    batch_size: int = 16
    use_retrieval: bool = True
    retrieval_alpha: float = 0.35
    corpus_emb_path: str = "./data/embeddings/kure_corpus_emb.npy"
    passages_meta_path: str = "./data/embeddings/kure_passages_meta.jsonl"


class MRCEnsemble:
    """MRC 모델 앙상블 클래스"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = []
        self.tokenizers = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 가중치 설정 (None이면 균등 분배)
        if config.weights is None:
            self.weights = [1.0 / len(config.model_paths)] * len(config.model_paths)
        else:
            # 가중치 정규화
            total = sum(config.weights)
            self.weights = [w / total for w in config.weights]
        
    
    def load_models(self):
        """모든 모델 로드"""
        for model_path in self.config.model_paths:
            config = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoModelForQuestionAnswering.from_pretrained(model_path, config=config)
            model.to(self.device)
            model.eval()
            self.models.append(model)
            self.tokenizers.append(tokenizer)
    
    def load_dataset(self) -> DatasetDict:
        """데이터셋 로드 (test 또는 train의 validation)"""
        if self.config.use_train_validation:
            train_datasets = load_from_disk(self.config.train_dataset_path)
            datasets = DatasetDict({"validation": train_datasets["validation"]})
        else:
            datasets = load_from_disk(self.config.test_dataset_path)
        return datasets
    
    def run_retrieval(self, datasets: DatasetDict) -> DatasetDict:
        """Weighted Hybrid Retrieval 수행 (BM25 + KURE)"""
        if not self.config.use_retrieval:
            return datasets
        
        tokenizer = self.tokenizers[0]
        corpus_emb_path = self.config.corpus_emb_path or "./data/embeddings/kure_corpus_emb.npy"
        passages_meta_path = self.config.passages_meta_path or "./data/embeddings/kure_passages_meta.jsonl"
        
        retriever = WeightedHybridRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents_normalized.json",
            corpus_emb_path=corpus_emb_path,
            passages_meta_path=passages_meta_path,
            alpha=self.config.retrieval_alpha,
        )
        retriever.build()
        
        df = retriever.retrieve(datasets["validation"], topk=self.config.top_k_retrieval)
        
        features_dict = {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
        
        if "original_context" in df.columns:
            features_dict["original_context"] = Value(dtype="string", id=None)
        if "answers" in df.columns:
            features_dict["answers"] = Sequence(
                feature={
                    "text": Value(dtype="string", id=None),
                    "answer_start": Value(dtype="int32", id=None),
                },
                length=-1,
                id=None,
            )
        
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=Features(features_dict))})
        return datasets
    
    def prepare_features(self, examples, tokenizer):
        """토큰화 및 feature 생성"""
        pad_on_right = tokenizer.padding_side == "right"
        
        tokenized = tokenizer(
            examples["question"] if pad_on_right else examples["context"],
            examples["context"] if pad_on_right else examples["question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.config.max_seq_length,
            stride=self.config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []
        
        for i in range(len(tokenized["input_ids"])):
            sequence_ids = tokenized.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])
            
            tokenized["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        
        return tokenized
    
    def get_logits_from_model(
        self, 
        model, 
        tokenizer, 
        dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """단일 모델에서 logits 추출"""
        
        # Feature 준비
        features = dataset.map(
            lambda x: self.prepare_features(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
        )
        
        # token_type_ids 처리
        model_type = getattr(model.config, "model_type", "").lower()
        type_vocab_size = getattr(model.config, "type_vocab_size", 0)
        use_token_type_ids = type_vocab_size > 1
        
        # DataLoader 준비
        data_collator = DataCollatorWithPadding(tokenizer)
        
        all_start_logits = []
        all_end_logits = []
        
        # 배치 처리
        batch_size = self.config.batch_size
        
        for i in tqdm(range(0, len(features), batch_size), desc="Inference"):
            batch_indices = range(i, min(i + batch_size, len(features)))
            
            # 배치 데이터 준비
            batch = {
                "input_ids": torch.tensor([features[j]["input_ids"] for j in batch_indices]),
                "attention_mask": torch.tensor([features[j]["attention_mask"] for j in batch_indices]),
            }
            
            if use_token_type_ids and "token_type_ids" in features.column_names:
                batch["token_type_ids"] = torch.tensor(
                    [features[j]["token_type_ids"] for j in batch_indices]
                )
            
            # GPU로 이동
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 추론
            with torch.no_grad():
                outputs = model(**batch)
            
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())
        
        start_logits = np.concatenate(all_start_logits, axis=0)
        end_logits = np.concatenate(all_end_logits, axis=0)
        
        return start_logits, end_logits, features
    
    def ensemble_logits(
        self, 
        all_start_logits: List[np.ndarray], 
        all_end_logits: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted sum으로 logits 앙상블"""
        ensembled_start = np.zeros_like(all_start_logits[0])
        ensembled_end = np.zeros_like(all_end_logits[0])
        
        for i, (start, end) in enumerate(zip(all_start_logits, all_end_logits)):
            ensembled_start += self.weights[i] * start
            ensembled_end += self.weights[i] * end
        
        return ensembled_start, ensembled_end
    
    def run(self):
        """앙상블 실행"""
        self.load_models()
        datasets = self.load_dataset()
        datasets = self.run_retrieval(datasets)
        
        all_start_logits = []
        all_end_logits = []
        features = None
        
        for model, tokenizer in zip(self.models, self.tokenizers):
            start_logits, end_logits, features = self.get_logits_from_model(
                model, tokenizer, datasets["validation"]
            )
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)
        
        ensembled_start, ensembled_end = self.ensemble_logits(all_start_logits, all_end_logits)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        prefix = "ensemble_validation" if self.config.use_train_validation else "ensemble"
        
        predictions = postprocess_qa_predictions(
            examples=datasets["validation"],
            features=features,
            predictions=(ensembled_start, ensembled_end),
            max_answer_length=self.config.max_answer_length,
            output_dir=self.config.output_dir,
            prefix=prefix,
        )
        
        csv_filename = "ensemble_predictions_validation.csv" if self.config.use_train_validation else "ensemble_predictions.csv"
        csv_path = os.path.join(self.config.output_dir, csv_filename)
        with open(csv_path, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for key, value in predictions.items():
                writer.writerow([key, value])
        
        print(f"\n✅ Ensemble complete!")
        dataset_type = "validation" if self.config.use_train_validation else "test"
        print(f"   📊 Dataset type: {dataset_type}")
        print(f"   📄 Predictions: {os.path.join(self.config.output_dir, f'predictions_{prefix}.json')}")
        print(f"   📄 CSV: {csv_path}")
        
        return predictions


def main():
    # EnsembleConfig의 기본값 가져오기
    defaults = {f.name: f.default for f in EnsembleConfig.__dataclass_fields__.values()}
    
    parser = argparse.ArgumentParser(description="MRC Model Ensemble")
    parser.add_argument("--model-paths", nargs="+", default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])
    parser.add_argument("--test-dataset", type=str, default=defaults["test_dataset_path"])
    parser.add_argument("--train-dataset", type=str, default=defaults["train_dataset_path"])
    parser.add_argument("--use-train-validation", action="store_true", default=defaults["use_train_validation"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k_retrieval"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--no-retrieval", action="store_true", default=False)
    parser.add_argument("--retrieval-alpha", type=float, default=defaults["retrieval_alpha"])
    parser.add_argument("--corpus-emb-path", type=str, default=defaults["corpus_emb_path"])
    parser.add_argument("--passages-meta-path", type=str, default=defaults["passages_meta_path"])
    
    args = parser.parse_args()
    
    model_paths = args.model_paths if args.model_paths else [path for path, _ in ENSEMBLE_MODELS]
    weights = args.weights if args.weights else [weight for _, weight in ENSEMBLE_MODELS]
    
    config = EnsembleConfig(
        model_paths=model_paths,
        weights=weights,
        output_dir=args.output_dir,
        test_dataset_path=args.test_dataset if not args.use_train_validation else None,
        train_dataset_path=args.train_dataset if args.use_train_validation else None,
        use_train_validation=args.use_train_validation,
        top_k_retrieval=args.top_k,
        batch_size=args.batch_size,
        use_retrieval=not args.no_retrieval,
        retrieval_alpha=args.retrieval_alpha,
        corpus_emb_path=args.corpus_emb_path,
        passages_meta_path=args.passages_meta_path,
    )
    
    ensemble = MRCEnsemble(config)
    ensemble.run()


if __name__ == "__main__":
    main()


