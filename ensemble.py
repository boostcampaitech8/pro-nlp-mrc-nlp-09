"""
MRC 모델 앙상블 (Soft Voting with Weighted Sum)

여러 학습된 모델의 start/end logits를 weighted sum하여 앙상블 수행
"""

import os
import json
import csv
import argparse
from typing import List, Dict, Tuple, Optional
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


# ============================================================
# 🎯 여기서 앙상블할 모델들을 설정하세요!
# ============================================================
ENSEMBLE_MODELS = [
    # (모델 경로, 가중치)
    # 가중치는 자동으로 정규화됩니다 (합이 1이 되도록)
    ("./outputs/taewon/oceann315", 1.0),
    ("./outputs/taewon/roberta-large", 1.0),
    # ("./outputs/dahyeong/model", 0.5),
    
    # 💡 가중치 예시:
    # - 균등: 모두 1.0
    # - 성능 기반: EM 점수에 비례 (예: 75점 → 0.75, 80점 → 0.80)
    # - 수동 조절: 원하는 비율로 설정
]
# ============================================================


@dataclass
class EnsembleConfig:
    """앙상블 설정"""
    model_paths: List[str]          # 모델 경로 리스트
    weights: Optional[List[float]]  # 모델별 가중치 (None이면 균등)
    output_dir: str                 # 결과 저장 경로
    test_dataset_path: str          # 테스트 데이터셋 경로
    max_seq_length: int = 384
    doc_stride: int = 128
    max_answer_length: int = 30
    top_k_retrieval: int = 10
    batch_size: int = 16
    use_retrieval: bool = True
    retrieval_alpha: float = 0.35  # WeightedHybridRetrieval의 BM25 가중치 (base.yaml과 동일)
    corpus_emb_path: Optional[str] = "./data/embeddings/kure_corpus_emb.npy"  # KURE corpus embedding 경로
    passages_meta_path: Optional[str] = "./data/embeddings/kure_passages_meta.jsonl"  # KURE passages meta 경로


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
        
        print(f"🔧 Device: {self.device}")
        print(f"📊 Model weights: {self.weights}")
    
    def load_models(self):
        """모든 모델 로드"""
        print("\n" + "=" * 60)
        print("📦 Loading models for ensemble...")
        print("=" * 60)
        
        for i, model_path in enumerate(self.config.model_paths):
            print(f"\n[{i+1}/{len(self.config.model_paths)}] Loading: {model_path}")
            
            # 모델과 토크나이저 로드
            config = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoModelForQuestionAnswering.from_pretrained(model_path, config=config)
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            self.tokenizers.append(tokenizer)
            
            print(f"   ✅ Loaded: {config.model_type}")
        
        print(f"\n✅ Total {len(self.models)} models loaded!")
    
    def load_dataset(self) -> DatasetDict:
        """테스트 데이터셋 로드"""
        print(f"\n📂 Loading dataset from: {self.config.test_dataset_path}")
        datasets = load_from_disk(self.config.test_dataset_path)
        print(f"   Dataset: {datasets}")
        return datasets
    
    def run_retrieval(self, datasets: DatasetDict) -> DatasetDict:
        """Weighted Hybrid Retrieval 수행 (BM25 + KURE)"""
        if not self.config.use_retrieval:
            return datasets
        
        print("\n🔍 Running Weighted Hybrid Retrieval (BM25 + KURE)...")
        
        # 첫 번째 토크나이저 사용
        tokenizer = self.tokenizers[0]
        
        # 기본 경로 설정 (base.yaml과 동일)
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
        
        df = retriever.retrieve(
            datasets["validation"], 
            topk=self.config.top_k_retrieval
        )
        
        # DataFrame을 Dataset으로 변환
        f = Features({
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        })
        
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        print(f"   ✅ Retrieval complete: {len(datasets['validation'])} examples")
        
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
        
        print("\n🎯 Ensembling logits with weighted sum...")
        
        # Weighted sum
        ensembled_start = np.zeros_like(all_start_logits[0])
        ensembled_end = np.zeros_like(all_end_logits[0])
        
        for i, (start, end) in enumerate(zip(all_start_logits, all_end_logits)):
            ensembled_start += self.weights[i] * start
            ensembled_end += self.weights[i] * end
            print(f"   Model {i+1}: weight={self.weights[i]:.3f}")
        
        return ensembled_start, ensembled_end
    
    def run(self):
        """앙상블 실행"""
        print("\n" + "=" * 60)
        print("🚀 MRC Ensemble (Soft Voting)")
        print("=" * 60)
        
        # 1. 모델 로드
        self.load_models()
        
        # 2. 데이터셋 로드
        datasets = self.load_dataset()
        
        # 3. Retrieval 수행
        datasets = self.run_retrieval(datasets)
        
        # 4. 각 모델에서 logits 추출
        print("\n" + "=" * 60)
        print("📊 Extracting logits from each model...")
        print("=" * 60)
        
        all_start_logits = []
        all_end_logits = []
        features = None
        
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            print(f"\n[Model {i+1}/{len(self.models)}]")
            start_logits, end_logits, features = self.get_logits_from_model(
                model, tokenizer, datasets["validation"]
            )
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)
            print(f"   Logits shape: start={start_logits.shape}, end={end_logits.shape}")
        
        # 5. 앙상블
        ensembled_start, ensembled_end = self.ensemble_logits(
            all_start_logits, all_end_logits
        )
        
        # 6. 후처리 및 답변 생성
        print("\n" + "=" * 60)
        print("📝 Post-processing predictions...")
        print("=" * 60)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        predictions = postprocess_qa_predictions(
            examples=datasets["validation"],
            features=features,
            predictions=(ensembled_start, ensembled_end),
            max_answer_length=self.config.max_answer_length,
            output_dir=self.config.output_dir,
            prefix="ensemble",
        )
        
        # 7. CSV 저장
        csv_path = os.path.join(self.config.output_dir, "ensemble_predictions.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for key, value in predictions.items():
                writer.writerow([key, value])
        
        print(f"\n✅ Ensemble complete!")
        print(f"   📄 Predictions: {os.path.join(self.config.output_dir, 'predictions_ensemble.json')}")
        print(f"   📄 CSV: {csv_path}")
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description="MRC Model Ensemble")
    parser.add_argument(
        "--model_paths", 
        nargs="+", 
        default=None,
        help="학습된 모델 경로들 (미지정시 ENSEMBLE_MODELS 사용)"
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="모델별 가중치 (미지정시 ENSEMBLE_MODELS 또는 균등 분배)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/taewon/ensemble",
        help="결과 저장 경로"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="./data/test_dataset",
        help="테스트 데이터셋 경로"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Retrieval top-k"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="배치 사이즈"
    )
    parser.add_argument(
        "--no_retrieval",
        action="store_true",
        help="Retrieval 사용 안함 (validation용)"
    )
    parser.add_argument(
        "--retrieval_alpha",
        type=float,
        default=0.35,
        help="WeightedHybridRetrieval의 BM25 가중치 (0~1, 기본값: 0.35, base.yaml과 동일)"
    )
    parser.add_argument(
        "--corpus_emb_path",
        type=str,
        default="./data/embeddings/kure_corpus_emb.npy",
        help="KURE corpus embedding 경로 (기본값: ./data/embeddings/kure_corpus_emb.npy, base.yaml과 동일)"
    )
    parser.add_argument(
        "--passages_meta_path",
        type=str,
        default="./data/embeddings/kure_passages_meta.jsonl",
        help="KURE passages meta 경로 (기본값: ./data/embeddings/kure_passages_meta.jsonl, base.yaml과 동일)"
    )
    
    args = parser.parse_args()
    
    # 모델 경로와 가중치 결정
    if args.model_paths is not None:
        # 커맨드라인에서 지정한 경우
        model_paths = args.model_paths
        weights = args.weights
    else:
        # ENSEMBLE_MODELS에서 가져오기
        if not ENSEMBLE_MODELS:
            raise ValueError(
                "❌ 앙상블할 모델이 없습니다!\n"
                "💡 ensemble.py 상단의 ENSEMBLE_MODELS에 모델을 추가하거나\n"
                "   --model_paths 인자를 사용하세요."
            )
        model_paths = [path for path, _ in ENSEMBLE_MODELS]
        weights = [weight for _, weight in ENSEMBLE_MODELS]
    
    print("\n" + "=" * 60)
    print("📋 Ensemble Configuration")
    print("=" * 60)
    for i, (path, w) in enumerate(zip(model_paths, weights or [1.0]*len(model_paths))):
        print(f"   [{i+1}] {path} (weight: {w})")
    print("=" * 60)
    
    # 설정 생성
    config = EnsembleConfig(
        model_paths=model_paths,
        weights=weights,
        output_dir=args.output_dir,
        test_dataset_path=args.test_dataset,
        top_k_retrieval=args.top_k,
        batch_size=args.batch_size,
        use_retrieval=not args.no_retrieval,
        retrieval_alpha=args.retrieval_alpha,
        corpus_emb_path=args.corpus_emb_path,
        passages_meta_path=args.passages_meta_path,
    )
    
    # 앙상블 실행
    ensemble = MRCEnsemble(config)
    predictions = ensemble.run()
    
    print("\n" + "=" * 60)
    print("🎉 Ensemble finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

