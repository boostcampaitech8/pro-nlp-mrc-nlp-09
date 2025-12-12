"""
MRC 모델 앙상블 (Soft Voting with Weighted Sum)

여러 학습된 모델의 start/end logits를 weighted sum하여 앙상블 수행

사용 예시:
    # Test 데이터셋 사용 (기본)
    python ensemble.py --output_dir ./outputs/ensemble/test
    
    # Train 데이터셋의 validation 셋 사용
    python ensemble.py --use_train_validation --train_dataset ./data/train_dataset --output_dir ./outputs/ensemble/validation --no_retrieval
    
    # 커맨드라인에서 모델 경로 지정
    python ensemble.py --model_paths ./outputs/model1 ./outputs/model2 --weights 0.6 0.4
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

from src.retrieval import get_retriever
from src.retrieval.reranker import CrossEncoderReranker # Reranker 임포트 추가
from src.utils.tokenization import get_tokenizer
from src.utils.qa import postprocess_qa_predictions
from transformers import AutoTokenizer as HFAutoTokenizer


# ============================================================
# 🎯 여기서 앙상블할 모델들을 설정하세요!
# ============================================================
ENSEMBLE_MODELS = [
    # (모델 경로, 가중치)
    # 가중치는 자동으로 정규화됩니다 (합이 1이 되도록)
    ("/data/ephemeral/home/junbeom/MRC/outputs/teawon/hanteck2", 1.0),
    ("/data/ephemeral/home/junbeom/MRC/outputs/teawon/oceann2", 1.0),
    ("/data/ephemeral/home/junbeom/MRC/outputs/teawon/roberta2", 1.0),
    ("/data/ephemeral/home/junbeom/MRC/outputs/teawon/uomnf2", 1.0),
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
    test_dataset_path: Optional[str] = None  # 테스트 데이터셋 경로
    train_dataset_path: Optional[str] = None  # 학습 데이터셋 경로 (validation 셋 사용 시)
    use_train_validation: bool = False  # train_dataset의 validation 셋 사용 여부
    max_seq_length: int = 384
    doc_stride: int = 128
    max_answer_length: int = 30
    top_k_retrieval: int = 10
    batch_size: int = 16
    use_retrieval: bool = True
    retrieval_alpha: float = 0.5  # Hybrid Retrieval의 BM25 가중치
    retrieval_tokenizer_name: str = "kiwi"  # kiwi or auto
    bm25_impl: str = "rank_bm25"  # rank_bm25 or bm25s
    bm25_k1: float = 1.2
    bm25_b: float = 0.6
    bm25_delta: float = 0.5
    fusion_method: str = "rrf"  # rrf or score
    corpus_emb_path: Optional[str] = None  # KoE5 corpus embedding 경로 (None이면 기본 경로 사용)
    dense_retriever_type: str = "koe5" # Hybrid 내부에서 사용할 Dense Retriever 타입 ("koe5" or "kure")
    # Reranker Settings
    reranker_name: Optional[str] = "BAAI/bge-reranker-v2-m3"
    rerank_topk: int = 50


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
        """데이터셋 로드 (test 또는 train의 validation)"""
        if self.config.use_train_validation:
            if self.config.train_dataset_path is None:
                raise ValueError("❌ use_train_validation=True인 경우 train_dataset_path를 지정해야 합니다.")
            print(f"\n📂 Loading train dataset from: {self.config.train_dataset_path}")
            train_datasets = load_from_disk(self.config.train_dataset_path)
            print(f"   Train dataset splits: {list(train_datasets.keys())}")
            
            # validation 셋이 있는지 확인
            if "validation" not in train_datasets:
                raise ValueError(
                    f"❌ train_dataset에 'validation' split이 없습니다.\n"
                    f"   Available splits: {list(train_datasets.keys())}"
                )
            
            # validation 셋만 사용
            datasets = DatasetDict({"validation": train_datasets["validation"]})
            print(f"   ✅ Using validation split: {len(datasets['validation'])} examples")
        else:
            if self.config.test_dataset_path is None:
                raise ValueError("❌ use_train_validation=False인 경우 test_dataset_path를 지정해야 합니다.")
            print(f"\n📂 Loading test dataset from: {self.config.test_dataset_path}")
            datasets = load_from_disk(self.config.test_dataset_path)
            print(f"   Dataset: {datasets}")
        
        return datasets
    
    def run_retrieval(self, datasets: DatasetDict) -> DatasetDict:
        """Hybrid Retrieval 수행 (BM25Plus + KoE5) + Reranking"""
        if not self.config.use_retrieval:
            return datasets
        
        print("\n🔍 Running Hybrid Retrieval (BM25Plus + KoE5)...")
        
        # Tokenizer 설정
        print(f"[INIT] Setting up tokenizer: {self.config.retrieval_tokenizer_name}")
        model_tokenizer = HFAutoTokenizer.from_pretrained("klue/roberta-large")  # Default fallback
        tokenize_fn = get_tokenizer(self.config.retrieval_tokenizer_name, model_tokenizer)
        
        # Hybrid Retrieval 생성
        print(f"[INIT] Setting up Hybrid Retriever")
        print(f"       - BM25 Impl: {self.config.bm25_impl} (k1={self.config.bm25_k1}, b={self.config.bm25_b}, delta={self.config.bm25_delta})")
        print(f"       - Hybrid Alpha: {self.config.retrieval_alpha}")
        print(f"       - Fusion Method: {self.config.fusion_method}")
        print(f"       - Dense Retriever Type: {self.config.dense_retriever_type}") # 추가된 부분
        
        retriever = get_retriever(
            retrieval_type="hybrid",
            tokenize_fn=tokenize_fn,
            data_path="./data",
            context_path="wikipedia_documents_normalized.json",
            # Hybrid Args
            alpha=self.config.retrieval_alpha,
            fusion_method=self.config.fusion_method,
            dense_retriever_type=self.config.dense_retriever_type, # 추가된 부분
            # BM25 Args
            impl=self.config.bm25_impl,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
            delta=self.config.bm25_delta,
            # KoE5/Kure Args
            corpus_emb_path=self.config.corpus_emb_path,
            passages_meta_path=None, # Kure가 ensemble.py에서 필요하면 추가해줘야 함. 현재는 없음.
                                     # but get_path() in retrieval/hybrid.py will handle default.
        )
        
        print("[INIT] Building retriever index...")
        retriever.build()
        
        # Reranker 초기화
        reranker = None
        if self.config.reranker_name:
            print(f"[INIT] Setting up Reranker: {self.config.reranker_name}")
            reranker = CrossEncoderReranker(model_name=self.config.reranker_name)
        
        # Retrieval 수행
        # Reranker가 있으면 더 많이 가져와서 재정렬
        top_k = self.config.rerank_topk if reranker else self.config.top_k_retrieval
        print(f"   - Retrieving top-{top_k} candidates...")
        
        queries = datasets["validation"]["question"]
        doc_scores, doc_indices = retriever.get_relevant_doc_bulk(queries, k=top_k)
        
        # Context 구성 (Reranking 포함)
        final_contexts = []
        print(f"   - Constructing contexts{' (with Reranking)' if reranker else ''}...")
        
        for i in tqdm(range(len(queries)), desc="Context Processing"):
            query = queries[i]
            indices = doc_indices[i]
            passages = [retriever.contexts[idx] for idx in indices]
            
            if reranker:
                # Reranking
                r_scores = reranker.rerank(query, passages)
                scored = sorted(zip(passages, r_scores), key=lambda x: x[1], reverse=True)
                # 최종 Top-K 선택
                selected_passages = [p for p, _ in scored][:self.config.top_k_retrieval]
                final_contexts.append(" ".join(selected_passages))
            else:
                # No Reranking
                final_contexts.append(" ".join(passages))
        
        # Dataset 재구성 (DataFrame 생성 없이 직접)
        # answers가 있는 경우와 없는 경우 처리
        data_dict = {
            "id": datasets["validation"]["id"],
            "question": queries,
            "context": final_contexts
        }
        
        if "answers" in datasets["validation"].column_names:
            data_dict["answers"] = datasets["validation"]["answers"]
            f = Features({
                "id": Value(dtype="string"),
                "question": Value(dtype="string"),
                "context": Value(dtype="string"),
                "answers": Sequence(feature={"text": Value(dtype="string"), "answer_start": Value(dtype="int32")})
            })
        else:
            f = Features({
                "id": Value(dtype="string"),
                "question": Value(dtype="string"),
                "context": Value(dtype="string")
            })
            
        new_ds = Dataset.from_dict(data_dict, features=f)
        datasets = DatasetDict({"validation": new_ds})
        
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
        
        dataset_type = "train/validation" if self.config.use_train_validation else "test"
        print(f"📋 Dataset type: {dataset_type}")
        
        # 1. 모델 로드
        self.load_models()
        
        # 2. 데이터셋 로드
        datasets = self.load_dataset()
        
        # 3. Retrieval 수행
        if self.config.use_train_validation and self.config.use_retrieval:
            print("\n⚠️  Warning: validation 셋 사용 시 일반적으로 retrieval을 사용하지 않습니다.")
            print("   (validation 셋은 gold context를 포함하고 있습니다)")
            print("   retrieval을 건너뛰려면 --no_retrieval 플래그를 사용하세요.")
        
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
        
        # prefix 설정 (validation 셋인지 test 셋인지 구분)
        prefix = "ensemble_validation" if self.config.use_train_validation else "ensemble"
        
        predictions = postprocess_qa_predictions(
            examples=datasets["validation"],
            features=features,
            predictions=(ensembled_start, ensembled_end),
            max_answer_length=self.config.max_answer_length,
            output_dir=self.config.output_dir,
            prefix=prefix,
        )
        
        # 7. CSV 저장
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
        default="./outputs/taewon/ensemble/3_3_3_1",
        help="결과 저장 경로"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="./data/test_dataset",
        help="테스트 데이터셋 경로 (use_train_validation=False일 때 사용)"
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        help="학습 데이터셋 경로 (use_train_validation=True일 때 사용, validation 셋을 가져옴)"
    )
    parser.add_argument(
        "--use_train_validation",
        action="store_true",
        help="train_dataset의 validation 셋 사용 (기본값: False, test_dataset 사용)"
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
        default=0.5,
        help="Hybrid Retrieval의 BM25 가중치 (0~1, 기본값: 0.5)"
    )
    parser.add_argument(
        "--retrieval_tokenizer_name",
        type=str,
        default="kiwi",
        help="Retrieval용 tokenizer (kiwi or auto, 기본값: kiwi)"
    )
    parser.add_argument(
        "--bm25_impl",
        type=str,
        default="rank_bm25",
        help="BM25 구현체 (rank_bm25 or bm25s, 기본값: rank_bm25)"
    )
    parser.add_argument(
        "--bm25_k1",
        type=float,
        default=1.2,
        help="BM25 k1 파라미터 (기본값: 1.2)"
    )
    parser.add_argument(
        "--bm25_b",
        type=float,
        default=0.6,
        help="BM25 b 파라미터 (기본값: 0.6)"
    )
    parser.add_argument(
        "--bm25_delta",
        type=float,
        default=0.5,
        help="BM25Plus delta 파라미터 (기본값: 0.5)"
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="rrf",
        help="Hybrid fusion 방법 (rrf or score, 기본값: rrf)"
    )
    parser.add_argument(
        "--corpus_emb_path",
        type=str,
        default=None,
        help="KoE5 corpus embedding 경로 (None이면 기본 경로 사용)"
    )
    parser.add_argument(
        "--dense_retriever_type", # 추가된 인자
        type=str,
        default="koe5",
        help="Hybrid Retrieval 내부에서 사용할 Dense Retriever 타입 (koe5 or kure, 기본값: koe5)"
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
        test_dataset_path=args.test_dataset if not args.use_train_validation else None,
        train_dataset_path=args.train_dataset if args.use_train_validation else None,
        use_train_validation=args.use_train_validation,
        top_k_retrieval=args.top_k,
        batch_size=args.batch_size,
        use_retrieval=not args.no_retrieval,
        retrieval_alpha=args.retrieval_alpha,
        retrieval_tokenizer_name=args.retrieval_tokenizer_name,
        bm25_impl=args.bm25_impl,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        bm25_delta=args.bm25_delta,
        fusion_method=args.fusion_method,
        corpus_emb_path=args.corpus_emb_path,
        dense_retriever_type=args.dense_retriever_type, # 추가된 부분
    )
    
    # 앙상블 실행
    ensemble = MRCEnsemble(config)
    predictions = ensemble.run()
    
    print("\n" + "=" * 60)
    print("🎉 Ensemble finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

