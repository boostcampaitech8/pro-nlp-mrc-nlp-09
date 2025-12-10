"""
MRC 모델 앙상블 (Soft Voting with Weighted Sum)

여러 학습된 모델의 start/end logits를 weighted sum하여 앙상블 수행

사용법:
  1. 직접 모델 경로 지정:
     python ensemble.py --model_paths ./outputs/model1 ./outputs/model2 --weights 0.5 0.5

  2. YAML config 파일 사용 (여러 실험 결과 앙상블):
     python ensemble.py --configs configs/active/exp1.yaml configs/active/exp2.yaml

  3. 파일 상단의 ENSEMBLE_MODELS 리스트 사용:
     python ensemble.py

제약사항:
  - 같은 토크나이저/모델 아키텍처끼리만 앙상블 가능 (텐서 shape 일치 필요)
"""

import os
import sys
import json
import csv
import glob
import argparse
import yaml
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

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
from src.retrieval.paths import get_path
from src.utils.qa import postprocess_qa_predictions
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================
# 🎯 여기서 앙상블할 모델들을 설정하세요!
# ============================================================
ENSEMBLE_MODELS = [
    # (모델 경로, 가중치)
    # 가중치는 자동으로 정규화됩니다 (합이 1이 되도록)
    # ("./outputs/dahyeong/exp_ra_k3_ds128", 1.0),
    # ("./outputs/dahyeong/exp_ra_k5_ds128", 1.0),
    # 💡 가중치 예시:
    # - 균등: 모두 1.0
    # - 성능 기반: EM 점수에 비례 (예: 75점 → 0.75, 80점 → 0.80)
    # - 수동 조절: 원하는 비율로 설정
]
# ============================================================


def find_best_checkpoint(output_dir: str) -> str:
    """
    output_dir에서 best checkpoint 경로를 찾습니다.

    탐색 우선순위:
    1. best_checkpoint_path.txt 파일이 있으면 그 내용 사용
    2. checkpoint-* 폴더 중 가장 최신 것
    3. output_dir 자체 (model.safetensors/pytorch_model.bin이 있는 경우)
    """
    # 1. best_checkpoint_path.txt 확인
    best_path_file = os.path.join(output_dir, "best_checkpoint_path.txt")
    if os.path.exists(best_path_file):
        with open(best_path_file, "r") as f:
            checkpoint_path = f.read().strip()
            if os.path.exists(checkpoint_path):
                return checkpoint_path

    # 2. checkpoint-* 폴더 탐색
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if checkpoint_dirs:
        # 숫자로 정렬하여 가장 큰 것 선택
        def get_step(path):
            try:
                return int(os.path.basename(path).split("-")[1])
            except:
                return 0

        checkpoint_dirs.sort(key=get_step, reverse=True)
        return checkpoint_dirs[0]

    # 3. output_dir 자체 확인
    model_files = ["model.safetensors", "pytorch_model.bin"]
    for model_file in model_files:
        if os.path.exists(os.path.join(output_dir, model_file)):
            return output_dir

    raise FileNotFoundError(
        f"❌ 모델을 찾을 수 없습니다: {output_dir}\n"
        f"💡 체크포인트 또는 model.safetensors/pytorch_model.bin이 필요합니다."
    )


def load_config_from_yaml(yaml_path: str) -> Dict:
    """YAML config 파일 로드"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_path_from_config(config: Dict) -> str:
    """YAML config에서 모델 경로 추출 (best checkpoint 자동 탐색)"""
    output_dir = config.get("output_dir", "")
    if not output_dir:
        raise ValueError("config에 output_dir이 없습니다.")

    return find_best_checkpoint(output_dir)


@dataclass
class EnsembleConfig:
    """앙상블 설정"""

    model_paths: List[str]  # 모델 경로 리스트
    weights: Optional[List[float]]  # 모델별 가중치 (None이면 균등)
    output_dir: str  # 결과 저장 경로
    test_dataset_path: str  # 테스트 데이터셋 경로
    max_seq_length: int = 384
    doc_stride: int = 128
    max_answer_length: int = 30
    top_k_retrieval: int = 10
    batch_size: int = 16
    use_retrieval: bool = True
    use_cache: bool = True  # Retrieval 캐시 사용 여부
    retrieval_alpha: float = 0.35  # WeightedHybridRetrieval의 BM25 가중치
    corpus_emb_path: Optional[str] = None
    passages_meta_path: Optional[str] = None
    inference_split: str = "test"  # test / validation


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

        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"📊 Model weights: {self.weights}")

    def load_models(self):
        """모든 모델 로드"""
        print("\n" + "=" * 60)
        print("📦 Loading models for ensemble...")
        print("=" * 60)

        for i, model_path in enumerate(self.config.model_paths):
            print(f"\n[{i + 1}/{len(self.config.model_paths)}] Loading: {model_path}")

            # Best checkpoint 자동 탐색
            try:
                actual_path = find_best_checkpoint(model_path)
                if actual_path != model_path:
                    print(f"   📍 Found checkpoint: {actual_path}")
            except FileNotFoundError:
                actual_path = model_path  # 그대로 시도

            # 모델과 토크나이저 로드
            config = AutoConfig.from_pretrained(actual_path)
            tokenizer = AutoTokenizer.from_pretrained(actual_path, use_fast=True)
            model = AutoModelForQuestionAnswering.from_pretrained(
                actual_path, config=config
            )
            model.to(self.device)
            model.eval()

            self.models.append(model)
            self.tokenizers.append(tokenizer)

            print(f"   ✅ Loaded: {config.model_type}")

        # 토크나이저 일관성 검증
        if len(self.tokenizers) > 1:
            base_vocab_size = len(self.tokenizers[0])
            for i, tok in enumerate(self.tokenizers[1:], 2):
                if len(tok) != base_vocab_size:
                    logger.warning(
                        f"⚠️ 토크나이저 vocab size 불일치: "
                        f"Model 1={base_vocab_size}, Model {i}={len(tok)}"
                    )

        print(f"\n✅ Total {len(self.models)} models loaded!")

    def load_dataset(self) -> DatasetDict:
        """테스트 데이터셋 로드"""
        print(f"\n📂 Loading dataset from: {self.config.test_dataset_path}")
        datasets = load_from_disk(self.config.test_dataset_path)
        print(f"   Dataset: {datasets}")
        return datasets

    def load_retrieval_from_cache(self, dataset: Dataset) -> Dataset:
        """캐시된 retrieval 결과 로드 (inference.py와 동일한 로직)"""
        # 캐시 경로 결정
        if self.config.inference_split == "test":
            cache_path = get_path("test_cache")
        else:
            cache_path = get_path("val_cache")

        if not os.path.exists(cache_path):
            return None

        logger.info(f"📦 Loading retrieval cache from: {cache_path}")

        # 캐시 로드
        cache = {}
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                cache[item["id"]] = item

        # Passages corpus 로드
        passages_meta_path = self.config.passages_meta_path or get_path(
            "kure_passages_meta"
        )
        wiki_path = get_path("wiki_corpus")

        if passages_meta_path and os.path.exists(passages_meta_path):
            passage_texts = []
            with open(passages_meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    meta = json.loads(line.strip())
                    passage_texts.append(meta["text"])
        else:
            with open(wiki_path, "r", encoding="utf-8") as f:
                wiki = json.load(f)
            unique_texts = {}
            for doc_id, doc_info in wiki.items():
                text = doc_info["text"]
                if text not in unique_texts:
                    unique_texts[text] = text
            passage_texts = list(unique_texts.keys())

        # 결과 구성
        result_data = {"id": [], "question": [], "context": []}
        top_k = self.config.top_k_retrieval
        alpha = self.config.retrieval_alpha

        for example in dataset:
            qid = example["id"]
            cache_entry = cache.get(qid)

            if cache_entry is None:
                logger.warning(f"⚠️ Cache miss for {qid}")
                context = ""
            else:
                candidates = cache_entry["retrieved"]
                if candidates:
                    bm25_scores = np.array([c["score_bm25"] for c in candidates])
                    dense_scores = np.array([c["score_dense"] for c in candidates])

                    eps = 1e-9
                    bm25_n = (bm25_scores - bm25_scores.min()) / (
                        bm25_scores.max() - bm25_scores.min() + eps
                    )
                    dense_n = (dense_scores - dense_scores.min()) / (
                        dense_scores.max() - dense_scores.min() + eps
                    )
                    hybrid_scores = alpha * bm25_n + (1 - alpha) * dense_n

                    sorted_indices = np.argsort(hybrid_scores)[::-1][:top_k]
                    contexts = []
                    for idx in sorted_indices:
                        passage_id = candidates[idx]["passage_id"]
                        if passage_id < len(passage_texts):
                            contexts.append(passage_texts[passage_id])
                    context = " ".join(contexts)
                else:
                    context = ""

            result_data["id"].append(qid)
            result_data["question"].append(example["question"])
            result_data["context"].append(context)

        features = Features(
            {
                "id": Value(dtype="string"),
                "question": Value(dtype="string"),
                "context": Value(dtype="string"),
            }
        )

        return Dataset.from_dict(result_data, features=features)

    def run_retrieval(self, datasets: DatasetDict) -> DatasetDict:
        """Weighted Hybrid Retrieval 수행 (BM25 + KURE)"""
        if not self.config.use_retrieval:
            return datasets

        print("\n🔍 Running Weighted Hybrid Retrieval (BM25 + KURE)...")

        # 캐시 사용 시도
        if self.config.use_cache:
            cached_dataset = self.load_retrieval_from_cache(datasets["validation"])
            if cached_dataset is not None:
                datasets = DatasetDict({"validation": cached_dataset})
                print(
                    f"   ✅ Loaded from cache: {len(datasets['validation'])} examples"
                )
                return datasets
            else:
                logger.info("⚠️ Cache not found, running live retrieval...")

        # 실시간 retrieval
        tokenizer = self.tokenizers[0]

        corpus_emb_path = self.config.corpus_emb_path or get_path("kure_corpus_emb")
        passages_meta_path = self.config.passages_meta_path or get_path(
            "kure_passages_meta"
        )

        retriever = WeightedHybridRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="./data",
            context_path="wikipedia_documents.json",
            corpus_emb_path=corpus_emb_path,
            passages_meta_path=passages_meta_path,
            alpha=self.config.retrieval_alpha,
        )
        retriever.build()

        df = retriever.retrieve(
            datasets["validation"], topk=self.config.top_k_retrieval
        )

        # DataFrame을 Dataset으로 변환
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

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
        self, model, tokenizer, dataset
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
                "input_ids": torch.tensor(
                    [features[j]["input_ids"] for j in batch_indices]
                ),
                "attention_mask": torch.tensor(
                    [features[j]["attention_mask"] for j in batch_indices]
                ),
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
        self, all_start_logits: List[np.ndarray], all_end_logits: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted sum으로 logits 앙상블"""

        print("\n🎯 Ensembling logits with weighted sum...")

        # Weighted sum
        ensembled_start = np.zeros_like(all_start_logits[0])
        ensembled_end = np.zeros_like(all_end_logits[0])

        for i, (start, end) in enumerate(zip(all_start_logits, all_end_logits)):
            ensembled_start += self.weights[i] * start
            ensembled_end += self.weights[i] * end
            print(f"   Model {i + 1}: weight={self.weights[i]:.3f}")

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
            print(f"\n[Model {i + 1}/{len(self.models)}]")
            start_logits, end_logits, features = self.get_logits_from_model(
                model, tokenizer, datasets["validation"]
            )
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)
            print(
                f"   Logits shape: start={start_logits.shape}, end={end_logits.shape}"
            )

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
        print(
            f"   📄 Predictions: {os.path.join(self.config.output_dir, 'predictions_ensemble.json')}"
        )
        print(f"   📄 CSV: {csv_path}")

        return predictions


def main():
    parser = argparse.ArgumentParser(
        description="MRC Model Ensemble (Soft Voting with Weighted Sum)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 1. 직접 모델 경로 지정
  python ensemble.py --model_paths ./outputs/model1 ./outputs/model2 --weights 0.5 0.5

  # 2. YAML config 파일 사용 (여러 실험 결과 앙상블)
  python ensemble.py --configs configs/active/exp1.yaml configs/active/exp2.yaml

  # 3. 파일 상단의 ENSEMBLE_MODELS 리스트 사용
  python ensemble.py
        """,
    )
    parser.add_argument(
        "--model_paths",
        nargs="+",
        default=None,
        help="학습된 모델 경로들 (best checkpoint 자동 탐색)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="YAML config 파일 경로들 (output_dir에서 모델 자동 탐색)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="모델별 가중치 (미지정시 균등 분배)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/ensemble", help="결과 저장 경로"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="./data/test_dataset",
        help="테스트 데이터셋 경로",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "validation"],
        default="test",
        help="inference split (test: 제출용, validation: 평가용)",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Retrieval top-k")
    parser.add_argument(
        "--doc_stride", type=int, default=128, help="Document stride for tokenization"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="배치 사이즈")
    parser.add_argument(
        "--no_retrieval",
        action="store_true",
        help="Retrieval 사용 안함 (gold context 사용)",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Retrieval 캐시 사용 안함 (항상 실시간 retrieval)",
    )
    parser.add_argument(
        "--retrieval_alpha",
        type=float,
        default=0.35,
        help="WeightedHybridRetrieval의 BM25 가중치 (0~1)",
    )

    args = parser.parse_args()

    # 모델 경로와 가중치 결정 (우선순위: --model_paths > --configs > ENSEMBLE_MODELS)
    model_paths = []
    weights = args.weights

    if args.model_paths is not None:
        # 1. 직접 경로 지정
        model_paths = args.model_paths
        logger.info("📋 Using model paths from command line")

    elif args.configs is not None:
        # 2. YAML config에서 추출
        logger.info("📋 Extracting model paths from YAML configs...")
        for config_path in args.configs:
            config = load_config_from_yaml(config_path)
            try:
                model_path = get_model_path_from_config(config)
                model_paths.append(model_path)
                logger.info(f"   ✅ {config_path} -> {model_path}")
            except Exception as e:
                logger.error(f"   ❌ {config_path}: {e}")
                sys.exit(1)

    elif ENSEMBLE_MODELS:
        # 3. 상단의 ENSEMBLE_MODELS 리스트 사용
        model_paths = [path for path, _ in ENSEMBLE_MODELS]
        weights = [weight for _, weight in ENSEMBLE_MODELS]
        logger.info("📋 Using ENSEMBLE_MODELS from script")

    else:
        raise ValueError(
            "❌ 앙상블할 모델이 없습니다!\n"
            "💡 다음 중 하나를 사용하세요:\n"
            "   1. --model_paths ./outputs/model1 ./outputs/model2\n"
            "   2. --configs configs/exp1.yaml configs/exp2.yaml\n"
            "   3. ensemble.py 상단의 ENSEMBLE_MODELS 리스트"
        )

    if not model_paths:
        raise ValueError("❌ 유효한 모델 경로가 없습니다.")

    # 데이터셋 경로 결정
    if args.split == "validation":
        test_dataset_path = "./data/train_dataset"  # validation split 포함
    else:
        test_dataset_path = args.test_dataset

    print("\n" + "=" * 60)
    print("📋 Ensemble Configuration")
    print("=" * 60)
    print(f"   Split: {args.split}")
    print(f"   Dataset: {test_dataset_path}")
    print(
        f"   Retrieval: {'Enabled' if not args.no_retrieval else 'Disabled (gold context)'}"
    )
    print(f"   Cache: {'Enabled' if not args.no_cache else 'Disabled'}")
    print(f"   Top-k: {args.top_k}")
    print(f"   Alpha: {args.retrieval_alpha}")
    print(f"   Doc stride: {args.doc_stride}")
    print("-" * 60)
    print("   Models:")
    for i, path in enumerate(model_paths):
        w = weights[i] if weights else 1.0
        print(f"   [{i + 1}] {path} (weight: {w})")
    print("=" * 60)

    # 설정 생성
    config = EnsembleConfig(
        model_paths=model_paths,
        weights=weights,
        output_dir=args.output_dir,
        test_dataset_path=test_dataset_path,
        top_k_retrieval=args.top_k,
        doc_stride=args.doc_stride,
        batch_size=args.batch_size,
        use_retrieval=not args.no_retrieval,
        use_cache=not args.no_cache,
        retrieval_alpha=args.retrieval_alpha,
        inference_split=args.split,
    )

    # 앙상블 실행
    ensemble = MRCEnsemble(config)
    predictions = ensemble.run()

    print("\n" + "=" * 60)
    print("🎉 Ensemble finished successfully!")
    print(f"   📄 Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
