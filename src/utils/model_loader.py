"""
모델 경로 자동 탐색 유틸리티

Train 시: model_name_or_path를 그대로 사용 (pretrained model)
Inference 시: use_trained_model=True이면 output_dir에서 best checkpoint 자동 탐색
"""

import os
import json
from typing import Optional


def get_model_path(model_args, training_args, for_inference: bool = False) -> str:
    """
    학습/추론 상황에 맞는 모델 경로를 반환합니다.

    Args:
        model_args: ModelArguments 인스턴스
        training_args: TrainingArguments 인스턴스
        for_inference: inference 모드인지 여부

    Returns:
        사용할 모델 경로 (pretrained model name 또는 checkpoint path)
    """
    # Train 모드이거나 use_trained_model=False인 경우
    if not for_inference or not model_args.use_trained_model:
        return model_args.model_name_or_path

    # Inference 모드에서 trained model 사용
    output_dir = training_args.output_dir

    if not os.path.exists(output_dir):
        raise FileNotFoundError(
            f"Output directory not found: {output_dir}\n"
            f"Set use_trained_model=false in YAML to use pretrained model directly."
        )

    # 1순위: best_checkpoint_path.txt 파일 확인
    best_checkpoint_file = os.path.join(output_dir, "best_checkpoint_path.txt")
    if os.path.exists(best_checkpoint_file):
        with open(best_checkpoint_file, "r") as f:
            checkpoint_path = f.read().strip()
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(
                    f"✅ Using best checkpoint from best_checkpoint_path.txt: {checkpoint_path}"
                )
                return checkpoint_path

    # 2순위: trainer_state.json에서 best_model_checkpoint 읽기
    trainer_state_file = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(trainer_state_file):
        with open(trainer_state_file, "r") as f:
            trainer_state = json.load(f)
            best_checkpoint = trainer_state.get("best_model_checkpoint")
            if best_checkpoint and os.path.exists(best_checkpoint):
                print(
                    f"✅ Using best checkpoint from trainer_state.json: {best_checkpoint}"
                )
                return best_checkpoint

    # 3순위: checkpoint-* 폴더 중 숫자가 가장 큰 것 선택 (fallback)
    checkpoint_dirs = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]

    if checkpoint_dirs:
        # checkpoint-1234 형식에서 숫자 추출하여 정렬
        def get_checkpoint_number(dirname):
            try:
                return int(dirname.split("-")[-1])
            except ValueError:
                return -1

        latest_checkpoint = max(checkpoint_dirs, key=get_checkpoint_number)
        checkpoint_path = os.path.join(output_dir, latest_checkpoint)
        print(
            f"⚠️  Best checkpoint info not found. Using latest checkpoint: {checkpoint_path}"
        )
        return checkpoint_path

    # 모든 시도 실패
    raise FileNotFoundError(
        f"No checkpoint found in {output_dir}\n"
        f"Please run training first, or set use_trained_model=false to use pretrained model."
    )


def load_inference_dataset(data_args, inference_split: Optional[str] = None):
    """
    inference_split에 따라 적절한 데이터셋을 로드합니다.

    Args:
        data_args: DataTrainingArguments 인스턴스
        inference_split: 'train', 'validation', 또는 'test' (None이면 data_args.inference_split 사용)

    Returns:
        로드된 데이터셋 (DatasetDict)
    """
    from datasets import load_from_disk

    split = inference_split or data_args.inference_split

    if split == "test":
        # test split: infer_dataset_name 사용
        dataset_path = data_args.infer_dataset_name
        print(f"📦 Loading test dataset from: {dataset_path}")
        return load_from_disk(dataset_path)

    elif split in ["train", "validation"]:
        # train/validation split: train_dataset_name에서 해당 split 사용
        dataset_path = data_args.train_dataset_name
        print(f"📦 Loading {split} split from: {dataset_path}")
        datasets = load_from_disk(dataset_path)

        if split not in datasets:
            raise ValueError(
                f"Split '{split}' not found in {dataset_path}. "
                f"Available splits: {list(datasets.keys())}"
            )

        # validation split을 "validation" 키로 반환 (inference.py 기대 형식)
        from datasets import DatasetDict

        return DatasetDict({"validation": datasets[split]})

    else:
        raise ValueError(
            f"Invalid inference_split: {split}. "
            f"Must be one of: 'train', 'validation', 'test'"
        )
