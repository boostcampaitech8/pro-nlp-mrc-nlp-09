import json
import sys
import os
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from transformers import HfArgumentParser

# ---------------------------------------------------------------------------
# YAML 기반 설정 파서
#   - 사용법: parser = HfArgumentParser(...)
#            model_args, data_args, training_args = parse_args_from_yaml(parser)
#   - python train.py configs/train/xxx.yaml 처럼 호출하면,
#     해당 yaml 내용을 기반으로 dataclass 인스턴스를 생성한다.
#   - YAML이 아니면 기존 CLI 방식으로 fallback.
# ---------------------------------------------------------------------------

# TODO: 추후에 여러 실험 동시에 작업 가능하도록 개선


def get_config(parser: HfArgumentParser):
    """
    HfArgumentParser에 대해 YAML 또는 CLI 인자를 파싱하는 헬퍼.

    - python train.py configs/train/exp.yaml
      -> exp.yaml을 읽어서 parse_dict로 dataclass 생성
    - python train.py --output_dir ... --do_train ...
      -> 기존 parse_args_into_dataclasses() 경로로 동작
    """

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        return parser.parse_yaml_file(config_path)

    # 그 외에는 기존 방식 유지 (CLI 인자)
    return parser.parse_args_into_dataclasses()


def to_serializable(value: Any):
    """dataclass 인스턴스를 JSON-friendly 구조로 변환"""
    if is_dataclass(value):
        return {k: to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def print_section(title: str, payload: Any):
    """섹션 타이틀과 함께 JSON 형태로 출력"""
    divider = "=" * 80
    print(divider)
    print(f"  {title}")
    print(divider)
    print(json.dumps(to_serializable(payload), indent=2, ensure_ascii=False))
    print()


if __name__ == "__main__":
    from src.arguments import ModelArguments, DataTrainingArguments
    from transformers import TrainingArguments

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = get_config(parser)

    print("Configuration loaded successfully.")

    print("======== Configurations =========")
    print_section("Model Arguments", model_args)
    print_section("Data Arguments", data_args)
    print_section("Training Arguments", training_args)

    print("\n\n======== Raw dataclass instances =========")
    print("model_args:", model_args)
    print("data_args:", data_args)
    print("training_args:", training_args)
