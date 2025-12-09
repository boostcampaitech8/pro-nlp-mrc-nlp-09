"""
Prediction 분석 유틸리티
- Logits 저장 및 confidence 계산
- 사후 분석을 위한 상세 정보 추출
"""

import os
import json
import logging
from typing import Dict, List, Optional

import torch
import numpy as np
import pandas as pd
from transformers.trainer_utils import PredictionOutput
from scipy.special import softmax

logger = logging.getLogger(__name__)


def save_prediction_analysis(
    predictions: PredictionOutput,
    examples: List[Dict],
    output_dir: str,
    split: str,
    answer_column_name: str = "answers",
):
    """
    Prediction 결과에서 logits, confidence scores 추출 및 저장

    저장 파일:
    - {split}_logits.pt: Raw logits (torch tensor, 재현/앙상블용)
    - {split}_confidence.csv: ID별 confidence scores (빠른 필터링용)

    Args:
        predictions: trainer.predict() 결과 (PredictionOutput)
        examples: 원본 dataset examples
        output_dir: 저장 디렉토리
        split: 'train', 'validation', 'test'
        answer_column_name: 정답 필드 이름
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Logits 추출
    # predictions.predictions는 (start_logits, end_logits) 튜플 또는
    # postprocessing 후 dict 리스트일 수 있음
    # 여기서는 QuestionAnsweringTrainer의 predict()가 이미 postprocessing 완료한
    # 상태를 받으므로, 실제로는 logits가 아닌 predictions만 있음
    # 따라서 logits 저장은 trainer 내부에서 해야 하거나,
    # 여기서는 prediction confidence만 계산

    # predictions.predictions는 [{"id": ..., "prediction_text": ...}, ...] 형태
    pred_list = (
        predictions.predictions if hasattr(predictions, "predictions") else predictions
    )

    # 예측 결과를 dict로 변환
    if isinstance(pred_list, list) and len(pred_list) > 0:
        if isinstance(pred_list[0], dict):
            pred_dict = {p["id"]: p["prediction_text"] for p in pred_list}
        else:
            # 혹시 다른 형태일 경우 대비
            logger.warning(f"Unexpected prediction format: {type(pred_list[0])}")
            pred_dict = {}
    else:
        pred_dict = {}

    # Logits 파일 로드 시도
    logits_dict = {}
    logits_file = os.path.join(output_dir, f"logits_{split}.json")
    if os.path.exists(logits_file):
        try:
            with open(logits_file, "r", encoding="utf-8") as f:
                logits_dict = json.load(f)
            logger.info(f"✅ Loaded logits from {logits_file}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load logits: {e}")

    # 2. Example별 정보 수집
    results = []
    correct_count = 0

    for example in examples:
        example_id = example["id"]
        prediction = pred_dict.get(example_id, "")

        # 정답 확인
        is_correct = False
        if answer_column_name in example and example[answer_column_name]:
            answers = example[answer_column_name]
            if isinstance(answers, dict) and "text" in answers:
                answer_texts = answers["text"]
            elif isinstance(answers, list):
                answer_texts = answers
            else:
                answer_texts = []

            # Exact match 체크
            is_correct = any(prediction.strip() == ans.strip() for ans in answer_texts)
            if is_correct:
                correct_count += 1

        # Confidence 계산 (logits가 있으면)
        max_prob = -1.0
        avg_prob = -1.0
        if example_id in logits_dict:
            logit_info = logits_dict[example_id]
            start_logit = logit_info.get("start_logit", 0.0)
            end_logit = logit_info.get("end_logit", 0.0)

            # Probability from logits: exp(logit) / sum(exp(all_logits))
            # 여기서는 단일 값만 있으므로 softmax 대신 sigmoid 사용
            # 또는 이미 저장된 probability 사용
            if "probability" in logit_info:
                max_prob = logit_info["probability"]
                avg_prob = logit_info["probability"]
            else:
                # Simple approximation: sigmoid of combined score
                combined_score = start_logit + end_logit
                max_prob = 1.0 / (1.0 + np.exp(-combined_score))
                avg_prob = max_prob

        results.append(
            {
                "id": example_id,
                "prediction": prediction,
                "max_prob": max_prob,
                "avg_prob": avg_prob,
                "is_correct": 1 if is_correct else 0,
                "pred_length": len(prediction),
            }
        )

    # 3. CSV 저장
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{split}_confidence.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"✅ Confidence analysis saved to {csv_path}")

    # 4. 통계 로깅
    accuracy = correct_count / len(examples) if len(examples) > 0 else 0
    avg_pred_length = df["pred_length"].mean()

    logger.info(f"📊 Analysis stats ({split}):")
    logger.info(f"   Total samples: {len(examples)}")
    logger.info(f"   Correct predictions: {correct_count} ({accuracy:.2%})")
    logger.info(f"   Average prediction length: {avg_pred_length:.1f}")

    # Bottom 10% (is_correct 기준)
    if len(df) > 0:
        incorrect_df = df[df["is_correct"] == 0]
        if len(incorrect_df) > 0:
            logger.info(
                f"   Incorrect predictions: {len(incorrect_df)} ({len(incorrect_df) / len(df):.2%})"
            )

            # 예측 길이가 짧은 순으로 정렬 (confidence proxy)
            bottom_10_pct = max(1, int(len(incorrect_df) * 0.1))
            shortest_incorrect = incorrect_df.nsmallest(bottom_10_pct, "pred_length")
            logger.info(
                f"   Bottom 10% incorrect (shortest predictions): {bottom_10_pct} samples"
            )
            logger.info(
                f"   Avg length in bottom 10%: {shortest_incorrect['pred_length'].mean():.1f}"
            )

    return csv_path


# TODO: confidence 계산 오류 존재함
def save_prediction_analysis_with_logits(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    predictions: List[Dict],
    examples: List[Dict],
    output_dir: str,
    split: str,
    answer_column_name: str = "answers",
):
    """
    Logits를 포함한 상세 분석 (trainer 내부에서 호출 시 사용)

    Args:
        start_logits: Start position logits (N, seq_len)
        end_logits: End position logits (N, seq_len)
        predictions: Postprocessed predictions [{"id": ..., "prediction_text": ...}]
        examples: 원본 examples
        output_dir: 저장 디렉토리
        split: 'train', 'validation', 'test'
        answer_column_name: 정답 필드 이름
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Logits 저장 (.pt - 재현/앙상블용)
    logits_path = os.path.join(output_dir, f"{split}_logits.pt")
    torch.save(
        {
            "start_logits": torch.from_numpy(start_logits)
            if isinstance(start_logits, np.ndarray)
            else start_logits,
            "end_logits": torch.from_numpy(end_logits)
            if isinstance(end_logits, np.ndarray)
            else end_logits,
            "metadata": {
                "split": split,
                "num_samples": len(examples),
            },
        },
        logits_path,
    )
    logger.info(f"✅ Logits saved to {logits_path}")

    # 2. Softmax 계산
    start_probs = softmax(start_logits, axis=-1)  # (N, seq_len)
    end_probs = softmax(end_logits, axis=-1)

    # 각 샘플의 최대 확률
    max_start_probs = np.max(start_probs, axis=-1)  # (N,)
    max_end_probs = np.max(end_probs, axis=-1)

    # Max와 Average confidence
    max_probs = np.maximum(max_start_probs, max_end_probs)
    avg_probs = (max_start_probs + max_end_probs) / 2

    # 3. Predictions를 dict로 변환
    pred_dict = {p["id"]: p["prediction_text"] for p in predictions}

    # 4. Example별 정보 수집
    results = []
    correct_count = 0

    for i, example in enumerate(examples):
        example_id = example["id"]
        prediction = pred_dict.get(example_id, "")

        # 정답 확인
        is_correct = False
        if answer_column_name in example and example[answer_column_name]:
            answers = example[answer_column_name]
            if isinstance(answers, dict) and "text" in answers:
                answer_texts = answers["text"]
            elif isinstance(answers, list):
                answer_texts = answers
            else:
                answer_texts = []

            is_correct = any(prediction.strip() == ans.strip() for ans in answer_texts)
            if is_correct:
                correct_count += 1

        results.append(
            {
                "id": example_id,
                "prediction": prediction,
                "max_prob": float(max_probs[i]),
                "avg_prob": float(avg_probs[i]),
                "is_correct": 1 if is_correct else 0,
            }
        )

    # 5. CSV 저장
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{split}_confidence.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"✅ Confidence analysis saved to {csv_path}")

    # 6. 통계 로깅
    accuracy = correct_count / len(examples) if len(examples) > 0 else 0
    mean_conf = np.mean(avg_probs)
    std_conf = np.std(avg_probs)
    bottom_10_pct_threshold = np.percentile(avg_probs, 10)

    logger.info(f"📊 Confidence stats ({split}):")
    logger.info(f"   Mean: {mean_conf:.3f}, Std: {std_conf:.3f}")
    logger.info(f"   Bottom 10% threshold: {bottom_10_pct_threshold:.3f}")
    logger.info(f"   Accuracy: {correct_count}/{len(examples)} ({accuracy:.2%})")

    # Bottom 10% 분석
    bottom_10_mask = avg_probs <= bottom_10_pct_threshold
    bottom_10_df = df[bottom_10_mask]
    bottom_10_incorrect = bottom_10_df[bottom_10_df["is_correct"] == 0]

    logger.info(f"   Bottom 10% samples: {len(bottom_10_df)}")
    logger.info(f"   Bottom 10% incorrect: {len(bottom_10_incorrect)}")

    return logits_path, csv_path
