"""
앙상블 스크립트: nbest_predictions 기반 텍스트 레벨 score voting

기능:
- 여러 모델의 nbest_predictions JSON 파일을 입력받음
- 정규화된 answer text를 기준으로 score voting
- 최종 predictions을 TSV 포맷으로 저장 (리더보드 제출용)

사용 예시:
python scripts/ensemble_predictions.py \\
  --nbest_paths outputs/exp1/nbest_predictions.json outputs/exp2/nbest_predictions.json \\
  --output_path outputs/ensemble/ens_test_pred.csv \\
  --weights 0.4 0.6 \\
  --score_key probability
"""

import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import csv
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def normalize_answer(text: str) -> str:
    """
    Answer 정규화 (공백/구두점 제거, 소문자화)

    정규화된 text가 key로 사용되어 같은 answer를 그룹화합니다.

    Args:
        text: 원본 answer text

    Returns:
        정규화된 text
    """
    # 소문자화
    text = text.lower()
    # 구두점 제거 (공백 제외)
    text = re.sub(r"[^\w\s]", "", text)
    # 연속 공백을 단일 공백으로
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_nbest(path: str) -> Dict[str, List[Dict]]:
    """
    nbest_predictions.json 로드

    Format: {
        "qid1": [
            {"text": "answer1", "probability": 0.9},
            {"text": "answer2", "probability": 0.05},
            ...
        ],
        ...
    }

    Args:
        path: JSON 파일 경로

    Returns:
        {qid: [{"text": ..., "probability"/"score": ...}]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 다양한 포맷 지원 (nbest_predictions이 리스트 또는 dict일 수 있음)
    if isinstance(data, list):
        # List of dicts: [{"id": qid, "text": answer, ...}]
        result = defaultdict(list)
        for item in data:
            qid = item.get("id")
            if qid:
                result[qid].append(item)
        return dict(result)
    else:
        # Dict of qid -> list
        return data


def ensemble_nbest(
    nbest_paths: List[str],
    weights: Optional[List[float]] = None,
    score_key: str = "probability",
) -> Dict[str, str]:
    """
    앙상블 로직: 정규화된 answer 기준 score voting

    Args:
        nbest_paths: nbest_predictions.json 파일 경로 리스트
        weights: 각 모델의 가중치 (None이면 균등)
        score_key: score 필드명 ("probability" or "score")

    Returns:
        {qid: best_answer_text}
    """
    num_models = len(nbest_paths)

    if weights is None:
        weights = [1.0 / num_models] * num_models
    else:
        weights = [w / sum(weights) for w in weights]  # 정규화

    logger.info(f"🎯 Ensemble with {num_models} models, weights: {weights}")
    logger.info(f"   Score key: {score_key}")

    # 모든 nbest 파일 로드
    nbest_list = []
    for path in nbest_paths:
        logger.info(f"📖 Loading {path}...")
        nbest_data = load_nbest(path)
        nbest_list.append(nbest_data)

    # 모든 question ID 수집
    all_qids = set()
    for nbest_data in nbest_list:
        all_qids.update(nbest_data.keys())

    logger.info(f"📊 Total questions: {len(all_qids)}")

    # 각 question에 대해 앙상블 수행
    ensemble_predictions = {}

    for qid in all_qids:
        # 정규화된 answer -> 누적 score
        answer_scores = defaultdict(float)
        # 정규화된 answer -> (최고 score, 원본 text) - 가장 높은 score의 원본 보존
        answer_best_original = {}

        for model_idx, nbest_data in enumerate(nbest_list):
            weight = weights[model_idx]

            if qid not in nbest_data:
                logger.warning(f"⚠️ qid {qid} not found in model {model_idx}")
                continue

            candidates = nbest_data[qid]
            if not candidates:
                continue

            # Top candidate에서만 score 가져오기 (또는 모든 후보 - 여기서는 top 3까지 처리)
            for candidate in candidates[:3]:  # top-3 후보만 고려
                text = candidate.get("text", "")
                score = candidate.get(score_key, 0.0)

                if not text:
                    continue

                normalized = normalize_answer(text)
                answer_scores[normalized] += weight * score
                answer_original[normalized] = text

        # 가장 높은 score를 가진 answer 선택
        if answer_scores:
            best_normalized = max(answer_scores, key=answer_scores.get)
            best_answer = answer_original[best_normalized]
            ensemble_predictions[qid] = best_answer
        else:
            logger.warning(f"⚠️ No valid candidates for qid {qid}")
            ensemble_predictions[qid] = ""

    logger.info(f"✅ Ensemble complete: {len(ensemble_predictions)} predictions")
    return ensemble_predictions


def save_predictions_csv(predictions: Dict[str, str], output_path: str):
    """
    Predictions을 TSV 형식으로 저장 (리더보드 제출용)

    Format:
    id\tanswer
    qid1\tanswer1
    qid2\tanswer2

    Args:
        predictions: {qid: answer} dict
        output_path: 저장 경로
    """
    import os

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "answer"])  # 헤더
        for qid, answer in sorted(predictions.items()):
            writer.writerow([qid, answer])

    logger.info(f"💾 Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="앙상블: nbest_predictions 기반 텍스트 레벨 score voting"
    )

    parser.add_argument(
        "--nbest_paths",
        nargs="+",
        required=True,
        help="nbest_predictions.json 파일 경로들 (여러 개 가능)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="출력 파일 경로 (TSV 포맷)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="각 모델의 가중치 (미지정 시 균등)",
    )
    parser.add_argument(
        "--score_key",
        type=str,
        choices=["probability", "score"],
        default="probability",
        help="사용할 score 필드명",
    )

    args = parser.parse_args()

    # Validation
    if args.weights and len(args.weights) != len(args.nbest_paths):
        raise ValueError(
            f"weights 개수({len(args.weights)}) != nbest_paths 개수({len(args.nbest_paths)})"
        )

    # 앙상블 수행
    ensemble_results = ensemble_nbest(
        nbest_paths=args.nbest_paths,
        weights=args.weights,
        score_key=args.score_key,
    )

    # 결과 저장
    save_predictions_csv(ensemble_results, args.output_path)

    logger.info("🎉 Ensemble complete!")


if __name__ == "__main__":
    main()
