"""
이종 모델 앙상블 (Heterogeneous Model Ensemble)

서로 다른 tokenizer/architecture를 가진 모델들의 예측을 Text-level Voting으로 앙상블

지원 모델 조합 예시:
- RoBERTa-Large + KoELECTRA + BERT-Base + KoBigBird
- 서로 다른 vocab_size, hidden_size를 가진 모델들도 조합 가능

앙상블 방식:
1. 각 모델이 독립적으로 inference 수행 → nbest_predictions.json 생성
2. 정규화된 answer text 기준으로 weighted voting
3. 가장 높은 점수의 answer 선택 (원본 텍스트 중 최고 확률 것 반환)

사용 예시:
  # 기본 사용 (nbest 파일들 직접 지정)
  python scripts/hetero_ensemble.py \\
    --nbest_paths outputs/roberta/nbest_predictions.json \\
                  outputs/koelectra/nbest_predictions.json \\
                  outputs/bert/nbest_predictions.json \\
    --weights 0.5 0.3 0.2 \\
    --output_path outputs/hetero_ensemble/predictions.json

  # output_dir들로 지정 (자동으로 nbest_predictions.json 탐색)
  python scripts/hetero_ensemble.py \\
    --output_dirs outputs/roberta outputs/koelectra outputs/bert \\
    --weights 0.5 0.3 0.2 \\
    --output_path outputs/hetero_ensemble/predictions.json

  # Validation 모드 (정답과 비교하여 EM/F1 계산)
  python scripts/hetero_ensemble.py \\
    --output_dirs outputs/roberta outputs/koelectra \\
    --weights 0.6 0.4 \\
    --output_path outputs/hetero_ensemble/predictions.json \\
    --eval_file ./data/train_dataset/validation

차이점 (기존 ensemble.py vs 이 스크립트):
- ensemble.py: Logit-level 앙상블 (동일 tokenizer 필수)
- hetero_ensemble.py: Text-level 앙상블 (이종 모델 조합 가능)
"""

import os
import sys
import json
import argparse
import logging
import re
import glob
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================
# 정규화 함수
# ============================================================


def normalize_answer(text: str) -> str:
    """
    Answer 정규화 (EM 평가와 동일한 방식)

    - 소문자화
    - 구두점 제거
    - 연속 공백 → 단일 공백
    - 앞뒤 공백 제거
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 데이터 로딩
# ============================================================


def find_nbest_file(output_dir: str) -> Optional[str]:
    """
    output_dir에서 nbest_predictions.json 파일 탐색

    탐색 순서:
    1. output_dir/nbest_predictions.json
    2. output_dir/checkpoint-*/nbest_predictions.json (최신 것)
    3. output_dir/**/nbest_predictions.json
    """
    # 1. 직접 경로
    direct_path = os.path.join(output_dir, "nbest_predictions.json")
    if os.path.exists(direct_path):
        return direct_path

    # 2. checkpoint 폴더 내
    checkpoint_pattern = os.path.join(
        output_dir, "checkpoint-*", "nbest_predictions.json"
    )
    checkpoint_files = glob.glob(checkpoint_pattern)
    if checkpoint_files:
        # 가장 최신 checkpoint
        def get_step(path):
            try:
                return int(os.path.basename(os.path.dirname(path)).split("-")[1])
            except:
                return 0

        checkpoint_files.sort(key=get_step, reverse=True)
        return checkpoint_files[0]

    # 3. 재귀 탐색
    recursive_pattern = os.path.join(output_dir, "**", "nbest_predictions.json")
    found = glob.glob(recursive_pattern, recursive=True)
    if found:
        return found[0]

    return None


def load_nbest(path: str) -> Dict[str, List[Dict]]:
    """
    nbest_predictions.json 로드

    Expected format:
    {
        "qid1": [
            {"text": "answer1", "probability": 0.9, "start_logit": ..., "end_logit": ...},
            {"text": "answer2", "probability": 0.05, ...},
            ...
        ],
        ...
    }
    """
    logger.info(f"📖 Loading: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # List 형태인 경우 dict로 변환
    if isinstance(data, list):
        result = defaultdict(list)
        for item in data:
            qid = item.get("id")
            if qid:
                result[qid].append(item)
        return dict(result)

    return data


def load_answers(eval_file: str) -> Dict[str, List[str]]:
    """
    정답 파일 로드 (validation 평가용)

    Returns:
        {qid: [answer1, answer2, ...]}  # 복수 정답 지원
    """
    from datasets import load_from_disk

    dataset = load_from_disk(eval_file)

    answers = {}
    for example in dataset:
        qid = example["id"]
        answer_texts = example.get("answers", {}).get("text", [])
        if answer_texts:
            answers[qid] = answer_texts

    return answers


# ============================================================
# 앙상블 로직
# ============================================================


@dataclass
class EnsembleConfig:
    """앙상블 설정"""

    top_k_candidates: int = 5  # 각 모델에서 고려할 후보 수
    score_key: str = "probability"  # "probability" or "start_logit" + "end_logit"
    use_rank_score: bool = False  # True면 rank 기반 점수 사용


def ensemble_predictions(
    nbest_list: List[Dict[str, List[Dict]]],
    weights: List[float],
    config: EnsembleConfig = None,
) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """
    Text-level Weighted Voting 앙상블

    Args:
        nbest_list: 각 모델의 nbest predictions [{qid: [candidates]}, ...]
        weights: 모델별 가중치 (정규화됨)
        config: 앙상블 설정

    Returns:
        predictions: {qid: best_answer}
        details: {qid: {answer_scores, selected_answer, ...}}
    """
    if config is None:
        config = EnsembleConfig()

    # 모든 question ID 수집
    all_qids = set()
    for nbest_data in nbest_list:
        all_qids.update(nbest_data.keys())

    logger.info(f"📊 Total questions: {len(all_qids)}")
    logger.info(f"🎯 Top-k candidates per model: {config.top_k_candidates}")

    predictions = {}
    details = {}

    for qid in all_qids:
        # 정규화된 answer → 누적 weighted score
        answer_scores = defaultdict(float)
        # 정규화된 answer → (best_prob, original_text)
        answer_originals = {}

        for model_idx, nbest_data in enumerate(nbest_list):
            weight = weights[model_idx]

            if qid not in nbest_data:
                continue

            candidates = nbest_data[qid][: config.top_k_candidates]

            for rank, candidate in enumerate(candidates):
                text = candidate.get("text", "")
                if not text:
                    continue

                # Score 계산
                if config.use_rank_score:
                    # Rank 기반: 1위=1.0, 2위=0.8, 3위=0.6, ...
                    score = max(0.2, 1.0 - rank * 0.2)
                else:
                    # Probability 기반
                    score = candidate.get(config.score_key, 0.0)
                    if score <= 0:
                        # logit 합산 fallback
                        start_logit = candidate.get("start_logit", 0)
                        end_logit = candidate.get("end_logit", 0)
                        score = start_logit + end_logit

                normalized = normalize_answer(text)
                if not normalized:
                    continue

                # Weighted score 누적
                answer_scores[normalized] += weight * score

                # 가장 높은 probability를 가진 원본 텍스트 보존
                prob = candidate.get("probability", score)
                if (
                    normalized not in answer_originals
                    or prob > answer_originals[normalized][0]
                ):
                    answer_originals[normalized] = (prob, text)

        # 최고 점수 answer 선택
        if answer_scores:
            best_normalized = max(answer_scores, key=answer_scores.get)
            best_answer = answer_originals[best_normalized][1]
            predictions[qid] = best_answer

            # 상세 정보 저장 (디버깅용)
            details[qid] = {
                "selected": best_answer,
                "normalized": best_normalized,
                "score": answer_scores[best_normalized],
                "all_scores": dict(
                    sorted(answer_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                ),  # top-5만 저장
            }
        else:
            predictions[qid] = ""
            details[qid] = {"selected": "", "error": "no_candidates"}
            logger.warning(f"⚠️ No valid candidates for qid: {qid}")

    return predictions, details


# ============================================================
# 평가
# ============================================================


def compute_em_f1(
    predictions: Dict[str, str], answers: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    EM, F1 계산
    """
    em_scores = []
    f1_scores = []

    for qid, pred in predictions.items():
        if qid not in answers:
            continue

        gold_answers = answers[qid]
        pred_normalized = normalize_answer(pred)

        # EM: 하나라도 일치하면 1
        em = max(
            int(normalize_answer(gold) == pred_normalized) for gold in gold_answers
        )
        em_scores.append(em)

        # F1: 최대 F1
        def token_f1(pred_tokens, gold_tokens):
            common = set(pred_tokens) & set(gold_tokens)
            if not common:
                return 0.0
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(gold_tokens) if gold_tokens else 0
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

        pred_tokens = pred_normalized.split()
        f1 = max(
            token_f1(pred_tokens, normalize_answer(gold).split())
            for gold in gold_answers
        )
        f1_scores.append(f1)

    return {
        "em": sum(em_scores) / len(em_scores) * 100 if em_scores else 0,
        "f1": sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0,
        "total": len(em_scores),
    }


# ============================================================
# 출력
# ============================================================


def save_predictions(
    predictions: Dict[str, str],
    output_path: str,
    details: Optional[Dict] = None,
    ordered_ids: Optional[List[str]] = None,
    ensemble_config: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
):
    """
    예측 결과를 단일 모델과 동일한 구조로 저장

    저장되는 파일들:
    - predictions.json: {qid: answer}
    - predictions_submit.csv: TSV 형식 (리더보드 제출용)
    - nbest_predictions.json: 앙상블 상세 정보 (다른 앙상블의 입력으로 사용 가능)
    - eval_results.json: EM/F1 평가 결과
    - config.json: 앙상블 설정 정보
    """
    # output_path에서 디렉토리 추출
    if output_path.endswith(".json") or output_path.endswith(".csv"):
        output_dir = os.path.dirname(output_path) or "."
    else:
        output_dir = output_path

    os.makedirs(output_dir, exist_ok=True)

    # 순서가 지정된 경우 OrderedDict 사용
    if ordered_ids:
        from collections import OrderedDict

        ordered_preds = OrderedDict()
        for qid in ordered_ids:
            if qid in predictions:
                ordered_preds[qid] = predictions[qid]
        predictions = ordered_preds

    # 1. predictions.json 저장
    pred_path = os.path.join(output_dir, "predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 predictions.json saved: {pred_path}")

    # 2. predictions_submit.csv 저장 (제출용)
    csv_path = os.path.join(output_dir, "predictions_submit.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for qid, answer in predictions.items():
            answer = answer.replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{answer}\n")
    logger.info(f"💾 predictions_submit.csv saved: {csv_path}")

    # 3. nbest_predictions.json 저장 (다른 앙상블 입력으로 재사용 가능)
    if details:
        nbest_path = os.path.join(output_dir, "nbest_predictions.json")
        # details를 nbest 형식으로 변환
        nbest_format = {}
        for qid, detail in details.items():
            nbest_format[qid] = [
                {
                    "text": detail.get("selected", ""),
                    "probability": detail.get("score", 0.0),
                    "normalized": detail.get("normalized", ""),
                }
            ]
            # all_scores에서 추가 후보들도 포함
            if "all_scores" in detail:
                for norm_text, score in list(detail["all_scores"].items())[1:5]:
                    nbest_format[qid].append(
                        {
                            "text": norm_text,  # 정규화된 텍스트
                            "probability": score,
                            "normalized": norm_text,
                        }
                    )

        with open(nbest_path, "w", encoding="utf-8") as f:
            json.dump(nbest_format, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 nbest_predictions.json saved: {nbest_path}")

        # 상세 정보도 별도 저장
        details_path = os.path.join(output_dir, "ensemble_details.json")
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 ensemble_details.json saved: {details_path}")

    # 4. eval_results.json 저장 (다른 모델과 동일한 형식)
    if metrics:
        eval_path = os.path.join(output_dir, "eval_results.json")
        eval_results = {
            "eval_exact_match": metrics.get("em", 0),
            "eval_f1": metrics.get("f1", 0),
            "eval_total": metrics.get("total", 0),
        }
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 eval_results.json saved: {eval_path}")

    # 5. config.json 저장 (앙상블 설정)
    if ensemble_config:
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(ensemble_config, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 config.json saved: {config_path}")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="이종 모델 앙상블 (Text-level Weighted Voting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # nbest 파일 직접 지정
  python scripts/hetero_ensemble.py \\
    --nbest_paths outputs/roberta/nbest_predictions.json \\
                  outputs/koelectra/nbest_predictions.json \\
    --weights 0.6 0.4 \\
    --output_path outputs/ensemble/hetero_pred.json

  # output_dir로 지정 (자동 탐색)
  python scripts/hetero_ensemble.py \\
    --output_dirs outputs/roberta outputs/koelectra \\
    --weights 0.6 0.4 \\
    --output_path outputs/ensemble/hetero_pred.json

  # Validation 평가
  python scripts/hetero_ensemble.py \\
    --output_dirs outputs/roberta outputs/koelectra \\
    --weights 0.6 0.4 \\
    --output_path outputs/ensemble/hetero_pred.json \\
    --eval_file ./data/train_dataset/validation
        """,
    )

    # 입력 (둘 중 하나 필수)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--nbest_paths",
        nargs="+",
        help="nbest_predictions.json 파일 경로들",
    )
    input_group.add_argument(
        "--output_dirs",
        nargs="+",
        help="모델 output directory들 (자동으로 nbest 파일 탐색)",
    )

    # 가중치
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="각 모델의 가중치 (미지정 시 균등)",
    )
    parser.add_argument(
        "--auto_weight_by_em",
        action="store_true",
        help="EM 점수 기반 자동 가중치 설정 (--eval_file 필요)",
    )

    # 출력
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="출력 파일 경로 (.json 또는 .csv)",
    )

    # 평가
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="정답 파일 경로 (지정 시 EM/F1 계산)",
    )

    # 앙상블 설정
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="각 모델에서 고려할 후보 수 (default: 5)",
    )
    parser.add_argument(
        "--score_key",
        type=str,
        default="probability",
        choices=["probability", "score"],
        help="사용할 score 필드 (default: probability)",
    )
    parser.add_argument(
        "--use_rank_score",
        action="store_true",
        help="Rank 기반 점수 사용 (1위=1.0, 2위=0.8, ...)",
    )
    parser.add_argument(
        "--save_details",
        action="store_true",
        help="앙상블 상세 정보 저장",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/train_dataset/validation",
        help="원본 데이터셋 경로 (순서 유지용, default: ./data/train_dataset/validation)",
    )

    args = parser.parse_args()

    # nbest 파일 경로 수집
    if args.nbest_paths:
        nbest_paths = args.nbest_paths
    else:
        nbest_paths = []
        for output_dir in args.output_dirs:
            nbest_file = find_nbest_file(output_dir)
            if nbest_file:
                nbest_paths.append(nbest_file)
            else:
                logger.error(f"❌ nbest_predictions.json not found in: {output_dir}")
                sys.exit(1)

    num_models = len(nbest_paths)

    # 가중치 처리
    if args.auto_weight_by_em:
        # EM 점수 기반 자동 가중치
        if not args.eval_file:
            logger.error("❌ --auto_weight_by_em 사용시 --eval_file 필요")
            sys.exit(1)

        logger.info("📊 Computing EM-based weights...")
        answers = load_answers(args.eval_file)
        em_scores = []

        for nbest_path in nbest_paths:
            # predictions.json 찾기
            pred_path = os.path.join(os.path.dirname(nbest_path), "predictions.json")
            if os.path.exists(pred_path):
                with open(pred_path) as f:
                    preds = json.load(f)
                em = compute_em_f1(preds, answers)["em"]
                em_scores.append(em)
                model_name = os.path.basename(os.path.dirname(nbest_path))
                logger.info(f"   {model_name}: EM = {em:.2f}%")
            else:
                em_scores.append(50.0)  # 기본값
                logger.warning(f"   predictions.json not found, using default EM=50")

        # EM 점수를 가중치로 변환 (정규화)
        weights = [em / sum(em_scores) for em in em_scores]
        logger.info(f"📊 Auto weights: {[f'{w:.3f}' for w in weights]}")
    elif args.weights:
        if len(args.weights) != num_models:
            logger.error(
                f"❌ weights 개수({len(args.weights)}) != 모델 수({num_models})"
            )
            sys.exit(1)
        weights = [w / sum(args.weights) for w in args.weights]
    else:
        weights = [1.0 / num_models] * num_models

    # 설정 출력
    print("\n" + "=" * 60)
    print("🔀 Heterogeneous Model Ensemble")
    print("=" * 60)
    print(f"   Models: {num_models}")
    for i, (path, w) in enumerate(zip(nbest_paths, weights)):
        print(
            f"   [{i + 1}] {os.path.basename(os.path.dirname(path))} (weight: {w:.3f})"
        )
        print(f"       → {path}")
    print(f"   Top-k candidates: {args.top_k}")
    print(f"   Score key: {args.score_key}")
    print(f"   Use rank score: {args.use_rank_score}")
    print("=" * 60)

    # nbest 로드
    nbest_list = [load_nbest(path) for path in nbest_paths]

    # 앙상블 수행
    config = EnsembleConfig(
        top_k_candidates=args.top_k,
        score_key=args.score_key,
        use_rank_score=args.use_rank_score,
    )

    predictions, details = ensemble_predictions(nbest_list, weights, config)

    logger.info(f"✅ Ensemble complete: {len(predictions)} predictions")

    # 평가 (옵션)
    metrics = None
    if args.eval_file:
        logger.info(f"\n📊 Evaluating against: {args.eval_file}")
        answers = load_answers(args.eval_file)
        metrics = compute_em_f1(predictions, answers)
        print("\n" + "=" * 40)
        print("📈 Evaluation Results")
        print("=" * 40)
        print(f"   EM:  {metrics['em']:.2f}%")
        print(f"   F1:  {metrics['f1']:.2f}%")
        print(f"   Total: {metrics['total']} questions")
        print("=" * 40)

    # 원본 데이터셋 순서 가져오기
    ordered_ids = None
    if os.path.exists(args.dataset_path):
        try:
            from datasets import load_from_disk

            dataset = load_from_disk(args.dataset_path)
            ordered_ids = [ex["id"] for ex in dataset]
            logger.info(
                f"📝 Loaded original order from: {args.dataset_path} ({len(ordered_ids)} examples)"
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not load dataset order: {e}")

    # 앙상블 설정 저장용
    model_names = [os.path.basename(os.path.dirname(p)) for p in nbest_paths]
    ensemble_config = {
        "ensemble_type": "hetero_text_voting",
        "models": model_names,
        "weights": weights,
        "top_k_candidates": args.top_k,
        "score_key": args.score_key,
        "use_rank_score": args.use_rank_score,
        "eval_file": args.eval_file,
        "dataset_path": args.dataset_path,
    }

    # 저장
    save_predictions(
        predictions,
        args.output_path,
        details if args.save_details else None,
        ordered_ids=ordered_ids,
        ensemble_config=ensemble_config,
        metrics=metrics,
    )

    print("\n🎉 Heterogeneous ensemble complete!")
    print(f"📂 Output directory: {os.path.dirname(args.output_path) or '.'}")


if __name__ == "__main__":
    main()
