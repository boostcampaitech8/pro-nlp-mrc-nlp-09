"""
헤테로 앙상블 가중치 Grid Search 최적화

Validation EM 점수 기준으로 최적의 가중치 조합을 찾습니다.

사용법:
  python scripts/grid_search_weights.py
  python scripts/grid_search_weights.py --step 0.05  # 더 정밀한 탐색
  python scripts/grid_search_weights.py --models model1 model2 model3
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize_answer(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_nbest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensemble_em(
    nbest_list: List[Dict],
    weights: List[float],
    answers: Dict[str, List[str]],
    top_k: int = 5,
) -> float:
    """주어진 가중치로 앙상블 후 EM 계산"""
    all_qids = set()
    for nb in nbest_list:
        all_qids.update(nb.keys())

    correct = 0
    total = 0

    for qid in all_qids:
        if qid not in answers:
            continue

        answer_scores = defaultdict(float)
        answer_originals = {}

        for model_idx, nbest in enumerate(nbest_list):
            w = weights[model_idx]
            if qid not in nbest:
                continue

            for cand in nbest[qid][:top_k]:
                text = cand.get("text", "")
                prob = cand.get("probability", 0)
                if not text:
                    continue

                norm = normalize_answer(text)
                if not norm:
                    continue

                answer_scores[norm] += w * prob
                if norm not in answer_originals or prob > answer_originals[norm][0]:
                    answer_originals[norm] = (prob, text)

        if answer_scores:
            best_norm = max(answer_scores, key=answer_scores.get)
            pred = answer_originals[best_norm][1]
            pred_norm = normalize_answer(pred)
            if any(normalize_answer(g) == pred_norm for g in answers[qid]):
                correct += 1

        total += 1

    return correct / total * 100 if total else 0


def generate_weight_combinations(
    n_models: int, step: float = 0.1, min_weight: float = 0.0
):
    """가중치 조합 생성 (합이 1이 되도록)"""
    candidates = [
        round(i * step, 2) for i in range(int(min_weight / step), int(1 / step) + 1)
    ]

    if n_models == 2:
        for w1 in candidates:
            w2 = round(1.0 - w1, 2)
            if w2 >= min_weight:
                yield [w1, w2]

    elif n_models == 3:
        for w1 in candidates:
            for w2 in candidates:
                w3 = round(1.0 - w1 - w2, 2)
                if w3 >= min_weight and w3 <= 1.0:
                    yield [w1, w2, w3]

    elif n_models == 4:
        for w1 in candidates:
            for w2 in candidates:
                for w3 in candidates:
                    w4 = round(1.0 - w1 - w2 - w3, 2)
                    if w4 >= min_weight and w4 <= 1.0:
                        yield [w1, w2, w3, w4]

    else:
        # 5개 이상은 균등 가중치만
        yield [1.0 / n_models] * n_models


def main():
    parser = argparse.ArgumentParser(description="Grid Search for ensemble weights")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["roberta_large_vanilla", "koelectra", "kobigbird"],
        help="모델 이름들 (shared outputs 기준)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="가중치 탐색 단위 (default: 0.1)",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=0.0,
        help="최소 가중치 (default: 0.0)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="각 모델에서 고려할 후보 수",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/ephemeral/home/shared/outputs/dahyeong",
        help="모델 출력 디렉토리",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🔍 Grid Search for Optimal Ensemble Weights")
    print("=" * 60)

    # 데이터 로드
    print("\n📂 Loading validation data...")
    from datasets import load_from_disk

    val = load_from_disk("./data/train_dataset/validation")
    answers = {ex["id"]: ex["answers"]["text"] for ex in val}
    print(f"   Loaded {len(answers)} examples")

    # nbest 파일 로드
    print("\n📂 Loading nbest predictions...")
    nbest_list = []
    model_names = []

    for model_name in args.models:
        nbest_path = os.path.join(args.output_dir, model_name, "nbest_predictions.json")
        if os.path.exists(nbest_path):
            nbest_list.append(load_nbest(nbest_path))
            model_names.append(model_name)
            print(f"   ✅ {model_name}")
        else:
            print(f"   ❌ {model_name} - nbest not found")

    if len(nbest_list) < 2:
        print("\n❌ Need at least 2 models for ensemble!")
        sys.exit(1)

    n_models = len(nbest_list)

    # 단일 모델 성능 계산
    print("\n📊 Single model performance:")
    print("-" * 40)
    for i, name in enumerate(model_names):
        single_weights = [0.0] * n_models
        single_weights[i] = 1.0
        em = ensemble_em(nbest_list, single_weights, answers, args.top_k)
        print(f"   {name:30s}: EM = {em:.2f}%")

    # Grid Search
    print(f"\n🔍 Searching with step={args.step}, min_weight={args.min_weight}...")

    results = []
    total_combinations = 0

    for weights in generate_weight_combinations(n_models, args.step, args.min_weight):
        em = ensemble_em(nbest_list, weights, answers, args.top_k)
        results.append((em, weights))
        total_combinations += 1

    print(f"   Evaluated {total_combinations} combinations")

    # 결과 정렬
    results.sort(key=lambda x: -x[0])

    # Top 10 출력
    print("\n" + "=" * 60)
    print("📊 Top 10 Weight Combinations")
    print("=" * 60)

    header = f"{'Rank':>4} | {'EM':>7} | Weights"
    print(header)
    print("-" * 60)

    for i, (em, weights) in enumerate(results[:10], 1):
        weight_str = ", ".join([f"{w:.2f}" for w in weights])
        print(f"{i:4d} | {em:6.2f}% | ({weight_str})")

    # Best 결과
    best_em, best_weights = results[0]
    print("\n" + "=" * 60)
    print("🏆 BEST RESULT")
    print("=" * 60)
    print(f"   EM: {best_em:.2f}%")
    print(f"   Weights:")
    for name, w in zip(model_names, best_weights):
        print(f"      {name}: {w:.2f}")

    # 명령어 출력
    print("\n📋 Command to run ensemble with best weights:")
    print("-" * 60)
    weight_args = " ".join([f"{w:.2f}" for w in best_weights])
    model_args = " ".join(model_names)
    print(f'make hetero-ensemble MODELS="{model_args}" WEIGHTS="{weight_args}"')

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
