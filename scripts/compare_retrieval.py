"""
Retrieval 성능 비교 스크립트

Gold context vs Retrieval context로 예측한 결과를 비교하여
retrieval의 영향을 분석합니다.

Usage:
    python scripts/compare_retrieval.py <output_dir>
    python scripts/compare_retrieval.py ./outputs/dahyeong/model_name
"""

import json
import csv
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import evaluate


def load_csv_predictions(csv_path: str) -> Dict[str, str]:
    """CSV 파일에서 예측 결과 로드 (TSV 형식)"""
    predictions = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) == 2:
                predictions[row[0]] = row[1]
    return predictions


def load_json_labels(json_path: str) -> Dict[str, Dict]:
    """JSON 파일에서 정답 레이블 로드"""
    with open(json_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels


def compute_metrics(predictions: Dict[str, str], references: Dict[str, Dict]) -> Dict:
    """SQuAD 메트릭 계산"""
    metric = evaluate.load("squad")

    # 형식 맞추기
    formatted_predictions = [
        {"id": id_, "prediction_text": pred_text}
        for id_, pred_text in predictions.items()
    ]
    formatted_references = [
        {"id": id_, "answers": ref} for id_, ref in references.items()
    ]

    return metric.compute(
        predictions=formatted_predictions, references=formatted_references
    )


def analyze_differences(
    gold_preds: Dict[str, str], retrieval_preds: Dict[str, str], labels: Dict[str, Dict]
) -> Dict:
    """Gold와 Retrieval 예측의 차이 분석"""

    metric = evaluate.load("squad")

    # 카테고리별 분류
    both_correct = []  # 둘 다 맞음
    gold_only_correct = []  # gold만 맞음 (retrieval 실패)
    retrieval_only_correct = []  # retrieval만 맞음 (드문 케이스)
    both_wrong = []  # 둘 다 틀림

    for id_ in gold_preds.keys():
        if id_ not in labels:
            continue

        gold_pred = gold_preds.get(id_, "")
        retrieval_pred = retrieval_preds.get(id_, "")
        ref = labels[id_]

        # 개별 EM 계산
        gold_result = metric.compute(
            predictions=[{"id": id_, "prediction_text": gold_pred}],
            references=[{"id": id_, "answers": ref}],
        )
        retrieval_result = metric.compute(
            predictions=[{"id": id_, "prediction_text": retrieval_pred}],
            references=[{"id": id_, "answers": ref}],
        )

        gold_correct = gold_result["exact_match"] == 100.0
        retrieval_correct = retrieval_result["exact_match"] == 100.0

        example_info = {
            "id": id_,
            "gold_pred": gold_pred,
            "retrieval_pred": retrieval_pred,
            "answer": ref["text"][0] if ref["text"] else "",
            "gold_em": gold_result["exact_match"],
            "retrieval_em": retrieval_result["exact_match"],
            "gold_f1": gold_result["f1"],
            "retrieval_f1": retrieval_result["f1"],
        }

        if gold_correct and retrieval_correct:
            both_correct.append(example_info)
        elif gold_correct and not retrieval_correct:
            gold_only_correct.append(example_info)
        elif not gold_correct and retrieval_correct:
            retrieval_only_correct.append(example_info)
        else:
            both_wrong.append(example_info)

    return {
        "both_correct": both_correct,
        "gold_only_correct": gold_only_correct,
        "retrieval_only_correct": retrieval_only_correct,
        "both_wrong": both_wrong,
    }


def print_summary(
    gold_metrics: Dict, retrieval_metrics: Dict, analysis: Dict, output_dir: Path
):
    """결과 요약 출력 및 저장"""

    total = sum(len(v) for v in analysis.values())
    both_correct = len(analysis["both_correct"])
    gold_only = len(analysis["gold_only_correct"])
    retrieval_only = len(analysis["retrieval_only_correct"])
    both_wrong = len(analysis["both_wrong"])

    # Retrieval 성공률 계산
    retrieval_success_rate = (
        (both_correct + retrieval_only) / total * 100 if total > 0 else 0
    )
    retrieval_failure_rate = gold_only / total * 100 if total > 0 else 0

    summary = f"""
================================================================================
📊 RETRIEVAL PERFORMANCE COMPARISON
================================================================================

1. Overall Metrics
   {"=" * 76}
   Gold Context (Upper Bound):
      - EM: {gold_metrics["exact_match"]:.2f}
      - F1: {gold_metrics["f1"]:.2f}
   
   With Retrieval (Actual Performance):
      - EM: {retrieval_metrics["exact_match"]:.2f}
      - F1: {retrieval_metrics["f1"]:.2f}
   
   Performance Gap:
      - EM Drop: {gold_metrics["exact_match"] - retrieval_metrics["exact_match"]:.2f} points
      - F1 Drop: {gold_metrics["f1"] - retrieval_metrics["f1"]:.2f} points

2. Detailed Analysis (Total: {total} examples)
   {"=" * 76}
   ✅ Both Correct:           {both_correct:4d} ({both_correct / total * 100:5.1f}%)
   ⚠️  Gold Only Correct:     {gold_only:4d} ({gold_only / total * 100:5.1f}%) ← Retrieval Failed
   🔄 Retrieval Only Correct: {retrieval_only:4d} ({retrieval_only / total * 100:5.1f}%)
   ❌ Both Wrong:             {both_wrong:4d} ({both_wrong / total * 100:5.1f}%)
   
3. Retrieval Impact
   {"=" * 76}
   Retrieval Success Rate:  {retrieval_success_rate:.1f}% (정답 유지 + retrieval만 맞춤)
   Retrieval Failure Rate:  {retrieval_failure_rate:.1f}% (gold는 맞았지만 retrieval 실패)
   
   💡 Interpretation:
      - {retrieval_failure_rate:.1f}%의 경우, 잘못된 context를 retrieval하여 성능 저하
      - 이론적 최대 성능 향상 가능성: {retrieval_failure_rate:.1f} EM points

================================================================================
📁 Detailed results saved to:
   - {output_dir}/retrieval_comparison.json
   - {output_dir}/retrieval_failures.json (gold는 맞았지만 retrieval 실패한 케이스)
================================================================================
"""

    print(summary)

    # 결과 저장
    comparison_result = {
        "gold_metrics": gold_metrics,
        "retrieval_metrics": retrieval_metrics,
        "performance_gap": {
            "em_drop": gold_metrics["exact_match"] - retrieval_metrics["exact_match"],
            "f1_drop": gold_metrics["f1"] - retrieval_metrics["f1"],
        },
        "counts": {
            "total": total,
            "both_correct": both_correct,
            "gold_only_correct": gold_only,
            "retrieval_only_correct": retrieval_only,
            "both_wrong": both_wrong,
        },
        "rates": {
            "retrieval_success_rate": retrieval_success_rate,
            "retrieval_failure_rate": retrieval_failure_rate,
        },
    }

    with open(output_dir / "retrieval_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_result, f, indent=2, ensure_ascii=False)

    # Retrieval 실패 케이스 상세 저장 (개선 포인트)
    with open(output_dir / "retrieval_failures.json", "w", encoding="utf-8") as f:
        json.dump(analysis["gold_only_correct"], f, indent=2, ensure_ascii=False)

    print(f"✅ Comparison complete! Check {output_dir} for detailed results.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_retrieval.py <output_dir>")
        print(
            "Example: python scripts/compare_retrieval.py ./outputs/dahyeong/model_name"
        )
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    # 파일 경로
    gold_csv = output_dir / "eval_pred_gold.csv"
    retrieval_csv = output_dir / "eval_pred_retrieval.csv"
    labels_json = output_dir / "eval_labels.json"

    # 파일 존재 확인
    missing_files = []
    if not gold_csv.exists():
        missing_files.append(str(gold_csv))
    if not retrieval_csv.exists():
        missing_files.append(str(retrieval_csv))
    if not labels_json.exists():
        missing_files.append(str(labels_json))

    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print(
            "\n💡 Tip: Run training with eval_retrieval=true to generate all required files"
        )
        sys.exit(1)

    print(f"📂 Loading predictions from {output_dir}")

    # 데이터 로드
    gold_preds = load_csv_predictions(gold_csv)
    retrieval_preds = load_csv_predictions(retrieval_csv)
    labels = load_json_labels(labels_json)

    print(f"   ✅ Loaded {len(gold_preds)} gold predictions")
    print(f"   ✅ Loaded {len(retrieval_preds)} retrieval predictions")
    print(f"   ✅ Loaded {len(labels)} labels")

    # 메트릭 계산
    print("\n📊 Computing metrics...")
    gold_metrics = compute_metrics(gold_preds, labels)
    retrieval_metrics = compute_metrics(retrieval_preds, labels)

    # 차이 분석
    print("🔍 Analyzing differences...")
    analysis = analyze_differences(gold_preds, retrieval_preds, labels)

    # 결과 출력
    print_summary(gold_metrics, retrieval_metrics, analysis, output_dir)


if __name__ == "__main__":
    main()
