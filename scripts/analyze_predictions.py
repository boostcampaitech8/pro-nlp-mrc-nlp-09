#!/usr/bin/env python2
"""
두 개의 prediction CSV 파일을 비교 분석하는 스크립트

Usage:
    python analyze_predictions.py <csv1> <csv2> [--output-dir <dir>]
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from datetime import datetime


def normalize_answer(s: str) -> str:
    """정답 문자열을 정규화합니다."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        # 한글, 영문, 숫자를 제외한 문자 제거
        return re.sub(r"[^\w\s]", "", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact Match 점수를 계산합니다."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """F1 점수를 계산합니다."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def read_csv(file_path: Path) -> Dict[str, str]:
    """CSV 파일을 읽어 {id: prediction} 딕셔너리로 반환합니다."""
    predictions = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 탭, 쉼표, 공백 등으로 분리 시도
            parts = None
            if "\t" in line:
                parts = line.split("\t", 1)
            elif "," in line:
                parts = line.split(",", 1)
            else:
                # 공백으로 분리 (최소 2개 이상의 공백)
                parts = line.split(None, 1)

            if parts and len(parts) >= 2:
                predictions[parts[0].strip()] = parts[1].strip()

    return predictions


def analyze_predictions(csv1_path: Path, csv2_path: Path) -> Tuple[List[Dict], Dict]:
    """두 CSV 파일을 분석하여 결과를 반환합니다."""
    pred1 = read_csv(csv1_path)
    pred2 = read_csv(csv2_path)

    # 파일 길이 체크
    len1 = len(pred1)
    len2 = len(pred2)

    if len1 != len2:
        print(f"\n⚠️  경고: 두 CSV 파일의 행 개수가 다릅니다!")
        print(f"  - CSV 1: {len1}개")
        print(f"  - CSV 2: {len2}개")

        # 데이터셋 타입 추론
        if len1 == 240 or len2 == 240:
            print(f"  → Validation 데이터셋으로 추정됩니다 (기대값: 240개)")
        elif len1 == 600 or len2 == 600:
            print(f"  → Test 데이터셋으로 추정됩니다 (기대값: 600개)")
        print()

    # 모든 ID 수집
    all_ids = sorted(set(pred1.keys()) | set(pred2.keys()))

    results = []
    stats = {
        "total": len(all_ids),
        "both_correct": 0,
        "only_pred1_correct": 0,
        "only_pred2_correct": 0,
        "both_wrong": 0,
        "agreement": 0,
        "disagreement": 0,
        "pred1_em_sum": 0,
        "pred2_em_sum": 0,
        "pred1_f1_sum": 0,
        "pred2_f1_sum": 0,
        "errors": [],
        "answer_length_dist": defaultdict(int),
        "diff_patterns": defaultdict(int),
    }

    for qid in all_ids:
        p1 = pred1.get(qid, "")
        p2 = pred2.get(qid, "")

        # 두 예측이 동일한지 확인
        em_between = compute_exact_match(p1, p2)
        f1_between = compute_f1(p1, p2)

        result = {
            "id": qid,
            "pred1": p1,
            "pred2": p2,
            "em": em_between,
            "f1": f1_between,
        }

        results.append(result)

        # 통계 수집
        stats["pred1_em_sum"] += em_between
        stats["pred1_f1_sum"] += f1_between

        if em_between == 1.0:
            stats["agreement"] += 1
        else:
            stats["disagreement"] += 1
            stats["errors"].append(
                {
                    "id": qid,
                    "pred1": p1,
                    "pred2": p2,
                    "f1": f1_between,
                }
            )

        # 답변 길이 분포
        len1 = len(p1.strip())
        len2 = len(p2.strip())
        stats["answer_length_dist"][f"{len1}-{len2}"] += 1

        # 차이 패턴 분석
        if p1 and p2:
            if len(p1) > len(p2) * 2:
                stats["diff_patterns"]["pred1_much_longer"] += 1
            elif len(p2) > len(p1) * 2:
                stats["diff_patterns"]["pred2_much_longer"] += 1
            elif p1[:10] == p2[:10]:
                stats["diff_patterns"]["same_prefix"] += 1
            elif p1[-10:] == p2[-10:]:
                stats["diff_patterns"]["same_suffix"] += 1

    return results, stats


def save_comparison_csv(results: List[Dict], output_path: Path):
    """비교 결과를 CSV로 저장합니다."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pred1", "pred2", "EM", "F1"])

        for result in results:
            writer.writerow(
                [
                    result["id"],
                    result["pred1"],
                    result["pred2"],
                    f"{result['em']:.4f}",
                    f"{result['f1']:.4f}",
                ]
            )


def generate_analysis_report(
    results: List[Dict], stats: Dict, csv1_name: str, csv2_name: str, output_path: Path
):
    """분석 보고서를 마크다운으로 생성합니다."""
    total = stats["total"]
    agreement_rate = stats["agreement"] / total * 100 if total > 0 else 0
    avg_f1 = stats["pred1_f1_sum"] / total if total > 0 else 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Prediction 비교 분석 보고서\n\n")
        f.write(f"**생성 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**비교 파일**:\n")
        f.write(f"- Prediction 1: `{csv1_name}`\n")
        f.write(f"- Prediction 2: `{csv2_name}`\n\n")

        f.write(f"---\n\n")
        f.write(f"## 📊 전체 통계\n\n")
        f.write(f"| 항목 | 값 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| 전체 예시 수 | {total} |\n")
        f.write(
            f"| 일치 (Agreement) | {stats['agreement']} ({agreement_rate:.2f}%) |\n"
        )
        f.write(
            f"| 불일치 (Disagreement) | {stats['disagreement']} ({100 - agreement_rate:.2f}%) |\n"
        )
        f.write(f"| 평균 F1 Score | {avg_f1:.4f} |\n\n")

        f.write(f"---\n\n")
        f.write(f"## 🔍 불일치 예시 분석\n\n")
        f.write(
            f"두 prediction이 다른 경우는 총 **{stats['disagreement']}개**입니다.\n\n"
        )

        if stats["errors"]:
            # F1 점수가 낮은 순으로 정렬
            sorted_errors = sorted(stats["errors"], key=lambda x: x["f1"])

            f.write(f"### Top 20 불일치 예시 (F1 낮은 순)\n\n")
            for i, error in enumerate(sorted_errors[:20], 1):
                f.write(f"#### {i}. ID: `{error['id']}`\n\n")
                f.write(f"- **Prediction 1**: {error['pred1']}\n")
                f.write(f"- **Prediction 2**: {error['pred2']}\n")
                f.write(f"- **F1 Score**: {error['f1']:.4f}\n\n")

        f.write(f"---\n\n")
        f.write(f"## 📈 차이 패턴 분석\n\n")

        if stats["diff_patterns"]:
            f.write(f"| 패턴 | 빈도 |\n")
            f.write(f"|------|------|\n")
            for pattern, count in sorted(
                stats["diff_patterns"].items(), key=lambda x: x[1], reverse=True
            ):
                pattern_name = {
                    "pred1_much_longer": "Pred1이 훨씬 긺 (2배 이상)",
                    "pred2_much_longer": "Pred2가 훨씬 긺 (2배 이상)",
                    "same_prefix": "동일한 접두사",
                    "same_suffix": "동일한 접미사",
                }.get(pattern, pattern)
                f.write(f"| {pattern_name} | {count} |\n")
            f.write(f"\n")

        f.write(f"---\n\n")
        f.write(f"## 📏 답변 길이 분포\n\n")
        f.write(f"상위 10개 (Pred1 길이 - Pred2 길이 쌍):\n\n")

        if stats["answer_length_dist"]:
            sorted_lengths = sorted(
                stats["answer_length_dist"].items(), key=lambda x: x[1], reverse=True
            )[:10]
            f.write(f"| Pred1 길이 - Pred2 길이 | 빈도 |\n")
            f.write(f"|-------------------------|------|\n")
            for length_pair, count in sorted_lengths:
                f.write(f"| {length_pair} | {count} |\n")
            f.write(f"\n")

        f.write(f"---\n\n")
        f.write(f"## 💡 인사이트 및 권장사항\n\n")

        # 자동 인사이트 생성
        insights = []

        if agreement_rate > 90:
            insights.append(
                f"✅ 두 모델의 예측이 {agreement_rate:.1f}% 일치하여 매우 높은 일관성을 보입니다."
            )
        elif agreement_rate > 70:
            insights.append(
                f"⚠️ 두 모델의 예측이 {agreement_rate:.1f}% 일치합니다. 불일치 케이스를 검토하여 개선점을 찾을 수 있습니다."
            )
        else:
            insights.append(
                f"🚨 두 모델의 예측이 {agreement_rate:.1f}%만 일치합니다. 큰 차이가 있으므로 원인 분석이 필요합니다."
            )

        if stats["diff_patterns"].get("pred1_much_longer", 0) > 10:
            insights.append(
                f"📝 Pred1이 Pred2보다 훨씬 긴 경우가 많습니다. 답변 추출 범위를 조정해볼 수 있습니다."
            )

        if stats["diff_patterns"].get("same_prefix", 0) > 10:
            insights.append(
                f"🔤 동일한 접두사를 가진 경우가 많습니다. 답변의 시작점은 유사하나 끝점에서 차이가 발생합니다."
            )

        if avg_f1 < 0.5:
            insights.append(
                f"📉 평균 F1 점수가 {avg_f1:.4f}로 낮습니다. 두 모델의 예측이 크게 다릅니다."
            )
        elif avg_f1 > 0.8:
            insights.append(
                f"📈 평균 F1 점수가 {avg_f1:.4f}로 높습니다. 두 모델이 유사한 패턴을 학습했습니다."
            )

        for insight in insights:
            f.write(f"- {insight}\n")

        f.write(f"\n---\n\n")
        f.write(f"## 🎯 다음 단계\n\n")
        f.write(
            f"1. **불일치 예시 검토**: 위의 Top 20 불일치 예시를 상세히 분석하여 패턴을 파악합니다.\n"
        )
        f.write(
            f"2. **앙상블 고려**: 두 모델의 예측을 결합하여 성능을 향상시킬 수 있습니다.\n"
        )
        f.write(
            f"3. **하이퍼파라미터 조정**: 차이가 큰 영역에서 모델 설정을 재검토합니다.\n"
        )
        f.write(
            f"4. **데이터 분석**: 특정 질문 유형이나 도메인에서 차이가 큰지 확인합니다.\n"
        )


def main():
    parser = argparse.ArgumentParser(
        description="두 개의 prediction CSV 파일을 비교 분석합니다."
    )
    parser.add_argument("csv1", type=str, help="첫 번째 CSV 파일 경로")
    parser.add_argument("csv2", type=str, help="두 번째 CSV 파일 경로")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_results",
        help="결과를 저장할 디렉토리 (기본값: ./analysis_results)",
    )

    args = parser.parse_args()

    csv1_path = Path(args.csv1)
    csv2_path = Path(args.csv2)
    output_dir = Path(args.output_dir)

    # 입력 파일 검증
    if not csv1_path.exists():
        print(f"❌ 오류: {csv1_path} 파일을 찾을 수 없습니다.")
        return

    if not csv2_path.exists():
        print(f"❌ 오류: {csv2_path} 파일을 찾을 수 없습니다.")
        return

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 분석 시작...")
    print(f"  - CSV 1: {csv1_path}")
    print(f"  - CSV 2: {csv2_path}")

    # 분석 수행
    results, stats = analyze_predictions(csv1_path, csv2_path)

    # 결과 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 데이터셋 타입 자동 추론
    num_samples = stats["total"]
    if num_samples == 240:
        dataset_type = "val"
    elif num_samples == 600:
        dataset_type = "test"
    else:
        dataset_type = "unknown"

    comparison_csv = output_dir / f"comparison_{timestamp}.csv"
    report_md = output_dir / f"analysis_report_{timestamp}.md"

    # 결과 저장
    print(f"\n💾 결과 저장 중...")
    save_comparison_csv(results, comparison_csv)
    print(f"  ✅ 비교 CSV 저장: {comparison_csv}")

    generate_analysis_report(results, stats, csv1_path.name, csv2_path.name, report_md)
    print(f"  ✅ 분석 보고서 저장: {report_md}")

    # 요약 출력
    print(f"\n" + "=" * 60)
    print(f"📊 분석 완료!")
    print(f"=" * 60)
    print(f"전체 예시 수: {stats['total']}")
    print(
        f"일치 (Agreement): {stats['agreement']} ({stats['agreement'] / stats['total'] * 100:.2f}%)"
    )
    print(
        f"불일치 (Disagreement): {stats['disagreement']} ({stats['disagreement'] / stats['total'] * 100:.2f}%)"
    )
    print(f"평균 F1 Score: {stats['pred1_f1_sum'] / stats['total']:.4f}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
