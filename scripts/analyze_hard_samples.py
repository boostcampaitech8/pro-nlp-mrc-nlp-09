"""
Hard Sample 분석 스크립트

Validation 결과를 상세히 분석하여 다양한 형태의 아웃풋을 생성합니다.

Usage:
    python scripts/analyze_hard_samples.py <output_dir>
    python scripts/analyze_hard_samples.py ./outputs/dahyeong/HANTAEK_roberta_large_vanilla

Outputs:
    1. val_simple_comparison.csv       - ground_truth vs prediction 단순 비교
    2. val_detailed_analysis.csv       - 문서정보, EM/F1, retrieval 성공여부 포함
    3. val_hard_samples.csv            - 틀린 샘플만 (hard samples)
    4. val_retrieval_failures.csv      - retrieval이 gold context 못 찾은 케이스
    5. val_error_analysis.json         - 에러 유형별 분류 및 통계
    6. val_analysis_summary.md         - 전체 분석 요약 (마크다운)
"""

import json
import csv
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datasets import load_from_disk

# 프로젝트 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.retrieval.paths import get_path, get_analysis_dir, DATA_ROOT


def normalize_answer(s: str) -> str:
    """정답 정규화 (EM 계산용)"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        # 한글은 유지, 영어 punctuation만 제거
        return re.sub(r"[^\w\s가-힣]", "", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def compute_em(prediction: str, ground_truths: List[str]) -> float:
    """Exact Match 계산"""
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == norm_pred:
            return 100.0
    return 0.0


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """F1 Score 계산"""

    def get_tokens(s):
        return normalize_answer(s).split()

    def compute_single_f1(pred_tokens, gt_tokens):
        common = set(pred_tokens) & set(gt_tokens)
        if len(common) == 0:
            return 0.0
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gt_tokens) if gt_tokens else 0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall) * 100

    pred_tokens = get_tokens(prediction)
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = get_tokens(gt)
        f1 = compute_single_f1(pred_tokens, gt_tokens)
        best_f1 = max(best_f1, f1)
    return best_f1


def load_wikipedia_documents() -> Dict[int, Dict]:
    """Wikipedia 문서 로드 (paths 모듈 사용)"""
    wiki_path = Path(get_path("wiki_corpus"))
    if not wiki_path.exists():
        print(f"⚠️ Wikipedia documents not found at {wiki_path}")
        return {}

    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki_data = json.load(f)

    # document_id를 key로 하는 dict 생성
    docs = {}
    for doc_id, doc in wiki_data.items():
        docs[int(doc_id)] = doc
    return docs


def load_retrieval_cache(cache_path: str = None) -> Dict[str, Dict]:
    """Retrieval 캐시 로드 (paths 모듈 사용)"""
    if cache_path is None:
        cache_path = get_path("val_cache")

    cache = {}
    if not Path(cache_path).exists():
        print(f"⚠️ Retrieval cache not found at {cache_path}")
        return cache

    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cache[item["id"]] = item
    return cache


def check_retrieval_found_gold(
    retrieved_docs: List[Dict], gold_doc_id: int, top_k: int = 10
) -> Tuple[bool, int]:
    """
    Retrieval이 gold document를 찾았는지 확인
    Returns: (found, rank) - rank는 못찾으면 -1
    """
    for rank, doc in enumerate(retrieved_docs[:top_k], 1):
        if doc.get("doc_id") == gold_doc_id:
            return True, rank
    return False, -1


def categorize_error(
    em: float,
    f1: float,
    retrieval_found: bool,
    prediction: str,
    ground_truths: List[str],
) -> str:
    """에러 유형 분류"""
    if em == 100.0:
        return "correct"

    # Retrieval 실패
    if not retrieval_found:
        return "retrieval_failure"

    # Partial match (F1은 높은데 EM은 0)
    if f1 >= 50:
        # 예측이 정답의 일부인 경우
        norm_pred = normalize_answer(prediction)
        for gt in ground_truths:
            norm_gt = normalize_answer(gt)
            if norm_pred in norm_gt:
                return "partial_subset"  # 예측이 정답의 부분집합
            if norm_gt in norm_pred:
                return "partial_superset"  # 예측이 정답을 포함
        return "partial_overlap"

    # 완전히 다른 답
    if f1 < 20:
        return "completely_wrong"

    return "low_overlap"


def analyze_samples(
    output_dir: Path,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    메인 분석 함수

    Args:
        output_dir: 모델 출력 디렉토리
        top_k: Retrieval top-k 기준
    """
    print("=" * 80)
    print("🔍 Hard Sample Analysis Tool")
    print("=" * 80)

    # 1. 데이터 로드
    print("\n[1/6] Loading data...")

    # Validation dataset (paths 모듈 사용)
    ds = load_from_disk(get_path("train_dataset"))
    val_ds = ds["validation"]
    print(f"   ✓ Loaded {len(val_ds)} validation samples")

    # Predictions
    pred_path = output_dir / "predictions.json"
    if not pred_path.exists():
        pred_path = output_dir / "val_predictions.json"

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    print(f"   ✓ Loaded {len(predictions)} predictions")

    # Wikipedia documents (paths 모듈 사용)
    wiki_docs = load_wikipedia_documents()
    print(f"   ✓ Loaded {len(wiki_docs)} wikipedia documents")

    # Retrieval cache (paths 모듈 사용)
    retrieval_cache = load_retrieval_cache()
    print(f"   ✓ Loaded {len(retrieval_cache)} retrieval cache entries")

    # Labels (이미 생성된 파일 사용)
    labels_path = output_dir / "eval_labels.json"
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        print(f"   ✓ Loaded {len(labels)} labels from eval_labels.json")
    else:
        # validation dataset에서 직접 생성
        labels = {}
        for ex in val_ds:
            labels[ex["id"]] = {
                "text": ex["answers"]["text"],
                "answer_start": ex["answers"]["answer_start"],
            }
        print(f"   ✓ Generated {len(labels)} labels from dataset")

    # 2. 샘플별 분석
    print("\n[2/6] Analyzing each sample...")

    analysis_results = []
    error_categories = defaultdict(list)
    retrieval_stats = {"found": 0, "not_found": 0, "no_cache": 0}

    for ex in val_ds:
        qid = ex["id"]
        question = ex["question"]
        gold_context = ex.get("context", "")
        gold_doc_id = ex.get("document_id", None)
        ground_truths = ex["answers"]["text"]

        # 예측
        prediction = predictions.get(qid, "")

        # EM/F1 계산
        em = compute_em(prediction, ground_truths)
        f1 = compute_f1(prediction, ground_truths)

        # Retrieval 분석
        retrieval_found = False
        retrieval_rank = -1
        retrieved_doc_ids = []
        retrieved_titles = []

        if qid in retrieval_cache:
            retrieved = retrieval_cache[qid].get("retrieved", [])
            retrieved_doc_ids = [d.get("doc_id") for d in retrieved[:top_k]]

            # 제목 가져오기
            for doc_id in retrieved_doc_ids[:3]:  # 상위 3개만
                if doc_id in wiki_docs:
                    retrieved_titles.append(wiki_docs[doc_id].get("title", "Unknown"))

            if gold_doc_id is not None:
                retrieval_found, retrieval_rank = check_retrieval_found_gold(
                    retrieved, gold_doc_id, top_k
                )
                if retrieval_found:
                    retrieval_stats["found"] += 1
                else:
                    retrieval_stats["not_found"] += 1
            else:
                retrieval_stats["no_cache"] += 1
        else:
            retrieval_stats["no_cache"] += 1

        # 에러 유형 분류
        error_type = categorize_error(
            em, f1, retrieval_found, prediction, ground_truths
        )
        error_categories[error_type].append(qid)

        # Gold document 정보
        gold_title = ""
        gold_text_snippet = ""
        if gold_doc_id and gold_doc_id in wiki_docs:
            gold_title = wiki_docs[gold_doc_id].get("title", "")
            gold_text_snippet = wiki_docs[gold_doc_id].get("text", "")[:200] + "..."
        elif gold_context:
            gold_text_snippet = gold_context[:200] + "..."

        result = {
            "id": qid,
            "question": question,
            "gold_doc_id": gold_doc_id,
            "gold_title": gold_title,
            "gold_context_snippet": gold_text_snippet,
            "ground_truth": " | ".join(ground_truths),
            "prediction": prediction,
            "em": em,
            "f1": f1,
            "retrieval_found": retrieval_found,
            "retrieval_rank": retrieval_rank,
            "retrieved_top3_titles": " | ".join(retrieved_titles),
            "error_type": error_type,
        }
        analysis_results.append(result)

    print(f"   ✓ Analyzed {len(analysis_results)} samples")

    # 3. Output 1: Simple comparison
    print("\n[3/6] Generating outputs...")

    # val_analysis 하위 디렉토리 생성 (paths 모듈 사용)
    analysis_dir = get_analysis_dir(output_dir, "val_analysis")

    simple_path = analysis_dir / "val_simple_comparison.csv"
    with open(simple_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "ground_truth", "prediction", "match"])
        for r in analysis_results:
            match = "✓" if r["em"] == 100.0 else "✗"
            writer.writerow([r["id"], r["ground_truth"], r["prediction"], match])
    print(f"   📄 {simple_path}")

    # 4. Output 2: Detailed analysis
    detailed_path = analysis_dir / "val_detailed_analysis.csv"
    with open(detailed_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "question",
                "gold_doc_id",
                "gold_title",
                "gold_context_snippet",
                "ground_truth",
                "prediction",
                "em",
                "f1",
                "retrieval_found",
                "retrieval_rank",
                "retrieved_top3_titles",
                "error_type",
            ]
        )
        for r in analysis_results:
            writer.writerow(
                [
                    r["id"],
                    r["question"],
                    r["gold_doc_id"],
                    r["gold_title"],
                    r["gold_context_snippet"],
                    r["ground_truth"],
                    r["prediction"],
                    f"{r['em']:.1f}",
                    f"{r['f1']:.1f}",
                    "Yes" if r["retrieval_found"] else "No",
                    r["retrieval_rank"] if r["retrieval_rank"] > 0 else "N/A",
                    r["retrieved_top3_titles"],
                    r["error_type"],
                ]
            )
    print(f"   📄 {detailed_path}")

    # 5. Output 3: Hard samples only (wrong predictions)
    hard_samples = [r for r in analysis_results if r["em"] < 100.0]
    hard_samples.sort(key=lambda x: x["f1"])  # F1 낮은 순으로 정렬

    hard_path = analysis_dir / "val_hard_samples.csv"
    with open(hard_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "question",
                "ground_truth",
                "prediction",
                "em",
                "f1",
                "retrieval_found",
                "error_type",
                "gold_title",
            ]
        )
        for r in hard_samples:
            writer.writerow(
                [
                    r["id"],
                    r["question"],
                    r["ground_truth"],
                    r["prediction"],
                    f"{r['em']:.1f}",
                    f"{r['f1']:.1f}",
                    "Yes" if r["retrieval_found"] else "No",
                    r["error_type"],
                    r["gold_title"],
                ]
            )
    print(f"   📄 {hard_path} ({len(hard_samples)} samples)")

    # 6. Output 4: Retrieval failures
    retrieval_failures = [
        r
        for r in analysis_results
        if not r["retrieval_found"] and r["retrieval_rank"] == -1
    ]

    retrieval_fail_path = analysis_dir / "val_retrieval_failures.csv"
    with open(retrieval_fail_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "question",
                "gold_title",
                "retrieved_top3_titles",
                "ground_truth",
                "prediction",
                "em",
                "f1",
            ]
        )
        for r in retrieval_failures:
            writer.writerow(
                [
                    r["id"],
                    r["question"],
                    r["gold_title"],
                    r["retrieved_top3_titles"],
                    r["ground_truth"],
                    r["prediction"],
                    f"{r['em']:.1f}",
                    f"{r['f1']:.1f}",
                ]
            )
    print(f"   📄 {retrieval_fail_path} ({len(retrieval_failures)} samples)")

    # 7. Output 5: Error analysis JSON
    error_analysis = {
        "total_samples": len(analysis_results),
        "correct_count": len(error_categories["correct"]),
        "wrong_count": len(analysis_results) - len(error_categories["correct"]),
        "retrieval_stats": retrieval_stats,
        "error_categories": {
            cat: {
                "count": len(ids),
                "percentage": len(ids) / len(analysis_results) * 100,
                "sample_ids": ids[:10],  # 처음 10개만
            }
            for cat, ids in error_categories.items()
        },
        "metrics": {
            "overall_em": sum(r["em"] for r in analysis_results)
            / len(analysis_results),
            "overall_f1": sum(r["f1"] for r in analysis_results)
            / len(analysis_results),
            "em_when_retrieval_found": (
                sum(r["em"] for r in analysis_results if r["retrieval_found"])
                / max(1, sum(1 for r in analysis_results if r["retrieval_found"]))
            ),
            "em_when_retrieval_not_found": (
                sum(r["em"] for r in analysis_results if not r["retrieval_found"])
                / max(1, sum(1 for r in analysis_results if not r["retrieval_found"]))
            ),
        },
    }

    error_json_path = analysis_dir / "val_error_analysis.json"
    with open(error_json_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False)
    print(f"   📄 {error_json_path}")

    # 8. Output 6: Summary markdown
    summary_md = generate_summary_markdown(
        error_analysis, analysis_results, analysis_dir
    )
    summary_path = analysis_dir / "val_analysis_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"   📄 {summary_path}")

    return error_analysis


def generate_summary_markdown(
    error_analysis: Dict, results: List[Dict], output_dir: Path
) -> str:
    """분석 요약 마크다운 생성"""

    total = error_analysis["total_samples"]
    correct = error_analysis["correct_count"]
    wrong = error_analysis["wrong_count"]
    metrics = error_analysis["metrics"]
    retrieval = error_analysis["retrieval_stats"]
    categories = error_analysis["error_categories"]

    # 에러 유형별 정렬
    sorted_cats = sorted(
        [(k, v) for k, v in categories.items() if k != "correct"],
        key=lambda x: x[1]["count"],
        reverse=True,
    )

    md = f"""# 📊 Validation Analysis Summary

> Output directory: `{output_dir}`
> Generated by `analyze_hard_samples.py`

---

## 1. Overall Performance

| Metric | Value |
|--------|-------|
| **Total Samples** | {total} |
| **Correct (EM=100)** | {correct} ({correct / total * 100:.1f}%) |
| **Wrong** | {wrong} ({wrong / total * 100:.1f}%) |
| **Overall EM** | {metrics["overall_em"]:.2f} |
| **Overall F1** | {metrics["overall_f1"]:.2f} |

---

## 2. Retrieval Impact

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Gold in Top-{10} | {retrieval["found"]} | {retrieval["found"] / total * 100:.1f}% |
| ❌ Gold NOT in Top-{10} | {retrieval["not_found"]} | {retrieval["not_found"] / total * 100:.1f}% |
| ⚠️ No Cache/Doc ID | {retrieval["no_cache"]} | {retrieval["no_cache"] / total * 100:.1f}% |

### Performance by Retrieval Success

| Condition | EM |
|-----------|-----|
| Retrieval Found Gold | {metrics["em_when_retrieval_found"]:.2f} |
| Retrieval Missed Gold | {metrics["em_when_retrieval_not_found"]:.2f} |

> 💡 **Insight**: Retrieval이 gold document를 찾았을 때 EM이 {metrics["em_when_retrieval_found"] - metrics["em_when_retrieval_not_found"]:.1f}점 더 높습니다.

---

## 3. Error Type Analysis

| Error Type | Count | % | Description |
|------------|-------|---|-------------|
"""

    error_descriptions = {
        "correct": "정답과 일치",
        "retrieval_failure": "Retrieval이 gold context를 못 찾음",
        "partial_subset": "예측이 정답의 부분집합 (더 짧게 예측)",
        "partial_superset": "예측이 정답을 포함 (더 길게 예측)",
        "partial_overlap": "부분적으로 겹침 (F1 >= 50)",
        "low_overlap": "낮은 겹침 (F1 20-50)",
        "completely_wrong": "완전히 다른 답 (F1 < 20)",
    }

    for cat, info in sorted_cats:
        desc = error_descriptions.get(cat, cat)
        md += f"| {cat} | {info['count']} | {info['percentage']:.1f}% | {desc} |\n"

    md += f"""
---

## 4. Sample Hard Cases

### 4.1 Retrieval Failures (Top 5)

"""

    # Retrieval 실패 케이스
    ret_failures = [r for r in results if r["error_type"] == "retrieval_failure"][:5]
    for i, r in enumerate(ret_failures, 1):
        md += f"""**{i}. [{r["id"]}]**
- **Question**: {r["question"][:100]}...
- **Gold Title**: {r["gold_title"]}
- **Retrieved**: {r["retrieved_top3_titles"]}
- **Answer**: {r["ground_truth"]} → **Pred**: {r["prediction"]}
- **F1**: {r["f1"]:.1f}

"""

    md += """### 4.2 Partial Match Cases (Top 5)

"""

    # Partial match 케이스
    partial_cases = [r for r in results if "partial" in r["error_type"]][:5]
    for i, r in enumerate(partial_cases, 1):
        md += f"""**{i}. [{r["id"]}]** ({r["error_type"]})
- **Question**: {r["question"][:100]}...
- **Answer**: `{r["ground_truth"]}` → **Pred**: `{r["prediction"]}`
- **EM**: {r["em"]:.0f}, **F1**: {r["f1"]:.1f}

"""

    md += """### 4.3 Completely Wrong Cases (Top 5)

"""

    # 완전히 틀린 케이스
    wrong_cases = [r for r in results if r["error_type"] == "completely_wrong"][:5]
    for i, r in enumerate(wrong_cases, 1):
        md += f"""**{i}. [{r["id"]}]**
- **Question**: {r["question"][:100]}...
- **Answer**: `{r["ground_truth"]}` → **Pred**: `{r["prediction"]}`
- **Retrieval Found**: {r["retrieval_found"]}

"""

    md += f"""---

## 5. Generated Files

| File | Description |
|------|-------------|
| `val_simple_comparison.csv` | ground_truth vs prediction 단순 비교 |
| `val_detailed_analysis.csv` | 문서정보, EM/F1, retrieval 성공여부 전체 |
| `val_hard_samples.csv` | 틀린 샘플만 (F1 낮은 순 정렬) |
| `val_retrieval_failures.csv` | Retrieval이 gold를 못 찾은 케이스 |
| `val_error_analysis.json` | 에러 유형별 통계 (프로그래밍용) |
| `val_analysis_summary.md` | 이 파일 |

---

## 6. Recommendations

"""

    # 추천사항 생성
    if retrieval["not_found"] > total * 0.05:  # 5% 이상 retrieval 실패
        md += f"""### 🔧 Retrieval 개선 필요
- Retrieval 실패율이 {retrieval["not_found"] / total * 100:.1f}%로 높습니다.
- top_k 증가, reranking 추가, 또는 hybrid 가중치 조정을 고려하세요.

"""

    partial_count = sum(1 for r in results if "partial" in r["error_type"])
    if partial_count > total * 0.1:  # 10% 이상 partial
        md += f"""### 🔧 Answer Span 경계 개선 필요
- Partial match가 {partial_count}개 ({partial_count / total * 100:.1f}%)입니다.
- doc_stride 조정 또는 start/end 모델 성능 개선이 필요합니다.

"""

    wrong_count = len([r for r in results if r["error_type"] == "completely_wrong"])
    if wrong_count > total * 0.1:
        md += f"""### 🔧 Reader 모델 개선 필요
- 완전히 틀린 예측이 {wrong_count}개 ({wrong_count / total * 100:.1f}%)입니다.
- 모델 fine-tuning, 데이터 augmentation, 또는 더 큰 모델 사용을 고려하세요.

"""

    return md


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_hard_samples.py <output_dir> [top_k]")
        print(
            "Example: python scripts/analyze_hard_samples.py ./outputs/dahyeong/HANTAEK_roberta_large_vanilla"
        )
        print(
            "         python scripts/analyze_hard_samples.py ./outputs/dahyeong/model_name 20"
        )
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        sys.exit(1)

    # 옵션 파라미터
    top_k = 10
    if len(sys.argv) > 2:
        top_k = int(sys.argv[2])

    try:
        result = analyze_samples(output_dir, top_k)

        print("\n" + "=" * 80)
        print("✅ Analysis Complete!")
        print("=" * 80)
        print(f"\n📊 Quick Summary:")
        print(f"   - Total: {result['total_samples']} samples")
        print(
            f"   - Correct: {result['correct_count']} ({result['correct_count'] / result['total_samples'] * 100:.1f}%)"
        )
        print(f"   - EM: {result['metrics']['overall_em']:.2f}")
        print(f"   - F1: {result['metrics']['overall_f1']:.2f}")
        print(f"\n📂 Check {output_dir}/val_analysis/ for detailed analysis files!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
