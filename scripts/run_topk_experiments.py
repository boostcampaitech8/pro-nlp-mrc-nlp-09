"""
Top-K 실험 스크립트

다양한 top_k_retrieval 값으로 inference를 실행하고 결과를 비교합니다.
- Recall 분석 후 각 top-k에 대해 inference 실행
- EM, F1 스코어 자동 수집 및 비교표 생성
- 로그 파일 자동 저장

사용법:
    python scripts/run_topk_experiments.py --topk_values 10,20,30,40,50
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict
import pandas as pd

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).parent.parent


def run_retrieval_analysis(retriever: str = "koe5") -> None:
    """Retrieval recall 분석 실행"""
    print("\n" + "=" * 80)
    print("📊 STEP 1: Retrieval Recall 분석")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(ROOT_DIR / "tests" / "test_retrieval_recall.py"),
        "--retriever",
        retriever,
        "--analyze_full",
    ]

    subprocess.run(cmd, check=True)


def update_config_topk(config_path: Path, topk: int) -> None:
    """YAML config의 top_k_retrieval 값 수정"""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # top_k_retrieval은 최상위에 위치
    config["top_k_retrieval"] = topk

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )

    print(f"   ✓ Config updated: top_k_retrieval = {topk}")


def run_inference(config_path: Path, topk: int, backup_dir: Path) -> Dict:
    """Inference 실행 및 결과 수집"""
    print(f"\n{'=' * 80}")
    print(f"🔬 STEP 2-{topk}: Inference with top_k={topk}")
    print("=" * 80)

    # Config 수정 (validation으로 변경)
    update_config_topk(config_path, topk)
    update_config_split(config_path, "validation")  # validation으로 강제 설정

    # Inference 실행 (실시간 출력)
    start_time = time.time()
    cmd = [sys.executable, str(ROOT_DIR / "inference.py"), str(config_path)]
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print(f"   ❌ Inference failed for top_k={topk}")
        return None

    print(f"   ✓ Inference completed in {elapsed_time:.1f}s")

    # 결과 파일 찾기 (validation 결과)
    output_dir = (
        ROOT_DIR
        / "outputs"
        / "dahyeong"
        / "HANTAEK_rob-large-kq-v1-qa-finetuned_stride64"
    )

    # 📌 Top-K별 결과 파일 백업 (덮어쓰기 방지)
    import shutil

    val_files = [
        "val_results.json",
        "predictions_val.json",
        "nbest_predictions_val.json",
        "val_pred.csv",
    ]

    print(f"   💾 Backing up results for top_k={topk}...")
    for filename in val_files:
        src = output_dir / filename
        if src.exists():
            dst = backup_dir / f"{filename.replace('val', f'val_topk{topk}')}"
            shutil.copy2(src, dst)

    # trainer.evaluate()가 생성한 val_results.json에서 메트릭 읽기
    eval_results_path = output_dir / "val_results.json"

    if not eval_results_path.exists():
        print(f"   ⚠️  val_results.json not found")
        return None

    with open(eval_results_path, "r") as f:
        metrics = json.load(f)

    # 키 형식: eval_exact_match, eval_f1
    em = metrics.get("eval_exact_match", 0.0)
    f1 = metrics.get("eval_f1", 0.0)
    print(f"   📈 EM: {em:.2f}% | F1: {f1:.2f}%")
    print(f"   📈 EM: {em:.2f}% | F1: {f1:.2f}%")

    return {"top_k": topk, "em": em, "f1": f1, "time": elapsed_time}


def update_config_split(config_path: Path, split: str) -> None:
    """YAML config의 inference_split 값 수정"""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["inference_split"] = split

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )

    print(f"   ✓ Config updated: inference_split = {split}")


def save_comparison_table(results: List[Dict], log_dir: Path) -> None:
    """결과 비교표 생성 및 저장"""
    if not results:
        print("   ⚠️  No results to save")
        return

    # DataFrame 생성
    df = pd.DataFrame(results)
    df = df.sort_values("top_k")

    # 증감율 계산 (baseline: top_k=10)
    baseline_em = (
        df[df["top_k"] == 10]["em"].values[0]
        if 10 in df["top_k"].values
        else df.iloc[0]["em"]
    )
    baseline_f1 = (
        df[df["top_k"] == 10]["f1"].values[0]
        if 10 in df["top_k"].values
        else df.iloc[0]["f1"]
    )

    df["em_delta"] = df["em"] - baseline_em
    df["f1_delta"] = df["f1"] - baseline_f1

    # 타임스탬프
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # CSV 저장
    csv_path = log_dir / f"topk_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n   💾 CSV saved: {csv_path}")

    # 텍스트 로그 저장
    log_path = log_dir / f"topk_comparison_{timestamp}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Top-K Retrieval Comparison Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Baseline: top_k=10 (EM={baseline_em:.2f}%, F1={baseline_f1:.2f}%)\n")
        f.write("\n")

        f.write(
            f"{'Top-K':<8} {'EM (%)':<10} {'F1 (%)':<10} {'EM Δ':<10} {'F1 Δ':<10} {'Time (s)':<12}\n"
        )
        f.write("-" * 80 + "\n")

        for _, row in df.iterrows():
            f.write(f"{row['top_k']:<8} ")
            f.write(f"{row['em']:<10.2f} ")
            f.write(f"{row['f1']:<10.2f} ")
            f.write(f"{row['em_delta']:+10.2f} ")
            f.write(f"{row['f1_delta']:+10.2f} ")
            f.write(f"{row['time']:<12.1f}\n")

        f.write("=" * 80 + "\n")

        # 최고 성능
        best_em_row = df.loc[df["em"].idxmax()]
        best_f1_row = df.loc[df["f1"].idxmax()]

        f.write("\n🏆 Best Results:\n")
        f.write(
            f"  - Best EM: top_k={int(best_em_row['top_k'])} with {best_em_row['em']:.2f}%\n"
        )
        f.write(
            f"  - Best F1: top_k={int(best_f1_row['top_k'])} with {best_f1_row['f1']:.2f}%\n"
        )

    print(f"   💾 Log saved: {log_path}")

    # 콘솔 출력
    print("\n" + "=" * 80)
    print("📊 Top-K Comparison Summary")
    print("=" * 80)
    print(
        f"{'Top-K':<8} {'EM (%)':<10} {'F1 (%)':<10} {'EM Δ':<10} {'F1 Δ':<10} {'Time (s)':<12}"
    )
    print("-" * 80)

    for _, row in df.iterrows():
        print(f"{row['top_k']:<8} ", end="")
        print(f"{row['em']:<10.2f} ", end="")
        print(f"{row['f1']:<10.2f} ", end="")
        print(f"{row['em_delta']:+10.2f} ", end="")
        print(f"{row['f1_delta']:+10.2f} ", end="")
        print(f"{row['time']:<12.1f}")

    print("=" * 80)
    print(
        f"\n🏆 Best EM: top_k={int(best_em_row['top_k'])} with {best_em_row['em']:.2f}%"
    )
    print(
        f"🏆 Best F1: top_k={int(best_f1_row['top_k'])} with {best_f1_row['f1']:.2f}%"
    )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Top-K Retrieval 실험 - 다양한 top_k 값으로 inference 실행 및 비교"
    )
    parser.add_argument(
        "--topk_values",
        type=str,
        default="10,20,30,40,50",
        help="실험할 top_k 값들 (쉼표로 구분, 예: 10,20,30,40,50)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/active/HANTAEK_roberta-large-korquad-v1-qa-finetuned_stride64.yaml",
        help="Inference config 파일 경로",
    )
    parser.add_argument(
        "--skip_recall", action="store_true", help="Retrieval recall 분석 건너뛰기"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="koe5",
        choices=["koe5", "tfidf"],
        help="Retrieval 방식",
    )

    args = parser.parse_args()

    # Top-K 값 파싱
    topk_values = sorted([int(k.strip()) for k in args.topk_values.split(",")])
    config_path = Path(args.config)

    print("=" * 80)
    print("🚀 Top-K Retrieval Experiment")
    print("=" * 80)
    print(f"Config: {config_path.name}")
    print(f"Top-K values: {topk_values}")
    print(f"Retriever: {args.retriever}")
    print("=" * 80)

    # 로그 디렉토리 생성
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    # 백업 디렉토리 생성 (각 top_k의 predictions 저장)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = log_dir / f"topk_experiment_{timestamp}"
    backup_dir.mkdir(exist_ok=True)
    print(f"\n📁 Predictions backup directory: {backup_dir}")

    # Step 1: Retrieval Recall 분석
    if not args.skip_recall:
        run_retrieval_analysis(args.retriever)
    else:
        print("\n⏭️  Retrieval recall 분석 건너뛰기")

    # Step 2: 각 top_k에 대해 inference 실행
    results = []
    for topk in topk_values:
        result = run_inference(config_path, topk, backup_dir)
        if result:
            results.append(result)

    # Step 3: 결과 비교표 생성
    if results:
        save_comparison_table(results, log_dir)
    else:
        print("\n   ⚠️  No successful results to compare")

    print("\n✅ 실험 완료!")


if __name__ == "__main__":
    main()
