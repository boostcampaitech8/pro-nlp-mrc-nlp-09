#!/usr/bin/env python3
"""
Batch 실험 결과 종합 분석 스크립트

각 모델의 학습 결과를 수집하고, 성능 비교 및 분석 리포트를 생성합니다.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


def load_experiment_results(output_dir: str, user: str = "dahyeong") -> List[Dict]:
    """모든 실험 결과 로드"""
    results = []
    base_path = Path(output_dir) / user

    if not base_path.exists():
        print(f"⚠️  Output directory not found: {base_path}")
        return results

    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        result = {"experiment": exp_name, "path": str(exp_dir)}

        # 1. eval_results.json (validation 성능)
        eval_results_path = exp_dir / "eval_results.json"
        if eval_results_path.exists():
            with open(eval_results_path, "r") as f:
                eval_data = json.load(f)
                result["eval_f1"] = eval_data.get("eval_f1", None)
                result["eval_em"] = eval_data.get("eval_exact_match", None)
                result["eval_samples"] = eval_data.get("eval_samples", None)

        # 2. train_results.txt (학습 성능)
        train_results_path = exp_dir / "train_results.txt"
        if train_results_path.exists():
            with open(train_results_path, "r") as f:
                for line in f:
                    if "train_loss" in line:
                        result["train_loss"] = float(line.split("=")[1].strip())
                    elif "train_runtime" in line:
                        result["train_runtime"] = float(line.split("=")[1].strip())
                    elif "epoch" in line:
                        result["num_epochs"] = float(line.split("=")[1].strip())

        # 3. all_results.json (전체 결과)
        all_results_path = exp_dir / "all_results.json"
        if all_results_path.exists():
            with open(all_results_path, "r") as f:
                all_data = json.load(f)
                result["final_em"] = all_data.get("eval_exact_match", None)
                result["final_f1"] = all_data.get("eval_f1", None)

        # 4. config_used.yaml 읽기
        config_path = exp_dir / "config_used.yaml"
        if config_path.exists():
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                result["model"] = config.get("model_name_or_path", "unknown")
                result["learning_rate"] = config.get("learning_rate", None)
                result["batch_size"] = config.get("per_device_train_batch_size", None)
                result["retrieval_type"] = config.get("retrieval_type", None)

                # DHN 사용 여부
                dhn_config = config.get("dynamic_hard_negative", {})
                result["use_dhn"] = dhn_config.get("enabled", False)

        # 최소한 eval 결과가 있어야 유효한 실험으로 간주
        if "eval_em" in result or "final_em" in result:
            results.append(result)

    return results


def create_summary_table(results: List[Dict]) -> pd.DataFrame:
    """결과를 DataFrame으로 정리"""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # EM/F1 우선순위: final > eval
    if "final_em" in df.columns:
        df["EM"] = df["final_em"].fillna(df.get("eval_em", None))
    else:
        df["EM"] = df.get("eval_em", None)

    if "final_f1" in df.columns:
        df["F1"] = df["final_f1"].fillna(df.get("eval_f1", None))
    else:
        df["F1"] = df.get("eval_f1", None)

    # 주요 컬럼만 선택
    display_cols = [
        "experiment",
        "model",
        "EM",
        "F1",
        "learning_rate",
        "batch_size",
        "use_dhn",
        "train_loss",
        "train_runtime",
    ]

    available_cols = [col for col in display_cols if col in df.columns]
    df = df[available_cols]

    # EM 기준 정렬
    if "EM" in df.columns:
        df = df.sort_values("EM", ascending=False)

    return df


def print_summary_report(df: pd.DataFrame, output_path: Optional[str] = None):
    """요약 리포트 출력 및 저장"""
    if df.empty:
        print("⚠️  No experiment results found.")
        return

    report = []
    report.append("=" * 100)
    report.append("📊 BATCH TRAINING RESULTS SUMMARY")
    report.append("=" * 100)
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Experiments: {len(df)}")
    report.append("")

    # Top 3 모델
    report.append("🏆 TOP 3 MODELS (by Exact Match)")
    report.append("-" * 100)

    top3 = df.head(3)
    for idx, (i, row) in enumerate(top3.iterrows(), 1):
        medal = ["🥇", "🥈", "🥉"][idx - 1]
        report.append(f"{medal} Rank {idx}: {row['experiment']}")
        report.append(f"   Model: {row.get('model', 'N/A')}")
        report.append(
            f"   EM: {row.get('EM', 'N/A'):.2f}%  |  F1: {row.get('F1', 'N/A'):.2f}%"
        )

        if "learning_rate" in row:
            report.append(
                f"   LR: {row.get('learning_rate', 'N/A')}  |  Batch Size: {row.get('batch_size', 'N/A')}"
            )

        if "use_dhn" in row:
            dhn_status = "✓ DHN" if row.get("use_dhn") else "✗ Vanilla"
            report.append(f"   Training: {dhn_status}")

        if "train_runtime" in row and row["train_runtime"]:
            runtime_min = row["train_runtime"] / 60
            report.append(f"   Training Time: {runtime_min:.1f} min")

        report.append("")

    # 전체 결과 테이블
    report.append("=" * 100)
    report.append("📋 ALL RESULTS")
    report.append("-" * 100)

    # 테이블 형식으로 출력
    table_str = df.to_string(
        index=False, max_colwidth=30, float_format=lambda x: f"{x:.2f}"
    )
    report.append(table_str)
    report.append("")

    # 통계 요약
    if "EM" in df.columns and df["EM"].notna().any():
        report.append("=" * 100)
        report.append("📈 STATISTICS")
        report.append("-" * 100)
        report.append(f"Average EM: {df['EM'].mean():.2f}%")
        report.append(f"Best EM: {df['EM'].max():.2f}%")
        report.append(f"Worst EM: {df['EM'].min():.2f}%")
        report.append(f"Std Dev: {df['EM'].std():.2f}%")
        report.append("")

    # DHN vs Vanilla 비교
    if "use_dhn" in df.columns:
        dhn_results = df[df["use_dhn"] == True]
        vanilla_results = df[df["use_dhn"] == False]

        if not dhn_results.empty and not vanilla_results.empty:
            report.append("=" * 100)
            report.append("🔬 DHN vs VANILLA COMPARISON")
            report.append("-" * 100)
            report.append(f"DHN Models: {len(dhn_results)} experiments")
            report.append(f"  Average EM: {dhn_results['EM'].mean():.2f}%")
            report.append(f"  Best EM: {dhn_results['EM'].max():.2f}%")
            report.append("")
            report.append(f"Vanilla Models: {len(vanilla_results)} experiments")
            report.append(f"  Average EM: {vanilla_results['EM'].mean():.2f}%")
            report.append(f"  Best EM: {vanilla_results['EM'].max():.2f}%")
            report.append("")

            em_diff = dhn_results["EM"].mean() - vanilla_results["EM"].mean()
            if em_diff > 0:
                report.append(f"✅ DHN shows +{em_diff:.2f}% improvement on average")
            else:
                report.append(
                    f"⚠️  Vanilla shows +{abs(em_diff):.2f}% improvement on average"
                )
            report.append("")

    report.append("=" * 100)

    # 콘솔 출력
    full_report = "\n".join(report)
    print(full_report)

    # 파일 저장
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n✅ Report saved to: {output_path}")

        # CSV도 함께 저장
        csv_path = output_path.replace(".txt", ".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✅ CSV saved to: {csv_path}")


def main():
    """메인 실행 함수"""
    output_dir = "./outputs"
    user = "dahyeong"

    # 커맨드라인 인자로 user 지정 가능
    if len(sys.argv) > 1:
        user = sys.argv[1]

    print("🔍 Collecting experiment results...")
    results = load_experiment_results(output_dir, user)

    if not results:
        print(f"❌ No experiment results found in {output_dir}/{user}/")
        sys.exit(1)

    print(f"✅ Found {len(results)} experiments\n")

    # DataFrame 생성
    df = create_summary_table(results)

    # 리포트 생성 및 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"./logs/batch_results_{timestamp}.txt"

    print_summary_report(df, report_path)


if __name__ == "__main__":
    main()
