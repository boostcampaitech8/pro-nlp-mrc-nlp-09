#!/usr/bin/env python3
"""
MRC 프로젝트 통합 실행 스크립트

사용법:
    # 단일 실험
    python run.py --mode train --config configs/experiment.yaml
    python run.py --mode inference --config configs/experiment.yaml
    python run.py --mode pipeline --config configs/experiment.yaml

    # 여러 실험 순차 실행 (밤새 GPU 돌리기)
    python run.py --mode batch --configs configs/exp1.yaml configs/exp2.yaml configs/exp3.yaml
    python run.py --mode batch --configs configs/experiments/*.yaml
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


def run_train(config_path: str) -> int:
    """Train 모드: train.py 실행"""
    print("=" * 80)
    print("🚀 Starting TRAINING")
    print("=" * 80)

    cmd = [sys.executable, "train.py", config_path]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
    else:
        print(f"\n❌ Training failed with exit code {result.returncode}")

    return result.returncode


def run_inference(config_path: str) -> int:
    """Inference 모드: inference.py 실행"""
    print("=" * 80)
    print("🔍 Starting INFERENCE")
    print("=" * 80)

    cmd = [sys.executable, "inference.py", config_path]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Inference completed successfully!")
    else:
        print(f"\n❌ Inference failed with exit code {result.returncode}")

    return result.returncode


def run_pipeline(config_path: str) -> int:
    """Pipeline 모드: train → inference 순차 실행"""
    print("=" * 80)
    print("🔄 Starting PIPELINE (Train → Inference)")
    print("=" * 80)

    # Step 1: Training
    train_exit_code = run_train(config_path)

    if train_exit_code != 0:
        print("\n⚠️  Training failed. Skipping inference.")
        return train_exit_code

    print("\n" + "=" * 80)
    print("📊 Training done. Starting inference with trained model...")
    print("=" * 80 + "\n")

    # Step 2: Inference (use_trained_model은 기본값 True이므로 자동으로 best checkpoint 사용)
    inference_exit_code = run_inference(config_path)

    if inference_exit_code == 0:
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("⚠️  Pipeline completed with errors in inference")
        print("=" * 80)

    return inference_exit_code


class ExperimentResult:
    """실험 결과를 저장하는 클래스"""

    def __init__(self, config_path: str, mode: str):
        self.config_path = config_path
        self.config_name = Path(config_path).stem
        self.mode = mode
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.exit_code = None
        self.status = "pending"

    def start(self):
        self.status = "running"
        self.start_time = datetime.now()

    def finish(self, exit_code: int):
        self.end_time = datetime.now()
        self.exit_code = exit_code
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = "success" if exit_code == 0 else "failed"


def run_single_experiment(config_path: str, mode: str) -> Tuple[int, float]:
    """
    단일 실험을 실행하고 결과를 반환

    Returns:
        (exit_code, duration_seconds)
    """
    start_time = time.time()

    print("\n" + "=" * 80)
    print(f"🚀 Starting experiment: {Path(config_path).stem}")
    print(f"   Config: {config_path}")
    print(f"   Mode: {mode}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # 모드에 따라 실행
    if mode == "train":
        exit_code = run_train(config_path)
    elif mode == "inference":
        exit_code = run_inference(config_path)
    elif mode == "pipeline":
        exit_code = run_pipeline(config_path)
    else:
        exit_code = 1

    duration = time.time() - start_time

    if exit_code == 0:
        print(f"\n✅ Experiment completed: {Path(config_path).stem}")
        print(f"   Duration: {duration:.1f}s ({duration / 60:.1f}min)")
    else:
        print(f"\n❌ Experiment failed: {Path(config_path).stem}")
        print(f"   Exit code: {exit_code}")
        print(f"   Duration: {duration:.1f}s")

    return exit_code, duration


def print_batch_progress(current: int, total: int, result: ExperimentResult):
    """배치 실행 진행 상황 출력"""
    progress = (current / total) * 100
    status_icon = {"success": "✅", "failed": "❌", "running": "🔄", "pending": "⏳"}

    print("\n" + "─" * 80)
    print(f"📊 Progress: {current}/{total} ({progress:.1f}%)")
    print(
        f"{status_icon.get(result.status, '❓')} {result.config_name}: {result.status.upper()}"
    )
    if result.duration:
        print(f"   Duration: {result.duration:.1f}s ({result.duration / 60:.1f}min)")
    print("─" * 80)


def print_batch_summary(results: List[ExperimentResult]):
    """배치 실행 최종 요약 리포트"""
    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")

    total_duration = sum(r.duration for r in results if r.duration)

    print("\n\n" + "=" * 80)
    print("📈 BATCH RUN SUMMARY")
    print("=" * 80)
    print(f"\n📊 Overall Statistics:")
    print(f"   Total experiments: {total}")
    print(f"   ✅ Succeeded: {success}")
    print(f"   ❌ Failed: {failed}")
    print(
        f"   ⏱️  Total time: {total_duration:.1f}s ({total_duration / 60:.1f}min / {total_duration / 3600:.1f}h)"
    )

    if success > 0:
        avg_duration = (
            sum(r.duration for r in results if r.status == "success") / success
        )
        print(
            f"   📊 Avg time per experiment: {avg_duration:.1f}s ({avg_duration / 60:.1f}min)"
        )

    print(f"\n📝 Detailed Results:")
    print(f"{'No.':<5} {'Status':<10} {'Config':<50} {'Duration':<15}")
    print("-" * 80)

    for idx, result in enumerate(results, 1):
        status_icon = {"success": "✅", "failed": "❌"}
        icon = status_icon.get(result.status, "❓")
        duration_str = (
            f"{result.duration:.1f}s ({result.duration / 60:.1f}min)"
            if result.duration
            else "N/A"
        )

        print(
            f"{idx:<5} {icon} {result.status:<8} {result.config_name:<50} {duration_str:<15}"
        )

    print("=" * 80)

    # 실패한 실험 상세
    if failed > 0:
        print(f"\n⚠️  Failed Experiments:")
        for result in results:
            if result.status == "failed":
                print(f"   • {result.config_name} (exit code: {result.exit_code})")
        print("\n   💡 Tip: Re-run failed experiments individually to debug")

    # 최종 성공률
    if total > 0:
        success_rate = (success / total) * 100
        print("\n")
        if success_rate == 100:
            print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉")
        elif success_rate >= 80:
            print(f"✨ Great! {success_rate:.1f}% success rate")
        elif success_rate >= 50:
            print(f"⚠️  Mixed results: {success_rate:.1f}% success rate")
        else:
            print(f"❌ Many failures: {success_rate:.1f}% success rate")

    print("=" * 80 + "\n")


def run_batch(
    config_paths: List[str], mode: str, continue_on_error: bool = True
) -> int:
    """
    여러 실험을 순차적으로 실행

    Args:
        config_paths: YAML 설정 파일 경로 리스트
        mode: 실행 모드 (train/inference/pipeline)
        continue_on_error: 실패해도 계속 진행할지 여부

    Returns:
        exit_code (0: 모두 성공, 1: 하나 이상 실패)
    """
    total = len(config_paths)

    print("\n" + "=" * 80)
    print("🚀 BATCH MODE STARTED")
    print("=" * 80)
    print(f"📋 Experiments to run: {total}")
    print(f"🎯 Mode: {mode}")
    print(f"⚙️  Continue on error: {continue_on_error}")
    print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nExperiment list:")
    for idx, config in enumerate(config_paths, 1):
        print(f"  {idx}. {Path(config).stem}")
    print("=" * 80)

    results = []

    for idx, config_path in enumerate(config_paths, 1):
        result = ExperimentResult(config_path, mode)
        results.append(result)

        result.start()
        exit_code, duration = run_single_experiment(config_path, mode)
        result.finish(exit_code)

        print_batch_progress(idx, total, result)

        # 실패 시 중단 여부 확인
        if exit_code != 0 and not continue_on_error:
            print("\n⚠️  Experiment failed and continue-on-error is disabled.")
            print(f"   Stopping batch run at {idx}/{total}")
            break

    # 최종 요약
    print_batch_summary(results)

    # Exit code 결정
    failed_count = sum(1 for r in results if r.status == "failed")
    return 0 if failed_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="MRC Project Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # === Single Experiment ===
  # Train only
  python run.py --mode train --config configs/my_experiment.yaml
  
  # Inference only
  python run.py --mode inference --config configs/my_experiment.yaml
  
  # Full pipeline (train → inference)
  python run.py --mode pipeline --config configs/my_experiment.yaml
  
  # === Batch Mode (Multiple Experiments) ===
  # 여러 실험 순차 실행 (밤새 GPU 돌리기)
  python run.py --mode batch --batch-mode pipeline --configs configs/exp1.yaml configs/exp2.yaml
  
  # 와일드카드로 모든 실험 자동 실행
  python run.py --mode batch --batch-mode pipeline --configs configs/experiments/*.yaml
  
  # Train만 (여러 모델 학습)
  python run.py --mode batch --batch-mode train --configs configs/*.yaml
  
  # 실패하면 중단 (기본은 계속 진행)
  python run.py --mode batch --batch-mode pipeline --configs configs/*.yaml --stop-on-error

Tips:
  • Batch mode는 GPU를 쉬지 않고 계속 돌릴 때 유용합니다
  • tmux/screen과 함께 사용하면 SSH 연결이 끊겨도 실험이 계속됩니다
  • 실패한 실험만 따로 재실행하려면 해당 config로 단일 모드 실행하세요
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "inference", "pipeline", "batch"],
        help="Execution mode",
    )

    parser.add_argument(
        "--config", type=str, help="Path to YAML config file (for single mode)"
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        help="Paths to multiple YAML config files (for batch mode)",
    )

    parser.add_argument(
        "--batch-mode",
        type=str,
        choices=["train", "inference", "pipeline"],
        default="pipeline",
        help="Mode to use for each experiment in batch (default: pipeline)",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch run if any experiment fails (default: continue)",
    )

    args = parser.parse_args()

    # 모드별 argument 검증
    if args.mode == "batch":
        if not args.configs:
            print("❌ Error: --configs is required for batch mode")
            parser.print_help()
            sys.exit(1)

        # Config 파일 검증
        valid_configs = []
        for config in args.configs:
            config_path = Path(config)
            if not config_path.exists():
                print(f"⚠️  Warning: Config file not found: {config} (skipping)")
                continue
            valid_configs.append(str(config_path))

        if not valid_configs:
            print("❌ Error: No valid config files found")
            sys.exit(1)

        # Batch 실행
        exit_code = run_batch(
            valid_configs, args.batch_mode, continue_on_error=not args.stop_on_error
        )

    else:
        # Single 모드
        if not args.config:
            print("❌ Error: --config is required for single mode")
            parser.print_help()
            sys.exit(1)

        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Error: Config file not found: {args.config}")
            sys.exit(1)

        # 단일 실행
        if args.mode == "train":
            exit_code = run_train(str(config_path))
        elif args.mode == "inference":
            exit_code = run_inference(str(config_path))
        elif args.mode == "pipeline":
            exit_code = run_pipeline(str(config_path))
        else:
            print(f"❌ Unknown mode: {args.mode}")
            sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
