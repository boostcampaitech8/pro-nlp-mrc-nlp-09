"""
학습 과정의 메트릭을 추적하고 시각화하는 유틸리티
"""

import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
from transformers import TrainerCallback


class MetricsTracker(TrainerCallback):
    """
    학습 과정의 메트릭을 추적하고 저장하는 Callback
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.train_losses = []
        self.eval_metrics = []
        self.steps = []
        self.epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 발생 시마다 호출"""
        if logs is None:
            return

        # Training loss 기록
        if "loss" in logs:
            self.train_losses.append(
                {"step": state.global_step, "epoch": state.epoch, "loss": logs["loss"]}
            )

        # Evaluation metrics 기록
        if "eval_exact_match" in logs:
            self.eval_metrics.append(
                {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "exact_match": logs["eval_exact_match"],
                    "f1": logs["eval_f1"],
                    "eval_loss": logs.get("eval_loss", None),
                }
            )

    def on_train_end(self, args, state, control, **kwargs):
        """학습 종료 시 메트릭 저장 및 시각화"""
        self.save_metrics()
        self.save_epoch_summary()  # 에포크별 요약 저장
        self.plot_metrics()

    def save_metrics(self):
        """메트릭을 JSON 파일로 저장 (전체 로그)"""
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {"train_losses": self.train_losses, "eval_metrics": self.eval_metrics},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"✅ Training metrics saved to {metrics_path}")

    def save_epoch_summary(self):
        """에포크별 EM/F1 스코어를 사람이 보기 쉬운 형태로 저장"""
        if not self.eval_metrics:
            return

        # 에포크별 메트릭 정리
        epoch_summary = []
        for metric in self.eval_metrics:
            epoch_summary.append(
                {
                    "epoch": round(metric["epoch"], 2),
                    "exact_match": round(metric["exact_match"], 2),
                    "f1": round(metric["f1"], 2),
                    "eval_loss": round(metric.get("eval_loss", 0), 4)
                    if metric.get("eval_loss")
                    else None,
                    "step": metric["step"],
                }
            )

        # Best 메트릭 찾기
        best_em = max(epoch_summary, key=lambda x: x["exact_match"])
        best_f1 = max(epoch_summary, key=lambda x: x["f1"])

        summary = {
            "epoch_metrics": epoch_summary,
            "best_performance": {
                "best_exact_match": {
                    "score": best_em["exact_match"],
                    "epoch": best_em["epoch"],
                    "step": best_em["step"],
                    "f1_at_best_em": best_em["f1"],
                },
                "best_f1": {
                    "score": best_f1["f1"],
                    "epoch": best_f1["epoch"],
                    "step": best_f1["step"],
                    "em_at_best_f1": best_f1["exact_match"],
                },
            },
        }

        # JSON 저장
        summary_path = os.path.join(self.output_dir, "epoch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Markdown 테이블 저장 (사람이 읽기 편한 형태)
        md_path = os.path.join(self.output_dir, "epoch_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Training Epoch Summary\n\n")
            f.write("## Epoch-by-Epoch Performance\n\n")
            f.write("| Epoch | EM Score | F1 Score | Eval Loss | Step |\n")
            f.write("|-------|----------|----------|-----------|------|\n")
            for m in epoch_summary:
                eval_loss_str = f"{m['eval_loss']:.4f}" if m["eval_loss"] else "N/A"
                f.write(
                    f"| {m['epoch']:.2f} | {m['exact_match']:.2f} | {m['f1']:.2f} | {eval_loss_str} | {m['step']} |\n"
                )

            f.write(f"\n## Best Performance\n\n")
            f.write(f"**Best Exact Match:** {best_em['exact_match']:.2f}%\n")
            f.write(f"- Epoch: {best_em['epoch']:.2f}\n")
            f.write(f"- Step: {best_em['step']}\n")
            f.write(f"- F1 at this point: {best_em['f1']:.2f}%\n\n")

            f.write(f"**Best F1 Score:** {best_f1['f1']:.2f}%\n")
            f.write(f"- Epoch: {best_f1['epoch']:.2f}\n")
            f.write(f"- Step: {best_f1['step']}\n")
            f.write(f"- EM at this point: {best_f1['exact_match']:.2f}%\n")

        print(f"✅ Epoch summary saved to {summary_path} and {md_path}")

    def plot_metrics(self):
        """메트릭을 그래프로 시각화"""
        if not self.eval_metrics:
            print("⚠️  No evaluation metrics to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

        # 1. Train Loss
        if self.train_losses:
            steps = [m["step"] for m in self.train_losses]
            losses = [m["loss"] for m in self.train_losses]
            axes[0, 0].plot(steps, losses, "b-", linewidth=2)
            axes[0, 0].set_title("Training Loss", fontsize=12, fontweight="bold")
            axes[0, 0].set_xlabel("Steps")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Exact Match (EM)
        eval_steps = [m["step"] for m in self.eval_metrics]
        em_scores = [m["exact_match"] for m in self.eval_metrics]
        axes[0, 1].plot(eval_steps, em_scores, "g-o", linewidth=2, markersize=6)
        axes[0, 1].set_title("Exact Match (EM) Score", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Steps")
        axes[0, 1].set_ylabel("EM Score")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 100])

        # 3. F1 Score
        f1_scores = [m["f1"] for m in self.eval_metrics]
        axes[1, 0].plot(eval_steps, f1_scores, "r-o", linewidth=2, markersize=6)
        axes[1, 0].set_title("F1 Score", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Steps")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 100])

        # 4. Eval Loss
        eval_losses = [
            m.get("eval_loss") for m in self.eval_metrics if m.get("eval_loss")
        ]
        if eval_losses:
            eval_loss_steps = [
                m["step"] for m in self.eval_metrics if m.get("eval_loss")
            ]
            axes[1, 1].plot(
                eval_loss_steps, eval_losses, "m-o", linewidth=2, markersize=6
            )
            axes[1, 1].set_title("Evaluation Loss", fontsize=12, fontweight="bold")
            axes[1, 1].set_xlabel("Steps")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "training_metrics.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Training metrics plot saved to {plot_path}")

    def print_summary(self):
        """학습 요약 출력 - Train/Eval 메트릭 모두 표시"""
        if not self.eval_metrics:
            return

        print("\n" + "=" * 80)
        print("📊 TRAINING SUMMARY")
        print("=" * 80)

        # Best metrics 찾기
        best_em_metric = max(self.eval_metrics, key=lambda x: x["exact_match"])
        best_f1_metric = max(self.eval_metrics, key=lambda x: x["f1"])
        final_eval_metric = self.eval_metrics[-1]

        # Train loss 정보
        if self.train_losses:
            final_train_loss = self.train_losses[-1]
            best_train_loss = min(self.train_losses, key=lambda x: x["loss"])
            print(f"\n📉 Training Loss:")
            print(
                f"   - Final: {final_train_loss['loss']:.4f} (Epoch {final_train_loss['epoch']:.2f}, Step {final_train_loss['step']})"
            )
            print(
                f"   - Best: {best_train_loss['loss']:.4f} (Epoch {best_train_loss['epoch']:.2f}, Step {best_train_loss['step']})"
            )

        # Eval 메트릭
        print(f"\n📊 Validation Metrics:")
        print(f"\n🏆 Best Exact Match: {best_em_metric['exact_match']:.2f}")
        print(f"   - Epoch: {best_em_metric['epoch']:.2f}")
        print(f"   - Step: {best_em_metric['step']}")
        print(f"   - F1: {best_em_metric['f1']:.2f}")
        if best_em_metric.get("eval_loss"):
            print(f"   - Eval Loss: {best_em_metric['eval_loss']:.4f}")

        print(f"\n🏆 Best F1 Score: {best_f1_metric['f1']:.2f}")
        print(f"   - Epoch: {best_f1_metric['epoch']:.2f}")
        print(f"   - Step: {best_f1_metric['step']}")
        print(f"   - EM: {best_f1_metric['exact_match']:.2f}")
        if best_f1_metric.get("eval_loss"):
            print(f"   - Eval Loss: {best_f1_metric['eval_loss']:.4f}")

        print(
            f"\n📈 Final Validation Metrics (Epoch {final_eval_metric['epoch']:.2f}):"
        )
        print(f"   - EM: {final_eval_metric['exact_match']:.2f}")
        print(f"   - F1: {final_eval_metric['f1']:.2f}")
        if final_eval_metric.get("eval_loss"):
            print(f"   - Eval Loss: {final_eval_metric['eval_loss']:.4f}")

        print(f"\n💡 Test 메트릭은 inference 후에 확인할 수 있습니다.")

        print("=" * 80 + "\n")
