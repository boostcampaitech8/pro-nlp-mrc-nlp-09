"""
최종 모델 성능 평가 및 결과 저장 유틸리티
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional
import evaluate


class FinalEvaluator:
    """
    학습 완료 후 train/validation/test에 대한 종합 평가 수행
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metric = evaluate.load("squad")
        self.results = {
            "evaluation_time": datetime.now().isoformat(),
            "train_performance": {},
            "validation_performance": {},
            "validation_with_retrieval_performance": {},
            "test_performance": {},
        }

    def evaluate_split(
        self,
        predictions: Dict,
        references: Dict,
        split_name: str,
        with_retrieval: bool = False,
    ) -> Dict:
        """
        특정 split에 대한 평가 수행

        Args:
            predictions: {id: prediction_text}
            references: {id: answers}
            split_name: 'train', 'validation', 'test'
            with_retrieval: retrieval 사용 여부
        """
        # predictions를 squad 형식으로 변환
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        formatted_references = [{"id": k, "answers": v} for k, v in references.items()]

        # 메트릭 계산
        metrics = self.metric.compute(
            predictions=formatted_predictions, references=formatted_references
        )

        result = {
            "exact_match": metrics["exact_match"],
            "f1": metrics["f1"],
            "total_samples": len(predictions),
            "with_retrieval": with_retrieval,
        }

        # 결과 저장
        if split_name == "train":
            self.results["train_performance"] = result
        elif split_name == "validation":
            if with_retrieval:
                self.results["validation_with_retrieval_performance"] = result
            else:
                self.results["validation_performance"] = result
        elif split_name == "test":
            self.results["test_performance"] = result

        return result

    def save_summary(self):
        """종합 평가 결과를 JSON으로 저장"""
        summary_path = os.path.join(self.output_dir, "final_evaluation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✅ Final evaluation summary saved to {summary_path}")

    def print_summary(self):
        """종합 평가 결과를 보기 좋게 출력"""
        print("\n" + "=" * 80)
        print("🎯 FINAL MODEL PERFORMANCE SUMMARY")
        print("=" * 80)

        def print_performance(title: str, perf: Dict):
            if not perf:
                print(f"\n{title}: Not evaluated")
                return
            print(f"\n{title}:")
            print(f"  📊 Exact Match: {perf['exact_match']:.2f}")
            print(f"  📊 F1 Score: {perf['f1']:.2f}")
            print(f"  📝 Total Samples: {perf['total_samples']}")
            if perf.get("with_retrieval") is not None:
                print(
                    f"  🔍 With Retrieval: {'Yes' if perf['with_retrieval'] else 'No'}"
                )

        print_performance(
            "📘 Train Performance", self.results.get("train_performance", {})
        )
        print_performance(
            "📗 Validation Performance (Direct Context)",
            self.results.get("validation_performance", {}),
        )
        print_performance(
            "📙 Validation Performance (With Retrieval)",
            self.results.get("validation_with_retrieval_performance", {}),
        )
        print_performance(
            "📕 Test Performance", self.results.get("test_performance", {})
        )

        print("=" * 80 + "\n")


def save_predictions(
    predictions: Dict, output_path: str, split_name: str = "predictions"
):
    """
    Predictions를 JSON 파일로 저장

    Args:
        predictions: {id: prediction_text}
        output_path: 저장할 디렉토리 경로
        split_name: 파일명에 사용할 split 이름
    """
    os.makedirs(output_path, exist_ok=True)
    pred_file = os.path.join(output_path, f"{split_name}_predictions.json")

    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"✅ Predictions saved to {pred_file}")
    return pred_file


def save_detailed_results(
    predictions: Dict,
    examples: list,
    output_path: str,
    split_name: str = "detailed",
):
    """
    사후 분석을 위한 상세 결과 저장

    Args:
        predictions: {id: prediction_text}
        examples: 원본 examples (question, context, answers 포함)
        output_path: 저장할 디렉토리 경로
        split_name: 파일명에 사용할 split 이름
    """
    os.makedirs(output_path, exist_ok=True)
    detailed_file = os.path.join(output_path, f"{split_name}_detailed_results.json")

    detailed_results = []
    metric = evaluate.load("squad")

    for example in examples:
        example_id = example["id"]
        prediction = predictions.get(example_id, "")

        # 개별 메트릭 계산
        if "answers" in example and example["answers"]["text"]:
            individual_metric = metric.compute(
                predictions=[{"id": example_id, "prediction_text": prediction}],
                references=[{"id": example_id, "answers": example["answers"]}],
            )
            em_score = individual_metric["exact_match"]
            f1_score = individual_metric["f1"]
        else:
            em_score = None
            f1_score = None

        detailed_results.append(
            {
                "id": example_id,
                "question": example.get("question", ""),
                "context": example.get("context", "")[:500]
                + "...",  # context는 앞 500자만
                "prediction": prediction,
                "ground_truth": example.get("answers", {}).get("text", []),
                "em_score": em_score,
                "f1_score": f1_score,
            }
        )

    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Detailed results saved to {detailed_file}")
    return detailed_file
