#!/usr/bin/env python
"""
자동 최적 앙상블 탐색 및 실행 스크립트

메타데이터(nbest_predictions.json, eval_results.json)만 있으면
최적의 앙상블 조합을 자동으로 찾아서 제출 파일을 생성합니다.

사용법:
    # Val 기준 최적 조합 탐색만
    python scripts/auto_ensemble.py --mode search

    # Val 기준 최적 조합으로 Test 앙상블 실행
    python scripts/auto_ensemble.py --mode run

    # Test nbest 있는 모델만 사용
    python scripts/auto_ensemble.py --mode run --test-only

    # 특정 모델들로 제한
    python scripts/auto_ensemble.py --mode run --models oceann315 HANTAEK_roberta_large_vanilla roberta-large
"""

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger

logger = get_logger(__name__)


class AutoEnsemble:
    def __init__(
        self,
        models_dir: str = "/data/ephemeral/home/shared/outputs/dahyeong",
        data_dir: str = "./data",
        output_dir: str = "./outputs/ensemble",
    ):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 정답 로드
        self.answers = self._load_answers()

        # 모델 메타데이터 로드
        self.models = self._scan_models()

    def _load_answers(self) -> Dict[str, List[str]]:
        """Validation 정답 로드"""
        from datasets import load_from_disk

        ds = load_from_disk(str(self.data_dir / "train_dataset"))
        return {ex["id"]: ex["answers"]["text"] for ex in ds["validation"]}

    def _load_test_order(self) -> List[str]:
        """Test dataset 원본 순서 로드"""
        from datasets import load_from_disk

        ds = load_from_disk(str(self.data_dir / "test_dataset"))
        return ds["validation"]["id"]

    def _scan_models(self) -> Dict[str, dict]:
        """모델 디렉토리 스캔하여 메타데이터 수집"""
        models = {}

        for d in self.models_dir.iterdir():
            if not d.is_dir():
                continue

            val_nbest = d / "nbest_predictions.json"
            test_nbest = d / "nbest_predictions_test.json"
            eval_file = d / "eval_results.json"

            if not val_nbest.exists():
                continue

            model_info = {
                "name": d.name,
                "path": str(d),
                "has_val_nbest": val_nbest.exists(),
                "has_test_nbest": test_nbest.exists(),
                "val_em": None,
                "val_nbest_path": str(val_nbest) if val_nbest.exists() else None,
                "test_nbest_path": str(test_nbest) if test_nbest.exists() else None,
            }

            # Val EM 로드
            if eval_file.exists():
                try:
                    r = json.loads(eval_file.read_text())
                    model_info["val_em"] = r.get(
                        "eval_exact_match", r.get("exact_match", 0)
                    )
                except:
                    pass

            # nbest 개수 확인
            try:
                val_data = json.loads(val_nbest.read_text())
                model_info["val_count"] = len(val_data)
            except:
                model_info["val_count"] = 0

            if test_nbest.exists():
                try:
                    test_data = json.loads(test_nbest.read_text())
                    model_info["test_count"] = len(test_data)
                except:
                    model_info["test_count"] = 0
            else:
                model_info["test_count"] = 0

            # 유효한 모델만 추가 (val 240개)
            if model_info["val_count"] == 240:
                models[d.name] = model_info

        return models

    def get_available_models(self, test_only: bool = False) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        if test_only:
            return [
                m
                for m, info in self.models.items()
                if info["has_test_nbest"] and info["test_count"] == 600
            ]
        return list(self.models.keys())

    def print_models(self):
        """모델 목록 출력"""
        print("\n" + "=" * 70)
        print("📋 사용 가능한 모델 목록")
        print("=" * 70)
        print(f"{'모델명':<40} | {'Val EM':>8} | {'Test':>6}")
        print("-" * 70)

        sorted_models = sorted(
            self.models.items(), key=lambda x: -(x[1]["val_em"] or 0)
        )

        for name, info in sorted_models:
            em_str = f"{info['val_em']:.2f}%" if info["val_em"] else "N/A"
            test_str = (
                "✅" if info["has_test_nbest"] and info["test_count"] == 600 else "❌"
            )
            print(f"{name:<40} | {em_str:>8} | {test_str:>6}")

        print("-" * 70)
        print(f"총 {len(self.models)}개 모델")
        test_available = len(self.get_available_models(test_only=True))
        print(f"Test 앙상블 가능: {test_available}개")
        print()

    def _load_nbest(self, model_name: str, use_test: bool = False) -> dict:
        """nbest predictions 로드"""
        info = self.models[model_name]
        path = info["test_nbest_path"] if use_test else info["val_nbest_path"]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ensemble_predict(
        self,
        model_weights: List[Tuple[str, float]],
        use_test: bool = False,
        top_k: int = 5,
    ) -> Dict[str, str]:
        """앙상블 예측 수행"""
        # nbest 로드
        nbest_data = {m: self._load_nbest(m, use_test) for m, _ in model_weights}

        predictions = {}
        first_model = model_weights[0][0]

        for qid in nbest_data[first_model].keys():
            vote = {}

            for model_name, weight in model_weights:
                nbest = nbest_data[model_name].get(qid, [])
                for pred in nbest[:top_k]:
                    text = pred["text"]
                    prob = pred.get("probability", pred.get("score", 0))
                    if text not in vote:
                        vote[text] = 0
                    vote[text] += prob * weight

            if vote:
                predictions[qid] = max(vote.items(), key=lambda x: x[1])[0]
            else:
                predictions[qid] = ""

        return predictions

    def _calc_em(self, predictions: Dict[str, str]) -> float:
        """EM 계산"""
        correct = 0
        total = 0
        for qid, pred in predictions.items():
            if qid in self.answers:
                total += 1
                if pred in self.answers[qid]:
                    correct += 1
        return (correct / total * 100) if total > 0 else 0

    def search_best_combinations(
        self,
        model_names: Optional[List[str]] = None,
        test_only: bool = False,
        max_models: int = 3,
        top_n: int = 10,
    ) -> List[Tuple[float, List[Tuple[str, float]]]]:
        """최적 앙상블 조합 탐색"""

        if model_names:
            available = [m for m in model_names if m in self.models]
        else:
            available = self.get_available_models(test_only=test_only)

        if len(available) < 2:
            logger.error(f"앙상블에 필요한 모델이 부족합니다: {len(available)}개")
            return []

        logger.info(f"🔍 {len(available)}개 모델로 최적 조합 탐색...")

        all_results = []

        # 2개 조합
        logger.info("  2개 모델 조합 탐색 중...")
        for m1, m2 in combinations(available, 2):
            for w1 in [0.4, 0.5, 0.6, 0.7]:
                w2 = round(1 - w1, 2)
                preds = self._ensemble_predict([(m1, w1), (m2, w2)])
                em = self._calc_em(preds)
                all_results.append((em, [(m1, w1), (m2, w2)]))

        # 3개 조합
        if max_models >= 3 and len(available) >= 3:
            logger.info("  3개 모델 조합 탐색 중...")
            for m1, m2, m3 in combinations(available, 3):
                for w1 in [0.3, 0.4, 0.5, 0.6]:
                    for w2 in [0.2, 0.3, 0.4]:
                        w3 = round(1 - w1 - w2, 2)
                        if w3 > 0.05:
                            preds = self._ensemble_predict(
                                [(m1, w1), (m2, w2), (m3, w3)]
                            )
                            em = self._calc_em(preds)
                            all_results.append((em, [(m1, w1), (m2, w2), (m3, w3)]))

        # 정렬
        all_results.sort(key=lambda x: -x[0])

        return all_results[:top_n]

    def print_search_results(
        self, results: List[Tuple[float, List[Tuple[str, float]]]]
    ):
        """탐색 결과 출력"""
        print("\n" + "=" * 70)
        print("🏆 최적 앙상블 조합 (Val EM 기준)")
        print("=" * 70)

        for i, (em, weights) in enumerate(results, 1):
            weight_str = " + ".join([f"{m}({w:.1f})" for m, w in weights])
            print(f"  {i:2d}. {em:.2f}% | {weight_str}")

        print("=" * 70)
        print()

    def run_ensemble(
        self,
        model_weights: List[Tuple[str, float]],
        output_name: str,
        use_test: bool = True,
    ) -> dict:
        """앙상블 실행 및 결과 저장"""

        logger.info(f"🔀 앙상블 실행: {output_name}")
        for m, w in model_weights:
            logger.info(f"   - {m}: {w:.1%}")

        # 예측
        predictions = self._ensemble_predict(model_weights, use_test=use_test)
        logger.info(f"✅ 예측 완료: {len(predictions)}개")

        # 출력 디렉토리
        out_dir = self.output_dir / output_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # predictions.json 저장
        pred_path = out_dir / "predictions.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 {pred_path}")

        # CSV 저장 (원본 순서)
        csv_path = out_dir / "predictions_submit.csv"
        if use_test:
            order = self._load_test_order()
        else:
            order = list(self.answers.keys())

        with open(csv_path, "w", encoding="utf-8") as f:
            for qid in order:
                answer = predictions.get(qid, "")
                f.write(f"{qid}\t{answer}\n")
        logger.info(f"💾 {csv_path}")

        # config.json 저장
        config = {
            "ensemble_type": "auto_ensemble",
            "models": [m for m, _ in model_weights],
            "weights": [w for _, w in model_weights],
            "use_test": use_test,
            "prediction_count": len(predictions),
        }
        config_path = out_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # Val 기준 EM 계산 (test가 아닐 때만)
        result = {"output_dir": str(out_dir), "predictions": len(predictions)}

        if not use_test:
            em = self._calc_em(predictions)
            result["val_em"] = em

            eval_results = {"eval_exact_match": em, "eval_total": len(predictions)}
            eval_path = out_dir / "eval_results.json"
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=2)
            logger.info(f"📊 Val EM: {em:.2f}%")

        return result

    def auto_run(
        self, test_only: bool = True, output_name: Optional[str] = None
    ) -> dict:
        """자동으로 최적 조합 찾아서 실행"""

        # 최적 조합 탐색
        results = self.search_best_combinations(test_only=test_only)

        if not results:
            logger.error("유효한 앙상블 조합을 찾지 못했습니다.")
            return {}

        self.print_search_results(results)

        # 최고 조합 선택
        best_em, best_weights = results[0]

        # 출력 이름 생성
        if output_name is None:
            model_abbrevs = [m[:10] for m, _ in best_weights]
            output_name = f"auto_{'_'.join(model_abbrevs)}"

        # 실행
        return self.run_ensemble(
            model_weights=best_weights, output_name=output_name, use_test=test_only
        )


def main():
    parser = argparse.ArgumentParser(description="자동 최적 앙상블 탐색 및 실행")
    parser.add_argument(
        "--mode",
        choices=["search", "run", "list"],
        default="search",
        help="실행 모드: search(탐색만), run(탐색+실행), list(모델목록)",
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Test nbest가 있는 모델만 사용"
    )
    parser.add_argument("--models", nargs="+", help="특정 모델들만 사용")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="--models와 함께 사용할 가중치 (순서대로)",
    )
    parser.add_argument("--output-name", help="출력 디렉토리 이름")
    parser.add_argument(
        "--models-dir",
        default="/data/ephemeral/home/shared/outputs/dahyeong",
        help="모델 출력 디렉토리",
    )
    parser.add_argument("--top-n", type=int, default=10, help="상위 N개 조합 출력")

    args = parser.parse_args()

    # AutoEnsemble 초기화
    auto_ens = AutoEnsemble(models_dir=args.models_dir)

    if args.mode == "list":
        auto_ens.print_models()
        return

    if args.mode == "search":
        results = auto_ens.search_best_combinations(
            model_names=args.models, test_only=args.test_only, top_n=args.top_n
        )
        auto_ens.print_search_results(results)

    elif args.mode == "run":
        if args.models and args.weights:
            # 수동 지정
            if len(args.models) != len(args.weights):
                logger.error("--models와 --weights 개수가 일치해야 합니다.")
                return

            model_weights = list(zip(args.models, args.weights))
            auto_ens.run_ensemble(
                model_weights=model_weights,
                output_name=args.output_name or "manual_ensemble",
                use_test=args.test_only,
            )
        else:
            # 자동 탐색 후 실행
            auto_ens.auto_run(test_only=args.test_only, output_name=args.output_name)


if __name__ == "__main__":
    main()
