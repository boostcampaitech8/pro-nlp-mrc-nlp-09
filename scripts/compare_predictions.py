"""
Validation 정답과 예측 결과 비교 스크립트

Usage:
    python scripts/compare_predictions.py
    python scripts/compare_predictions.py --predictions ./outputs/.../predictions_val.json
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from datasets import load_from_disk


def compare_predictions(
    predictions_path: str,
    dataset_path: str = "./data/train_dataset",
    output_path: str = None
):
    """
    Validation 정답과 예측 결과를 비교하여 CSV로 저장
    
    Args:
        predictions_path: predictions_val.json 경로
        dataset_path: train_dataset 경로 (validation split 포함)
        output_path: 출력 CSV 경로 (None이면 자동 생성)
    """
    print("=" * 80)
    print("📊 Prediction Comparison Tool")
    print("=" * 80)
    
    # 1. Dataset 로드
    print(f"\n[1/4] Loading dataset from {dataset_path}...")
    ds = load_from_disk(dataset_path)
    val_ds = ds['validation']
    print(f"   ✓ Loaded {len(val_ds)} validation samples")
    
    # 2. Predictions 로드
    print(f"\n[2/4] Loading predictions from {predictions_path}...")
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    print(f"   ✓ Loaded {len(predictions)} predictions")
    
    # 2-1. val_results.json에서 전체 EM/F1 점수 로드
    results_path = Path(predictions_path).parent / "val_results.json"
    overall_em = overall_f1 = None
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            overall_em = results.get('eval_exact_match', 0)
            overall_f1 = results.get('eval_f1', 0)
        print(f"   ✓ Loaded overall metrics: EM={overall_em:.2f}%, F1={overall_f1:.2f}")
    
    # 3. 비교 데이터 생성
    print(f"\n[3/4] Comparing predictions with ground truth answers...")
    comparison_data = []
    correct_count = 0
    
    for ex in val_ds:
            qid = ex['id']
            question = ex['question']
            context = ex.get('context', '')
            ground_truth = ex['answers']['text']
            # gold_pred: gold context에서 추론한 정답 (실제 gold context 기반 예측이 있으면 여기서 로드)
            # retrieval_pred: retrieval로 찾은 context에서 추론한 정답
            retrieval_pred = predictions.get(qid, "")
            # gold_pred는 별도 predictions_gold.json 등에서 불러올 수 있음 (현재는 미사용)
            comparison_data.append({
                "index": qid,
                "question": question,
                "context_snippet": context[:100] + "..." if len(context) > 100 else context,
                "ground_truth": " | ".join(ground_truth),
                "retrieval_pred": retrieval_pred,
            })
    
    # EM/F1 계산은 val_results.json에서만 로드
    
    # 4. CSV 저장
    df = pd.DataFrame(comparison_data)
    if output_path is None:
        pred_dir = Path(predictions_path).parent
        output_path = pred_dir / "val_comparison_detailed.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # simple: ground_truth, retrieval_pred
    simple_df = df[['ground_truth', 'retrieval_pred']].copy()
    simple_output = Path(output_path).parent / "val_comparison_simple.csv"
    simple_df.to_csv(simple_output, index=False, encoding='utf-8-sig')

    # wrong_only: ground_truth, retrieval_pred (정답과 일치하지 않는 것만)
    def is_wrong(row):
        return row['retrieval_pred'] not in row['ground_truth'].split(' | ')
    wrong_df = df[simple_df.apply(is_wrong, axis=1)].copy()
    wrong_output = Path(output_path).parent / "val_comparison_wrong_only.csv"
    wrong_df[['ground_truth', 'retrieval_pred']].to_csv(wrong_output, index=False, encoding='utf-8-sig')

    print(f"\n[4/4] Saved comparison results:")
    print(f"   📄 Detailed: {output_path}")
    print(f"   📄 Simple: {simple_output}")
    print(f"   📄 Wrong only: {wrong_output}")
    print(f"\n   📊 Official Metrics (from val_results.json):")
    if overall_em is not None and overall_f1 is not None:
        print(f"      EM: {overall_em:.2f}%")
        print(f"      F1: {overall_f1:.2f}")
    print("\n" + "=" * 80)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare validation predictions with gold answers")
    parser.add_argument(
        "--predictions",
        type=str,
        default="./outputs/dahyeong/HANTAEK_rob-large-kq-v1-qa-finetuned_stride64/predictions_val.json",
        help="Path to predictions_val.json"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/train_dataset",
        help="Path to train_dataset (containing validation split)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: same dir as predictions)"
    )
    
    args = parser.parse_args()
    
    compare_predictions(
        predictions_path=args.predictions,
        dataset_path=args.dataset,
        output_path=args.output
    )
