# MRC 프로젝트 실험 관리 가이드

## 개요

이 프로젝트는 YAML 기반 설정으로 train과 inference를 통합 관리하며, **Confidence Score 기반 예측 분석**을 지원합니다.

**🎯 핵심 기능:**
- ✅ **Confidence Score 계산**: 모델 확신도 기반 예측 품질 평가
- ✅ **YAML 기반 설정**: 실험 재현성 보장
- ✅ **Makefile 자동화**: 간편한 명령어로 실험 관리
- ✅ **Batch 모드**: 여러 실험 자동 순차 실행 (밤새 GPU 가동)
- ✅ **Best Checkpoint 자동 탐색**: 수동 경로 지정 불필요
- ✅ **상세 분석 도구**: Epoch별 메트릭, Retrieval 성능 비교

## ⚡ 빠른 시작 (Makefile)

### 기본 워크플로우

```bash
# 1. 도움말 확인
make help

# 2. 단일 실험 (train → inference 자동 연결)
make train-pipeline CONFIG=configs/my_experiment.yaml

# 3. 여러 실험 밤새 돌리기 (GPU 최대 활용)
# Step 1: 실험할 config들을 active 폴더에 준비
cp configs/exp1.yaml configs/active/
cp configs/exp2.yaml configs/active/

# Step 2: Batch 실행
make batch

# 4. 결과 분석
make compare-results          # 모든 실험 F1/EM 비교
make show-best                # 최고 성능 실험 찾기
```

### 주요 명령어 요약

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `make train-pipeline` | Train + Inference 자동 실행 | `make train-pipeline CONFIG=configs/exp.yaml` |
| `make batch` | configs/active/*.yaml 순차 실행 | `make batch` |
| `make compare-results` | 모든 실험 결과 비교 | `make compare-results` |
| `make gpu-status` | GPU 사용 현황 확인 | `make gpu-status` |

## 📂 출력 파일 구조

Train/Inference 실행 후 생성되는 파일들:

```
outputs/dahyeong/my_experiment/
├── 📁 checkpoint-*/                      # 학습 체크포인트
├── 📄 best_checkpoint_path.txt          # Best checkpoint 경로 (자동 생성)
├── 📄 trainer_state.json                # Trainer 상태 (HuggingFace)
├── 📄 config_used.yaml                  # 실험에 사용된 설정
│
├── 📊 학습 메트릭
│   ├── training_metrics.json           # 전체 학습 로그
│   ├── training_metrics.png            # 학습 곡선 그래프
│   ├── epoch_summary.json              # 에포크별 요약 (JSON)
│   └── epoch_summary.md                # 에포크별 요약 (Markdown 테이블)
│
├── 🎯 예측 결과
│   ├── test_pred.csv                   # Test 제출 파일 (id, prediction)
│   ├── predictions_test.json           # Test 상세 예측
│   ├── predictions_val.json            # Validation 예측
│   └── predictions_train.json          # Train 예측
│
├── 🧠 Confidence 분석 (NEW!)
│   ├── logits_test.json                # Test logits (start/end/probability)
│   ├── logits_val.json                 # Validation logits
│   ├── logits_train.json               # Train logits
│   ├── test_confidence.csv             # Test confidence scores
│   ├── val_confidence.csv              # Validation confidence scores
│   └── train_confidence.csv            # Train confidence scores
│       # 구조: id, prediction, max_prob, avg_prob, is_correct, pred_length
│
└── 📋 상세 분석
    ├── val_detailed_results.json       # Validation 상세 (question, context, prediction, confidence 포함)
    ├── train_detailed_results.json     # Train 상세
    ├── eval_results.json               # Validation 메트릭 (EM, F1)
    └── train_results.json              # Train 메트릭
```

### 핵심 파일 설명

#### 1. **Confidence CSV** (예: `val_confidence.csv`)
모델이 얼마나 확신하는지 수치화:
```csv
id,prediction,max_prob,avg_prob,is_correct,pred_length
mrc-0-000000,서울,0.987,0.945,1,2
mrc-0-000001,1950년,0.234,0.189,0,5
```
- `max_prob`: 최대 확률 (모델의 확신도)
- `avg_prob`: 평균 확률
- `is_correct`: 정답 여부 (1=정답, 0=오답)

**활용**:
```bash
# Low confidence 오답 찾기 (모델도 헷갈리는 케이스)
awk -F, '$5==0 && $4<0.5' val_confidence.csv

# High confidence 오답 찾기 (체계적 오류 - 위험!)
awk -F, '$5==0 && $4>0.8' val_confidence.csv
```

#### 2. **Detailed Results JSON** (예: `val_detailed_results.json`)
각 example의 모든 정보:
```json
[
  {
    "id": "mrc-0-000000",
    "question": "대한민국의 수도는?",
    "context": "대한민국의 수도는 서울이다...",
    "prediction": "서울",
    "ground_truth": ["서울", "서울특별시"],
    "em_score": 100.0,
    "f1_score": 100.0,
    "confidence_max": 0.987,  // ← NEW!
    "confidence_avg": 0.945   // ← NEW!
  }
]
```

**활용**: 오답 분석, 질문 유형별 성능 평가

#### 3. **Epoch Summary Markdown** (`epoch_summary.md`)
사람이 읽기 쉬운 학습 진행 상황:
```markdown
| Epoch | EM Score | F1 Score | Eval Loss | Step |
|-------|----------|----------|-----------|------|
| 1.00  | 68.50    | 75.20    | 0.8234    | 499  |
| 2.00  | 70.30    | 76.80    | 0.7123    | 998  |
| 3.00  | 72.10    | 78.30    | 0.6891    | 1497 |

**Best Exact Match:** 72.10% (Epoch 3.00)
**Best F1 Score:** 78.30% (Epoch 3.00)
```



## 🔧 주요 기능

### 1. Train + Inference Pipeline

**한 줄 명령으로 학습부터 추론까지**:
```bash
make train-pipeline CONFIG=configs/my_experiment.yaml
```

실행 과정:
1. ✅ GPU 메모리 확인 (10GB 미만 대기)
2. 🚀 Training 시작 (3 epochs)
3. 💾 Best checkpoint 자동 저장
4. 🔍 Validation set inference (EM/F1 계산)
5. 📊 Confidence score 계산
6. 📁 Test set inference (제출용 test_pred.csv 생성)

생성 파일:
- `test_pred.csv` (제출용)
- `val_confidence.csv` (분석용)
- `val_detailed_results.json` (상세 분석용)
- `epoch_summary.md` (학습 진행 요약)

---

### 2. Batch 실험 (밤새 GPU 돌리기)

**여러 실험을 순차적으로 자동 실행**:

```bash
# Step 1: 실험할 config들을 active 폴더에 준비
ls configs/active/
# exp1_bert_lr3e5.yaml
# exp2_bert_lr5e5.yaml
# exp3_electra_lr3e5.yaml

# Step 2: Batch 실행
make batch

# 실행 결과:
# 📦 [1/3] Processing: exp1_bert_lr3e5.yaml
#   ✅ Train completed (57.5min)
#   ✅ Inference completed
# 📦 [2/3] Processing: exp2_bert_lr5e5.yaml
#   ✅ Train completed (58.5min)
#   ✅ Inference completed
# 📦 [3/3] Processing: exp3_electra_lr3e5.yaml
#   ✅ Train completed (61.6min)
#   ✅ Inference completed
#
# 🎉 ALL EXPERIMENTS COMPLETED! (Total: 3h 2min)
```

**주요 특징**:
- ✅ 실패해도 다음 실험 계속 진행
- ✅ 각 실험 소요 시간 자동 추적
- ✅ 최종 Summary 리포트 생성
- ✅ GPU 공백 시간 최소화

---

### 3. Confidence 기반 예측 분석

**모델의 확신도를 수치화하여 품질 평가**:

```bash
# 1. 학습 완료 후 confidence.csv 생성됨
cat outputs/dahyeong/my_exp/val_confidence.csv
# id,prediction,max_prob,avg_prob,is_correct,pred_length
# mrc-0-000000,서울,0.987,0.945,1,2
# mrc-0-000001,1950년,0.234,0.189,0,5

# 2. Low confidence 오답 찾기 (모델도 헷갈리는 케이스)
awk -F, '$5==0 && $4<0.5 {print $1,$2,$4}' val_confidence.csv | head -10
# mrc-0-000123 김영삼 0.234
# mrc-0-000456 1980년 0.312

# 3. High confidence 오답 찾기 (체계적 오류 - 위험!)
awk -F, '$5==0 && $4>0.8 {print $1,$2,$4}' val_confidence.csv
# mrc-0-000789 서울특별시 0.892  # "서울"이 정답인데 "서울특별시"로 예측

# 4. Python으로 상세 분석
python << EOF
import pandas as pd
df = pd.read_csv('outputs/dahyeong/my_exp/val_confidence.csv')

# 정확도
accuracy = df['is_correct'].mean()
print(f"Accuracy: {accuracy:.2%}")

# Confidence 분포
errors = df[df['is_correct'] == 0]
print(f"\nError confidence distribution:")
print(errors['avg_prob'].describe())
EOF
```

**Confidence Score 활용**:
1. **Low confidence 오답**: 데이터 품질 문제 또는 어려운 질문
2. **High confidence 오답**: 모델의 체계적 오류 (재학습 필요)
3. **Low confidence 정답**: 운으로 맞춘 케이스 (불안정)
4. **High confidence 정답**: 모델이 잘 학습한 케이스

---

### 4. 결과 비교 및 분석

```bash
# 1. 모든 실험 결과 비교 (F1, EM)
make compare-results

# 출력 예시:
# 📊 Comparing experiment results:
#   exp1_bert_lr3e5       F1: 68.45    EM: 55.32
#   exp2_bert_lr5e5       F1: 71.23    EM: 59.87
#   exp3_electra_lr3e5    F1: 69.87    EM: 57.45

# 2. 최고 성능 실험 찾기
make show-best

# 출력 예시:
# 🏆 Best experiment (by F1 score):
#   Experiment: exp2_bert_lr5e5
#   F1 Score: 71.23
#   Path: ./outputs/dahyeong/exp2_bert_lr5e5/

# 3. GPU 상태 확인
make gpu-status

# 출력 예시:
# 🖥️  GPU Status:
#   GPU 0: NVIDIA A100 | Util: 85% | Mem: 25631/40960 MB
```

---

### 5. Active Config 관리

`configs/active/` 폴더로 실험 대상 관리:

```bash
# 1. 모든 config 목록 보기
make list-active

# 2. Config 유효성 검증
make check-config CONFIG=configs/my_experiment.yaml

# 3. Active 폴더 비우기 (배치 실행 전 정리)
make clean-active
```



## 📋 Makefile 명령어 전체 목록

### 실험 실행

| 명령어 | 설명 | 용도 |
|--------|------|------|
| `make train-pipeline CONFIG=...` | Train + Test inference | **가장 많이 사용** |
| `make train CONFIG=...` | Train만 실행 | 학습만 |
| `make inference CONFIG=...` | Inference만 실행 | 이미 학습된 모델 |
| `make eval-val CONFIG=...` | Validation 분석 (gold vs retrieval) | Retrieval 성능 비교 |
| `make eval-test CONFIG=...` | Test inference | 제출용 |

### Batch 실험

| 명령어 | 설명 |
|--------|------|
| `make batch` | configs/active/*.yaml 순차 실행 |
| `make list-active` | Active config 목록 보기 |
| `make check-active` | Active config 유효성 검증 |

### 결과 분석

| 명령어 | 설명 |
|--------|------|
| `make compare-results` | 모든 실험 F1/EM 비교 |
| `make show-best` | 최고 F1 실험 찾기 |

### 유틸리티

| 명령어 | 설명 |
|--------|------|
| `make gpu-status` | GPU 사용 현황 |
| `make clean-checkpoints` | checkpoint 폴더만 삭제 |
| `make check-config CONFIG=...` | YAML 유효성 검증 |
| `make help` | 도움말 |



### 모델 설정 (ModelArguments)

```yaml
##################################
# --- model (ModelArguments) ---
##################################
model_name_or_path: klue/bert-base  # 학습 시작 모델 (pretrained 또는 경로)
# config_name: null                 # 모델과 동일하면 생략
# tokenizer_name: null              # 모델과 동일하면 생략

# [Inference 전용]
use_trained_model: true             # true: output_dir에서 best checkpoint 자동 탐색
                                    # false: model_name_or_path 직접 사용
```

### 데이터 설정 (DataTrainingArguments)

```yaml
##################################
# --- data (DataTrainingArguments) ---
##################################
train_dataset_name: ./data/train_dataset     # 학습용 데이터 (train/validation split 포함)
infer_dataset_name: ./data/test_dataset      # 테스트 데이터 (정답 없음)

# [Inference 전용]
inference_split: validation                   # 추론할 데이터 선택
                                              # - train: train_dataset의 train split
                                              # - validation: train_dataset의 validation split (기본)
                                              # - test: infer_dataset_name 사용 (제출용)
```

### Inference 동작 방식

| inference_split | 데이터셋 | do_eval | do_predict | 용도 |
|----------------|---------|---------|------------|------|
| train | train_dataset/train | ✅ | ✅ | train set 성능 분석 |
| validation | train_dataset/validation | ✅ | ✅ | validation set 성능 확인 (기본) |
| test | infer_dataset_name | ❌ | ✅ | 제출용 predictions.json 생성 |

## Best Checkpoint 자동 탐색 로직

`use_trained_model=true`로 설정하면 다음 순서로 checkpoint를 탐색합니다:

1. **`best_checkpoint_path.txt`** (train.py가 자동 생성)
   - train 완료 시 best checkpoint 경로가 저장됨
   - 최우선 참조

2. **`trainer_state.json`의 `best_model_checkpoint`**
   - HuggingFace Trainer가 자동 생성
   - 2순위 참조

3. **`checkpoint-*` 폴더 중 가장 큰 숫자**
   - 위 파일들이 없을 때 fallback
   - 예: checkpoint-1234 > checkpoint-123

## 🎯 실전 워크플로우 예시

### Case 1: 첫 실험 (전체 파이프라인)

```bash
# 1. Base config 복사
cp configs/base.yaml configs/my_first_exp.yaml

# 2. 설정 수정
vim configs/my_first_exp.yaml
# - model_name_or_path: monologg/koelectra-small-v3-discriminator
# - output_dir: ./outputs/dahyeong/koelectra_baseline
# - num_train_epochs: 3

# 3. YAML 유효성 검증
make check-config CONFIG=configs/my_first_exp.yaml

# 4. Train + Inference 실행
make train-pipeline CONFIG=configs/my_first_exp.yaml

# 5. 결과 확인
ls outputs/dahyeong/koelectra_baseline/
# test_pred.csv              ← 제출용!
# val_confidence.csv         ← 분석용
# val_detailed_results.json  ← 오답 분석용
# epoch_summary.md           ← 학습 진행 확인
```

---

### Case 2: 여러 실험 비교 (Hyperparameter Tuning)

```bash
# 1. 여러 config 준비
configs/
  ├── exp1_lr2e5.yaml   # learning_rate: 2e-5
  ├── exp2_lr3e5.yaml   # learning_rate: 3e-5
  └── exp3_lr5e5.yaml   # learning_rate: 5e-5

# 2. Active 폴더로 복사
cp configs/exp*.yaml configs/active/

# 3. Batch 실행 (밤새 돌리기)
make batch

# 다음날 아침 확인:
make compare-results
# 📊 Comparing experiment results:
#   exp1_lr2e5    F1: 68.45    EM: 55.32
#   exp2_lr3e5    F1: 71.23    EM: 59.87  ← Best!
#   exp3_lr5e5    F1: 69.12    EM: 56.78

make show-best
# 🏆 Best experiment (by F1 score):
#   Experiment: exp2_lr3e5
#   F1 Score: 71.23

# 4. Active 폴더 정리
rm configs/active/*.yaml
```

---

### Case 3: Confidence 기반 오답 분석

```bash
# 1. Validation confidence 확인
cat outputs/dahyeong/my_exp/val_confidence.csv | head

# 2. 오답만 필터링
awk -F, '$5==0' outputs/dahyeong/my_exp/val_confidence.csv > errors.csv

# 3. Confidence 분포 확인 (Python)
python << EOF
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('errors.csv', names=['id','pred','max_prob','avg_prob','is_correct','length'])

# Confidence 히스토그램
df['avg_prob'].hist(bins=20)
plt.xlabel('Confidence Score')
plt.ylabel('Error Count')
plt.title('Error Confidence Distribution')
plt.savefig('error_confidence.png')

# Low vs High confidence 오답 비율
low_conf = len(df[df['avg_prob'] < 0.5])
high_conf = len(df[df['avg_prob'] > 0.8])
print(f"Low confidence errors: {low_conf} ({low_conf/len(df)*100:.1f}%)")
print(f"High confidence errors: {high_conf} ({high_conf/len(df)*100:.1f}%)")
EOF

# 4. High confidence 오답 상세 분석 (체계적 오류)
python << EOF
import pandas as pd
import json

# Confidence 로드
conf_df = pd.read_csv('outputs/dahyeong/my_exp/val_confidence.csv')

# Detailed results 로드
with open('outputs/dahyeong/my_exp/val_detailed_results.json') as f:
    details = {item['id']: item for item in json.load(f)}

# High confidence 오답 찾기
high_conf_errors = conf_df[(conf_df['is_correct'] == 0) & (conf_df['avg_prob'] > 0.8)]

print(f"Found {len(high_conf_errors)} high confidence errors (systematic errors)\n")

# 상위 10개 출력
for idx, row in high_conf_errors.head(10).iterrows():
    detail = details[row['id']]
    print(f"ID: {row['id']}")
    print(f"Question: {detail['question']}")
    print(f"Prediction: {detail['prediction']} (confidence: {row['avg_prob']:.3f})")
    print(f"Ground Truth: {detail['ground_truth']}")
    print(f"---")
EOF
```

---

### Case 4: Best Model로 Test 제출

```bash
# 1. 최고 성능 실험 찾기
make show-best
# 🏆 Best experiment: exp2_lr3e5
#   Path: ./outputs/dahyeong/exp2_lr3e5/

# 2. 해당 실험의 test_pred.csv 확인
head outputs/dahyeong/exp2_lr3e5/test_pred.csv
# mrc-0-000000	서울
# mrc-0-000001	1950년

# 3. 제출
cp outputs/dahyeong/exp2_lr3e5/test_pred.csv submission.csv
# Kaggle/Competition 사이트에 업로드
```

---

### Case 5: Retrieval 성능 비교

```bash
# 1. Validation set에 대해 gold context vs retrieval context 비교
make eval-val CONFIG=configs/my_exp.yaml

# 실행 과정:
# Step 1: Gold context로 inference (상한선 측정)
# Step 2: Retrieval context로 inference (실제 성능)
# Step 3: 두 결과 비교

# 2. 비교 결과 확인
cat outputs/dahyeong/my_exp/retrieval_comparison.json

# 예시 출력:
# {
#   "gold_metrics": {"exact_match": 72.08, "f1": 80.23},
#   "retrieval_metrics": {"exact_match": 65.34, "f1": 74.56},
#   "performance_gap": {"em_drop": 6.74, "f1_drop": 5.67},
#   "rates": {
#     "retrieval_success_rate": 90.63,
#     "retrieval_failure_rate": 6.74
#   }
# }

# 3. Retrieval 실패 케이스 상세 분석
cat outputs/dahyeong/my_exp/retrieval_failures.json | head -20
# Gold context로는 맞았지만 Retrieval로 틀린 케이스들
# → Retrieval 개선 필요!
```



**Makefile 사용:**
```bash
# 1. YAML 설정 작성
cp configs/base.yaml configs/bert_lr3e5.yaml
vim configs/bert_lr3e5.yaml  # 설정 수정

# 2. Pipeline 실행 (train + inference)
make pipeline CONFIG=configs/bert_lr3e5.yaml
```

**Python 직접 실행:**
```bash
python run.py --mode pipeline --config configs/bert_lr3e5.yaml
```

**결과물:**
- `./outputs/{username}/{exp_name}/` 폴더에 모델, 체크포인트, 메트릭 저장
- `best_checkpoint_path.txt`: best checkpoint 경로
- `predictions.json`: validation set 예측 결과 (do_eval + do_predict)
- `eval_results.json`: validation 성능 메트릭

### Case 2: 🔥 여러 실험 밤새 돌리기 (GPU 최대 활용)

**Makefile 사용 (가장 간단):**
```bash
# 1. 여러 실험 YAML을 active 폴더에 준비
make activate-config CONFIG=configs/exp1_bert_lr3e5.yaml
make activate-config CONFIG=configs/exp2_bert_lr5e5.yaml
make activate-config CONFIG=configs/exp3_electra_lr3e5.yaml
make activate-config CONFIG=configs/exp4_roberta_lr3e5.yaml

# 2. tmux에서 밤새 돌리기 (SSH 끊겨도 안전)
make tmux-start

# 또는 tmux 없이 바로 실행
make batch
```

**Python 직접 실행:**
```bash
# 1. 여러 실험 YAML 준비
configs/
  ├── exp1_bert_lr3e5.yaml
  ├── exp2_bert_lr5e5.yaml
  ├── exp3_electra_lr3e5.yaml
  └── exp4_roberta_lr3e5.yaml

# 2. Batch 실행
python run.py --mode batch --batch-mode pipeline \
    --configs configs/exp*.yaml

# 또는 tmux와 함께 사용
tmux new -s experiments
python run.py --mode batch --batch-mode pipeline --configs configs/exp*.yaml
# Ctrl+B, D로 detach
# 나중에 tmux attach -t experiments로 재접속
```

**실행 결과:**
```
🚀 BATCH MODE STARTED
================================================================================
📋 Experiments to run: 4
🎯 Mode: pipeline
⚙️  Continue on error: True
🕐 Start time: 2025-12-07 22:00:00

Experiment list:
  1. exp1_bert_lr3e5
  2. exp2_bert_lr5e5
  3. exp3_electra_lr3e5
  4. exp4_roberta_lr3e5
================================================================================

... (각 실험 실행) ...

📈 BATCH RUN SUMMARY
================================================================================

📊 Overall Statistics:
   Total experiments: 4
   ✅ Succeeded: 4
   ❌ Failed: 0
   ⏱️  Total time: 14523.2s (242.1min / 4.0h)
   📊 Avg time per experiment: 3630.8s (60.5min)

📝 Detailed Results:
No.   Status     Config                                             Duration       
--------------------------------------------------------------------------------
1     ✅ success   exp1_bert_lr3e5                                    3450.2s (57.5min)
2     ✅ success   exp2_bert_lr5e5                                    3512.1s (58.5min)
3     ✅ success   exp3_electra_lr3e5                                 3698.5s (61.6min)
4     ✅ success   exp4_roberta_lr3e5                                 3862.4s (64.4min)
================================================================================

🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉
================================================================================
```

### Case 3: 이미 학습된 모델로 다른 split 추론

**Makefile 사용:**
```bash
# validation set으로 추론 (기본)
make inference CONFIG=configs/bert_lr3e5.yaml

# train set으로 추론하려면 YAML 수정 필요
vim configs/bert_lr3e5.yaml
# inference_split: train 으로 변경

make inference CONFIG=configs/bert_lr3e5.yaml
```

**Python 직접 실행:**
```bash
python run.py --mode inference --config configs/bert_lr3e5.yaml
```

### Case 4: Test set 제출용 predictions.json 생성

```yaml
# YAML에서 inference_split 변경
inference_split: test
```

**Makefile 사용:**
```bash
make inference CONFIG=configs/bert_lr3e5.yaml
```

**Python 직접 실행:**
```bash
python run.py --mode inference --config configs/bert_lr3e5.yaml
```

**결과:** `./outputs/{username}/{exp_name}/predictions.json` (제출용)

### Case 5: 팀원의 모델로 inference

```yaml
# 팀원의 output_dir 지정
output_dir: ./outputs/seunghwan/bert_experiment

# use_trained_model이 true면 해당 폴더의 best checkpoint 자동 사용
use_trained_model: true
inference_split: validation
```

**Makefile 사용:**
```bash
make inference CONFIG=configs/use_teammate_model.yaml
```

**Python 직접 실행:**
```bash
python run.py --mode inference --config configs/use_teammate_model.yaml
```

### Case 6: 실패한 실험만 재실행

Batch 실행 후 일부 실험이 실패했다면:

**Makefile 사용:**
```bash
# 실패한 실험만 개별 실행
make pipeline CONFIG=configs/exp2_bert_lr5e5.yaml
make pipeline CONFIG=configs/exp4_roberta_lr3e5.yaml

# 또는 active 폴더 이용
make activate-config CONFIG=configs/exp2_bert_lr5e5.yaml
make activate-config CONFIG=configs/exp4_roberta_lr3e5.yaml
make batch
```

**Python 직접 실행:**
```bash
# 실패한 실험만 골라서 재실행
python run.py --mode pipeline --config configs/exp2_bert_lr5e5.yaml
python run.py --mode pipeline --config configs/exp4_roberta_lr3e5.yaml

# 또는 batch로 실패한 것들만 모아서
python run.py --mode batch --batch-mode pipeline \
    --configs configs/exp2_bert_lr5e5.yaml configs/exp4_roberta_lr3e5.yaml
```

### Case 7: GPU 상태 확인 및 실험 결과 비교

**Makefile 사용:**
```bash
# GPU 상태 확인
make gpu-status

# GPU 상태 실시간 모니터링 (2초마다 갱신)
make watch-gpu

# 모든 실험 결과 비교 (F1, EM 점수)
make compare-results

# 가장 높은 F1 점수를 가진 실험 찾기
make show-best
```

출력 예시:
```
🏆 Best experiment (by F1 score):
  Experiment: exp3_electra_lr3e5
  F1 Score: 71.23
  Path: ./outputs/dahyeong/exp3_electra_lr3e5/
```

## 🛠️ 디버깅 가이드

### 문제 1: "No checkpoint found" 에러

```
FileNotFoundError: No checkpoint found in ./outputs/dahyeong/exp1
```

**원인**: 학습이 완료되지 않았거나 checkpoint 저장 실패

**해결**:
```bash
# 1. 체크포인트 존재 확인
ls outputs/dahyeong/exp1/
# best_checkpoint_path.txt가 있는지 확인

# 2. 로그 확인
tail -n 100 outputs/dahyeong/exp1/*.log

# 3. 없으면 재학습
make train-pipeline CONFIG=configs/exp1.yaml
```

---

### 문제 2: Confidence score가 모두 -1.0

**원인**: `save_logits=False`로 설정되었거나 postprocessing 실패

**해결**:
```bash
# 1. logits_{split}.json 파일 존재 확인
ls outputs/dahyeong/exp1/logits_*.json

# 2. 없으면 inference 재실행 (logits 다시 생성)
make inference CONFIG=configs/exp1.yaml
```

---

### 문제 3: GPU Out of Memory

```
RuntimeError: CUDA out of memory
```

**해결**:
```yaml
# Config YAML 수정
per_device_train_batch_size: 4   # 8 → 4로 감소
gradient_accumulation_steps: 8   # 4 → 8로 증가
fp16: true                        # Mixed precision 활성화
```

또는:
```bash
# GPU 메모리 정리
make gpu-status
# 좀비 프로세스 확인 후 kill
kill -9 <PID>
```

---

### 문제 4: Batch 실행 중 일부만 성공

**상황**: 3개 실험 중 2번째만 실패

**해결**:
```bash
# 1. 실패한 실험만 재실행
make train-pipeline CONFIG=configs/exp2.yaml

# 2. 또는 active에 실패한 것만 추가
cp configs/exp2.yaml configs/active/
make batch
```

---

### 문제 5: YAML 파싱 에러

```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**해결**:
```bash
# YAML 유효성 검증
make check-config CONFIG=configs/my_exp.yaml

# 일반적인 오류:
# ❌ model_name_or_path:klue/bert-base  (콜론 뒤 공백 필수)
# ✅ model_name_or_path: klue/bert-base

# ❌ 들여쓰기 오류
# ✅ 일관된 들여쓰기 (space 2칸 또는 4칸)
```



## 폴더 구조

```
MRC/
├── run.py                          # 통합 실행 스크립트 (NEW)
├── train.py                        # 학습 스크립트
├── inference.py                    # 추론 스크립트 (개선됨)
├── configs/
│   ├── base.yaml                   # 기본 설정 템플릿 (업데이트됨)
│   └── my_experiment.yaml          # 사용자 실험 설정
├── src/
│   ├── arguments.py                # Arguments 정의 (업데이트됨)
│   └── utils/
│       ├── model_loader.py         # 모델 경로 자동 탐색 (NEW)
│       └── ...
└── outputs/
    └── {username}/
        └── {exp_name}/
            ├── checkpoint-123/
            ├── checkpoint-247/
            ├── best_checkpoint_path.txt   # Best checkpoint 경로 (NEW)
            ├── trainer_state.json
            ├── config_used.yaml
            ├── predictions.json
            └── eval_results.json
```

## 마이그레이션 가이드 (기존 코드 → 새 구조)

### 기존 방식
```bash
# Train
python train.py --output_dir ./outputs/dahyeong/exp1 \
                --model_name_or_path klue/bert-base \
                --do_train

# Inference (수동으로 checkpoint 경로 지정)
python inference.py --output_dir ./outputs/dahyeong/exp1_infer \
                    --model_name_or_path ./outputs/dahyeong/exp1/checkpoint-247 \
                    --do_predict
```

### 새 방식
```yaml
# configs/exp1.yaml
model_name_or_path: klue/bert-base
output_dir: ./outputs/dahyeong/exp1
use_trained_model: true
inference_split: validation
```

```bash
# Pipeline (train + inference 자동)
python run.py --mode pipeline --config configs/exp1.yaml
```

## 주요 변경사항 요약

1. ✅ **통합 YAML 설정**: 하나의 YAML로 train과 inference 모두 관리
2. ✅ **자동 checkpoint 탐색**: `use_trained_model=true`로 best model 자동 로드
3. ✅ **유연한 데이터셋 선택**: `inference_split`으로 train/validation/test 간편 전환
4. ✅ **Pipeline 모드**: train → inference 한 번에 실행
5. ✅ **Batch 모드**: 여러 실험 순차 실행으로 GPU 활용 극대화
6. ✅ **do_eval/do_predict 자동 설정**: split에 따라 자동 결정
7. ✅ **기존 방식 호환**: 기존 CLI 방식도 그대로 사용 가능

---

## Makefile 명령어 전체 목록

### 도움말
```bash
make help              # 모든 명령어와 설명 출력
```

### 단일 실험 실행
```bash
make train CONFIG=configs/my_exp.yaml         # Train 모드
make inference CONFIG=configs/my_exp.yaml     # Inference 모드
make pipeline CONFIG=configs/my_exp.yaml      # Pipeline 모드 (train → inference)
```

### 배치 실험 실행
```bash
make batch                                    # configs/active/*.yaml 모두 실행
make batch-custom CONFIGS='file1 file2'       # 지정한 파일들만 실행
make batch-train                              # Train만 batch 실행
make batch-infer                              # Inference만 batch 실행
make batch-stop-on-error                      # 실패 시 중단
```

### tmux 세션 관리
```bash
make tmux-start        # tmux 세션 시작 + batch 실행
make tmux-attach       # 실행 중인 tmux 세션 재접속
make tmux-kill         # tmux 세션 종료
```

### Config 관리
```bash
make check-config CONFIG=configs/my_exp.yaml  # YAML 유효성 검증
make list-configs                             # 사용 가능한 config 목록
make activate-config CONFIG=configs/my.yaml   # Active 폴더로 복사
make deactivate-config NAME=my.yaml           # Active에서 제거
make clear-active                             # Active 폴더 비우기
```

### 유틸리티
```bash
make gpu-status        # GPU 사용 현황 확인
make watch-gpu         # GPU 상태 2초마다 자동 갱신
make compare-results   # 모든 실험 결과 비교 (F1, EM)
make show-best         # 가장 높은 F1 점수 실험 출력
make clean-outputs     # outputs 폴더 전체 삭제
make clean-checkpoints # checkpoint 폴더만 삭제
```

### 개발 도구
```bash
make install           # 패키지 설치 (requirements.txt)
make format            # 코드 자동 포맷팅 (black, isort)
make lint              # 코드 포맷 검사
make test-config       # 테스트용 config 빠른 실행
```

### 디버깅
```bash
make debug-train CONFIG=configs/my.yaml       # Train 디버그 모드
make debug-inference CONFIG=configs/my.yaml   # Inference 디버그 모드
make tail-log OUTPUT_DIR_PATH=outputs/user/exp1  # 최근 로그 확인
```

## 디버깅 & 문제 해결

### 모델 로딩 문제

**증상**: `Model not found` 또는 `best_checkpoint_path.txt not found` 에러
```
FileNotFoundError: [Errno 2] No such file or directory: './outputs/username/exp1/best_checkpoint_path.txt'
```

**원인 & 해결**:
1. **학습이 완료되지 않음**: `train.py`가 끝까지 실행됐는지 확인
   ```bash
   ls -la ./outputs/username/exp1/
   # best_checkpoint_path.txt가 있는지 확인
   ```

2. **학습 중 에러 발생**: 로그 확인
   ```bash
   make tail-log OUTPUT_DIR_PATH=./outputs/username/exp1
   # 또는
   tail -n 50 ./outputs/username/exp1/log.txt
   ```

3. **수동으로 checkpoint 지정**: YAML에서 직접 경로 명시
   ```yaml
   use_trained_model: true
   model_name_or_path: ./outputs/username/exp1/checkpoint-247
   ```

### Batch 모드 문제

**증상**: 일부 실험만 실행되고 중단됨
```
Experiment 3/5: configs/exp3.yaml
Error: CUDA out of memory
```

**해결**:
1. **실패 시점부터 재실행**:
   ```bash
   # Makefile 사용
   make activate-config CONFIG=configs/exp3.yaml
   make activate-config CONFIG=configs/exp4.yaml
   make batch
   
   # 또는 직접 실행
   python run.py --mode batch --batch-mode pipeline \
       --configs configs/exp3.yaml configs/exp4.yaml
   ```

2. **메모리 설정 조정**:
   ```yaml
   # 실패한 실험의 YAML에서
   per_device_train_batch_size: 8  # 16 → 8로 감소
   gradient_accumulation_steps: 4   # 2 → 4로 증가 (동일한 effective batch size 유지)
   ```

3. **실험 사이 GPU 메모리 정리**:
   ```bash
   make gpu-status
   # 좀비 프로세스 확인 후
   kill -9 <PID>
   ```

**증상**: Batch 실행 중 SSH 연결 끊김으로 중단
```
client_loop: send disconnect: Broken pipe
```

**해결**: tmux 사용
```bash
# Makefile 사용 (권장)
make tmux-start      # 자동으로 batch 시작 + tmux 세션 생성

# 다시 접속 후
make tmux-attach

# 또는 직접 tmux 사용
tmux new -s mrc_batch
make batch
# Ctrl+B, D로 detach

# 재접속
tmux attach -t mrc_batch
```

### 데이터셋 관련 문제

**증상**: `inference_split` 설정했는데 엉뚱한 데이터셋으로 추론됨

**확인사항**:
```yaml
# YAML에서 확인
inference_split: test  # "test" 맞는지 확인 (오타 주의: "tset", "Test" 등)
use_trained_model: true  # true로 설정됐는지

# 로그에서 어떤 데이터셋을 로드했는지 확인
tail -n 100 ./outputs/username/exp1/log.txt | grep "Loading dataset"
```

### YAML 설정 문제

**증상**: YAML 파싱 에러
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**해결**: YAML 문법 확인
```bash
# Makefile로 유효성 검증
make check-config CONFIG=configs/my_experiment.yaml
```

```yaml
# ❌ 잘못된 예
model_name_or_path:klue/bert-base  # 콜론 뒤 공백 필수

# ✅ 올바른 예
model_name_or_path: klue/bert-base

# ❌ 들여쓰기 오류
model_name_or_path: klue/bert-base
 output_dir: ./outputs/exp1  # 앞에 불필요한 공백

# ✅ 일관된 들여쓰기
model_name_or_path: klue/bert-base
output_dir: ./outputs/exp1
```

---

## 부록: 추가 기능

### Config 유효성 검증

실험 실행 전 YAML 설정 검증:
```bash
# Makefile 사용 (권장)
make check-config CONFIG=configs/my_exp.yaml

# 또는 직접 Python 실행
python -c "
from transformers import HfArgumentParser
from src.arguments import ModelArguments, DataTrainingArguments, TrainingArguments

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_yaml_file('configs/my_exp.yaml')
print('✅ YAML 설정 유효')
"
```

### Best Checkpoint 경로 확인

```bash
# 학습 완료 후 어떤 checkpoint가 best인지 확인
cat ./outputs/username/exp1/best_checkpoint_path.txt
# 출력: ./outputs/username/exp1/checkpoint-247

# 또는 trainer_state.json에서 직접 확인
python -c "
import json
with open('./outputs/username/exp1/trainer_state.json') as f:
    state = json.load(f)
    print(state['best_model_checkpoint'])
"
```

### 실험 결과 비교

여러 실험의 결과 한눈에 비교:
```bash
# Makefile 사용 (권장)
make compare-results

# 가장 높은 F1 점수 실험 찾기
make show-best
```

출력 예시:
```
📊 Comparing experiment results:

  exp1_bert                                F1: 68.45    EM: 55.32   
  exp2_roberta                             F1: 71.23    EM: 59.87   
  exp3_electra                             F1: 69.87    EM: 57.45   

🏆 Best experiment (by F1 score):
  Experiment: exp2_roberta
  F1 Score: 71.23
  Path: ./outputs/dahyeong/exp2_roberta/
```

---

## ❓ FAQ

**Q1: 가장 먼저 실행해야 할 명령어는?**  
A: `make train-pipeline CONFIG=configs/base.yaml`로 기본 실험부터 시작하세요.

**Q2: Confidence score가 낮은 예측은 신뢰할 수 없나요?**  
A: 
- **Low confidence + 정답**: 운으로 맞춘 케이스 (불안정)
- **Low confidence + 오답**: 모델도 헷갈리는 어려운 문제
- **High confidence + 오답**: **위험!** 체계적 오류 (재학습 필요)

**Q3: Batch 실행 중 SSH 연결이 끊기면?**  
A: 이미 실행 중인 실험은 계속 진행되지만, 다음 실험은 실행되지 않습니다. 장시간 실험은 tmux/screen 사용을 권장합니다.

**Q4: GPU 메모리 부족 에러가 발생하면?**  
```yaml
# Config YAML에서 batch size 조정
per_device_train_batch_size: 8  # 16 → 8로 감소
gradient_accumulation_steps: 4   # 2 → 4로 증가 (effective batch size 유지)
```

**Q5: configs/active/ 폴더는 왜 필요한가요?**  
A: "지금 실행하고 싶은 실험들"을 모아두는 곳입니다. `make batch`는 이 폴더의 YAML만 실행합니다.

**Q6: Best checkpoint는 어떻게 자동 탐색되나요?**  
A: 
1. `best_checkpoint_path.txt` (train.py가 생성)
2. `trainer_state.json`의 `best_model_checkpoint`
3. `checkpoint-*` 폴더 중 가장 큰 숫자

**Q7: 여러 실험 중 하나만 실패하면 전체가 중단되나요?**  
A: 아닙니다. 기본값은 `continue_on_error=True`로 실패해도 다음 실험 계속 진행합니다.

**Q8: Confidence score는 어떻게 계산되나요?**  
A: 
```python
# Start/End logit의 softmax 확률을 평균
start_prob = softmax(start_logits)
end_prob = softmax(end_logits)
avg_prob = (max(start_prob) + max(end_prob)) / 2
```

**Q9: 팀원의 실험 결과를 내 설정으로 재현하려면?**  
```bash
# 팀원의 config_used.yaml 복사
cp outputs/teammate/exp5/config_used.yaml configs/reproduce_exp5.yaml

# output_dir만 변경 후 실행
vim configs/reproduce_exp5.yaml  # output_dir: ./outputs/myname/reproduce_exp5
make train-pipeline CONFIG=configs/reproduce_exp5.yaml
```

**Q10: YAML 설정이 올바른지 확인하려면?**  
```bash
make check-config CONFIG=configs/my_exp.yaml
```



---

## 마이그레이션 가이드 (기존 코드 → 새 구조)

### 기존 방식
```bash
# Train
python train.py --output_dir ./outputs/dahyeong/exp1 \
                --model_name_or_path klue/bert-base \
                --do_train

# Inference (수동으로 checkpoint 경로 지정)
python inference.py --output_dir ./outputs/dahyeong/exp1_infer \
                    --model_name_or_path ./outputs/dahyeong/exp1/checkpoint-247 \
                    --do_predict
```

### 새 방식 (Makefile)
```yaml
# configs/exp1.yaml
model_name_or_path: klue/bert-base
output_dir: ./outputs/dahyeong/exp1
use_trained_model: true
inference_split: validation
```

```bash
# Pipeline (train + inference 자동)
make pipeline CONFIG=configs/exp1.yaml
```

## 주요 변경사항 요약

1. ✅ **Makefile 추가**: 명령어 간편화 및 유틸리티 제공
2. ✅ **YAML 기반 설정**: 하나의 YAML로 train과 inference 모두 관리
3. ✅ **자동 checkpoint 탐색**: `use_trained_model=true`로 best model 자동 로드
4. ✅ **유연한 데이터셋 선택**: `inference_split`으로 train/validation/test 간편 전환
5. ✅ **Pipeline 모드**: train → inference 한 번에 실행
6. ✅ **Batch 모드**: 여러 실험 순차 실행으로 GPU 활용 극대화
7. ✅ **tmux 통합**: 장시간 실험을 안전하게 실행
8. ✅ **결과 비교 도구**: F1, EM 점수 자동 비교
9. ✅ **do_eval/do_predict 자동 설정**: split에 따라 자동 결정
10. ✅ **기존 방식 호환**: 기존 CLI 방식도 그대로 사용 가능

---

## 📚 추가 자료

### Confidence Score 활용 전략

#### 1. 예측 필터링 전략
```python
import pandas as pd

df = pd.read_csv('outputs/dahyeong/my_exp/val_confidence.csv')

# 전략 1: High confidence만 제출 (정밀도 우선)
high_conf = df[df['avg_prob'] > 0.9]
print(f"High confidence: {len(high_conf)} examples ({len(high_conf)/len(df)*100:.1f}%)")

# 전략 2: Low confidence 재검토
low_conf = df[df['avg_prob'] < 0.3]
print(f"Need review: {len(low_conf)} examples")
```

#### 2. 모델 앙상블 시 가중치
```python
# Confidence를 가중치로 사용
ensemble_pred = (
    model1_pred * model1_confidence +
    model2_pred * model2_confidence
) / (model1_confidence + model2_confidence)
```

#### 3. 능동 학습 (Active Learning)
```python
# Low confidence 케이스만 추가 라벨링
unlabeled_df = df[df['avg_prob'] < 0.5]
print(f"Need labeling: {len(unlabeled_df)} examples")
# → Human annotation → Retrain
```

---

### 성능 개선 체크리스트

- [ ] **Hyperparameter Tuning**
  - Learning rate: [2e-5, 3e-5, 5e-5]
  - Batch size 조정
  - Epochs: 2-5 범위

- [ ] **Data Augmentation**
  - Back-translation
  - Synonym replacement
  - Context paraphrasing

- [ ] **Model Selection**
  - KoELECTRA vs RoBERTa vs BERT
  - Large vs Base 비교

- [ ] **Retrieval 개선**
  - BM25 → Dense retrieval (DPR, ColBERT)
  - Top-K 조정 (30 vs 50 vs 100)
  - Reranking 추가

- [ ] **Post-processing**
  - 불필요한 공백 제거
  - 특수문자 정규화
  - Entity 보정

- [ ] **Confidence 기반 필터링**
  - High confidence 오답 재학습
  - Low confidence 데이터 추가 라벨링

---

### 유용한 스크립트 모음

#### 오답 분석 스크립트
```bash
# scripts/analyze_errors.py 생성
cat > scripts/analyze_errors.py << 'EOF'
import pandas as pd
import json

def analyze_errors(output_dir):
    # Load confidence
    conf_df = pd.read_csv(f'{output_dir}/val_confidence.csv')
    
    # Load detailed results
    with open(f'{output_dir}/val_detailed_results.json') as f:
        details = {item['id']: item for item in json.load(f)}
    
    errors = conf_df[conf_df['is_correct'] == 0]
    
    print(f"Total errors: {len(errors)}")
    print(f"\nError breakdown:")
    print(f"  Low confidence (<0.5): {len(errors[errors['avg_prob'] < 0.5])}")
    print(f"  Medium confidence (0.5-0.8): {len(errors[(errors['avg_prob'] >= 0.5) & (errors['avg_prob'] < 0.8)])}")
    print(f"  High confidence (>0.8): {len(errors[errors['avg_prob'] >= 0.8])}")
    
    # High confidence errors (systematic errors)
    high_conf_errors = errors[errors['avg_prob'] > 0.8]
    print(f"\n⚠️  {len(high_conf_errors)} high confidence errors (need investigation):")
    
    for idx, row in high_conf_errors.head(5).iterrows():
        detail = details[row['id']]
        print(f"\nID: {row['id']}")
        print(f"Question: {detail['question']}")
        print(f"Prediction: {detail['prediction']} (conf: {row['avg_prob']:.3f})")
        print(f"Ground Truth: {detail['ground_truth']}")

if __name__ == '__main__':
    import sys
    analyze_errors(sys.argv[1])
EOF

# 실행
python scripts/analyze_errors.py outputs/dahyeong/my_exp
```

#### 실험 비교 스크립트
```bash
# scripts/compare_experiments.py
cat > scripts/compare_experiments.py << 'EOF'
import json
import os
from pathlib import Path

def compare_experiments(base_dir='outputs/dahyeong'):
    results = []
    
    for exp_dir in Path(base_dir).iterdir():
        if not exp_dir.is_dir():
            continue
        
        eval_file = exp_dir / 'eval_results.json'
        if not eval_file.exists():
            continue
        
        with open(eval_file) as f:
            metrics = json.load(f)
        
        results.append({
            'name': exp_dir.name,
            'em': metrics.get('eval_exact_match', 0),
            'f1': metrics.get('eval_f1', 0)
        })
    
    # Sort by F1
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    print("Experiment Comparison (sorted by F1):")
    print(f"{'Rank':<5} {'Experiment':<50} {'EM':<8} {'F1':<8}")
    print("-" * 75)
    
    for rank, res in enumerate(results, 1):
        print(f"{rank:<5} {res['name']:<50} {res['em']:<8.2f} {res['f1']:<8.2f}")

if __name__ == '__main__':
    compare_experiments()
EOF

# 실행
python scripts/compare_experiments.py
```

---

### 체크포인트 관리

```bash
# 오래된 체크포인트 삭제 (best만 유지)
make clean-checkpoints

# 디스크 용량 확인
du -sh outputs/dahyeong/*/

# 특정 실험만 삭제
rm -rf outputs/dahyeong/failed_exp/
```

---

## 🎓 Best Practices

1. **실험 명명 규칙**
   ```
   {model}_{hyper}_{date}
   예: koelectra_lr3e5_1208
       roberta_bs16_1208
   ```

2. **Config 버전 관리**
   - `configs/` 폴더는 Git 커밋
   - `configs/active/` 폴더는 .gitignore (임시 작업 공간)

3. **결과 백업**
   ```bash
   # 중요한 실험 결과 백업
   tar -czf experiments_backup_1208.tar.gz outputs/dahyeong/
   ```

4. **GPU 에티켓**
   - 학습 전 `make gpu-status` 확인
   - 장시간 실험은 밤에 실행
   - 완료 후 프로세스 종료 확인

5. **문서화**
   - 각 실험 후 `epoch_summary.md` 확인
   - 중요한 발견사항은 노트 기록
   - Best model의 `config_used.yaml` 백업

---

## 📞 문제 해결

### 도움이 필요할 때

1. **로그 확인**
   ```bash
   tail -n 100 outputs/dahyeong/my_exp/*.log
   ```

2. **GPU 상태 확인**
   ```bash
   make gpu-status
   nvidia-smi
   ```

3. **Config 검증**
   ```bash
   make check-config CONFIG=configs/my_exp.yaml
   ```

4. **Issue 생성**
   - 에러 메시지 전체 복사
   - 사용한 명령어 기록
   - 환경 정보 (GPU, Python 버전)

---

**이 가이드가 도움이 되셨나요? 추가 질문은 언제든 환영합니다! 🎉**
