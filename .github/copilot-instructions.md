# GitHub Copilot Instructions for MRC (ODQA) Project — Korean-Centric

## 프로젝트 개요 (Project Overview)

이 레포지토리는 Open-Domain Question Answering (ODQA) 대회용 프로젝트입니다.

* **목표**: Query(질문)를 입력받아, 사전에 구축된 Knowledge base에서 관련 문서를 검색하고, 최종적으로 정답 문자열을 반환하는 시스템을 만드는 것.
* **아키텍처**: 2-stage 시스템

  1. **Retriever**: `wikipedia_documents.json` (약 5.7만 개 문서)에서 관련 Passage를 검색.
  2. **Reader (MRC 모델)**: Retriever가 가져온 문서를 읽고 정답을 추출.

사용자는 한국어가 모국어이며, 영어 코드/에러/논문을 읽고 이해할 수 있지만, **설명과 커뮤니케이션은 한국어를 기본적으로 선호**합니다.

---

## 대회 규칙 & 제약 (CRITICAL)

아래 규칙은 절대 위반하면 안 됩니다. 코드를 제안하거나 리팩터링할 때 항상 이 규칙을 만족해야 합니다.

* **Test Set 분석 금지**:

  * Test set을 분석, 시각화, 라벨링하여 학습에 사용하는 행위는 절대 금지입니다.
  * “테스트셋 일부를 눈으로 보고 라벨링해서 학습에 쓰자” 같은 아이디어를 제안하지 마십시오.
  * Test set은 오직 **최종 평가 / 제출 파일 생성**에만 사용해야 합니다.

* **Pretrained Weights 제약**:

  * **KLUE-MRC 데이터셋으로 학습된 기학습 가중치(pretrained weight)를 사용하는 것은 금지**입니다.
  * 그 외 공개된 pretrained weight는 사용 가능하며, 반드시 공개적으로 접근 가능하고 저작권 문제가 없어야 합니다.
  * 새로운 가중치를 제안할 때는:

    * 모델 이름
    * HuggingFace Hub 혹은 GitHub 등 접근 가능한 링크(설명 수준)
      를 명시적으로 언급해 주세요.

* **External Data 사용**:

  * KLUE-MRC를 제외한 외부 데이터셋은 사용 가능합니다.
  * 외부 데이터를 제안할 경우:

    * 데이터 출처
    * 사용 목적 (pretraining, data augmentation 등)
      을 간략히 설명해 주세요.

---

## 평가 지표 (Evaluation Metrics)

* **Exact Match (EM)** — 리더보드 기준 주요 지표:

  * 공백/문장부호를 정규화한 후, 예측 답이 정답 문자열과 **완전히 일치**하면 1점, 그렇지 않으면 0점입니다.
  * 정답이 여러 개일 수 있으며, **하나라도 일치하면 정답**으로 처리합니다.

* **F1 Score** — 참고용 보조 지표:

  * 예측/정답 간의 토큰 단위 겹침 정도를 기반으로 부분 점수를 부여합니다.
  * 리더보드 순위에는 직접 사용되지 않지만, 모델 특성 분석, 오류 분석에 유용합니다.

실험 설계와 결과 분석 시에는 **EM을 우선**으로 고려하되, **F1을 함께 보고** 모델 특성을 파악해야 합니다.

---

## 아키텍처 & 주요 컴포넌트

### 1. 핵심 스크립트 (Core Scripts)

* `train.py`

  * Reader(MRC) 모델 학습을 담당합니다.
  * HuggingFace `Trainer`를 상속한 커스텀 `QuestionAnsweringTrainer`를 사용합니다. (자세한 구현은 `src/trainer_qa.py` 참조)
  * YAML 설정 파일을 `HfArgumentParser`로 파싱해 `ModelArguments`, `DataTrainingArguments`, `TrainingArguments`를 구성합니다.

* `inference.py`

  * 평가 및 제출 파일 생성을 담당합니다.
  * **Retrieval + Reader**를 통합하여 end-to-end ODQA 파이프라인을 실행합니다.
  * YAML의 `inference_split` 값에 따라:

    * `validation`: 검증 데이터 기준으로 EM/F1 계산
    * `test`: 레이블 없이 제출용 prediction 파일 생성
      를 수행합니다.

* `src/trainer_qa.py`

  * HF `Trainer`를 상속한 `QuestionAnsweringTrainer` 구현체입니다.
  * `evaluate`, `predict` 메서드를 오버라이드하여:

    * 모델의 raw logits를 text 형태의 정답으로 후처리
    * EM/F1 계산 및 로깅, 결과 저장
      를 수행합니다.

### 2. Retrieval 시스템 (`src/retrieval/`)

* **패턴**: `src/retrieval/__init__.py`의 `get_retriever`를 사용하는 factory 패턴.
* **베이스 클래스**: `src/retrieval/base.py`의 `BaseRetrieval`.
* **주요 구현체**:

  * `SparseRetrieval` (TF-IDF / BM25)
  * `KoE5Retrieval` (Dense Retrieval)
  * `HybridRetrieval` (Sparse + Dense 결합)
* **사용 방법**:

  * 항상 `get_retriever(retrieval_type=..., ...)`로 인스턴스를 생성합니다.
  * 사용 전 반드시 `.build()`를 호출하여 index를 구성합니다.
  * 성능 평가는 주로 **Recall@k (k = 1, 5, 10, 20, 50, 100)** 기준으로 봅니다.

### 3. 설정 관리 (Configuration Management)

* **단일 진실의 원천(Source of truth)**: `configs/` 디렉토리 내 YAML 파일들
  예: `configs/base.yaml`, `configs/exp/*.yaml`, `configs/active/*.yaml`
* **스키마(Schema)**: `src/arguments.py`에 정의

  * `ModelArguments`
  * `DataTrainingArguments`
  * `TrainingArguments`
* **파싱(Parsing)**:

  * `HfArgumentParser`를 사용하여 CLI 인자 + YAML 내용을 dataclass로 변환합니다.
  * 일반적인 실행 예:

    * `python train.py configs/your_experiment.yaml`
    * `python inference.py configs/your_experiment.yaml`

---

## 데이터 흐름 (Data Flow)

1. **로딩 (Loading)**:

   * HuggingFace `datasets`를 사용하여 `./data` 경로에서 Arrow 포맷 데이터를 로딩합니다.
   * Train: 3,952개 예시
   * Validation: 240개 예시
   * Test: 600개 예시 (Public 240, Private 360)
   * Test set에는 context/answers가 없으므로, **Retrieval이 필수**입니다.

2. **전처리 (Preprocessing)**:

   * `prepare_train_features`:

     * 토크나이징
     * `doc_stride`를 사용하는 sliding window 처리
     * 정답 start/end 위치(label) 정렬
   * `prepare_validation_features`:

     * 평가/추론용 토크나이징
     * 레이블 없이, 예시 ID와 tokenized context 매핑만 수행

3. **Retrieval 통합 (in `inference.py`)**:

   * `retrieve_and_build_dataset`:

     * 각 질문에 대해 top-k 문맥(context)를 Retriever로부터 받아옵니다.
     * `(question, retrieved_context)` 쌍으로 구성된 새 Dataset을 만듭니다.
   * Reader 모델은 이 Dataset을 기반으로 정답을 추론합니다.

---

## 핵심 워크플로 (Critical Workflows)

### 1. 학습 (Training)

```bash
python train.py configs/your_experiment.yaml
```

* YAML의 `output_dir`는 **실험마다 유일한 경로**로 설정하여 결과가 덮어쓰이지 않도록 해야 합니다.
* `TrainingArguments`를 통해 다음을 제어합니다:

  * `learning_rate`
  * `per_device_train_batch_size`
  * `per_device_eval_batch_size`
  * `gradient_accumulation_steps`
  * `num_train_epochs`
  * `save_strategy`, `evaluation_strategy`, `logging_steps` 등

### 2. 추론 / 제출 (Inference / Submission)

```bash
python inference.py configs/your_experiment.yaml
```

* YAML의 `inference_split`에 따라:

  * `validation`: 검증 데이터에 대해 EM/F1 계산 및 분석
  * `test`: 정답 없이 제출용 prediction 파일 생성
* 코드를 수정하거나 제안할 때는 다음 필드가 올바르게 유지되도록 해야 합니다:

  * `id`
  * `question`
  * `context` (retrieved context)
  * `prediction_text`

---

## 코딩 컨벤션 (Coding Conventions)

* **Type Hint 적극 사용**:

  * `List`, `Dict`, `Optional`, `Tuple`, `NoReturn` 등 `typing` 모듈을 적극적으로 사용합니다.
  * 함수의 반환 타입을 명시하는 것을 기본으로 합니다.

* **HuggingFace 중심 사용**:

  * 가능하면 raw PyTorch보다 `transformers`, `datasets` API를 우선 사용합니다.
  * `AutoModelForQuestionAnswering`, `AutoTokenizer`, `Trainer` 등을 활용합니다.

* **경로(Path) 처리**:

  * 프로젝트 루트를 기준으로 한 상대 경로나, 명확한 절대 경로를 사용합니다.
  * 특정 서버/사용자에 종속적인 하드코딩 경로는 피합니다.

* **로깅(Logging)**:

  * `print` 대신 `src.utils.get_logger`를 사용합니다.
  * “현재 단계, 데이터셋 크기, 주요 하이퍼파라미터, 평가 지표” 등을 충분히 로깅합니다.

* **코드 수정 방식**:

  * 사용자가 특별히 요구하지 않는 한, **큰 리팩터보다는 작은 단위의 수정/추가**를 선호합니다.
  * 대규모 리팩터가 필요하다면, 먼저 간단한 계획을 설명하고 난 뒤 코드를 제안합니다.

---

## 자주 발생하는 문제 (Common Pitfalls)

* **Token Type IDs**:

  * RoBERTa 계열 모델은 일반적으로 `token_type_ids`를 사용하지 않습니다.
  * `model.config.type_vocab_size` 등을 확인한 뒤, 필요 없으면 `token_type_ids`를 전달하지 않습니다.
  * 확실하지 않을 경우, RoBERTa 계열에는 `token_type_ids`를 제거하는 쪽을 우선 고려합니다.

* **정답 위치(Answer Alignment)**:

  * Retrieval로 context를 교체하거나, 전처리 방식을 바꿀 때는 `answer_start` 인덱스를 반드시 재계산해야 합니다.
  * 원래 정답 인덱스는 **원본 context** 기준이므로, 새로운 문맥에 그대로 사용할 수 없습니다.
  * 정답 오프셋이 잘못되면 EM/F1이 크게 떨어질 수 있습니다.

* **CUDA OOM (Out-Of-Memory)**:

  * GPU 메모리 부족이 발생하면:

    * `per_device_train_batch_size`를 줄입니다.
    * 동일한 effective batch size를 유지하고 싶다면 `gradient_accumulation_steps`를 늘립니다.
    * 필요하다면 `max_seq_length`를 줄이는 것도 고려합니다.

---

## Makefile 사용법 (Makefile Usage)

이 프로젝트는 반복적인 작업을 단순화하기 위해 `Makefile`을 제공합니다.

### 주요 커맨드

* `make train CONFIG=configs/exp.yaml`
  → 주어진 CONFIG로 학습만 실행.

* `make inference CONFIG=configs/exp.yaml`
  → 주어진 CONFIG로 추론만 실행.

* `make train-pipeline CONFIG=configs/exp.yaml`
  → 학습 후 바로 같은 설정으로 inference까지 실행 (전체 실험 파이프라인).

* `make eval-val CONFIG=configs/exp.yaml`
  → validation 기준으로 Retrieval + Reader 성능을 분석 (gold context vs retrieval 비교 등).

* `make batch`
  → `configs/active/`에 있는 모든 설정 파일을 순차적으로 실행.

* `make check-config CONFIG=configs/exp.yaml`
  → 해당 CONFIG가 스키마에 맞는지, 필수 필드가 있는지 등을 검증.

* `make list-active`
  → `configs/active/`에 등록된 활성 실험 설정 목록 출력.

* `make gpu-status`
  → 현재 GPU 사용량 확인.

* `make clean-checkpoints`
  → 중간 checkpoint를 삭제하여 디스크 공간을 확보.

* `make compare-results`
  → 여러 실험 결과의 F1/EM을 비교하여 출력.

### 예시

```bash
make train-pipeline CONFIG=configs/active/my_experiment.yaml
```

---

## 언어 & 커뮤니케이션 가이드라인 (매우 중요)

이 설정은 **한국어 중심 편의성**을 목표로 합니다.

### 1. 입력 언어 (User Input Language)

* 사용자는 **주로 한국어로 질문**합니다.
* 영어로 질문하는 경우도 있지만, 기본 가정은 **한국어**입니다.

### 2. 출력 언어 (Assistant Output Language)

* **기본 규칙**:

  * 사용자가 메시지에 한국어를 한 글자라도 사용했다면, **반드시 한국어로 답변**해야 합니다.
  * 코드, 함수명, 클래스명, 에러 메시지, CLI 명령어, 파일 경로 등은 **영어 그대로** 사용해도 됩니다.

* **영어로만 답변해야 하는 예외 상황**:

  * 사용자가 명시적으로 아래와 같이 요청한 경우에만 영어 전체 답변을 허용합니다:

    * “영어로만 답변해줘”
    * “Answer in English only”
    * “Please write everything in English”
  * 혹은:

    * 사용자가 메시지 전체를 영어로 작성했고,
    * “영어 설명을 선호한다”는 취지로 말한 경우
      에 한해 영어-only 답변을 허용할 수 있습니다.

* **혼합 모드 (한국어 설명 + 영어 코드)**:

  * 설명, 해설, 요약: **한국어**
  * 코드, 타입 힌트, 라이브러리 이름, 에러 로그: **영어**
  * 기술 용어(예: “Retriever”, “Reader”, “logits”, “Dataset”)는 필요 시 영어 그대로 사용합니다.

### 3. 설명 스타일

* 복잡한 내용일수록:

  * 먼저 **핵심 요약을 한국어로** 짧게 제공하고,
  * 이어서 상세한 단계별 설명을 한국어로 작성합니다.
* 필요 시, 난이도 높은 개념에 대해서는:

  * 간단한 영어 표현을 괄호 안에 병기할 수 있습니다.
    예: “정확도(accuracy)”, “학습률(learning rate)”

### 4. 명확한 형식 지정

* 사용자가 “코드 먼저 보여줘, 그다음 설명해줘”처럼 형식을 요구하면, **반드시 그 순서를 지킵니다**.
* 별도의 요구가 없더라도, 비-trivial한 작업(새 기능 추가, 리팩터링, 디버깅)에는 다음 구조를 권장합니다:

  1. **요약 / 계획 (한국어)**: 무엇을 할지 bullet point로 간단히.
  2. **코드 스니펫**: 전체 함수를 포함한 형태로.
  3. **세부 설명 (한국어)**: 왜 이렇게 구현했는지, ODQA 파이프라인에서 어떤 역할을 하는지.

### 5. 추가 질문 / Clarification

* 큰 리팩터나 파괴적인 변경(파일 구조 변경, 주요 API 변경 등)을 제안하기 전에:

  * 한국어로 간단한 확인 질문을 먼저 던질 수 있습니다.
  * 예: “이 모듈 전체를 리팩터링해도 될까요?”, “현재 사용 중인 모델 목록을 알려주실 수 있나요?”

---

## Reasoning & Answer Structure (추론 및 답변 구조)

* 단순한 문법 수정, 작은 버그 수정 → 바로 코드와 짧은 설명만 제공해도 됩니다.
* 복잡한 작업(새 실험 설계, 모델 선택, Retrieval 전략 논의 등)에서는 다음 구조를 따르는 것이 좋습니다:

1. **상황 정리 (한국어)**

   * 현재 상황/문제 요약
   * 주어진 로그, config, 코드 조각에서 읽어낸 핵심 정보 정리

2. **해결 전략 / 실험 플랜 (한국어)**

   * bullet point로 2~5개 정도의 구체적 액션 아이템
   * 각 항목마다 기대 효과, 리스크를 간단히 언급

3. **구체적인 코드 또는 설정 제안**

   * 해당 전략을 구현하는 코드 스니펫, YAML 예시, Makefile 명령어 등 제시
   * 코드와 설정은 영어를 사용해도 괜찮습니다.

4. **마무리 요약 (한국어)**

   * “정리하면, 지금 제안한 변경을 적용하면 ○○이 향상될 가능성이 있습니다.” 처럼 2~3문장으로 정리

이 instruction은 **“한국어로 자연스럽게 소통하면서도, 코드/설계 측면에서 최대한의 생산성을 내는 것”**을 목표로 합니다.
