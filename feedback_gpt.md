<!-- 모든 피드백은 시간순으로 작성돼있습니다. 즉 아래쪽으로 갈수록 더 최근입니다.-->
좋아요, 방향은 잘 잡으셨어요.

* “train도 retrieval을 쓰되”
* “지금 DHN 스타일은 킵하지 말고 다른 식으로 train/test를 맞추자”
* “gold를 항상 맨 앞에 박는 것도 재검토”

이 세 가지를 동시에 만족하는 설계를 같이 짜보면 됩니다.

---

## 1. DHN 대신 쓸 수 있는 “retrieval-aware” 학습 아이디어들

핵심은:

* **test 시나리오**:

  * retrieval (KURE+BM25 weighted_hybrid, top_k=10)
  * → reader가 **retrieved context**에서 span 뽑음
* **train 시나리오**도 최대한 이 구조와 비슷하게 만들기

  * 다만, **chunk 오프셋 계산 지옥**은 피하거나 최소화

를 만족하는 쪽으로 가야 합니다.

아래 3개는 “난이도/효과/현재 코드와의 적합성”을 같이 고려한 제안입니다.

---

### 아이디어 A. “Retrieval-aware Gold-only” (DHN 없이, 그러나 train에 retrieval 포함)

**컨셉**

* DHN은 버리고,
* 대신 **train에서도 retrieval을 한번 거친 뒤, 그 중 “정답을 포함하는 context 한 개만”을 골라서 gold로 학습**하는 방식.
* 즉, **“retrieval을 통과한 gold context”**로만 학습하고, negative는 아예 안 씀.

**왜 이게 괜찮냐?**

* 지금 가장 큰 distribution gap은

  > train: “깨끗한 gold context (원본 KorQuAD 스타일)”
  > test: “retrieval을 통해 나온 passage/chunk (prefix 붙고, truncation/절단, title 포함 등)”

  이 차이입니다.

* A는 이걸 상당 부분 줄입니다.

  * train에서도 실제 inference 시 사용하는 retrieval 포맷(예: `passage: {title} {text}`) 그대로 노출
  * 단지 그 중 answer가 포함된 context 하나만 쓰는 것.

**구체 구조**

1. `train_cache` + `kure_passages_meta`를 이용해서,

   * 각 train example에 대해 retrieval top-50 안에서
     `answer_text`를 포함하는 passage를 찾음.
   * 찾으면 그 passage를 **train 시 context로 사용**.
   * 못 찾으면 fallback:

     * (A1) 그냥 기존 gold context 사용
     * (A2) 아예 그 sample은 학습에서 제외 (7~8% 수준이라면 감당 가능)

2. YAML 변경 (예시)

```yaml
dynamic_hard_negative:
  enabled: false           # DHN 끄기

train_retrieval: true      # train도 retrieval 파이프라인 사용
retrieval_type: weighted_hybrid
top_k_retrieval: 10
```

3. Dataset 로직은:

```python
# 의사코드 느낌
for example in train:
    qid = example["id"]
    answers = example["answers"]["text"]
    cands = cache[qid]["retrieved"]

    pos_passage = None
    for cand in cands:
        p = passages[cand["passage_id"]]
        if any(ans in p["text"] for ans in answers):
            pos_passage = p["text"]
            break

    if pos_passage is None:
        # fallback: 기존 gold context or skip
        context = example["context"]
    else:
        context = pos_passage

    # 이후는 기존 prepare_train_features와 동일
```

**장점**

* 구현 난이도: DHN보다 훨씬 단순.
* train도 “retrieval을 한번 거친” context를 본다는 점에서 **train/test alignment가 상당히 좋아짐**.
* negative handling이 없기 때문에 **학습 안정적**.

**단점/주의**

* chunk 텍스트 안에서 `ans in p["text"]`로 찾는 방식이라,

  * answer가 여러 번 등장하는 특수 케이스에서는 ambiguous.
  * 하지만 현재 데이터 규모/특성상 “아예 못 쓰겠다” 수준은 아님.
* retrieval 실패 케이스(~7~8%)는 여전히 남음
  → 이건 나중에 top_k를 20으로 키우거나, BM25-only doc 수준 retrieval로 train할 수도 있음.

---

### 아이디어 B. “Simple Negative Sampling QA” (DHN보다 단순한 per-context 학습)

**컨셉**

* A에서 한 걸 바탕으로, **negative context도 조금 섞되** DHN처럼 복잡하게 dynamic pool을 돌리지 않고,

* 그냥:

  > “한 질문에 대해
  > (Q, positive_context) + (Q, negative_context 1~2개)
  > 를 각각 별도의 training example로 만들자”

* 그리고 negative에는 “no-answer (CLS)”를 정답으로 넣는 classic 방식.

**구체 구조**

1. train에서:

```python
for example in train:
    qid = example["id"]
    answers = example["answers"]["text"]
    cands = cache[qid]["retrieved"]

    pos_passage = None
    neg_passages = []
    for cand in cands:
        p = passages[cand["passage_id"]]
        if any(ans in p["text"] for ans in answers):
            if pos_passage is None:
                pos_passage = p["text"]
            else:
                neg_passages.append(p["text"])
        else:
            neg_passages.append(p["text"])

    # 1) positive sample
    if pos_passage is not None:
        add_train_example(question, pos_passage, answers, has_answer=True)

    # 2) negative sample 1~2개
    for neg in random.sample(neg_passages, k=min(len(neg_passages), N_NEG)):
        add_train_example(question, neg, answers=None, has_answer=False)
```

2. 모델 관점에서는:

* `has_answer=True`: start/end span supervised (KorQuAD 그대로)
* `has_answer=False`: start/end 둘 다 CLS 위치로, 또는 별도 no-answer logit 사용

3. test/inference:

* top_k contexts 각각에 대해 reader 돌려서 score를 구하고,
* 가장 confident한 span 선택 (지금 pipeline에 가까운 구조)

**장점**

* train에서 **실제 test distribution과 매우 비슷한 일을 함**:

  * 대부분 context에는 답이 없고, 일부에만 있음 → 이걸 학습.
* DHN처럼 "pool에서 하나 뽑아 학습"이 아니라,
  **모든 후보를 명시적으로 학습 데이터로 쓰기 때문에** stability가 좋음.

**단점/난이도**

* training dataset size가 늘어남 (질문당 3~5개 예제).

  * 다만 4k 질문 × 3~5 ≒ 1.2~2만 예제 수준이면 충분히 감당 가능.
* 구현은 **새 Dataset 클래스 하나 파는 수준**이라, 시간이 아주 촉박하면 부담될 수 있음.

---

### 아이디어 C. Curriculum Learning + Retrieval (팀원 아이디어 보강 버전)

팀원 아이디어:

> 1 epoch: gold context만
> 2~3 epoch: gold + negative

이 구조는 상당히 합리적입니다. 다만, 몇 가지 잡설만 더하면 안전해집니다.

**권장 설계**

* **Epoch 1: A 방식 (Retrieval-aware Gold-only)**

  * 위의 아이디어 A만 먼저 돌림.
  * 이때 negative는 전혀 섞지 않고,
    “retrieval에 의해 변형된 gold context”에만 완전히 적응하게 만들기.

* **Epoch 2~3: B 방식의 “약한” 버전**

  * 질문당:

    * positive: 1개 (retrieval에서 찾은 gold passage)
    * negative: 1개 정도만 (too many negative는 피함)
  * batch 구성: 항상 `pos : neg ≈ 1 : 1` 수준으로 유지 (정답 signal이 사라지지 않게)

이렇게 하면:

* 1 epoch 동안 “pure QA with retrieval-shaped context”에 적응
* 이후 1~2 epoch 동안 “no-answer 케이스도 조금씩 학습”
  → test 때 top_k 중 gold 외 다른 context를 잘 거르는 데 도움

**DHN과 차이**

* DHN은 “pool에서 random choice” + “어떤 step에서는 네거만 학습”처럼 distribution이 요동치는 반면,
* Curriculum B는 **각 epoch/step마다 pos/neg 비율을 꽤 안정적으로 유지**하게 설계할 수 있습니다.

---

## 2. “gold context를 항상 맨 앞에 붙이는” 것에 대한 의견

이 부분은 상당히 중요해서 명확히 짚고 갈게요.

### 2-1. 언제 문제가 되냐?

“gold context를 맨 앞에 배치하는” 상황이 아래처럼 쓰이면 위험합니다:

* train에서:

  ```text
  [Q] + [SEP] + [GOLD CONTEXT] + [SEP] + [NEG1] + [SEP] + [NEG2] ...
  ```

* test에서는:

  ```text
  [Q] + [SEP] + [RET1] + [SEP] + [RET2] + ...
  ```

  여기서 **RET1이 항상 gold가 아님** (retrieval ranking에 따라 달라짐).

이 경우, 모델은 아주 쉽게 **position bias**를 학습합니다:

* “답은 항상 첫 번째 블록에 있다”라는 편향
* train 때는 이게 항상 맞으니까 loss는 잘 내려가지만,
* test에서는 gold가 2번째/3번째 블록에 있을 때
  → 모델이 앞에 있는 비슷한 개체를 먼저 잡고 멈출 가능성이 커짐

즉, **train/test mismatch를 오히려 키우는 방향**입니다.

---

### 2-2. 언제 괜찮냐?

반대로, 아래처럼 쓰인다면 큰 문제는 덜합니다:

* **train에서도**, **test에서도** 항상 gold를 맨 앞에 가져오는 구조:

  * train:

    * label이 있으니 gold가 어딘지 알 수 있음 → 맨 앞으로 재배치
  * test:

    * “test에도 gold doc id가 있다”는 전제 하에 (현 대회는 아님 ㅋㅋ)
      gold를 앞에 박고 나머지를 뒤에 붙이는 구조

이러면 train/test distribution이 다시 맞고, position bias가 문제가 되지 않습니다.

하지만 **ODQA test는 gold를 모르는 상태**이므로
실제로는 이렇게 못 합니다.

---

### 2-3. 지금 상황에서의 결론

현재 프로젝트에서는:

* test 시점에 **gold를 모름**
* retrieval 순서는 **hybrid score 순 정렬**

따라서, “train에서만 gold를 항상 맨 앞에 붙이는” 구조는
**좋지 않은 선택**에 가깝습니다.

**더 나은 선택지**

1. **여러 context를 한 문장에 concat해야 한다면**

   * gold를 맨 앞에 강제로 옮기지 말고,
   * **retrieval ranking 그대로** 두는 쪽이 낫습니다.
   * (더 공격적으로는, 매 epoch마다 순서를 랜덤 셔플해서 position bias를 줄이는 것도 가능)

2. 하지만, 개인적으로는 **concat 방식 자체를 피하고**,

   * 아예 “context 하나당 QA 한 번” 구조 (아이디어 B/C)로 가는 게
   * 구현은 약간 늘어나도 **모델이 하려는 일(“여러 후보 중 하나 고르기”)과 더 잘 맞습니다.**

---

## 3. 지금 상황에서의 현실적인 추천 조합

질문 주신 것을 한 줄로 요약하면:

> “train도 retrieval을 써서 test와 맞추고 싶은데,
> DHN은 성능이 애매하니 다른 아이디어로 하고 싶다.
> 그리고 gold를 맨 앞에 두는 방식은 좀 불안하다.”

이 기준으로, **짧은 시간 대비 효과 + 구현 난이도**를 고려한 추천은:

1. **DHN 끄고, A 방식 먼저**

   * `dynamic_hard_negative.enabled: false`
   * `train_retrieval: true`
   * train에서 retrieval을 쓰되, **answer가 들어 있는 passage 한 개만 골라서 gold-only 학습**.
   * 이걸로 baseline 하나 다시 깔끔하게 만들기.

2. **시간이 허용되면 B/C를 소프트하게 도입**

   * 팀원이 커리큘럼 러닝 통합할 때,

     * Epoch 1은 A 그대로
     * Epoch 2~3에서 질문당

       * gold passage 1개 + negative passage 1개 정도만 추가
       * negative는 “no-answer”로 학습
     * 이때 여러 context를 concat하지 말고, **context별 sample로 분리해서 학습**하도록 요청하는 게 좋습니다.

       * (gold 맨 앞 bias 문제도 동시에 해소)

3. **“gold를 항상 맨 앞에 붙이는 concat 방식”은 지양**

   * 이미 구현돼 있다면:

     * 적어도 **retrieval rank대로 keep**하거나,
     * 가능하다면 concat 구조 자체를 버리고 per-context 학습으로 전환.

---

원하시면,
이제 이 아이디어들을 토대로:

* “가설 → 실험 설계 → 기대 효과 → 소요 시간” 형식으로 정리해서
* Claude Opus에 넘길 **실험 계획 프롬프트** 형태로도 정리해 줄 수 있습니다.

좋아요, 이제 퍼즐 조각이 거의 다 모였어요.
지금까지 나온 분석 + 코드 구조 + 보조 함수까지 전부 엮어서, **“짧은 시간 안에 의미 있게 점수를 올릴 수 있는” 최종 계획**을 정리해볼게요.

---

## 0. 현재 핵심 인사이트 정리 (짧게)

1. **Retrieval**

   * weighted_hybrid (KURE + BM25, α=0.35)으로
     R@10 ≈ 93–95%, R@20 ≈ 96–98% 수준 → **retrieval은 거의 해결**.
   * val 기준 retrieval failure는 7.5% 정도로 EM 상한선을 92~93% 정도로 제한.

2. **Reader**

   * gold context 기준 KorQuAD-finetuned roberta-large → EM ≈ 73 (예전 세팅).
   * 현재 파이프라인 (val에서 retrieval context 사용) 기준:

     * Vanilla: EM ≈ 61.25, F1 ≈ 70 / 리더보드 EM 65.42, F1 75.77.
     * DHN: EM ≈ 60.83, F1 ≈ 69.19 → **DHN은 유의미한 이득 없음 (오히려 손해)**.
   * Error breakdown:

     * retrieval_failure: 7.5%
     * completely_wrong: 약 20%
     * partial_superset/subset/overlap: 약 12~15%
   * 결론:
     **retrieval은 이미 상위 5% 수준, 병목은 reader 학습 분포 + context 구조.**

3. **핵심 과제**

   * 지금까지는 **train: gold context**, **val/test: retrieval context**였음.
   * 이제는 “**train에도 retrieval을 쓰되**, DHN처럼 복잡하게 섞지 않고,
     현재 제공된 `retrieve_and_build_dataset + realign_answers_in_retrieved_context`를 활용해서
     **train/test 컨텍스트 분포를 최대한 맞추는 것**이 핵심.

---

## 1. Train에서도 Retrieval을 쓰는 설계 정리

지금 코드 기준으로 train에서 retrieval을 쓰는 경로는 이미 준비되어 있어요.

### 1-1. 이미 있는 훅 정리

`train.py` 상단:

```python
if data_args.train_retrieval and training_args.do_train:
    ...
    retriever = get_retriever(...)
    retriever.build()

    new_train_dataset = retrieve_and_build_dataset(
        retriever=retriever,
        dataset=datasets["train"],
        data_args=data_args,
        split_name="train",
        is_train=True,
        tokenizer=tokenizer,
    )
    datasets["train"] = new_train_dataset
```

`retrieve_and_build_dataset` 내부:

```python
df = retriever.retrieve(dataset, topk=data_args.top_k_retrieval, tokenizer=tokenizer)
# df["context"]는 weighted_hybrid 결과로 뽑은 context (title 포함 가능)

new_dataset = Dataset.from_pandas(df, features=features)

if is_train and has_answers:
    new_dataset = new_dataset.map(realign_answers_in_retrieved_context)
    new_dataset = new_dataset.filter(filter_valid_answers)
```

`realign_answers_in_retrieved_context`:

```python
retrieved_context = example["context"]
for answer_text in original_answers["text"]:
    start_idx = retrieved_context.find(answer_text)
    if start_idx != -1:
        new_text.append(answer_text)
        new_answer_start.append(start_idx)
example["answers"] = {"text": new_text, "answer_start": new_answer_start}
```

→ **train 시점에 retrieval로 얻은 context 안에서 answer_start를 다시 잡고,
정답이 안 나온 sample은 필터링**하는 구조가 이미 구현되어 있음.

### 1-2. DHN과의 관계

현재 base.yaml:

```yaml
eval_retrieval: true
train_retrieval: false      # <- 여기
dynamic_hard_negative:
  enabled: true             # <- 여기
  k_ret: 20
  k_read: 3
```

* **DHN 켜져 있으면**: `MRCWithRetrievalDataset` 경로를 타고, gold context + neg들이 섞이는 구조.
* **우리가 원하는 것은**: **DHN OFF + train_retrieval ON** 조합.

즉, 새 실험에서는:

```yaml
eval_retrieval: true
train_retrieval: true           # train에도 retrieval 적용
dynamic_hard_negative:
  enabled: false                # DHN OFF
```

으로 두고, train / val 모두 **retrieval 기반 context**를 쓰게 만들면 됨.

---

## 2. “긴 컨텍스트에 안 헷갈리게” 만드는 설계 포인트

당신이 걱정한 것:

> “retrieval을 train에도 최대한 깔끔하게 적용하면서도
> train이 너무 긴 컨텍스트 때문에 안헷갈리도록 하는게 좋을 것 같아”

이 포인트를 기준으로 설계를 정리하면:

1. **top_k_retrieval (train용)는 너무 크지 않게**

   * val에서 R@5 ≈ 92.9%, R@10 ≈ 95% 수준인 걸 감안하면,
   * **train에는 top_k_retrieval=3~5 정도가 적당**.

     * 3: context 짧음, 노이즈 적음, recall 약 90% 수준.
     * 5: 조금 더 여유, recall ≈ 93% 이상.
   * test/inference에서는 그대로 top_k=10 유지 (또는 10 vs 20 비교 후 최종 선택).

2. **retrieved context 구성 방식**

   * 현재 `BaseRetrieval.retrieve`가 `context`를 어떻게 만들고 있는지에 따라 두 가지 경우:

     1. top-k 전체를 concat한 하나의 긴 context
     2. top-1 passage만 context로 선택
   * 어느 방식이든 `realign_answers_in_retrieved_context`는
     **“retrieved_context 문자열 안에서 answer_text를 직접 찾아 start index를 갱신”**하므로,

     * 길이만 과도하지 않으면, indexing 문제는 해결됨.
   * “안 헷갈리게”라는 관점에서는:

     * **train에서는 top_k_retrieval=3 (또는 5)** → context가 지나치게 길어지지 않게.
     * doc_stride=128 유지 or 약간 줄이기(96)로 boundary 문제 줄이기.

3. **gold context를 맨 앞에 박지 않는다**

   * 이번 설계는 **retriever.retrieve의 ranking 그대로 context를 가져오고**,
     gold를 앞에 강제로 끼워 넣지 않으므로,
   * 앞서 얘기했던 **position bias(“답은 항상 첫 번째 블록에 있다”) 문제**도 피할 수 있음.

---

## 3. 최종 실험 / 작업 계획 (우선순위 + 가설 + 구체 설정)

### [Tier 0] 기준선 정리 (이미 완료된 상태 고정)

* 기준 모델: `HANTAEK_roberta_large_vanilla`

  * train: gold context
  * val/test: retrieval context (α=0.35, top_k=10)
* 성능:

  * val: EM ≈ 61.25, F1 ≈ 70.03
  * test: EM 65.42, F1 75.77
* 이 모델과의 차이만 보면서 “의미 있는 개선” 여부 판단.

---

### [Tier 1] Train에도 Retrieval 적용 (DHN 제거) – **가장 중요한 실험**

**가설**

> train에서도 retrieval 결과로 구성된 context를 사용하고
> answer_start를 재정렬하면,
> gold-only train vs retrieval-based val/test 간 분포 차이가 줄어들어
> EM이 최소 +1~2pt 개선될 것이다.

**설정**

1. 새 YAML (예: `configs/active/roblarge_retrieval_train.yaml`) 생성:

```yaml
# 기존 base.yaml에서 파생

# --- data ---
eval_retrieval: true
train_retrieval: true
top_k_retrieval: 3          # ★ train용, 짧은 context 가정 (3 또는 5 추천)
doc_stride: 128             # 우선 baseline 유지

# --- dynamic_hard_negative ---
dynamic_hard_negative:
  enabled: false            # ★ DHN OFF

# --- training ---
num_train_epochs: 3
learning_rate: 2.0e-5
...
output_dir: ./outputs/dahyeong/HANTAEK_roberta_large_retrTrain_k3
```

2. `train.py` 그대로 사용:

   * `train_retrieval=True` → `retrieve_and_build_dataset(..., is_train=True)` 경로 활성화.
   * `realign_answers_in_retrieved_context`가 answer_start 재계산 + 필터링 수행.
   * DHN은 disabled, MRCWithRetrievalDataset 사용 안 함.

3. `inference.py`에서 동일 YAML로 val/test eval:

   * val: EM/F1 측정.
   * test: `test_pred.csv` 생성 후 제출.

**기대 효과**

* Train context도 이제 **retrieval 기반 분포** →
  gold vs retrieval gap이 줄어들어

  * val EM: **61.2 → 62~63** 정도 기대.
  * test EM: **65.4 → 66~67**까지 현실적인 범위.

**비용/난이도**

* 구현: YAML 수정만으로 가능 (코드 이미 준비됨).
* 시간: 현 로그 기준 3 epoch ≈ 6분 내외 → 시도 부담 낮음.

**추가 체크포인트**

* train 로그에 나오는 realign 필터링 통계 확인:

  * `Lost examples`가 10% 내라면 OK.
  * 너무 많으면 train용 `top_k_retrieval`을 5로 올리는 실험 추가.

---

### [Tier 1.5] doc_stride 조정 – **Reader 경계 오류 줄이기**

**가설**

> 현재 partial_superset/subset 오류가 10%대 존재한다.
> doc_stride를 줄여 chunk overlap을 늘리면,
> 경계에 걸린 정답들이 더 안정적으로 포함되어 EM/F1이 소폭 오른다.

**설정**

* 위 Tier 1 설정에서 단 하나만 변경:

```yaml
doc_stride: 96    # 128 → 96 (또는 80)
```

* 나머지 설정 동일 (`train_retrieval=True`, `top_k_retrieval=3`, DHN OFF).

**기대 효과**

* EM/F1 +0.3~0.8 수준의 미세한 개선 예상.
* 특히 partial_subset/superset 비율 감소.

**비용/난이도**

* YAML 한 줄 수정.
* 학습/평가 1회 추가.

---

### [Tier 2] Curriculum Learning (팀원 아이디어를 retrieval-train과 결합)

**가설**

> 1 epoch 동안 gold-only context로 “깨끗한 QA”에 적응시키고,
> 이후 1~2 epoch 동안 retrieval 기반 context에 fine-tune하면,
> 바로 retrieval 트레인만 하는 것보다 안정적이고 일반화가 더 좋아진다.

**설정 (2-stage 학습)**

1. **Stage 1 – Gold-only**

   ```yaml
   # configs/active/roblarge_curr_stage1.yaml

   train_retrieval: false
   eval_retrieval: false       # stage1은 gold context 기준 성능만 확인
   dynamic_hard_negative.enabled: false

   num_train_epochs: 1
   output_dir: ./outputs/dahyeong/roblarge_curr_stage1
   ```

   * train: gold context.
   * val: gold context (baseline KorQuAD-style).

2. **Stage 2 – Retrieval-aware fine-tuning**

   ```yaml
   # configs/active/roblarge_curr_stage2.yaml

   model_name_or_path: ./outputs/dahyeong/roblarge_curr_stage1  # stage1 결과 로딩

   train_retrieval: true
   eval_retrieval: true
   top_k_retrieval: 3 or 5
   dynamic_hard_negative.enabled: false

   num_train_epochs: 2         # 추가 2 epoch fine-tune
   output_dir: ./outputs/dahyeong/roblarge_curr_stage2
   ```

   * 이 단계는 사실상 “DHN 없는 Curriculum” 버전.
   * 팀원이 pipeline에 통합해준다면, 위 구조를 그대로 전달하면 됨.

**기대 효과**

* Tier 1에서 얻은 gain에 추가로 **+0.5~1pt 정도 EM 상승 여지**.
* 특히 completely_wrong 케이스 일부가 줄어들 가능성.

**비용/난이도**

* 학습 2번 (1 epoch + 2 epoch) → 전체 3 epoch와 큰 차이 없음.
* 설정만 잘 공유되면 구현 복잡도도 낮음.

---

### [Tier 3] Inference top_k 튜닝 (10 vs 20) – **Retrieval failure 줄이기**

**가설**

> R@10 = ~93–95%, R@20 = ~96–98% 이므로,
> inference에서 top_k_retrieval를 20으로 늘리면
> retrieval_failure(현재 7.5%)가 절반 이하로 줄고,
> reader가 해당 context에서 정답을 발견할 확률이 증가한다.

**설정**

* 학습은 Tier 1 또는 Tier 2 모델 활용.
* inference 전용 YAML에서만 수정:

```yaml
top_k_retrieval: 10    # 기존
# vs
top_k_retrieval: 20    # 새 버전
```

* `inference.py`의 `load_retrieval_from_cache`는 top_k개 passage를 concat하므로,

  * context 길이 ↑ → tokenization 시 슬라이딩 window로 나뉨.
  * 성능 지표: val에서 미리 `top_k=10` vs `top_k=20` 비교 후, test에 더 좋은 쪽 사용.

**기대 효과**

* 단독으로는 +0.5~1 EM 정도 기대.
* 단, context가 너무 길어져 오히려 혼란을 줄 수도 있으므로,
  반드시 **val 기준으로 10 vs 20 비교 후 선택**.

**비용/난이도**

* 학습 없음, inference만 2회.
* 캐시(`val_top50.jsonl`)는 score_dense/score_bm25 모두 가지고 있어 부담 없음.

---

### [Tier 4] 단순 앙상블 – **마지막 한 방울 짜내기**

**가설**

> 서로 조금 다른 에러 패턴을 가진 reader들을
> (예: vanilla gold-train vs retrieval-train vs curriculum)
> n-best 결과로 앙상블하면,
> 개별 모델보다 EM/F1이 0.5~1pt 정도 올라간다.

**설정**

* 대상 모델:

  * `HANTAEK_roberta_large_vanilla`
  * `roberta_large_retrTrain_k3` (Tier 1)
  * `roberta_large_curr_stage2` (Tier 2, 있다면)

* 각 모델에 대해 `inference.py`로 `nbest_predictions_test.json` 생성.

* 간단 앙상블 스키마 예시:

  * 같은 id에 대해 각 모델의 n-best 후보 리스트를 합치고,
  * `score`(또는 `start_logit + end_logit`)를 normalize 후,
  * 동일 `text` 기준으로 score를 평균 또는 가중합하여 가장 높은 answer 선택.

**기대 효과**

* 현재 리더보드 EM 65.4 기준,
  앙상블까지 포함하면 **66~67+ α** 정도까지 도달 가능성.

**비용/난이도**

* 별도 스크립트 하나 작성 필요 (Python 50~100줄 수준).
* 하지만 train은 재사용, inference만 몇 번 돌리면 됨.

---

### [Tier 5] 이번 라운드에서 *의도적으로* 제외하는 것들

짧은 시간 + 높은 확실성을 기준으로, 이번 라운드에서는 **우선순위에서 내리는 항목**도 명시해둘게요.

1. **Cross-Encoder Reranker 도입**

   * 효용성은 있지만, 별도 모델 학습 + 파이프라인 복잡도 증가.
   * 현재 retrieval이 이미 상위권이라 **ROI가 상대적으로 낮음**.

2. **완전히 새로운 Reader 아키텍처 (DeBERTa, ELECTRA, etc.)**

   * 현재 HANTAEK roberta-large가 이미 강한 KorQuAD 전이 성능을 보여줌.
   * 새 모델 검증 + 파이프라인 수정까지 고려하면 시간 대비 리스크 큼.

3. **DHN 변형 실험**

   * 현재 DHN의 첫 결과가 baseline보다 낮았고,
   * 이미 더 깔끔한 retrieval-train 경로가 마련된 상태.
   * DHN을 고치는 것보다는, 아예 **단순하고 직관적인 retrieval-train**에 집중하는 것이 합리적.

---

## 4. 요약 – 지금 당장 진행하면 좋은 순서

정리하면, “짧은 시간 안에 확실히 의미 있을 것 같은” 순서는 이렇게 추천합니다.

1. **[필수] Tier 1: Retrieval-aware Training (DHN OFF, train_retrieval=TRUE, top_k_train=3)**
   → 새 baseline 확보.

2. **[강추] Tier 1.5: doc_stride 조정 (128→96) with Tier 1 설정 그대로**
   → partial match 계열 개선.

3. **[가능하면] Tier 2: Curriculum (1 epoch gold-only → 2 epoch retrieval-train)**
   → 안정화 + 일반화 개선.

4. **[매우 쉬움] Tier 3: Inference에서 top_k=10 vs 20 비교 후, test에 더 좋은 값 사용**

5. **[여유가 되면] Tier 4: 위 2~3개 모델로 n-best 앙상블**

---

이제 다음 단계는:

* 위 계획에서 실제로 하고 싶은 Tier들을 골라주면,
* 그 선택을 반영해서 **Claude Opus 4.5용 “실험 계획 프롬프트”**를
  아주 정갈하게 작성해줄게요.
  (가설, 실험 구조, YAML 변경 포인트, 기대 효과까지 포함된 형태로.)

너는 한국어 MRC(ODQA) 대회에 참가 중인 팀의 모델/코드 설계 보조자 역할이다.
단순 설명이 아니라, **짧은 시간 안에 실제 리더보드 점수를 확실히 올릴 수 있는 실험 설계 + 코드 레벨 제안**을 해주는 것이 목적이다.

이번 프롬프트의 핵심은 **이미 어느 정도 정리된 전략(Tier 1, 1.5, 3, 4)을 바탕으로, 실제로 지금 코드베이스에 꽂을 수 있는 구체적인 실험 설계/수정안을 정리**하는 것이다.

---

# 1. 프로젝트/코드 개요 (요약 정보)

## 1.1 태스크/데이터 개요

* 태스크: Korean Open-Domain Question Answering (MRC + Retrieval)

* 데이터:

  * train: 3,952 QA pairs (KorQuAD 스타일, gold context 포함)
  * validation: 240 QA pairs (gold context 포함)
  * test: 600 questions (context/answers 없음, 리더보드 제출용)

* 파이프라인: 2-stage

  1. **Retriever**: KURE-v1 dense + BM25 sparse → weighted hybrid retrieval
  2. **Reader**: RoBERTa large, KorQuAD v1로 이미 finetune된 모델
     (`HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned` 기반 변형)

* 현재 성능 (중요):

  * Validation (retrieval context 기준, Vanilla):

    * EM ≈ 61.25, F1 ≈ 70
  * Test 리더보드 (Vanilla + DHN 둘 다 비슷):

    * EM 65.42, F1 75.77

* Error pattern (대략):

  * Retrieval failure (gold not in top-10): ~7.5%
  * Retrieval 성공 사례 중에서도:

    * completely_wrong (F1 < 20): ~20% 안팎
    * 부분 일치 subset/superset/overlap: 10%+
      → **retrieval은 거의 해결(상한선은 높음), 실제 병목은 Reader 쪽**이라고 보고 있다.

---

## 1.2 Retrieval 시스템 (핵심만)

* Dense: `nlpai-lab/KURE-v1`, 1024-d embedding
* Sparse: BM25(k1=1.5, b=0.75)
* Hybrid score (per-query 정규화 후):

```python
bm25_n  = (bm25 - bm25.min()) / (bm25.max() - bm25.min() + eps)
dense_n = (dense - dense.min()) / (dense.max() - dense.min() + eps)
hybrid  = alpha * bm25_n + (1 - alpha) * dense_n
# 현재 alpha ≈ 0.35 사용
```

* 대략적인 recall (weighted_hybrid, alpha≈0.35):

  * R@10: 93–95%
  * R@20: 96%+
  * R@50: 97–98%

* Retrieval 캐시:

  * `data/cache/retrieval/train_top50.jsonl`
  * `data/cache/retrieval/val_top50.jsonl`
  * `data/cache/retrieval/test_top50.jsonl`
  * 각 라인: question id, question text, `retrieved[]`(passage_id, score_bm25, score_dense …)

---

## 1.3 Reader/학습 구조 (핵심만)

* Reader: RoBERTa large KorQuAD v1 QA finetuned 기반.
* 토크나이저/슬라이딩 윈도우:

```python
tokenized_examples = tokenizer(
    question,
    context,
    truncation="only_second",
    max_length=max_seq_length,   # 기본 384
    stride=doc_stride,           # 기본 128
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length" or False,
    return_token_type_ids=False  # RoBERTa 특성 상 자동 처리
)
```

* 기본 설정:

  * `max_seq_length=384`
  * `doc_stride=128`
* 후처리: `postprocess_qa_predictions`에서 SQuAD-style span 검색
  (n-best 중 `start_logits + end_logits`가 가장 높은 span을 answer로 선택)

---

## 1.4 현재 base.yaml의 중요한 설정

```yaml
# Data args
max_seq_length: 384
doc_stride: 128
max_answer_length: 30

# Retrieval 설정
eval_retrieval: true            # val/test는 기본 retrieval 사용
train_retrieval: false          # 지금은 train에서 retrieval 사용 안 함 (Tier 1에서 바꿀 예정)
retrieval_type: weighted_hybrid
top_k_retrieval: 10             # inference/val 기본 top_k

# Dynamic Hard Negative (DHN)
dynamic_hard_negative:
  enabled: true                 # 과거 실험에서 켜서 사용했으나 효과 미미
  k_ret: 20
  k_read: 3
  alpha: 0.35
  use_title: true

# Training
num_train_epochs: 3
per_device_train_batch_size: 16
learning_rate: 2e-5
warmup_ratio: 0.1
weight_decay: 0.01
eval_strategy: epoch
save_strategy: epoch
load_best_model_at_end: true
metric_for_best_model: exact_match
fp16: true
```

---

## 1.5 중요한 코드 파일 요약

* `train.py`

  * `get_config(parser)`로 YAML/CLI args 파싱
  * `data_args.train_retrieval`이 True이면 `retrieve_and_build_dataset`로 train split을 retrieval 기반 dataset으로 변환 가능
  * `dynamic_hard_negative.enabled=True`인 경우 `MRCWithRetrievalDataset` 사용 (지금 라운드에서는 사용하지 않을 예정)
  * 마지막에 `FinalEvaluator` + `save_prediction_analysis`로 train/val 사후분석 파일들 생성

* `inference.py`

  * `inference_split`(train/validation/test)에 따라 `do_eval`/`do_predict` 제어
  * `load_retrieval_from_cache`로 캐시 기반 retrieval context 생성 (여기서는 top-k passages concat)
  * `postprocess_qa_predictions` 호출,
    `predictions_{split}.json`, `nbest_predictions_{split}.json`, `{split}_pred.csv` 생성

* `src/utils/retrieval_utils.py`

  * `retrieve_and_build_dataset(retriever, dataset, data_args, split_name, is_train, tokenizer)`
  * 내부에서 `retriever.retrieve(...)`를 호출하고,
    DataFrame → Dataset 변환 후 `is_train=True`이면 answer realignment + filtering 수행

* `realign_answers_in_retrieved_context` (중요 보조 함수):

```python
def realign_answers_in_retrieved_context(example):
    retrieved_context = example["context"]
    original_answers = example["answers"]
    new_text = []
    new_answer_start = []

    for answer_text in original_answers["text"]:
        start_idx = retrieved_context.find(answer_text)
        if start_idx != -1:
            new_text.append(answer_text)
            new_answer_start.append(start_idx)

    example["answers"] = {"text": new_text, "answer_start": new_answer_start}
    return example
```

* train에서 retrieval 기반 context를 사용할 때,
  concat된 context 안에서 answer text의 위치를 다시 맞춰준다.

---

# 2. 전략 방향 (이미 결정된 가이드라인)

아래 4가지 방향은 이미 팀과 합의된 내용이며,
너는 이 방향을 전제로 **“안에서의 설계/튜닝”**을 제안해야 한다.

1. **[반드시] Tier 1 – Retrieval-aware Training**

   * train에서도 반드시 retrieval 기반 context를 사용한다.
   * 핵심: **retrieval로 가져온 여러 passage를 train에서 어떻게 사용할지 (단순 concat이 최선인지)** 설계.
   * DHN(dynamic hard negative)은 이번 라운드에서는 **사용하지 않는다**
     (이미 실험했으나 성능 이득이 거의 없었고, 복잡도만 증가).

2. **[반드시] Tier 1.5 – doc_stride 튜닝**

   * 과거 sparse+gold baseline에서 `doc_stride=128 → 64` 변경만으로 EM이 크게 상승한 경험이 있다.
   * 현재도 partial_superset/subset 에러가 꽤 많으므로,
     **chunk overlap 증가(stride 감소)** 를 통해 span 경계 문제를 줄이는 것이 매우 유망하다.

3. **[중요] Tier 3 – Inference용 top_k 튜닝**

   * train과 inference에서 사용할 top_k를 분리하여 관리하고 싶다.

     * train: top_k_train = 1 또는 3 (context를 너무 길게 만들지 않기 위해)
     * inference: top_k_infer = 10 또는 20 (retrieval failure 감소 기대)

4. **[중요] Tier 4 – Ensemble**

   * 여러 reader 변종(예: baseline gold-only, retrieval-train, curriculum stage2 등)을
     `nbest_predictions_{split}.json` / logits 기반으로 앙상블해서
     EM/F1을 +0.5~1pt 정도 올리고 싶다.
   * val/test 모두 `nbest_predictions_*.json`, `logits_*.json` 파일은 이미 저장되고 있다.

5. **커리큘럼 러닝은 팀원이 구현 예정**

   * 예: 1 epoch gold-only, 2~3 epoch retrieval-train
   * 이 프롬프트에서는 커리큘럼 자체는 구현하지 말고,
     **우리가 설계하는 retrieval-train 구조가 커리큘럼 stage2에서 그대로 재사용 가능**하도록 설계해달라.

6. **시간 제약 상, 큰 구조 변경/새 모델 학습보다는,
   기존 코드 위에서 “현실적인 수정 + 실험” 위주로 계획을 세워야 한다.**

   * 예: Cross-Encoder reranker는 “시간이 되면” 수행하는 optional tier로만 취급.

---

# 3. 네가 설계해야 할 Tier별 실험/코드 계획

아래 4개 Tier에 대해, **가설 → 실험 설계 → 코드 수정 포인트 → YAML 설정 예시 → 평가 방법**까지
한 번에 정리해줘.

---

## [Tier 1 – 반드시 수행] Retrieval-aware Training (DHN OFF, train_retrieval=TRUE)

### 3.1 Tier 1 목적

* train에서도 retrieval 기반 context를 사용하여:

  * train vs val/test의 context distribution mismatch를 줄이고,
  * Reader가 “retrieval이 가져온 noisy multi-passage context”에 잘 적응하도록 학습.
* 단순 “top-k concat”이 아니라, **train에서 사용할 context 구성을 제대로 설계하는 것**이 핵심.

### 3.2 현재 도구/제약

* 이미 다음이 존재:

  * `retrieve_and_build_dataset(retriever, dataset, data_args, split_name, is_train, tokenizer)`
  * `realign_answers_in_retrieved_context`
  * `train_retrieval=True`로 두면 train에서도 retrieval 사용 가능
* 다만 현재 retrieval → context 구성 방식이 “단순 concat”에 가깝다고 보고 있고,
  이것이 Reader에게 혼란을 줄 가능성을 우려하고 있다.

### 3.3 네가 설계해야 할 것 (Tier 1)

1. **Train context 구성 전략 후보를 2~3개 정의하고, 장단점을 비교해줘.**

   예시는 아래와 같지만, 여기에 얽매이지 말고 **지금 코드 구조에 잘 맞고, 단기간에 구현 가능한 현실적인 전략 2~3개**를 제안해달라.

   * 전략 A: **Top-1 passage만 사용 (train에서)**

     * 각 질문에 대해 hybrid score가 가장 높은 passage 하나만 context로 사용.
     * 장점: context 짧음, train time/GPU 부담 적음, answer realignment 간단.
     * 단점: train에서 retrieval failure인 경우 answer를 잃어버려 필터 비율 증가.

   * 전략 B: **Top-k(예: 3~5) passage concat + 구조적 구분**

     * 예: `[title1] SEP text1 SEP [title2] SEP text2 SEP ...` 형태로 붙이고,
       sep token/제목으로 passage 경계를 명확히 드러내기.
     * `max_seq_length=384`를 넘지 않도록, 앞에서부터 priority 높은 passage 위주로 잘라내는 정책.
     * `realign_answers_in_retrieved_context`를 이 구조에 맞게 보정.
     * 장점: train 시에도 inference와 비슷한 multi-passage 상황 학습.
     * 단점: 구현 난이도/디버깅 비용 증가, context 길이 증가로 인한 noise 가능성.

   * 전략 C: **Passage-level multi-example 학습**

     * 한 question에 대해 top-3 passage를 각각 별도 training example로 생성.
     * gold answer가 포함된 passage에는 정상 answer_start를 부여,
     * gold answer가 없는 passage는 “no-answer (CLS index)”로 학습.
     * 장점: 문장 길이를 짧게 유지하면서도 retrieve된 후보를 여러 개 활용, no-answer 패턴 학습.
     * 단점: dataset size 2~3배 증가, 훈련 시간 증가.

   각 전략에 대해:

   * 구현 난이도
   * train time 증가량
   * EM/F1 개선에 대한 직관적 기대
     를 비교해서 정리해줘.

2. **위 전략들 중 “Tier 1에서 꼭 돌려볼 핵심 1~2개”를 추천하고, 구체 코드 수정 포인트를 제시해줘.**

   최소한 다음 레벨에서 수정 포인트/스니펫을 제시해달라:

   * `BaseRetrieval.retrieve` 또는 해당 구현에서
     DataFrame `df["context"]`를 어떻게 만들지 (top-1 vs concat vs multi-example).
   * `retrieve_and_build_dataset` 내부에서,
     `is_train=True`인 경우에 어떤 전처리/구성/필터링을 추가해야 하는지.
   * `realign_answers_in_retrieved_context`가
     multi-passage/sep token 구조에서도 안전하게 동작하도록 어떻게 바꿀지.

   “이 파일을 열고 아래 블록 근처에 이런 코드를 추가/수정하면 된다” 수준으로
   구체적인 pseudo-code 또는 부분 코드 스니펫을 보여줘.

3. **Tier 1 실험 세트 정의 (YAML 기반)**

   * 공통 설정:

     ```yaml
     eval_retrieval: true
     train_retrieval: true
     dynamic_hard_negative:
       enabled: false

     max_seq_length: 384
     doc_stride: 128   # Tier 1.5에서 조정, 여기선 baseline
     ```

   * 예시 실험 (필요시 더 적절히 조정/보완해도 좋다):

     * Exp 1-1: train_top_k=1, top-1 passage only, simple context (no concat)
     * Exp 1-2: train_top_k=3, 구조화된 concat 또는 multi-example

   각 실험에 대해:

   * exp_id
   * 변경되는 YAML 값 (특히 train_retrieval, top_k_retrieval, 관련 custom flag)
   * 기대되는 효과/위험요인
     을 짧은 표 형태로 정리해줘.

4. **평가 방법**

   * 각 Tier 1 실험에 대해 validation 기준:

     * val EM/F1
     * retrieval failure 케이스는 retriever 한계이므로 제외,
     * retrieval 성공 케이스 subset에서 Reader EM/F1 비교 등.
   * 이미 존재하는 `val_analysis` 아웃풋 구조
     (`val_*_analysis`, `val_hard_samples.csv`, `val_error_analysis.json` 등)를 활용해
     “어떤 지표를 봐야 이 전략이 성공/실패인지”를 정의해줘.

---

## [Tier 1.5 – 반드시 수행] doc_stride 튜닝

### 3.4 Tier 1.5 목적

* partial_superset/subset 에러가 꽤 많다.
* 과거 sparse+gold baseline에서 `doc_stride=128 → 64` 변경만으로 EM이 확 뛰는 경험이 있다.
* 현재 설정: `max_seq_length=384`, `doc_stride=128`.
* Tier 1의 retrieval-aware training과 결합하여,
  **span 경계 문제를 줄이는 방향**으로 stride를 줄이는 실험을 진행하고 싶다.

### 3.5 네가 설계해야 할 것 (Tier 1.5)

1. **doc_stride 후보 값 제안 & trade-off 분석**

   * 예: 128, 96, 64 (또는 네가 판단한 다른 값 포함 가능)
   * 각 값에 대해:

     * 평균적으로 question 하나당 몇 개의 span이 생기는지 (rough 추정),
     * GPU 메모리/학습 시간 증가량,
     * partial match 감소에 대한 직관적 기대
       를 설명해줘.

2. **Tier 1과 결합한 실험 매트릭스 제안**

   * 예시 (표 형태로 정리해줘):

     * Exp 1-1-128: train_top_k=1, doc_stride=128
     * Exp 1-1-64:  train_top_k=1, doc_stride=64
     * Exp 1-2-128: train_top_k=3, doc_stride=128
     * Exp 1-2-64:  train_top_k=3, doc_stride=64

   * 시간이 매우 부족하다는 가정 하에,
     “최소한 이 정도는 반드시 돌려야 한다”는 core 조합 2~4개를 추천해줘.

3. **YAML 수정 예시**

   * `configs/active/exp_xxx.yaml` 기준:

     * doc_stride 값을 어떻게 변경하면 되는지,
     * exp_name suffix를 어떻게 정의하면 관리가 쉬울지 (예: `_ds64`, `_ds96`)
   * 실제 YAML snippet을 작성해줘.

4. **평가 관점 정리**

   * stride 변경 시, 특히 partial_superset/subset 케이스에서 F1/EM이 어떻게 변하는지
     기존 `val_hard_samples.csv`, `val_error_analysis.json`과 연동해
     “어떤 에러 타입이 줄어드는지”를 보는 관점을 제안해줘.

---

## [Tier 3 – 중요] Inference top_k 튜닝 (train/infer 분리)

### 3.6 Tier 3 목적

* train에서는 context 길이/노이즈를 줄이기 위해 작은 k(예: 1~3)를 사용하고,
* inference/validation에서는 더 큰 k(예: 10 또는 20)를 사용하여 retrieval failure를 줄이고 싶다.
* 현재는 `data_args.top_k_retrieval` 하나로 train/val/test가 같이 움직이는 구조에 가깝다.

### 3.7 네가 설계해야 할 것 (Tier 3)

1. **train_top_k vs infer_top_k 분리 설계**

   * `DataTrainingArguments`에 다음과 같은 필드를 추가하는 식으로:

     ```python
     train_top_k_retrieval: Optional[int] = None  # None이면 top_k_retrieval와 동일
     infer_top_k_retrieval: Optional[int] = None  # None이면 top_k_retrieval와 동일
     ```

   * 그리고:

     * train 시 `retrieve_and_build_dataset(..., is_train=True)`에서는 `train_top_k_retrieval` 사용,
     * inference/validation에서는 `infer_top_k_retrieval` 사용,
     * 기존 `top_k_retrieval`은 default 역할만 수행.

   * 필요한 코드 수정 포인트:

     * `src/arguments.py` (DataTrainingArguments 추가 필드)
     * `train.py` – train_retrieval branch에서 top_k 선택 부분
     * `inference.py` – `load_retrieval_from_cache`/live retrieval 시 top_k 선택 부분

   * 이 수정이 실제로 어떻게 코드에 들어가야 하는지,
     pseudo code 또는 부분 코드 스니펫 수준으로 보여줘.

2. **실험 설계**

   * 예를 들어, Tier 1에서 가장 promising한 train 설정(예: train_top_k=3)을 고정하고
     inference에서만 top_k를 바꾸는 실험:

     * Exp 3-1: train_top_k=3, infer_top_k=10
     * Exp 3-2: train_top_k=3, infer_top_k=20

   * 필요하면 train_top_k=1인 모델에 대해서도 infer_top_k=10/20 비교 포함.

   * 각 조합에 대해, retrieval failure 감소 → EM 상한선 증가에 대한 rough 계산을 해줘
     (예: R@10=93.8%, R@20=96.3%라면 retrieval failure 18→9개로 줄어들고,
     Reader 효율이 그대로라고 가정했을 때 EM이 이론상 얼마나 늘어날 수 있는지 등).

3. **YAML 예시**

   * 실제 config에서:

     ```yaml
     top_k_retrieval: 10
     train_top_k_retrieval: 3
     infer_top_k_retrieval: 20
     ```

   * 이런 식으로 설정했을 때, 각 스크립트에서 값이 어떻게 사용되어야 하는지 설명해줘.

---

## [Tier 4 – 중요] Ensemble (여러 Reader 조합)

### 3.8 Tier 4 목적

* 여러 Reader 변종을 조합(앙상블)해서 마지막 0.5~1pt 정도를 더 끌어올리기.
* 각 모델은 이미:

  * `predictions_{split}.json`
  * `nbest_predictions_{split}.json`
  * `logits_{split}.json` (또는 유사 구조)
    를 저장하고 있다.

### 3.9 네가 설계해야 할 것 (Tier 4)

1. **실용적인 앙상블 스키마**

   * 간단하면서도 구현 가능한 방식으로 제안해달라:

     * 같은 question id에 대해 여러 모델의 n-best candidate를 모으고,
     * 동일 answer string에 대한 score를 합산/평균하거나,
     * 모델별 weight를 두고 weighted voting을 수행하는 방식 등.
   * 텍스트 normalization (공백/구두점 제거 등)도 적절히 포함.

2. **앙상블 조합 후보**

   * 예:

     * Ens-1: baseline gold-only train 모델 + best retrieval-train 모델 (2-model ensemble)
     * Ens-2: Ens-1 + curriculum stage2 모델 (3-model ensemble, 추후)
   * “에러 패턴이 서로 다른 모델끼리 조합하는 것이 좋다”는 관점에서,
     어떤 조합이 가장 유망할지 가설을 적어줘.

3. **구현 pseudo code / Python snippet**

   * 예:

     ```python
     # inputs: 여러 개의 nbest_predictions_test_modelX.json 파일 경로 리스트
     # output: ensemble_test_pred.csv

     def load_nbest(path):
         import json
         with open(path, "r", encoding="utf-8") as f:
             return json.load(f)  # {id: [ {text, prob, score, ...}, ...]}

     def normalize_answer(text):
         # 공백/구두점/대소문자 등 정규화
         ...

     def ensemble_nbest(nbest_paths, weights=None):
         # weights: 모델별 가중치 리스트 (None이면 균등가중)
         ...
     ```

   * 이 스크립트가:

     * 각 파일에서 `{id: [nbest entries...]}`를 읽고,
     * answer string을 key로 score를 누적/평균하고,
     * 최종 score가 가장 높은 후보를 선택하여 `{id, prediction}` CSV를 만드는 구조가 되도록,
       실제 Python 코드에 가깝게 작성해줘.

---

# 4. 추가로 고려해볼 수 있는 (Optional) Tier – Reranker

* 시간/리소스가 허락하면, retrieval 이후 top-50 후보에 대해 간단한 Cross-Encoder Reranker를 붙이는 것도 고려 중이다.
* 다만 이는 “지금 당장 반드시 할 일”은 아니고,
  상위 Tier 1/1.5/3/4 실험이 어느 정도 정리된 뒤에만 수행할 수 있다.

너는 이 reranker를 “Optional Tier 2” 정도로 간략히:

* 사용할 수 있는 한국어 기반 Cross-Encoder 후보 (예: klue-roberta 기반)
* 학습 데이터 구성 (train QA → (question, positive passage, negative passage) pairs)
* train/infer 시 코드에 어떻게 붙이면 되는지

를 high-level로만 제안해주면 된다.

---

# 5. 출력 형식 요청

최종 답변은 다음 구조를 따르길 원한다:

1. **요약**

   * 전체 전략(특히 Tier 1, 1.5, 3, 4)이 무엇인지 5~10줄로 정리

2. **Tier 1 – Retrieval-aware Training**

   * 가설
   * train context 구성 전략 후보(A/B/C 등), 장단점
   * 추천 전략 + 수정할 코드 포인트 + 코드 스니펫
   * Tier 1 실험 표 (exp_id, 설정, 기대효과)
   * 평가 지표/분석 관점

3. **Tier 1.5 – doc_stride 튜닝**

   * 후보 값/ trade-off
   * Tier 1과 결합한 실험 매트릭스
   * YAML 수정 예시
   * 어떤 에러 타입이 줄어드는지에 대한 가설

4. **Tier 3 – Inference top_k 튜닝**

   * train_top_k vs infer_top_k 분리 설계
   * 실험 조합 (train 설정 고정, infer_top_k만 변경)
   * YAML & 코드 변경 포인트
   * retrieval_failure 감소 → EM 상한선 증가에 대한 rough 계산

5. **Tier 4 – Ensemble**

   * 앙상블 전략 개요
   * 모델 조합 후보 (Ens-1, Ens-2 등)
   * 구현 pseudo code / Python snippet

6. **우선순위/로드맵**

   * 어떤 Tier/실험을 1순위, 2순위로 돌릴지,
   * 시간/자원 대비 기대 효과 기준으로 간략한 로드맵 정리

가능한 한 **지금 코드베이스(MRC repo)에 바로 적용 가능한 수준**으로 구체적인 제안을 해달라.
“이론적으로 ~할 수 있다”가 아니라,
**어떤 파일에서 어떤 함수를 어떻게 수정하고, 어떤 YAML을 새로 만들고, 어떤 CLI 명령을 돌리면 되는지**
까지 포함해서 답변해줘.

핵심부터 말하면,
**전체 전략 방향은 우리 둘이 정리했던 최종 그림과 잘 맞고, “승인해도 되는 수준”인데,
실전 투입 전에 꼭 짚고 넘어가야 할 리스크/수정 포인트가 몇 개 있습니다.**

그 포인트들만 보완하면, 이 설계서를 팀에 그대로 넘겨도 큰 문제 없다고 봅니다.

---

## 1. 전체 플랜에 대한 총평

좋은 점부터:

* 우리가 얘기했던 축 4개
  **(Tier 1: train_retrieval, Tier 1.5: doc_stride, Tier 3: train/infer top_k 분리, Tier 4: ensemble)**
  가 그대로 반영돼 있고, 우선순위도 현실적입니다.
* “코드 어디를 어떻게 고칠지”가 함수/파일 단위로 짚혀 있어서,
  현재 레포에 꽂기가 실제로 가능해 보입니다.
* Tier 1을 **Top-1 → 이후 Top-3 concat** 순서로 가져가는 것도,
  시간과 위험도 고려하면 나쁘지 않은 순서입니다.
* 앙상블 스크립트도 실제로 돌아갈 모양으로 나와 있고, 나중에 바로 돌리면 됩니다.

다만, **내가 꼭 짚고 싶은 핵심 리스크는 딱 두 가지**입니다.

1. **Top-1 retrieval 기반 train에서 “얼마나 많은 train 샘플이 날아가는지”**
2. **inference 코드 snippet에서 test 분기/데이터셋 지정 및 캐시 로직 살짝 의심 구간**

이 두 가지는 실제 코드에 넣기 전에 한 번 더 체크하고 들어가는 게 좋습니다.

아래에서 Tier별로 짧게 정리해 볼게요.

---

## 2. Tier 1 – Retrieval-aware Training

### 2.1 개념/방향성 타당성

* “train에서도 retrieval context로 학습해서 **train–inference distribution gap 줄인다**”
  → 우리가 계속 말하던 그 포인트랑 100% 일치합니다.
* DHN은 이번 라운드에서 끄고, **순수한 retrieval-aware training**만 보는 것도 합리적입니다.
* 전략 A/B/C 정의 자체도 괜찮습니다.

  * A: Top-1 only (simple, low risk)
  * B: Top-3 concat (실제 inference 분포와 더 유사)
  * C: multi-example (시간/리스크 때문에 pass)

### 2.2 Top-1 전략(A)에 대한 진짜 핵심 리스크

설계서에는 “R@1 ≈ 62% → 필터링 ~7%” 같은 느낌으로 쓰여 있는데,
우리 예전 리콜 테이블 기억하면:

* KoE5 dense 기준 R@1이 46–49% 정도였고,
* hybrid로 올려도 R@1이 **60–70% 이상**이라고 장담하긴 어려운 상태였죠.

그래서 실제로는:

* **Top-1 train_retrieval + `realign_answers` + filter**를 하면
  `lost_ratio`가 7%가 아니라 **30~40%까지도 튈 수 있습니다.**

즉, **train 데이터를 3~4천 개 중 1~2천 개만 쓰는 사태**도 가능해서,
이건 그냥 “retrieval-aware”를 넘어서 **데이터 부족** 문제가 돼버립니다.

> 그래서 “Top-1 only 전략을 완전히 베이스라인으로 박는 건 약간 위험하다”가 내 의견입니다.

다만, 이건 로그 한 번 찍어보면 바로 알 수 있습니다.
이미 `retrieve_and_build_dataset`에 lost ratio 로그를 넣는 설계가 있으니,
**한 번만 돌려보고 lost_ratio > 25~30% 나오면, Top-1 단독 전략은 버리고 바로 Top-3로 가는 게 좋습니다.**

→ 정리하면:

* **계획 자체는 타당하지만, R@1에 대한 낙관 가정이 섞여 있다.**
* 승인 시, “Top-1 lost_ratio를 반드시 check하고, 25~30% 이상이면 Top-3 전략으로 즉시 스위치”라는 조건을 붙이는 게 안전.

### 2.3 코드 설계 자체는 괜찮은지

* `DataTrainingArguments`에 `train_top_k_retrieval`, `infer_top_k_retrieval` 추가
  → 기존 YAML에 없는 필드는 default=None으로 잘 fallback 되니, **파싱 깨질 일은 없음**.
* `retrieve_and_build_dataset`에서 `is_train` flag로 top_k 분리
  → 구조도 깔끔하고, 나중에 Tier 3에도 그대로 재사용 가능.
* `train.py`에서 `train_retrieval` 분기로 `datasets["train"]` 교체
  → 지금 네 코드 구조와 잘 맞는 흐름입니다.
* `realign_answers_in_retrieved_context`는

  * Top-1 단일 passage에도 문제 없고,
  * Top-k concat에서도 `.find(answer_text)`만 맞으면 동작하므로,
    **sep 토큰/타이틀을 넣어도 크게 문제는 없습니다.**

다만 두 가지 주의:

1. **`retrieve_and_build_dataset` 안에서 `retriever.retrieve(..., tokenizer=tokenizer)` 인자 시그니처**
   → 네 현재 BaseRetrieval 구현이 이 시그니처를 받는지 한 번 더 확인해 두면 좋습니다.
   (이미 그렇게 쓰고 있는 걸로 알고 있지만, 안전 차원에서)

2. **realign 실패 케이스(log)**

   * plan대로 `filter`를 하고 있지만,
   * debug용으로 `lost_count` 외에 `예: 앞 5개 예시` 정도는 로그/CSV로 찍어두면
     segmentation 이유인지, retrieval failure인지 감이 잡히고,
   * 필요한 경우 gold context와 비교해서 더 미세하게 대응할 수 있습니다.

### 2.4 “train에서 concat이 너무 단순한 거 아닌가?” 걱정과의 정합성

너가 처음 말한 고민이 이거였죠:

> “현재 단순 concat이 안 좋을 수도 있을 것 같다. train에서 어떻게 학습시키느냐가 핵심이다.”

Opus 설계서는:

* **Tier 1 기본은 Top-1 only (멀티 passage X)**
* 그 다음 단계로 Top-3 concat을 “추가 실험”으로 둔 상태.

즉, **우리가 처음 말하던 ‘multi-passage train’ 문제는 Tier 1-2 이후에야 제대로 커버**되는 구조입니다.

나는 이렇게 보는 게 합리적이라고 생각합니다:

* 시간/리스크 때문에,
  **“(1) Top-1 train_retrieval로 일단 retr-aware 환경에서 학습이 잘 되는지 확인 → (2) 여유되면 Top-k concat으로 분포 더 맞추기”**
  라는 2단계 전략은 현실적인 타협입니다.
* 단, 위에서 말한 것처럼 **Top-1에서 데이터가 너무 많이 날아가면**
  이건 실험이 아니라 **데이터 수 감소 효과**에 가까워지니,
  그때는 바로 Top-3 concat을 Tier 1의 메인 실험으로 승격시키는 게 맞습니다.

이 조건만 명확히 깔고 간다면,
**너가 걱정하던 “너무 단순한 concat” 문제와도 크게 충돌하지 않는 방향**이라고 볼 수 있습니다.

---

## 3. Tier 1.5 – doc_stride 튜닝

이건 솔직히 **가장 확실하고, 리스크가 적고, 이미 네 경험으로 검증된 축**이라서
계획 내용 그대로 승인해도 된다고 봅니다.

* `doc_stride=128 → 64`가 partial_superset/subset 줄이는 데 효과가 있다는 건
  이전 sparse+gold 실험에서 이미 본 패턴이고,
* 현재 데이터 사이즈(4k 미만)에 RoBERTa large라도
  **배치 줄이고 gradient_accumulation 넣으면 충분히 감당 가능한 수준**입니다.

주의할 점:

* Top-3 concat과 `doc_stride=64`를 함께 쓰면
  feature 수가 꽤 많이 늘어날 수 있으니,

  * 순서는 **Top-1 + ds64**를 먼저 보고,
  * 그다음에 시간이 남을 때 **Top-3 + ds64**로 가는 것이 안전합니다.

이 부분은 설계서의 우선순위 표랑도 대체로 잘 맞습니다.

---

## 4. Tier 3 – Inference top_k 튜닝

기획 자체는 깔끔합니다.

* `train_top_k` / `infer_top_k` 분리 →
  우리가 원하던 “train은 noise 줄이고, infer는 recall 늘리기” 그대로입니다.
* `load_retrieval_from_cache`에 `top_k` 인자를 추가하고,
  캐시에 저장된 top-50 중 상위 top_k만 쓰는 구조도 합리적입니다.

여기서 내가 딱 하나 걸리는 부분:

### 4.1 inference.py snippet 중 dataset 인자

snippet에 이런 부분이 있었죠:

```python
if os.path.exists(test_cache_path):
    new_test_dataset = load_retrieval_from_cache(
        cache_path=test_cache_path,
        dataset=datasets["validation"],
        ...
    )
```

* 이게 실제 코드 구조와 다를 수 있지만,
  **“test split인데 validation dataset을 넘기는”** 부분은
  오타일 가능성이 높습니다.
* 실제 네 코드에서는 아마 `datasets["test"]` 혹은
  `datasets[inference_split]`을 넘기는 게 맞을 겁니다.

→ **승인 전에 이 부분은 꼭 네 실제 inference.py 열어 보고 맞춰줘야 합니다.**
(이건 계획의 논리가 아니라 단순 실수/타이핑 영역)

### 4.2 R@20 기반 EM 상한선 계산

* “이론상 +2.5pt” 이런 숫자는 계산 방법은 맞지만,
  **실제 EM 이득은 보통 이보다 작게 나옵니다.**

  * retrieval failure 케이스가 원래 reader에게도 어려운 케이스일 확률이 높고,
  * 추가 passage가 noise로 작용할 수도 있으니까요.
* 그러니 이건 **상한선** 정도로 받아들이고,
  “+1~2pt 정도 기대할 수 있다” 정도로 현실적으로 생각하면 될 것 같습니다.

이 부분은 계획을 바꾸라는 얘기가 아니라,
우리가 **기대치를 너무 공격적으로 잡지 말자**는 의미입니다.

---

## 5. Tier 4 – Ensemble

전체적으로 무난하고, 실용적인 설계입니다.

좋은 점:

* `nbest_predictions` 기반 score voting → 구현 난이도 대비 효과 좋은 방식.
* answer normalization도 과하지 않고 현실적인 수준입니다.
* 스크립트 구조(입력 경로, output_path, weights)는 실전에 바로 쓸 수 있게 설계돼 있습니다.

체크 포인트:

1. **`nbest_predictions` JSON의 실제 필드 이름**

   * 네 현재 `postprocess_qa_predictions`가
     `probability` / `score` 중 어떤 키를 쓰는지 반드시 확인해야 합니다.
   * 실제 코드에서 `score_key="probability"`로 되어 있는데,
     네 JSON이 `probability`가 없고 `prob`나 그냥 `score`만 있다면
     그대로 쓰면 다 0점이 됩니다.
   * 이건 단순히 한 번 `head nbest_predictions_val.json` 찍어보면 해결.

2. **제출 포맷**

   * 스크립트가 `\t` 구분자로 `id, answer`를 쓰는데,
     기존 제출 스크립트/대회 포맷이 이와 완전히 일치하는지 한 번만 확인하면 됩니다.

이 두 가지만 확인하면, Tier 4는 거의 “공짜 점수”에 가깝습니다.

---

## 6. 우선순위 / 로드맵에 대한 코멘트

설계서의 로드맵:

1. 1-1-128
2. 1-1-64
3. 1-2-128
4. 1-2-64
5. 3-2 (infer_top_k=20)
6. Ens-1

이 순서는 “실험 설계 관점”에서는 깔끔한데,
**실제 대회 마감까지 남은 시간**을 생각하면 한 가지 선택지가 있습니다.

### 선택지 A (지금 설계서 그대로 진행)

* 전제:

  * Top-1 lost_ratio가 생각보다 크지 않다.
  * 코드 수정에 큰 삽질이 없다.
* 이 경우 설계서 그대로 진행해도 좋습니다.
* 단, **1-1-128은 “필수”라기보다는 sanity check**에 가깝기 때문에,
  시간 없으면 1-1-128은 생략하고 1-1-64만 바로 가도 됩니다.

### 선택지 B (안전 위주 단축 버전 – 만약 진짜 시간이 빡빡하다면)

만약 너가 “오늘 밤에 크게 삽질할 여유 없다”고 느낀다면,
나는 아래 같이 살짝 리스케줄링 하는 것도 고려해볼 만하다고 봅니다:

1. **doc_stride 64 먼저 (Top-1 train_retrieval 없이)**

   * 이미 baseline 코드에서 doc_stride=64 효과를 봤으니,
     현재 구조에서 `doc_stride=64`만 바꾼 실험 하나를 바로 돌린다.
2. 그다음에 **Top-1 train_retrieval + ds64**를 한 번에 적용 (1-1-64에 해당)

   * 여기서 lost_ratio를 확인하고,
     너무 크면 Top-3 concat을 곧바로 메인으로 승격.
3. 그 후 **infer_top_k=20** 돌리기 (학습 없이 inference만)
4. 마지막으로 **Ens-1**

이건 “Tier 1을 반드시 한다”는 원칙은 유지하되,
**doc_stride 64라는 이미 검증된 카드의 효과를 최대한 빨리 확보**하는 쪽입니다.

---

## 7. 결론: 승인 여부에 대한 내 의견

정리하면:

* **전략 방향**: 우리가 길게 논의한 내용과 상당히 잘 맞고,
  Tier 1/1.5/3/4의 역할 분담도 명확해서 **개념적으로 타당**합니다.
* **코드 레벨 설계**:

  * `DataTrainingArguments` 확장
  * `retrieve_and_build_dataset` / `train.py` / `inference.py` 수정 포인트
    모두 현재 레포 구조와 어긋나지 않고, 구현 가능성이 높습니다.
* **주의해야 할 지점**:

  1. Top-1 train_retrieval에서 **lost_ratio 꼭 확인**
     → 25–30% 이상이면 Top-3 전략을 메인으로 승격.
  2. inference.py snippet의 `dataset=` 인자 및 캐시 로직을
     **실제 코드 기준으로 한 번 더 맞추기**.
  3. ensemble 스크립트에서 **실제 nbest JSON의 score 필드 이름 확인**.

이 세 가지만 머릿속에 체크하고 간다면,

> “약간의 주석/조건을 붙인 형태로 승인해도 된다”가 내 최종 의견입니다.

즉,
**“이 설계서를 그대로 팀에 공유하되, 위 3가지 리스크는 별도 메모로 적어두고,
첫 실험 돌릴 때 반드시 체크하자”** 정도로 정리해서 승인하면 괜찮을 것 같습니다.

좋아요, 이제 숫자가 보이니까 훨씬 얘기하기 편하네요.
일단 네가 잡은 baseline을 기준으로, “어떤 모델/세팅을 메인으로 삼고, 그 위에 어떤 Tier 실험을 먼저 얹을지” 관점에서 우선순위를 정리해볼게.

---

## 1. 지금 baseline 테이블 해석부터 정리

Top 3만 보면:

1. `roberta_large_vanilla`

   * EM 61.67 / F1 68.77 / train 10.4분

2. `oceann315`

   * EM 61.67 / F1 70.01 / train 21.7분

3. `HANTAEK_roberta_large_vanilla`

   * EM 61.25 / F1 70.03 / train 10.4분

여기서 중요한 포인트는:

* **F1 기준으론 oceann315 ≒ HANTAEK (둘 다 70 근처)**
* **학습 시간은 HANTAEK ≈ roberta_large_vanilla(10분대) < oceann315(21분대)**
* `kobigbird`, `koelectra` 등은 EM/F1 둘 다 확실히 밀리므로 **지금 시점에서 추가 투자할 가치 거의 없음**.

그래서 “이 위에 Tier1/1.5/3/4를 얹을 메인 베이스”는 이렇게 잡는 게 합리적이라고 본다:

* **메인 베이스 모델(P0): `HANTAEK_roberta_large_vanilla`**

  * KorQuAD 특화 + 짧은 학습시간 + 이미 여러 레포에서 검증된 조합.
* **서브 베이스(P1, 앙상블용): `oceann315`**

  * 성능 거의 같고, 표현이 살짝 다를 가능성이 있어 앙상블에서 이득 기대.
* **그 외 모델: 현재로선 “성능 비교용 기록만 유지”하고 추가 실험은 중단.**

---

## 2. 이 기준에서 “실험 방향” 우선순위

이제, 이 베이스라인 위에서 **어떤 실험을 먼저/나중에 할지**를 다시 정리하면:

### 우선순위 1: HANTAEK 기준 골드-only + doc_stride 검증 (아주 짧게)

> 목적: “Reader 자체 상한선”을 확인하는 골드-only 기준선.

* 설정

  * `HANTAEK` + gold context만 사용 (지금까지 쓰던 vanilla 세팅)
  * `doc_stride=64` vs `128` 비교 (이미 sparse+gold에서 EM 확 뛴 경험 있음)
* 이유

  * 이 실험은 **retrieval 영향 없는, 순수 Reader 능력의 상한선**을 다시 한 번 박아두는 역할.
  * train/infer 모두 gold일 때 EM이 어디까지 나오는지 대략 감을 잡아야,
    뒤에서 “retrieval-aware training + noisy context”가 얼마나 손해/이득인지 판단 가능.

→ 이미 비슷한 실험을 했다면 **결과만 팀에 공유**하고, 추가 학습은 안 해도 됨.

---

### 우선순위 2: Tier 1 + 1.5 핵심 – HANTAEK + Retrieval-train + `doc_stride=64`

> 여기부터가 진짜 “점수 올리는” 구간.

1. **Exp A (최우선)**

   * Base: `HANTAEK_roberta_large_vanilla`
   * 설정

     * `train_retrieval = true`
     * `train_top_k_retrieval = 1` (Top-1 only)
     * `infer_top_k_retrieval = 10`
     * `doc_stride = 64`
   * 체크포인트

     * `retrieve_and_build_dataset` 로그에서 **lost_ratio(필터링 비율)** 꼭 확인

       * 예: `Original: 3952 → Filtered: 3200 (lost 19.0%)`
     * lost_ratio가 **25~30% 이상**이면 → Top-1만으로는 학습 데이터 너무 많이 잃는 것이므로,

       * “Top-1 only”는 실험용으로만 두고,
       * **Top-3 concat을 메인 전략으로 승격**하는 판단 필요.

2. **Exp B (Top-3 concat 버전, 상황 보고 진행)**

   * Base: 동일 (`HANTAEK`)
   * 설정

     * `train_retrieval = true`
     * `train_top_k_retrieval = 3` (Top-3 concat)
     * `infer_top_k_retrieval = 10`
     * `doc_stride = 64`
   * 역할

     * Top-1 lost_ratio가 높았을 때 대안 / 혹은 최종 모델 후보.

> 요약하면:
> **HANTAEK + train_retrieval + ds=64**가 1순위,
> 그 안에서 **Top-1 vs Top-3**를 lost_ratio와 EM/F1을 보고 선택.

---

### 우선순위 3: Tier 3 – Inference `top_k` 튜닝 (학습 없이 빨리 하는 구간)

> 목적: retrieval failure 줄여서 EM 상한선 자체를 밀어 올리기.

* 대상: **위에서 가장 잘 나온 모델 1~2개** (예: Exp A, Exp B)
* 설정 예:

  * `train_top_k_retrieval = 1` or 3 (이미 학습된 값 유지)
  * `infer_top_k_retrieval = 20`로 바꿔서 **inference만 재실행**
* 이유

  * 이미 분석한 대로, R@20 기준으로 retrieval failure가 7.5% → 3.75%로 줄어들면,
    Reader 성능이 그대로여도 **이론상 EM +2~3pt**까지 가능.
  * **학습이 아니라 inference만 돌리면 되기 때문에, 시간 대비 효율이 매우 좋음.**

→ 정리: **“좋은 모델 1개 뽑히면 바로 top_k=20 inference 한 번은 무조건 돌린다.”**

---

### 우선순위 4: Tier 4 – Ensemble (Vanilla + Retrieval-train)

> 목적: 서로 다른 에러 패턴을 가진 모델 조합으로 마지막 0.5~1pt 뽑기.

* 조합 추천

  1. `HANTAEK_roberta_large_vanilla` (gold-only baseline)
  2. `HANTAEK + train_retrieval + ds=64` (베스트 실험)
  3. (여유 시) `oceann315` 변종 추가
* 로직

  * `nbest_predictions_test.json` 기반 score-voting (이미 스니펫 설계되어 있음)
* 타이밍

  * **대회 막판에 “학습 다 끝난 후”에만 한 번 잘 세팅해서 사용.**
  * 중간에는 실험 자원/시간을 Reader 개선 쪽에 집중.

---

## 3. “모델/실험” 관점에서 우선순위 정리 (한눈에)

**모델 축 우선순위**

1. **HANTAEK_roberta_large_vanilla**

   * 모든 Tier 1/1.5/3 실험의 기본 베이스.

2. **oceann315**

   * 별도 retrain까지는 옵션,
   * 앙상블용으로 최소한 gold-only(or 간단 세팅) 버전 하나 유지.

3. 그 외 (uomnf97, kobigbird, koelectra…)

   * **현재 성능 기준으로는 추가 실험 보류.**

---

**실험 축 우선순위**

1. **P0 (짧게):**

   * HANTAEK + gold-only + `doc_stride=64` 상한선 확인 (이미 했다면 스킵하고 결과만 공유)

2. **P1 (메인):**

   * HANTAEK + `train_retrieval=true` + `train_top_k=1` + `doc_stride=64`
   * lost_ratio, val EM/F1 분석

3. **P2 (상황에 따라):**

   * lost_ratio 높거나 EM이 기대만큼 안 나오면
     → HANTAEK + `train_top_k=3` + `doc_stride=64` 재학습

4. **P3 (값어치 대비 싸게):**

   * 위 P1/P2 중 최고 모델 대상으로
     `infer_top_k=20` inference-only 실험

5. **P4 (마무리):**

   * gold-only HANTAEK + best retrieval-train 모델 + (옵션: oceann315) 앙상블

---

## 4. 한 줄로 정리하면

> **“베이스 모델은 HANTAEK로 고정하고,
> 먼저 train_retrieval + doc_stride=64로 Reader를 끌어올린 뒤,
> inference에서 top_k만 키워 상한선을 밀고,
> 마지막에 gold-only + retrieval-train 앙상블로 마무리하자.”**

이 우선순위로 가져가면,
지금 네 baseline 결과(EM 61.6 ±)를 자연스럽게 발판으로 쓰면서
리더보드 70 근처까지 노리는 흐름이 잘 맞는다고 봐도 될 것 같아.


<!-- 네거티브 샘플과 configs/new_active에 있는 커리큘럼 러닝 -->
결론부터 말하면:

**“완전 새로 짜야 하는 수준은 아니고, 지금 파이프라인에서 거의 그대로 돌릴 수 있다.
다만 `train_dataset_name` 쪽 실제 데이터 포맷만 한번 확인하면 된다.”** 쪽에 가깝습니다.

---

## 1. 지금 올라온 YAML들의 상태 진단

실제로 파일들 까보면:

* 전부 맨 위에 `# base.yaml` / `# arguments.py에 지정된 dataclass 형식` 주석이 있고
* 필드 구조가 **지금 너가 쓰는 통합 YAML 포맷과 동일**합니다.

예시 (실제 내용):

* `model_name_or_path`
* `train_dataset_name`, `validation_dataset_name`, `infer_dataset_name`
* `max_seq_length`, `doc_stride`, `per_device_train_batch_size`, `learning_rate` …
* `eval_retrieval`, `retrieval_type`, `top_k_retrieval` 같이 **지금 파이프라인에서 쓰는 Retrieval 관련 필드도 이미 들어 있음**
* 2단계 학습도 이미 반영:

  * `roberta-large1.yaml` → `model_name_or_path: klue/roberta-large`
  * `roberta-large2.yaml` → `model_name_or_path: ./outputs/taewon/roberta1`

즉, “예전 완전 다른 베이스라인 포맷”이 아니라,
**지금 ML 파이프라인의 `base.yaml + arguments.py(dataclass)` 설계에 맞춰 작성된 YAML**들이라고 보면 됩니다.

그래서:

* HfArgumentParser + `get_config` 경로에 그대로 태워도
* 필드 이름 때문에 바로 깨질 가능성은 낮습니다.

---

## 2. 진짜 체크 포인트는 여기 하나

네가 짚은 것처럼, 핵심은 사실 이거 하나입니다.

> `train_dataset_name`이
>
> * 1단계: `./data/train_dataset` (혹은 `train_dataset_clean`)
> * 2단계: `./data/train_dataset_with_negatives`
>   로 갈려 있는 부분이 “지금 파이프라인에서도 그대로 유효하냐?”

이걸로 갈리는 건:

1. **데이터 로딩 코드에서 해당 경로/이름을 인식하느냐**

   * 예: `load_from_disk(train_dataset_name)` / `load_dataset(...)` 등
2. **`*_with_negatives` 버전의 컬럼 구조가 지금 파이프라인이 기대하는 포맷과 같냐**

   * 최소 요구:

     * `id`, `question`, `context` (또는 `context`가 gold로 고정)
     * `answers` (text, answer_start)
   * 만약 팀원이 커리큘럼용으로 컬럼을 하나 더 추가했다면:

     * 지금 파이프라인이 그 컬럼을 “무시해도 상관없는지” or
     * 해당 컬럼을 명시적으로 사용하는 코드가 이미 들어가 있는지

이 두 가지만 맞으면,

> **현재 파이프라인에서 이 YAML들을 “거의 그대로” 끼워 넣어서
> 커리큘럼 러닝 1단계/2단계 모델을 학습시키는 건 충분히 가능하다**

라고 봐도 됩니다.

---

## 3. 요약: 된다 / 안 된다

질문 스타일대로 요약해보면:

* **된다 쪽에 가깝다 (✅)**

  * YAML 포맷, 필드명, 구조가 지금 ML 파이프라인 설계(`base.yaml + arguments.py`)와 이미 일치
  * 2단계 학습도 `model_name_or_path: ./outputs/...` 식으로 잘 연결돼 있음
  * 바꿔야 할 건 주로 `output_dir` 네이밍, `train_dataset_name` 실제 경로 정도

* 다만, **반드시 한 번 확인해야 할 부분 (체크리스트)**:

  1. `train_dataset_name` / `validation_dataset_name`이 가리키는 실제 데이터가
     지금 파이프라인의 `DataTrainingArguments` + 데이터 로딩 함수에서 **동일 포맷**으로 처리 가능한지
  2. `*_with_negatives`가 단순히 “더 많은 negative context가 섞인 KorQuAD 형식”인지,
     아니면 **커리큘럼 전용 커스텀 컬럼**이 있는지

이 두 가지만 OK면:

> “팀원이 만들었던 `roberta-large1/2.yaml`, `oceann1/2.yaml` 같은 설정을
> 지금 네 ML 파이프라인에서 **거의 그대로 재현 + 확장**하는 것”은 충분히 현실적인 플랜이다.

그래서 한 줄로 답하면:

> **“웬만하면 된다.
> 단, `train_dataset_name`이 가리키는 실제 데이터 포맷만 확인하고 쓰자.”**
