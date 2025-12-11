dhn에 대한 결과인데 보고 함께 분석해줄래?

또 현재 baseline으로 정한 모델 성능과 사후 분석을 통해 "짧은 시간"내에 확실하게 성능 올릴 수 있는 가설 - 실험 구조를 제안해줘

가능해! 지금 준 5개 중에서 내가 직접 읽을 수 있는 건

* `val_analysis_summary.md`
* `val_error_analysis.json`

이고, 나머지 3개 CSV는 요약 정보가 요 문서 안에 다 들어가 있어서, 그걸 기반으로 좀 더 “정량적인 사후분석”으로 정리해볼게.

---

## 1. 숫자로 보는 현재 상태 정리

### 1) Validation vs Leaderboard

* **Val (retrieval + vanilla reader)**

  * EM **61.25**, F1 **70.00** 
* **Leaderboard (같은 세팅)**

  * EM **65.42**, F1 **75.77**

→ **리더보드가 val보다 약 4.2 EM, 5.7 F1 높음**
→ test 분포가 조금 더 “친절”하거나, retrieval/reader 세팅이 test에서 잘 맞아떨어졌다고 볼 수 있음.
(일단 “val가 심하게 overfit된 건 아니다” 쪽에 가까움)

### 2) 정답이 context 안에 있을 때 vs 없을 때

* Gold가 Top-10 안에 있는 경우: **222/240 (92.5%)** 
* 없는 경우: **18/240 (7.5%)**

조건별 EM:

| 조건                 | EM        |
| ------------------ | --------- |
| Gold in Top-10     | **64.41** |
| Gold NOT in Top-10 | **22.22** |



👉 현재 **61.25 EM**은 대략 이렇게 쪼개짐:

* “retrieval이 gold를 찾아줬는데 reader가 틀린 케이스”에서 **약 28~30 EM 손실**
* “retrieval이 아예 gold를 못 찾은 7.5% 케이스”에서 **약 5.8 EM 손실**

---

## 2. 오답 93개는 어디서 새고 있나?

총 240 중 **147 정답 / 93 오답** 

오답 93개를 타입별로 보면: 

| Error Type        | 개수 | 전체 대비 | Wrong 중 비율(대략) | 의미                    |
| ----------------- | -- | ----- | -------------- | --------------------- |
| completely_wrong  | 46 | 19.2% | **약 49%**      | 완전 다른 스팬 (F1 < 20)    |
| partial_subset    | 16 | 6.7%  | 약 17%          | 정답의 부분집합 (짧게 뽑음)      |
| partial_superset  | 10 | 4.2%  | 약 11%          | 정답을 포함한 더 긴 스팬        |
| partial_overlap   | 3  | 1.2%  | 약 3%           | 일부만 겹침 (F1 ≥ 50)      |
| low_overlap       | 4  | 1.7%  | 약 4%           | 겹치긴 하는데 애매 (F1 20–50) |
| retrieval_failure | 14 | 5.8%  | **약 15%**      | gold context 미포함      |

**요약하면:**

* **리더가 완전 틀린 케이스(completely_wrong)**:
  → 오답의 **절반(약 50%)**
  → retrieval은 gold를 찾아줬는데, **질문 이해 / 엔티티 선택 / 논리적 추론에서 틀린 경우**가 많음. 

* **스팬 경계 문제(부분정답)**:
  → partial_subset + superset + overlap 류 합치면 **29개 (약 12.1%)** (low_overlap까지 포함하면 33개, 13.8%)
  → **EM만 까먹는 케이스**. F1은 꽤 높게 나옴. 예:

  * `작가 베게티우스` vs `베게티우스` 
  * `옥음방송` vs `쇼와 천황의 옥음방송` 

* **retrieval_failure**:
  → 14개 (5.8%)지만, 이건 “정답이 context에 없으니 맞출 수 없는” 하드 리밋. 

---

## 3. “이론상” 어디까지 올라갈 수 있나? (Val 기준)

완전 비현실적인 상한선이지만, 대략적인 **개선 여지의 크기**를 보려고 계산해보면:

1. **Retrieval을 100%로 만든다고 가정**

   * retrieval_failure 14개가 모두 맞는다고 하면
   * EM + (14 / 240 × 100) ≈ **+5.8**
   * → 61.25 → **최대 ~67점** 가능

2. **Span 경계 문제(29개 partial)를 전부 EM 1로 만든다고 가정**

   * EM + (29 / 240 × 100) ≈ **+12.1**
   * → 61.25 → **~73.3점**

3. 둘 다 동시에 완벽하게 해결한다고 치면 이론상 **~79점**까지도 가능하지만,

   * retrieval_failure랑 partial가 일부 겹칠 가능성도 있고
   * 완전 틀린 46개(completely_wrong) 중 일부는 구조적으로 매우 어렵기 때문에
     → 현실적으로는 “좋아봐야 고 60대~저 70대” 정도가 ceiling일 가능성이 큼.

**현재 Leaderboard EM 65.42는**
→ 이 그림에서 보면 **“retrieval failure 조금 줄고 / 일부 partial 케이스에서 운 좋게 EM이 맞은 상태”에 해당**하는 느낌이야.

---

## 4. 에러 타입별로 조금 더 구체적으로 보면

### 4-1. Retrieval Failure 패턴

예시 몇 개만 봐도 성격이 드러남: 

1. **마리아 테레사 문제**

   * Q: 루이 14세의 왕비 마리아 테래사는 어느 나라 공주인가?
   * Gold: `스페인`
   * Retrieved: 다른 “마리아 ~” 계열 공주 문서들
   * Pred: `오스트리아`

   → **질문은 왕비/공주 “역할” + 국가**,
   → Dense가 “마리아 + 공주” 의미에 강하게 반응하면서 **근처 비슷한 인물 문서로 튀는** 전형적인 Dense retrieval 실패.

2. **‘광배’ 케이스 (청양 읍내리 석조여래삼존입상)** 

   * Gold title은 맞는데, retrieval이 다른 석상 문서들을 top에 올리며 헷갈리는 경우
   * Pred에서 `광배(光背)`와 같이 한자 표기를 포함하면서 평가 상 F1=0이 나오는 특수 케이스도 있음.

→ 여기서 보이는 특징:

* **이름/개체가 비슷한 문서들 사이에서 헷갈리는 Dense의 약점**
* 일부는 “document level은 맞지만 passage slicing 위치가 어긋난” 쪽일 가능성도 있음 (chunking을 나중에 쓰면 더 중요해짐)

### 4-2. Partial Match 패턴

Partial 상위 5개만 봐도 거의 전부 **Span 경계 문제 + annotation 스타일 차이**다: 

* `"옥음방송"` vs `"쇼와 천황의 옥음방송"` → 의미적으로 완전 정답
* `'진전(陳田)'이라 새겨진 기와조각` vs `기와조각`
* `작가 베게티우스` vs `베게티우스`
* `매년 음력 정월` vs `음력 정월 초하룻날`

여기서 알 수 있는 건:

* **모델이 중요한 핵심 엔티티는 제대로 집어내고 있음.**
* 다만,

  * gold는 “좀 더 긴 설명형 답변”
  * 모델은 “핵심 명사만 뽑거나, 앞/뒤 수식어를 붙이거나/빼 버리는” 경향.

→ 이건 **모델의 span 선택 정책 & doc_stride / max_seq_length / 학습된 KorQuAD 스타일**이 섞인 문제고,
metric이 EM이라 **한 글자라도 다르면 0점**이 되는 구조라 “체감보다 EM 손해가 크다”고도 볼 수 있음.

### 4-3. Completely Wrong 패턴

Top 5만 봐도 느낌 딱 옴: 

* `1965년` vs `1963년` (년도 혼동)
* `아리크 부케` vs `카라 훌레구` (비슷한 정치/역사 인물 헷갈림)
* `미타케성` vs `이이노야 성` (비슷한 지명/성 이름 헷갈림)
* `박트리아` vs `밀라노`
* `2011년 3월 19일` vs `2008년 9월 22일` (날짜 다른 이벤트)

공통점:

* context 안에 유사한 개체/날짜/지명이 **여러 개 나열되어 있을 때**,
* reader가 “질문에 특화된 조건(예: ‘다시 해산했던 년도’, ‘쿠릴타이에서 대칸의 지위를 얻은 사람’, ‘전쟁이 났을 때 살던 곳’ 등)을 제대로 follow하지 못하고, 그냥 가장 눈에 띄는/최근의 엔티티를 찍는 경향.

이건 span boundary라기보다는 **질문 이해 + reasoning + multi-sentence alignment 문제**에 가깝고,
단기간에 해결하기 가장 어렵지만, “큰 EM 점프”는 이쪽에 숨어있음.

---

## 5. Leaderboard 점수와의 관계

지금까지를 정리하면:

1. **Gold-only 학습/평가 시 Reader 능력**

   * 과거에 말해준 것 기준으로, gold context만 줄 때 EM ≈ 73 근방 (validation)

2. **이번 설정(Weighted Hybrid Retrieval + top_k=10 + vanilla reader)**

   * Val: EM 61.25 / F1 70.00
   * Test: EM 65.42 / F1 75.77

3. EM 손실 구조 (rough):

   * Gold-only 73 → Retrieval+Reader 61

     * 약 **−12 EM** 손실이 생김
     * 이 중 **~5.8 EM**은 retrieval failure에서 오는 **불가피한 손실**
     * 나머지 **~6~7 EM 정도는**

       * multi-passage concat 상황
       * span 경계가 더 복잡해짐
       * test 분포/어휘 차이
         등의 복합 효과라고 볼 수 있음.

4. 그런데 리더보드는 65.4라서,

   * “실제 test set에선 retrieval이 val보다 조금 더 잘 맞았거나,
     partial match가 조금 더 많이 EM으로 잡힌다” 정도의 효과가 있었던 것 같고,
   * **현재 파이프라인이 “망했다” 수준은 전혀 아니고, 꽤 건실한 baseline+ 개선** 상태라고 보는 게 맞음.

---

## 6. 정리: 현재 사후분석에서 읽히는 진짜 포인트

**1) Retrieval은 “상한선”을 90대 중반까지 올려줬다.**

* Gold in Top-10 92.5%, EM 상한선도 그 근처.
* Retrieval failure가 5~6점 정도의 EM을 먹고 있지만,
  지금 시점에서 가장 큰 병목은 **“retrieval이 찾고 난 뒤의 reader”** 쪽.

**2) Reader의 에러는 크게 3층으로 나뉜다.**

* (a) **완전 오답 (46개; 전체의 19.2%)**

  * 근본적인 질문 이해/추론 문제
  * EM을 크게 올리려면, 장기적으로는 이 부분을 줄여야 함.

* (b) **Partial Match (29개; 12.1%)**

  * 거의 다 맞았는데, annotation 스타일/스팬 경계 때문에 EM=0
  * “짧게/길게 뽑는 경향 조정”이나 “KorQuAD 스타일을 좀 더 반영”하면
    → **단기적으로 EM 몇 점을 가장 싸게 주워 올 수 있는 구간**.

* (c) **Retrieval Failure (14개; 5.8%)**

  * alpha, top_k, hybrid 방식, 혹은 reranking으로 **정교하게 건드릴 여지가 남아있음.**

**3) Leaderboard 점수와의 갭은 “어느 정도는 자연스러운 범위”**

* val 61 ↔ test 65는

  * 분포 차이 + 약간의 운 + retrieval/test 구조 차이 정도로 설명 가능
* 예전 “train 분포와 test 분포 차이 때문에 reader가 완전 망가진 상태”와 비교하면,
  **지금은 안정적인 generalization 쪽에 훨씬 더 가까워진 상태**라고 봐도 될 것 같아.

---

필요하면:

* “partial_superset/subset 29개만 골라서 직접 몇 개 보여주고,
  이걸 어떻게 하면 EM으로 끌어올릴 수 있을지”
* 혹은 “완전 오답 46개 중 10개 정도만 깐 다음 ‘질문 해석 vs 지문 구조’ 기준으로 더 쪼개서 분석”

같은 더 미세한 레벨의 분석도 해줄 수 있어.
일단은 지금까지 제공해준 5개 파일에서 나올 수 있는 **정량/정성적인 핵심 메시지**를 다 뽑아서 정리해봤어.

좋아요, 지금까지 나온 로그/분석 파일들 기준으로 한 번 “냉정하게” 정리해볼게요.
(말씀하신 목적: **확실히 의미 있는 요소만 선별 → 우선순위 → Opus용 계획 프롬프트 작성**)

---

## 1. 사후 분석 정리 (Vanilla + DHN 흐름 포함)

### 1-1. Validation 전체 분해 (Vanilla 기준)

`val_analysis_summary.md`, `val_detailed_analysis.csv` 기준:

* 전체: **240개**
* 정답(EM=100): **147개 (61.3%)**
* 오답: **93개 (38.8%)**

에러 타입 분포:

* **retrieval_failure**: 14개 (5.8%)
* **subset_match** (예측 ⊂ 정답): 16개 (6.7%)
* **superset_match** (예측 ⊃ 정답): 10개 (4.2%)
* **overlap_partial**: 4개 (1.7%)
* **overlap_other**: 3개 (1.3%)
* **completely_wrong**: 46개 (19.2%)

→ 크게 세 덩어리로 볼 수 있음:

1. **Retrieval 실패**: 최소 14개
2. **Span 경계만 애매한 케이스**(subset/superset/partial): 29개
3. **완전히 다른 답(completely_wrong)**: 46개

---

### 1-2. Retrieval 관점 분석

`val_retrieval_failures.csv`, 교차탭 기준:

* **gold doc이 top-k 안에 없는 질문**: 18개 (7.5%)

  * 이 중 **4개는 EM=100** (retrieval은 실패했지만, 다른 passage에서 우연히 맞춤)
  * **실제로 “retrieval 때문에 틀린 케이스”**: 14개 정도

즉:

* Retrieval이 아무리 좋아도, 현 시스템에서의 **EM 상한선은 대략 92.5%** (18/240은 gold context 자체가 안 옴)
* 현재 **EM=61.25** → **retrieval이 막고 있는 부분은 대략 3~4 EM 정도의 잠재력**이라고 보는 게 정직함

  * 예: top_k를 키워서 14개 모두 gold를 가져온다 해도, reader가 그 중 전부 맞추지는 못함
    → 14 × (현재 reader hit rate ~60%) ≒ 8~9개 추가 정답 → **EM +3~4 정도**

**정리**

* Retrieval은 이미 “상당히 잘 되는 편”이고,
* 지금 EM=61에서 **retrieval만으로 올릴 수 있는 여지는 많아야 +3~4포인트 수준**.

즉, 지금 병목은 “Retrieval”이 아니라 거의 확실하게 **Reader** 쪽.

---

### 1-3. Reader 관점 분석 (오류 유형)

#### (1) Partial match 29개

`val_hard_samples.csv` 길이 분석 기준:

* subset/superset/overlap 계열에서 **gold answer 길이가 평균적으로 더 길고**, 예측은 그 중 일부/혹은 더 긴 span인 경우가 많음.
* 전형적인 패턴:

  * 사람 이름 + 직책 중 일부만 뽑음
  * 날짜/범위 표현에서 앞/뒤 절반만 선택
  * 설명형 정답에서 문장 일부만 뽑음

→ 이건 **“reader가 정답이 있는 문장 주변까지는 잘 도착했는데, start/end 경계가 안 맞는 문제”**로 해석 가능.

이 29개는 **완전히 새로운 정보 이해 능력** 문제라기보다,

* `max_seq_length`, `doc_stride`,
* tokenizer alignment, post-processing (공백/조사/부호 제거) 같은
* **span extraction 세팅**과 강하게 연관된 오류.

> 이론상, span 경계만 잘 잡으면 **이 29개 중 절반 이상을 EM으로 바꿀 잠재력이 있음 → +7~8 EM까지도 ceiling**.
> 현실적으로는 그중 일부(3~5 EM 정도)가 현실적인 타깃.

#### (2) Completely wrong 46개

완전히 엉뚱한 답(전혀 다른 개체, 문장, 숫자)을 내는 케이스:

* Retrieval은 gold를 가져왔는데,
* reader가:

  * 질문과 상관없는 문장에 앵커링되거나,
  * 비슷한 개체/숫자에 홀려서 골라버린 경우.

이 부분은:

* **모델 capacity/표현력**
* **질문-컨텍스트 매칭 능력**
* **fine-tuning 세팅 (lr/epoch/loss)**
* **train 분포 vs test 분포 차이**

같은 **진짜 MRC 능력** 영역.

여기서 얻을 수 있는 현실적인 개선 폭은:

* 완벽히 때려잡기는 불가능.
* 좋은 세팅으로 **이 영역에서 20~30% 정도의 에러 감소**를 노린다고 하면,

  * 46개 × 0.25 ≒ 11~12개 → EM +4~5 정도가 ceiling.

---

### 1-4. DHN vs Vanilla (현재까지 관찰)

지금까지 로그/언급 종합:

* **Vanilla + 개선된 retrieval (top_k=10)**

  * Val: EM ≒ 61.25, F1 ≒ 70.0
  * Test: EM 65.42, F1 75.77

* **DHN(HANTAEK_roberta_large_kure_dhn)**

  * Val: EM 거의 유사 (61± 정도)
  * Test: **EM 65.42, F1 75.77로 결과 동일**

즉:

1. **EM/F1 수준은 두 모델이 사실상 동일**
2. prediction-level 비교(이전 대화 기준)에서는:

   * 대부분 질문에서 두 모델의 예측이 동일
   * “한쪽만 맞고 한쪽은 틀리는” 케이스는 **소수(대략 5~10개 수준)**

→ 결론:

* DHN은 **큰 카타스트로픽한 실패는 없지만, 눈에 띄는 개선도 없음**.
* “완전히 다른 종류의 모델”이라기보다는, **Vanilla의 변형 버전**에 가까움.
* 둘을 합친 ensemble의 최대 이득은 **+1~2 EM 정도가 ceiling**일 가능성이 높음
  (val 기준 차별적인 correct가 3개쯤이었으니).

---

### 1-5. 잠재적인 EM 개선 여지 정리

매우 러프하지만, “완전히 잘 되었을 때”의 ceiling을 감각 차원에서 짚어보면:

1. **Retrieval 개선만으로**

   * +3~4 EM (현재 7.5% gold 미회수 → 3~4% 수준으로 줄이고, 그 중 일부 맞춘다는 가정)
2. **Span 경계 튜닝(doc_stride 등)으로 partial → exact로 승격**

   * +3~5 EM (29개 partial 중 일부 회수)
3. **Reader 자체 개선 (학습 전략/regularization/커리큘럼)**

   * +3~5 EM (completely_wrong 46개 중 일부 회수)
4. **Vanilla + DHN ensemble**

   * +1~2 EM (예측 보완이 크지 않기 때문에 ceiling이 낮음)

이걸 다 더한 “이론상 최선”은 70대 중반 근처지만,
실제 남은 시간/리소스를 생각하면 **현실적인 목표는 EM 68~70 근방** 정도가 상식적인 타깃.

---

## 2. 방향성: “싱글 모델 극한 튜닝 vs 앙상블” 결정 기준

### 2-1. 일반적인 판단 기준

일반론부터 정리하면:

**A. 하나의 모델을 끝까지 튜닝해야 하는 상황**

1. **Retrieval/Reader 구조가 아직 많이 날 것**이고,

   * top_k, alpha, doc_stride, max_seq_length, loss 세팅 등
   * **구조적인 개선 여지가 분명한 경우**
2. **여러 모델을 만들더라도 에러 패턴이 비슷할 것 같은 경우**

   * 같은 아키텍처, 같은 데이터, 비슷한 하이퍼파라미터
     → ensemble을 해도 서로 보완보다 “같이 틀리는” 경향이 강함
3. **엔지니어링 복잡도를 최소화해야 하는 경우**

   * inference 속도, 메모리, 코드 복잡도 제약이 있을 때

이때는:

* **Retrieval/Reader 파이프라인 하나를 잘 다듬고**,
* 그 안에서 curriculum, regularization, 하이퍼 튜닝으로
* **단일 모델의 성능/안정성을 극대화**하는 게 낫다.

---

**B. 앙상블이 더 이득인 상황**

1. 이미 **여러 모델이 돌아가고 있고, 서로 에러 패턴이 다를 때**

   * 다른 아키텍처 (RoBERTa vs DeBERTa vs ELECTRA)
   * 다른 학습 목표 (gold-only vs DHN vs curriculum)
   * 다른 seed/augmentation 전략
2. **추가로 새 모델을 더 학습하는 게 상대적으로 싸고**,
   ensemble 코드는 한 번만 잘 짜놓으면 재활용 가능한 경우
3. 리더보드에서 **1~2 EM로도 순위가 크게 바뀌는 구간**에 있을 때

   * 이때는 “단일 모델을 신중하게 튜닝해서 +3~4 EM” vs “모델 3개 만들어서 앙상블로 +2 EM” 이런 선택 구조가 될 수 있음

이 경우에는:

* **각기 다른 bias를 가진 모델 2~3개**를 만들고,
* prediction-level / nbest-level ensemble로
* **안정적인 generalization을 얻는 전략**이 더 안전한 선택.

---

### 2-2. 지금 프로젝트 상황에 그대로 대입해 보면

현 상태 정리:

1. Retrieval

   * Hybrid(KURE + BM25)와 alpha 튜닝, top_k=10까지 이미 적용
   * doc-level Recall@10 ≒ 92.5%, Answer-level 기준으로 봐도 mid-90% 수준
     → **retrieval은 “거의 해낼 만큼 해냈다” 쪽에 가까움**

2. Reader

   * KorQuAD-finetuned RoBERTa-large가 **분명히 강한 베이스라인**
   * Gold context로 학습했을 때 val EM ≈ 73은 확인
   * offline retrieval 붙여 test에서 65.42까지 회복 → **일반화는 오히려 좋아진 편**

3. DHN

   * EM/F1이 Vanilla와 사실상 동일
   * 예측 차이가 큰 폭으로 갈라지지 않음 → **앙상블 이득 ceiling이 낮음**

4. 남은 시간

   * 대회 종료 D-1 수준 (실질적으로 남은 full training 기회 1~2번)

이걸 기준으로 보면:

> **“지금은 아직 단일 모델/단일 파이프라인을 완성도 있게 다듬는 단계에 더 가깝다”**는 게 합리적 판단임.

* Retrieval/Reader 구조가 어느 정도 갖춰지긴 했지만,

  * top_k, doc_stride, context 포맷(title/sep token), curriculum 등
    **파이프라인 내부에서 손볼 수 있는 “큰 레버”가 남아 있음**
* 반면,

  * DHN과 Vanilla의 에러 패턴이 크게 다르지 않아서,
    **現 상태에서 앙상블만으로 얻을 수 있는 이득은 매우 제한적** (체감 +1~2 EM 수준)

---

### 2-3. 따라서, 지금의 전략 방향 제안

지금 시점에서의 “전략 방향”을 한 줄로 요약하면:

> **“단일 파이프라인(현 Vanilla + Hybrid Retrieval)을 메인으로 가져가되,
> 그 안에서 2~3개의 ‘성격 다른 reader 변형’을 만들고,
> 필요하면 마지막에 가벼운 앙상블을 덧입히는 전략”**이 가장 합리적입니다.

좀 풀어서 말하면:

1. **메인 파이프라인은 지금의 Vanilla 구조 유지**

   * Hybrid Retrieval (alpha 최적값 적용)
   * top_k 확대(10 → 20) + doc_stride 튜닝 같은 구조적 개선 먼저

2. **Reader 쪽에서는 “강하게 다른 세팅” 1~2개만 추가**

   * 예:

     * (A) Gold-only 학습 (baseline 세팅 리런; seed만 다르게)
     * (B) 팀원이 제안한 Curriculum DHN

       * 1 epoch: gold-only
       * 2~3 epoch: gold + negative (DHN 스타일)
   * 이렇게 하면 **A/B 모델이 에러 패턴에서 “어느 정도” 다양성을 가질 가능성**이 생김.

3. **앙상블은 최종 마무리 단계에서 가볍게 적용**

   * 위 A/B (및 현재 Vanilla 기존 모델) 2~3개 결과를 모아
   * simple한 rule 기반 / score 기반 ensemble을 적용
   * 여기서는 “코스트 대비 +1~2 EM을 last touch로 얻는” 느낌으로 활용

즉,

* 지금은 “앙상블에 올인”하기보다는,
* **“메인 파이프라인 하나를 잘 설계하고, 그 위에서 2~3개 모델을 변주해서 약간의 다양성을 확보 → 끝에 ensemble로 generalization 보정”**
  이 패턴이 훨씬 안정적이고, D-1이라는 시간 제약에도 맞습니다.

---

## 3. 다음 대화에서 하면 좋을 것

질문하신 것처럼, 다음 단계는:

1. **지금 분석 결과를 기반으로**

   * 어떤 가설이 “짧은 시간 내에 의미 있는 성능 향상”을 줄 수 있을지 정리
   * 예:

     * top_k 10→20 + alpha=0.5 고정 (retrieval 측)
     * doc_stride 조절 (span boundary 측)
     * curriculum DHN 1 run (reader 학습 측)
     * very light ensemble (Vanilla vs Curriculum)

2. 이걸 **“가설 → 실험 구조 → 예상 효과 → 소요시간” 형태로 재정리**해서

   * Opus에게 넘길 “계획 프롬프트”를 작성하는 것

원하시면,
지금 분석을 바탕으로:

* “우선순위 1~3번만 뽑아서”
* 각 실험을 **가설/설명/실행 방법/필요 자원/성공 기준**까지 포함한
  **깔끔한 Opus용 프롬프트 템플릿**으로 정리해 줄게요.
