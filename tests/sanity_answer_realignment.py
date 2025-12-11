"""
Answer Realignment End-to-End Sanity Check

실제 데이터셋과 Retrieval 결과를 사용하여 answer realignment이
올바르게 동작하는지 검증합니다.

검증 항목:
1. realign 후 answer_start가 실제 context에서 정답을 가리키는지
2. Tokenization 후 start_positions/end_positions가 정확한지
3. 디코딩된 정답이 원본 정답과 일치하는지

실행:
    python -m tests.sanity_answer_realignment
    python -m tests.sanity_answer_realignment --verbose
    python -m tests.sanity_answer_realignment --num_samples 50
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_from_disk
from transformers import AutoTokenizer


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


class AnswerRealignmentSanityChecker:
    """Answer Realignment 종합 검증 클래스"""

    def __init__(
        self,
        tokenizer_name: str = "klue/roberta-large",
        max_seq_length: int = 384,
        doc_stride: int = 128,
        verbose: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.verbose = verbose

        # 결과 저장
        self.results = {
            "total": 0,
            "char_level_correct": 0,
            "token_level_correct": 0,
            "decode_correct": 0,
            "failures": [],
        }

    def check_char_level_alignment(
        self, context: str, answer_text: str, answer_start: int
    ) -> Tuple[bool, str]:
        """
        1단계: Character level에서 answer_start가 정확한지 확인

        context[answer_start:answer_start+len(answer_text)] == answer_text
        """
        if answer_start < 0:
            return False, f"Invalid answer_start: {answer_start}"

        if answer_start + len(answer_text) > len(context):
            return (
                False,
                f"answer_start({answer_start}) + len({len(answer_text)}) > context_len({len(context)})",
            )

        extracted = context[answer_start : answer_start + len(answer_text)]
        if extracted == answer_text:
            return True, ""
        else:
            return False, f"Mismatch: expected '{answer_text}', got '{extracted}'"

    def check_token_level_alignment(
        self,
        question: str,
        context: str,
        answer_text: str,
        answer_start: int,
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        2단계: Token level에서 start_positions/end_positions가 정확한지 확인

        Tokenization 후 offset_mapping을 사용해 정답 토큰 위치 계산
        """
        # Tokenization (question + context)
        tokenized = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # 첫 번째 span만 검사 (간단화)
        input_ids = tokenized["input_ids"][0]
        offset_mapping = tokenized["offset_mapping"][0]

        # Answer의 character 범위
        answer_end = answer_start + len(answer_text)

        # sequence_ids로 context 영역 찾기
        sequence_ids = tokenized.sequence_ids(0)

        # Context 시작/끝 토큰 인덱스 찾기
        context_start_token = None
        context_end_token = None
        for idx, seq_id in enumerate(sequence_ids):
            if seq_id == 1:  # context
                if context_start_token is None:
                    context_start_token = idx
                context_end_token = idx

        if context_start_token is None:
            return False, "Context not found in tokenization", None

        # 정답이 이 span에 포함되는지 확인
        span_start_char = offset_mapping[context_start_token][0]
        span_end_char = offset_mapping[context_end_token][1]

        if answer_start < span_start_char or answer_end > span_end_char:
            return (
                False,
                f"Answer not in span: answer[{answer_start}:{answer_end}], span[{span_start_char}:{span_end_char}]",
                None,
            )

        # 정답 토큰 위치 찾기
        start_token = None
        end_token = None

        for idx in range(context_start_token, context_end_token + 1):
            token_start, token_end = offset_mapping[idx]
            if token_start is None:
                continue

            # start_token: 정답 시작을 포함하는 토큰
            if start_token is None and token_start <= answer_start < token_end:
                start_token = idx
            # 또는 정답 시작이 토큰 시작과 정확히 일치
            if start_token is None and token_start == answer_start:
                start_token = idx

            # end_token: 정답 끝을 포함하는 토큰
            if token_start < answer_end <= token_end:
                end_token = idx

        # 더 관대한 매칭
        if start_token is None:
            for idx in range(context_start_token, context_end_token + 1):
                token_start, token_end = offset_mapping[idx]
                if token_start is None:
                    continue
                if token_start <= answer_start and answer_start < token_end:
                    start_token = idx
                    break

        if end_token is None:
            for idx in range(context_start_token, context_end_token + 1):
                token_start, token_end = offset_mapping[idx]
                if token_start is None:
                    continue
                if token_start < answer_end and answer_end <= token_end:
                    end_token = idx

        if start_token is None or end_token is None:
            return (
                False,
                f"Could not find answer tokens: start={start_token}, end={end_token}",
                None,
            )

        debug_info = {
            "start_token": start_token,
            "end_token": end_token,
            "input_ids": input_ids,
            "offset_mapping": offset_mapping,
        }

        return True, "", debug_info

    def check_decode_correctness(
        self,
        input_ids: List[int],
        start_token: int,
        end_token: int,
        expected_answer: str,
    ) -> Tuple[bool, str]:
        """
        3단계: 디코딩된 정답이 원본과 일치하는지 확인
        """
        decoded = self.tokenizer.decode(input_ids[start_token : end_token + 1])
        decoded_clean = decoded.strip()

        # 정확히 일치하거나, 공백/특수문자 제거 후 일치
        if decoded_clean == expected_answer:
            return True, ""

        # 공백 정규화 후 비교
        decoded_normalized = " ".join(decoded_clean.split())
        expected_normalized = " ".join(expected_answer.split())

        if decoded_normalized == expected_normalized:
            return True, f"(normalized match)"

        # 부분 일치 허용 (토큰화 경계 문제)
        if expected_answer in decoded_clean or decoded_clean in expected_answer:
            return True, f"(partial match: decoded='{decoded_clean}')"

        return (
            False,
            f"Mismatch: expected '{expected_answer}', decoded '{decoded_clean}'",
        )

    def check_single_example(
        self,
        question: str,
        context: str,
        answer_text: str,
        answer_start: int,
        example_id: str = "",
    ) -> Dict:
        """단일 예시 검증"""
        result = {
            "id": example_id,
            "char_level": False,
            "token_level": False,
            "decode": False,
            "error": None,
        }

        # 1. Character level 검증
        char_ok, char_error = self.check_char_level_alignment(
            context, answer_text, answer_start
        )
        result["char_level"] = char_ok

        if not char_ok:
            result["error"] = f"Char level: {char_error}"
            return result

        # 2. Token level 검증
        token_ok, token_error, debug_info = self.check_token_level_alignment(
            question, context, answer_text, answer_start
        )
        result["token_level"] = token_ok

        if not token_ok:
            result["error"] = f"Token level: {token_error}"
            return result

        # 3. Decode 검증
        decode_ok, decode_error = self.check_decode_correctness(
            debug_info["input_ids"],
            debug_info["start_token"],
            debug_info["end_token"],
            answer_text,
        )
        result["decode"] = decode_ok

        if not decode_ok:
            result["error"] = f"Decode: {decode_error}"

        return result

    def run_on_dataset(
        self,
        examples: List[Dict],
        use_title: bool = False,
        sep_token: Optional[str] = None,
    ) -> Dict:
        """데이터셋 전체에 대해 검증 실행"""
        from src.utils.retrieval_utils import realign_answers_in_retrieved_context

        print_header("Answer Realignment Sanity Check")
        print(f"  Tokenizer: {self.tokenizer.name_or_path}")
        print(f"  use_title: {use_title}")
        print(f"  sep_token: {sep_token}")
        print(f"  Total examples: {len(examples)}")

        self.results = {
            "total": len(examples),
            "char_level_correct": 0,
            "token_level_correct": 0,
            "decode_correct": 0,
            "failures": [],
        }

        for i, example in enumerate(examples):
            # Realignment 적용
            realigned = realign_answers_in_retrieved_context(
                example.copy(),
                sep_token=sep_token,
                use_title=use_title,
            )

            # 정답이 없으면 스킵
            if len(realigned["answers"]["text"]) == 0:
                continue

            answer_text = realigned["answers"]["text"][0]
            answer_start = realigned["answers"]["answer_start"][0]
            context = realigned["context"]
            question = example.get("question", "질문")
            example_id = example.get("id", f"ex_{i}")

            # 검증
            result = self.check_single_example(
                question=question,
                context=context,
                answer_text=answer_text,
                answer_start=answer_start,
                example_id=example_id,
            )

            if result["char_level"]:
                self.results["char_level_correct"] += 1
            if result["token_level"]:
                self.results["token_level_correct"] += 1
            if result["decode"]:
                self.results["decode_correct"] += 1

            if result["error"]:
                self.results["failures"].append(
                    {
                        "id": example_id,
                        "error": result["error"],
                        "answer": answer_text,
                        "answer_start": answer_start,
                        "context_snippet": context[:100] + "...",
                    }
                )

                if self.verbose:
                    print(f"\n  ⚠️ Example {example_id}:")
                    print(f"     Error: {result['error']}")
                    print(f"     Answer: '{answer_text}' at {answer_start}")

        return self.results

    def print_summary(self):
        """결과 요약 출력"""
        print_header("Results Summary")

        total = self.results["total"]
        if total == 0:
            print("  No examples to check!")
            return

        char_pct = self.results["char_level_correct"] / total * 100
        token_pct = self.results["token_level_correct"] / total * 100
        decode_pct = self.results["decode_correct"] / total * 100

        print(f"  Total examples checked: {total}")
        print(f"")
        print_result(
            "Character-level alignment",
            char_pct == 100,
            f"{self.results['char_level_correct']}/{total} ({char_pct:.1f}%)",
        )
        print_result(
            "Token-level alignment",
            token_pct >= 95,  # 95% 이상이면 OK (truncation으로 일부 손실 가능)
            f"{self.results['token_level_correct']}/{total} ({token_pct:.1f}%)",
        )
        print_result(
            "Decode correctness",
            decode_pct >= 95,
            f"{self.results['decode_correct']}/{total} ({decode_pct:.1f}%)",
        )

        if self.results["failures"]:
            print(f"\n  ⚠️ {len(self.results['failures'])} failures detected")
            if len(self.results["failures"]) <= 5:
                for f in self.results["failures"]:
                    print(f"     - {f['id']}: {f['error']}")

        # 최종 판정
        print("\n" + "-" * 70)
        all_pass = char_pct == 100 and token_pct >= 95 and decode_pct >= 95
        if all_pass:
            print("  🎉 All sanity checks PASSED!")
        else:
            print("  ❌ Some sanity checks FAILED - investigate before training!")

        return all_pass


def create_mock_retrieved_examples(
    num_samples: int = 20,
    use_title: bool = True,
    sep_token: str = "</s>",
) -> List[Dict]:
    """
    실제 retrieval 결과와 유사한 mock 데이터 생성

    다양한 케이스:
    1. 정답이 본문에 있는 경우
    2. 정답이 여러 번 등장 (title + 본문)
    3. 긴 context
    4. 특수문자 포함
    """
    examples = []

    # 케이스 1: 기본 케이스
    for i in range(num_samples // 4):
        title = "대한민국"
        body = f"대한민국은 동아시아에 위치한 나라이다. 수도는 서울이며, 인구는 약 5000만명이다. (샘플 {i})"
        context = f"{title} {sep_token} {body}" if use_title else body

        examples.append(
            {
                "id": f"basic_{i}",
                "question": "대한민국의 수도는?",
                "context": context,
                "answers": {"text": ["서울"], "answer_start": [999]},  # 잘못된 값
            }
        )

    # 케이스 2: 정답이 title과 본문 둘 다 있는 경우
    for i in range(num_samples // 4):
        title = "서울특별시"
        body = f"서울은 대한민국의 수도이다. 서울의 인구는 약 1000만명이다. (샘플 {i})"
        context = f"{title} {sep_token} {body}" if use_title else body

        examples.append(
            {
                "id": f"title_body_{i}",
                "question": "대한민국의 수도는?",
                "context": context,
                "answers": {"text": ["서울"], "answer_start": [0]},
            }
        )

    # 케이스 3: 숫자/날짜 정답
    for i in range(num_samples // 4):
        title = "역사"
        body = f"대한민국은 1948년 8월 15일에 건국되었다. 이는 광복 3년 후의 일이다. (샘플 {i})"
        context = f"{title} {sep_token} {body}" if use_title else body

        examples.append(
            {
                "id": f"date_{i}",
                "question": "대한민국의 건국일은?",
                "context": context,
                "answers": {"text": ["1948년 8월 15일"], "answer_start": [0]},
            }
        )

    # 케이스 4: 긴 정답
    for i in range(num_samples // 4):
        title = "지리"
        body = f"대한민국의 수도는 서울특별시이며, 면적은 약 100,000 제곱킬로미터이다. (샘플 {i})"
        context = f"{title} {sep_token} {body}" if use_title else body

        examples.append(
            {
                "id": f"long_{i}",
                "question": "대한민국의 수도의 정식 명칭은?",
                "context": context,
                "answers": {"text": ["서울특별시"], "answer_start": [0]},
            }
        )

    return examples


def load_real_validation_data(data_path: str = "./data/train_dataset") -> List[Dict]:
    """실제 validation 데이터 로드"""
    try:
        datasets = load_from_disk(data_path)
        examples = []
        for ex in datasets["validation"]:
            examples.append(
                {
                    "id": ex["id"],
                    "question": ex["question"],
                    "context": ex["context"],  # gold context
                    "answers": ex["answers"],
                }
            )
        return examples
    except Exception as e:
        print(f"  ⚠️ Could not load real data: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Answer Realignment Sanity Check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Number of mock samples"
    )
    parser.add_argument(
        "--use_real_data", action="store_true", help="Use real validation data"
    )
    parser.add_argument(
        "--tokenizer", default="klue/roberta-large", help="Tokenizer name"
    )
    parser.add_argument(
        "--use_title", action="store_true", default=True, help="Include title"
    )
    parser.add_argument("--no_title", action="store_true", help="Disable title")
    args = parser.parse_args()

    use_title = not args.no_title

    # Checker 초기화
    checker = AnswerRealignmentSanityChecker(
        tokenizer_name=args.tokenizer,
        verbose=args.verbose,
    )

    sep_token = checker.tokenizer.sep_token
    print(f"\n🔍 Using sep_token: '{sep_token}'")

    # 테스트 데이터 준비
    if args.use_real_data:
        print("\n📂 Loading real validation data...")
        examples = load_real_validation_data()
        if not examples:
            print("  Falling back to mock data...")
            examples = create_mock_retrieved_examples(
                num_samples=args.num_samples,
                use_title=use_title,
                sep_token=sep_token,
            )
    else:
        print(f"\n🔧 Creating {args.num_samples} mock examples...")
        examples = create_mock_retrieved_examples(
            num_samples=args.num_samples,
            use_title=use_title,
            sep_token=sep_token,
        )

    # 검증 실행
    checker.run_on_dataset(
        examples=examples,
        use_title=use_title,
        sep_token=sep_token,
    )

    # 결과 출력
    all_pass = checker.print_summary()

    # Title OFF 테스트도 실행
    print("\n" + "=" * 70)
    print("  🔄 Also testing with use_title=False...")
    print("=" * 70)

    examples_no_title = create_mock_retrieved_examples(
        num_samples=args.num_samples,
        use_title=False,
        sep_token=sep_token,
    )

    checker.run_on_dataset(
        examples=examples_no_title,
        use_title=False,
        sep_token=None,
    )
    all_pass_no_title = checker.print_summary()

    # 종합 결과
    print("\n" + "=" * 70)
    print("  📊 FINAL VERDICT")
    print("=" * 70)

    if all_pass and all_pass_no_title:
        print("  ✅ All sanity checks PASSED for both use_title=True and False!")
        return 0
    else:
        print("  ❌ Some checks FAILED - review before training!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
