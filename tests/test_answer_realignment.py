"""
Answer Realignment 단위 테스트

검색된 context에서 정답 위치를 재계산하는 realign_answers_in_retrieved_context()
함수의 정확성을 검증합니다.

실행:
    python -m pytest tests/test_answer_realignment.py -v
    python tests/test_answer_realignment.py  # 직접 실행
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
from src.utils.retrieval_utils import realign_answers_in_retrieved_context


class TestAnswerRealignment(unittest.TestCase):
    """Answer Realignment 기본 테스트"""

    def test_basic_realignment(self):
        """기본 케이스: 정답이 context에 존재하는 경우"""
        example = {
            "context": "서울은 대한민국의 수도이다.",
            "answers": {"text": ["서울"], "answer_start": [999]},  # 잘못된 원본 위치
        }

        result = realign_answers_in_retrieved_context(example)

        self.assertEqual(result["answers"]["text"], ["서울"])
        self.assertEqual(result["answers"]["answer_start"], [0])  # 올바른 위치

    def test_answer_not_found(self):
        """정답이 context에 없는 경우 → 빈 리스트"""
        example = {
            "context": "부산은 대한민국의 도시이다.",
            "answers": {"text": ["서울"], "answer_start": [0]},
        }

        result = realign_answers_in_retrieved_context(example)

        self.assertEqual(result["answers"]["text"], [])
        self.assertEqual(result["answers"]["answer_start"], [])

    def test_multiple_answers(self):
        """복수 정답 케이스"""
        example = {
            "context": "서울은 대한민국의 수도이다. 서울의 인구는 약 1000만명이다.",
            "answers": {
                "text": ["서울", "대한민국"],
                "answer_start": [100, 200],  # 잘못된 원본 위치
            },
        }

        result = realign_answers_in_retrieved_context(example)

        self.assertEqual(result["answers"]["text"], ["서울", "대한민국"])
        self.assertEqual(result["answers"]["answer_start"], [0, 4])

    def test_partial_match(self):
        """일부 정답만 존재하는 경우"""
        example = {
            "context": "서울은 대한민국의 수도이다.",
            "answers": {
                "text": ["서울", "도쿄"],  # 도쿄는 없음
                "answer_start": [0, 10],
            },
        }

        result = realign_answers_in_retrieved_context(example)

        self.assertEqual(result["answers"]["text"], ["서울"])
        self.assertEqual(result["answers"]["answer_start"], [0])

    def test_answer_in_middle(self):
        """정답이 중간에 있는 경우"""
        example = {
            "context": "대한민국의 수도는 서울이다.",
            "answers": {"text": ["서울"], "answer_start": [0]},
        }

        result = realign_answers_in_retrieved_context(example)

        self.assertEqual(result["answers"]["text"], ["서울"])
        self.assertEqual(result["answers"]["answer_start"], [10])  # "서울" 시작 위치


class TestTitleAwareRealignment(unittest.TestCase):
    """Title 포함 Context에서의 Realignment 테스트"""

    def test_title_with_sep_token(self):
        """Title [SEP] 본문 형태에서 본문의 정답 찾기"""
        example = {
            "context": "서울 [SEP] 서울은 대한민국의 수도이다.",
            "answers": {"text": ["서울"], "answer_start": [0]},
        }

        # use_title=True이면 title 영역을 건너뜀
        result = realign_answers_in_retrieved_context(
            example, sep_token="[SEP]", use_title=True
        )

        # "서울"이 title(0)이 아닌 본문(9)에서 찾아져야 함
        # "서울 [SEP] " = 9 characters (3+5+1)
        self.assertEqual(result["answers"]["text"], ["서울"])
        self.assertEqual(result["answers"]["answer_start"], [9])

    def test_title_without_sep_token(self):
        """use_title=True이지만 sep_token이 없는 경우"""
        example = {
            "context": "서울 [SEP] 서울은 대한민국의 수도이다.",
            "answers": {"text": ["서울"], "answer_start": [0]},
        }

        # sep_token이 None이면 처음부터 검색
        result = realign_answers_in_retrieved_context(
            example, sep_token=None, use_title=True
        )

        # Title 영역의 첫 번째 "서울"(0)을 찾음
        self.assertEqual(result["answers"]["text"], ["서울"])
        self.assertEqual(result["answers"]["answer_start"], [0])

    def test_use_title_false(self):
        """use_title=False이면 title 영역도 포함해서 검색"""
        example = {
            "context": "서울 [SEP] 부산은 대한민국의 도시이다.",
            "answers": {"text": ["서울"], "answer_start": [0]},
        }

        result = realign_answers_in_retrieved_context(
            example, sep_token="[SEP]", use_title=False
        )

        # use_title=False이면 title 영역의 "서울"(0)을 찾음
        self.assertEqual(result["answers"]["text"], ["서울"])
        self.assertEqual(result["answers"]["answer_start"], [0])

    def test_answer_only_in_title(self):
        """정답이 title에만 있고 본문에 없는 경우 → 빈 리스트"""
        example = {
            "context": "서울 [SEP] 부산은 대한민국의 도시이다.",
            "answers": {"text": ["서울"], "answer_start": [0]},
        }

        result = realign_answers_in_retrieved_context(
            example, sep_token="[SEP]", use_title=True
        )

        # 본문에 "서울"이 없으므로 빈 리스트
        self.assertEqual(result["answers"]["text"], [])
        self.assertEqual(result["answers"]["answer_start"], [])

    def test_roberta_sep_token(self):
        """RoBERTa의 </s> separator 테스트"""
        example = {
            "context": "서울 </s> 서울은 대한민국의 수도이다.",
            "answers": {"text": ["서울"], "answer_start": [0]},
        }

        result = realign_answers_in_retrieved_context(
            example, sep_token="</s>", use_title=True
        )

        # 본문의 "서울" 위치 (</s> 뒤)
        # "서울 </s> " = 8 characters (3+4+1)
        self.assertEqual(result["answers"]["text"], ["서울"])
        self.assertEqual(result["answers"]["answer_start"], [8])

    def test_multiple_passages_with_title(self):
        """여러 passage가 연결된 경우 (첫 번째 title만 건너뜀)"""
        example = {
            # 실제로는 "Title1 [SEP] passage1 Title2 [SEP] passage2" 형태
            "context": "서울 [SEP] 부산은 도시이다. 대구 [SEP] 서울은 수도이다.",
            "answers": {"text": ["서울"], "answer_start": [0]},
        }

        result = realign_answers_in_retrieved_context(
            example, sep_token="[SEP]", use_title=True
        )

        # 첫 번째 [SEP] 이후부터 검색하므로 두 번째 passage의 "서울"(31)을 찾음
        # "부산은 도시이다. 대구 [SEP] 서울은 수도이다." 에서 "서울" 위치
        self.assertEqual(result["answers"]["text"], ["서울"])
        # 실제 위치 계산: "서울 [SEP] " = 8, "부산은 도시이다. 대구 [SEP] " = 23, total=31
        self.assertTrue(result["answers"]["answer_start"][0] > 8)


class TestEdgeCases(unittest.TestCase):
    """엣지 케이스 테스트"""

    def test_empty_context(self):
        """빈 context"""
        example = {
            "context": "",
            "answers": {"text": ["서울"], "answer_start": [0]},
        }

        result = realign_answers_in_retrieved_context(example)

        self.assertEqual(result["answers"]["text"], [])
        self.assertEqual(result["answers"]["answer_start"], [])

    def test_empty_answers(self):
        """빈 answers"""
        example = {
            "context": "서울은 대한민국의 수도이다.",
            "answers": {"text": [], "answer_start": []},
        }

        result = realign_answers_in_retrieved_context(example)

        self.assertEqual(result["answers"]["text"], [])
        self.assertEqual(result["answers"]["answer_start"], [])

    def test_long_answer(self):
        """긴 정답 텍스트"""
        example = {
            "context": "대한민국의 수도는 서울특별시이며, 인구는 약 1000만명이다.",
            "answers": {"text": ["서울특별시"], "answer_start": [0]},
        }

        result = realign_answers_in_retrieved_context(example)

        self.assertEqual(result["answers"]["text"], ["서울특별시"])
        self.assertEqual(result["answers"]["answer_start"], [10])  # 한글 10자 후

    def test_special_characters_in_answer(self):
        """특수문자가 포함된 정답"""
        example = {
            "context": "2024년 1월 1일에 발표되었다.",
            "answers": {"text": ["2024년 1월 1일"], "answer_start": [0]},
        }

        result = realign_answers_in_retrieved_context(example)

        self.assertEqual(result["answers"]["text"], ["2024년 1월 1일"])
        self.assertEqual(result["answers"]["answer_start"], [0])

    def test_duplicate_answer_in_context(self):
        """같은 정답이 context에 여러 번 등장"""
        example = {
            "context": "서울은 서울특별시의 약칭이다. 서울의 인구는 많다.",
            "answers": {"text": ["서울"], "answer_start": [100]},
        }

        result = realign_answers_in_retrieved_context(example)

        # 첫 번째 등장 위치를 반환
        self.assertEqual(result["answers"]["text"], ["서울"])
        self.assertEqual(result["answers"]["answer_start"], [0])

    def test_unicode_normalization(self):
        """유니코드 정규화 관련 (한글 자모 분리 등)"""
        # 일반적인 한글 텍스트
        example = {
            "context": "한글은 세종대왕이 만들었다.",
            "answers": {"text": ["세종대왕"], "answer_start": [0]},
        }

        result = realign_answers_in_retrieved_context(example)

        self.assertEqual(result["answers"]["text"], ["세종대왕"])
        self.assertEqual(result["answers"]["answer_start"], [4])


class TestIntegration(unittest.TestCase):
    """통합 테스트: 실제 사용 시나리오"""

    def test_realistic_retrieval_scenario(self):
        """실제 retrieval 결과와 유사한 시나리오"""
        # Retrieved context: 여러 passage가 연결됨
        example = {
            "context": (
                "대한민국 [SEP] 대한민국은 동아시아에 위치한 나라이다. "
                "서울 [SEP] 서울은 대한민국의 수도이다. 인구는 약 1000만명이다."
            ),
            "answers": {
                "text": ["서울"],
                "answer_start": [31],
            },  # 원본 gold context 기준
        }

        # use_title=True로 첫 번째 title만 건너뜀
        result = realign_answers_in_retrieved_context(
            example, sep_token="[SEP]", use_title=True
        )

        # 첫 번째 [SEP] 이후부터 검색
        self.assertEqual(result["answers"]["text"], ["서울"])
        # "대한민국 [SEP] " 이후의 첫 번째 "서울" 위치
        self.assertTrue(result["answers"]["answer_start"][0] > 0)

    def test_filter_integration(self):
        """필터링 통합 테스트: 정답 없는 example 감지"""
        examples = [
            {
                "context": "서울은 수도이다.",
                "answers": {"text": ["서울"], "answer_start": [0]},
            },
            {
                "context": "부산은 도시이다.",
                "answers": {"text": ["서울"], "answer_start": [0]},  # 정답 없음
            },
        ]

        results = [realign_answers_in_retrieved_context(ex) for ex in examples]

        # 첫 번째는 정답 있음
        self.assertEqual(len(results[0]["answers"]["text"]), 1)
        # 두 번째는 정답 없음 → 필터링 대상
        self.assertEqual(len(results[1]["answers"]["text"]), 0)


def run_tests():
    """테스트 실행 및 결과 출력"""
    print("=" * 60)
    print("🧪 Answer Realignment Unit Tests")
    print("=" * 60)

    # unittest 실행
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 모든 테스트 클래스 추가
    suite.addTests(loader.loadTestsFromTestCase(TestAnswerRealignment))
    suite.addTests(loader.loadTestsFromTestCase(TestTitleAwareRealignment))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 결과 요약
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests PASSED!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
