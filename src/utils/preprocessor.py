"""
Text preprocessing utilities for MRC dataset cleaning.
"""
import re
from typing import Dict, Any


class TextPreprocessor:
    """
    텍스트 전처리를 위한 클래스.
    HTML 태그, 특수 문자, 공백 정규화 등을 수행합니다.
    """
    
    def __init__(self, apply_cleaning: bool = False):
        """
        Args:
            apply_cleaning: True이면 텍스트 정규화를 적용합니다.
        """
        self.apply_cleaning = apply_cleaning
    
    def normalize_text(self, text: str) -> str:
        """
        기본 텍스트 정규화: 줄바꿈 제거 및 공백 정규화
        
        사용 가이드:
        - normalize_text(): 가벼운 정규화 (줄바꿈 및 공백만 처리)
        - safe_normalize(): 강력한 정규화 (HTML, 특수문자, 참조번호 제거)
        - preprocess_example(): 데이터셋 예제에 safe_normalize 자동 적용
        
        Args:
            text: 정규화할 텍스트
            
        Returns:
            정규화된 텍스트
        """
        if not isinstance(text, str):
            return text
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text
    
    def safe_normalize(self, text: str) -> str:
        """
        안전한 텍스트 정규화: HTML 태그, 참조 번호, 특수 문자 제거
        
        Args:
            text: 정규화할 텍스트
            
        Returns:
            정규화된 텍스트
        """
        if not isinstance(text, str):
            return text
        
        # HTML 태그 제거
        text = re.sub(r"<[^>]+>", " ", text)
        # 참조 번호 제거 (예: [1], [2])
        text = re.sub(r"\[\d+\]", " ", text)
        # 특수 문자 제거
        text = re.sub(r"[●★■◆▼▲…]", " ", text)
        # 연속된 공백을 하나로
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 예제에 대한 전처리를 수행합니다.
        
        Args:
            example: context, question, answers를 포함하는 dictionary
            
        Returns:
            전처리된 example
        """
        if not self.apply_cleaning:
            return example
        
        return {
            **example,
            "context": self.safe_normalize(example["context"]),
            "question": self.safe_normalize(example["question"]),
        }
