"""
Utilities for tokenization.
Handles Kiwi initialization.
"""

from typing import Callable, List

def get_tokenizer(tokenizer_name: str, model_tokenizer=None) -> Callable[[str], List[str]]:
    """
    Returns a tokenization function based on name.
    """
    if tokenizer_name == "kiwi":
        try:
            from kiwipiepy import Kiwi
            kiwi = Kiwi()
            print("[Tokenizer] Using Kiwi Morphological Analyzer")
            
            def kiwi_tokenize(text: str) -> List[str]:
                return [token.form for token in kiwi.tokenize(text)]
            
            return kiwi_tokenize
        except ImportError:
            raise ImportError("Kiwi is not installed. `pip install kiwipiepy`")
            
    elif tokenizer_name == "auto" and model_tokenizer:
        return model_tokenizer.tokenize
        
    else:
        # Fallback to simple split
        return lambda x: x.split()
