"""
Cross-Encoder based Reranker.
Uses a pretrained Cross-Encoder model to rescore (query, passage) pairs.
"""

import torch
from typing import List, Tuple, NoReturn, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np

class CrossEncoderReranker:
    """
    Cross-Encoder Reranker.
    
    Usage:
        reranker = CrossEncoderReranker(
            model_name="monologg/koelectra-base-v3-discriminator",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        scores = reranker.rerank(query, passages)
    """
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> NoReturn:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"[Reranker] Loading Cross-Encoder: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        passages: List[str],
    ) -> List[float]:
        """
        Rerank a list of passages for a single query.
        Returns a list of scores corresponding to the passages.
        """
        scores = []
        
        # Pairs creation
        pairs = [[query, p] for p in passages]

        # Batch processing
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                # Assuming binary classification (irrelevant/relevant) or regression
                # If output is 1 dim -> regression score
                # If output is 2 dim -> take logit for 'relevant' class (usually index 1)
                
                if logits.shape[1] == 1:
                    batch_scores = logits.squeeze(-1).cpu().numpy().tolist()
                else:
                    # Typically index 1 is positive class
                    batch_scores = logits[:, 1].cpu().numpy().tolist()
            
            scores.extend(batch_scores)

        return scores
