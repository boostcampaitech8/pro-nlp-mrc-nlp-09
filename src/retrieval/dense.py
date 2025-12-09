import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from typing import List, NoReturn, Optional, Union

from transformers import AutoTokenizer, AutoModel

from src.retrieval.dense_hard_sampling import get_hard_sample
from src.retrieval.dense_train import dpr_train
from src.retrieval.dense_embed import dense_embedding
from datasets import Dataset

class DenseRetrieval:
    def __init__(
        self,
        bm25_candidate: int = 200,
        reranker_model: str = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
        hard_sample_k: int = 5,
        dpr_model: str = "snumin44/biencoder-ko-bert-question",
        dpr_model_output_dir: str = "./outputs/minseok",
        data_path: str = "./data",
        context_path: str = "./data/wikipedia_documents.json",
        device: Optional[str] = "cuda",
    ) -> NoReturn:
        
        self.bm25_candidate = bm25_candidate
        self.reranker_model = reranker_model
        self.hard_sample_k = hard_sample_k
        self.dpr_model = dpr_model
        self.dpr_model_output_dir = dpr_model_output_dir
        self.data_path = data_path
        self.context_path = context_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 하드샘플 및 DPR 체크포인트 경로
        self.hard_sample_path = os.path.join(self.data_path, "train_dataset","negative")

        # Dense embedding 경로
        self.embeddings_path = os.path.join(self.data_path, "embeddings/context_embeddings.npy")
        self.metadata_path = os.path.join(self.data_path, "embeddings/chunk_metadata.npy")
        self.wiki_texts_path = os.path.join(self.data_path, "embeddings/wiki_texts_dedup.pkl")

        # 모델/임베딩 초기값
        self.encoder = None
        self.tokenizer = None
        self.context_embeddings = None
        self.chunk_metadata = None
        self.wiki_texts = None

    # =====================================================
    # 필수 파일/체크포인트 생성 및 확인
    # =====================================================
    def _ensure_hard_samples(self):
        if not os.path.exists(self.hard_sample_path):
            print(">>> Hard samples missing. Generating...")
            get_hard_sample(
                data_path=self.data_path,
                context_path=self.context_path,
                reranker_model=self.reranker_model,
                bm25_candidate=self.bm25_candidate,
                hard_sample_k=self.hard_sample_k
            )
        else:
            print("Hard samples OK.")

    def _ensure_dpr_checkpoint(self):
        if not os.path.exists(os.path.join(self.dpr_model_output_dir, "question_encoder")):
            print(">>> DPR checkpoint missing. Training...")
            dpr_train(
                model_name=self.dpr_model,
                output_dir=self.dpr_model_output_dir,
                hard_sample_path=self.hard_sample_path
            )
        else:
            print("DPR checkpoint OK.")

    def _ensure_dense_embeddings(self):
        if not (os.path.exists(self.embeddings_path) and os.path.exists(self.metadata_path) and os.path.exists(self.wiki_texts_path)):
            print(">>> Dense embeddings missing. Building...")
            dense_embedding(
                model_path=self.dpr_model_output_dir,
                wiki_path=self.wiki_texts_path,
                embeddings_output_path=self.embeddings_path,
                metadata_output_path=self.metadata_path
            )
        else:
            print("Dense embeddings OK.")

    # =====================================================
    # Dense retrieval
    # =====================================================
    def retrieve(self, dataset: Union[List[dict], Dataset], topk: int = 50) -> pd.DataFrame:
        """
        run() 없이도 retrieve()만 호출하면 자동으로 필요한 파일 체크 후 반환
        """
        # 1) 필수 파일 체크 및 생성
        self._ensure_hard_samples()
        self._ensure_dpr_checkpoint()
        self._ensure_dense_embeddings()

        # 2) DPR 모델 로드 (최초 1회)
        if self.encoder is None:
            print("Loading DPR encoder...")
            self.encoder = AutoModel.from_pretrained(os.path.join(self.dpr_model_output_dir, "question_encoder")).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.dpr_model_output_dir, "question_encoder"))
            self.encoder.eval()

        # 3) Context embeddings 및 메타데이터 로드 (최초 1회)
        if self.context_embeddings is None:
            print("Loading wiki embeddings...")
            self.context_embeddings = F.normalize(
                torch.tensor(np.load(self.embeddings_path), device=self.device), p=2, dim=1
            )
            self.chunk_metadata = np.load(self.metadata_path)
            with open(self.wiki_texts_path, "rb") as f:
                self.wiki_texts = pickle.load(f)["wiki_texts"]

        # 4) Query embedding
        queries = [ex["question"] for ex in dataset]
        q_inputs = self.tokenizer(
            queries,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        q_inputs = {k: v.to(self.device) for k, v in q_inputs.items()}
        with torch.no_grad():
            q_outputs = self.encoder(**q_inputs)
            q_emb = q_outputs.pooler_output if hasattr(q_outputs, "pooler_output") else q_outputs.last_hidden_state[:,0]
            q_emb = F.normalize(q_emb, p=2, dim=1)

        # 5) Dense search (dot product)
        scores = q_emb @ self.context_embeddings.T
        topk_scores, topk_indices = torch.topk(scores, k=topk*2, dim=1)
        topk_scores = topk_scores.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()

        # 6) 결과 구성 (중복 제거 + top_k 보장)
        results = []
        for q_idx, example in enumerate(dataset):
            seen_docs = set()
            contexts = []
            for idx in topk_indices[q_idx]:
                doc_idx = self.chunk_metadata[idx]
                if doc_idx in seen_docs:
                    continue
                seen_docs.add(doc_idx)
                contexts.append(self.wiki_texts[doc_idx])
                if len(contexts) >= topk:
                    break

            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context": " ".join(contexts)
            }
            if "context" in example and "answers" in example:
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            results.append(tmp)

        return pd.DataFrame(results)
