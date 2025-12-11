from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="uomnf97/klue-roberta-finetuned-korquad-v2",
        metadata={
            "help": "speicfy the pretrained model name or path to local checkpoint"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_trained_model: bool = field(
        default=True,
        metadata={
            "help": "For inference: automatically load best checkpoint from output_dir instead of model_name_or_path"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset_name: Optional[str] = field(
        default="./data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )

    infer_dataset_name: Optional[str] = field(
        default="./data/test_dataset",
        metadata={"help": "The name of the inference dataset to use."},
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    train_retrieval: bool = field(
        default=False,
        metadata={
            "help": "Whether to run passage retrieval during training (Retrieval-Augmented Training)."
        },
    )
    retrieval_type: str = field(
        default="bm25",
        metadata={
            "help": "Retrieval type to use (sparse, dense, hybrid, bm25, koe5, kure, weighted_hybrid)."
        },
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    inference_split: str = field(
        default="test",
        metadata={
            "help": "Which split to use for inference: 'train', 'validation', or 'test'. "
            "If 'test', only do_predict is executed. Otherwise, both do_eval and do_predict are executed."
        },
    )
    compare_retrieval: bool = field(
        default=False,
        metadata={
            "help": "For validation inference: compare gold context vs retrieval performance. "
            "Only used when inference_split='validation'. Runs inference twice (gold + retrieval)."
        },
    )
    # --- Reranker & Tokenizer Options ---
    reranker_name: Optional[str] = field(
        default=None,
        metadata={"help": "Model name for Cross-Encoder Reranker (e.g. 'monologg/koelectra-base-v3-discriminator'). If None, no reranking."},
    )
    retrieval_tokenizer_name: str = field(
        default="auto",
        metadata={"help": "Tokenizer for retrieval: 'auto' (use model's tokenizer) or 'kiwi' (use Kiwi)."},
    )
    bm25_impl: str = field(
        default="bm25s",
        metadata={"help": "BM25 implementation: 'bm25s' (fast, default) or 'rank_bm25' (supports BM25Plus)."},
    )
    bm25_delta: float = field(
        default=0.5,
        metadata={"help": "Delta parameter for BM25Plus (only used if bm25_impl='rank_bm25')."},
    )
    # Dynamic Hard Negative Training 관련 설정 (YAML에서 dynamic_hard_negative 섹션으로 관리)
    # 아래 설정들은 DataTrainingArguments에서 직접 사용하지 않고,
    # YAML의 dynamic_hard_negative 섹션에서 읽어 사용합니다.
    # - enabled: bool (Dynamic Hard Negative 사용 여부)
    # - k_ret: int (retrieval top-k)
    # - k_read: int (train에서 사용할 context 수)
    # - alpha: float (hybrid score 계산용 BM25 가중치)
    # - use_title: bool (title을 context에 포함할지 여부)
