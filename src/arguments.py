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
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset_name: Optional[str] = field(
        default="./data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )

    val_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the validation dataset to use."},
    )

    test_dataset_name: Optional[str] = field(
        default="./data/test_dataset",
        metadata={"help": "The name of the test dataset to use."},
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


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    학습 하이퍼파라미터 기본값 설정
    CLI에서 별도로 지정하지 않으면 이 값들이 사용됩니다.
    """
    
    # 학습 기본 설정
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning rate for AdamW optimizer"}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/CPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/CPU for evaluation"}
    )
    
    # 최적화 설정
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of total training steps for warmup"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for AdamW optimizer"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before backward pass"}
    )
    
    # 저장/평가 설정
    save_strategy: str = field(
        default="no",
        metadata={"help": "Save strategy: 'no', 'steps', 'epoch'"}
    )
    save_total_limit: int = field(
        default=2,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"help": "Evaluation strategy: 'no', 'steps', 'epoch'"}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log every X steps"}
    )
    
    # 성능 최적화
    fp16: bool = field(
        default=True,
        metadata={"help": "Use mixed precision training"}
    )
