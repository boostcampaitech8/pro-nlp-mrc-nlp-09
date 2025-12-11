# GitHub Copilot Instructions for MRC (ODQA) Project

## Project Overview
This is an Open-Domain Question Answering (ODQA) project for a competition.
-   **Goal**: Answer questions using a knowledge base (`wikipedia_documents.json`).
-   **Architecture**: Two-stage system:
    1.  **Retriever**: Finds relevant passages (Sparse, Dense, Hybrid).
    2.  **Reader**: Extracts answers from passages (MRC model).

## Architecture & Key Components

### 1. Core Scripts
-   `run.py`: **Main Entry Point**. Wraps `train.py` and `inference.py`. Handles batch execution.
-   `train.py`: Trains the Reader model. Uses `QuestionAnsweringTrainer`.
-   `inference.py`: Evaluates the model and generates submission files.
-   `src/trainer_qa.py`: Custom `QuestionAnsweringTrainer` for QA post-processing.

### 2. Retrieval System (`src/retrieval/`)
-   **Factory**: `get_retriever` in `src/retrieval/__init__.py`.
-   **Types**: `sparse` (TF-IDF), `bm25`, `dense` (KoE5), `hybrid` (BM25 + Dense).
-   **Usage**: `retriever = get_retriever(retrieval_type="hybrid", ...)` then `retriever.build()`.

### 3. Configuration Management
-   **YAML-based**: All experiments are defined in `configs/*.yaml`.
-   **Base Config**: `configs/base.yaml` serves as the template.
-   **Active Configs**: `configs/active/` directory is used for batch processing (`make batch`).
-   **Arguments**: Defined in `src/arguments.py` (`ModelArguments`, `DataTrainingArguments`).

## Critical Workflows

### Makefile Automation (Recommended)
-   **Train + Inference**: `make train-pipeline CONFIG=configs/exp.yaml`
-   **Train Only**: `make train CONFIG=configs/exp.yaml`
-   **Inference Only**: `make inference CONFIG=configs/exp.yaml`
-   **Batch Experiments**:
    1.  Copy configs to `configs/active/`.
    2.  Run `make batch`.
-   **Compare Results**: `make compare-results` (Compares F1/EM of all outputs).
-   **GPU Status**: `make gpu-status`.

### Manual Execution (via `run.py`)
```bash
# Single Experiment
python run.py --mode pipeline --config configs/exp.yaml

# Batch Execution
python run.py --mode batch --configs configs/exp1.yaml configs/exp2.yaml
```

## Project-Specific Conventions

### 1. Experiment Management
-   **Config Files**: Create a new YAML file in `configs/` for each experiment. Do not modify `base.yaml` directly for experiments.
-   **Output Directory**: `outputs/{username}/{experiment_name}`.
-   **Best Checkpoint**: `inference.py` automatically finds the best checkpoint in `output_dir` if `use_trained_model: true`.

### 2. Data Flow
-   **Datasets**: `data/train_dataset`, `data/test_dataset`.
-   **Retrieval**: `wikipedia_documents.json` is the corpus.
-   **Confidence Score**: The system calculates confidence scores for predictions (see `outputs/.../val_confidence.csv`).

### 3. Coding Standards
-   **Logging**: Use `src.utils.get_logger`. Avoid `print`.
-   **Type Hinting**: Use `typing` module.
-   **HuggingFace**: Use `transformers` and `datasets` APIs.

## Common Pitfalls
-   **Token Type IDs**: RoBERTa models do not use `token_type_ids`. Check `model.config.type_vocab_size`.
-   **OOM Errors**: Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` in YAML.
-   **Retrieval Build**: Ensure `retriever.build()` is called before `retrieve()`.

## Communication
-   **Language**: Respond in **Korean** (User prefers Korean).
-   **Context**: The codebase is in English, but explanations should be in Korean.
