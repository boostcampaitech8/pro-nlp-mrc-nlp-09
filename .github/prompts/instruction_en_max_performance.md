# GitHub Copilot Instructions for MRC (ODQA) Project — Performance (English-Centric)

## Project Overview

This repository is an Open-Domain Question Answering (ODQA) competition project.

* **Goal**: Build a system that takes a query and returns an answer string by searching a knowledge base.
* **Architecture**: Two-stage system:

  1. **Retriever**: Finds relevant passages from `wikipedia_documents.json` (approx. 57k documents).
  2. **Reader**: Extracts answers from the retrieved passages using a Machine Reading Comprehension (MRC) model.

The user is an advanced ML engineer working on this project. Assume they are comfortable reading English, but they may sometimes ask questions in Korean.

---

## Competition Rules & Constraints (CRITICAL)

You MUST obey these rules at all times. Never propose code or workflows that violate them.

* **No Test Set Analysis**:

  * It is strictly forbidden to analyze, visualize, or use the test set for training/labeling.
  * Do NOT suggest "peeking" at test labels or manually relabeling test questions.

* **Pretrained Weights**:

  * Usage of weights pretrained specifically on the **KLUE-MRC** dataset is **prohibited**.
  * Other public pretrained weights are allowed.
  * All pretrained weights must be publicly available and free of copyright issues.

* **External Data**:

  * External datasets are allowed **except** KLUE-MRC.
  * If you propose using any external data, clearly state:

    * Source
    * Rough usage plan (pretraining, augmentation, etc.)

---

## Evaluation Metrics

* **Exact Match (EM)** — Primary metric:

  * Score is 1 if the prediction exactly matches the ground truth (after cleaning punctuation/whitespace), 0 otherwise.
  * Multiple acceptable gold answers may exist; if the prediction matches **any** of them, it is correct.

* **F1 Score** — Secondary (reference only):

  * Provides partial credit for token-level overlap between prediction and ground truth.
  * EM is used for leaderboard ranking; F1 is used only for analysis.

When designing experiments or analyzing results, explicitly consider both EM and F1.

---

## Architecture & Key Components

### 1. Core Scripts

* `train.py`

  * Handles training of the Reader model.
  * Uses a custom `QuestionAnsweringTrainer` (see `src/trainer_qa.py`).
  * Reads configuration from YAML using `HfArgumentParser`.

* `inference.py`

  * Handles evaluation and submission file generation.
  * Integrates **Retrieval + Reader**.
  * Controls whether to run on validation (with labels) or test (without labels) via `inference_split` in the YAML.

* `src/trainer_qa.py`

  * Custom `QuestionAnsweringTrainer` inheriting from HF `Trainer`.
  * Overrides `evaluate` and `predict` to handle QA post-processing (logits → text answers).
  * Responsible for computing EM/F1 and saving predictions.

### 2. Retrieval System (`src/retrieval/`)

* **Pattern**: Factory pattern via `get_retriever` in `src/retrieval/__init__.py`.
* **Base Class**: `BaseRetrieval` in `src/retrieval/base.py`.
* **Implementations**:

  * `SparseRetrieval` (TF-IDF / BM25)
  * `KoE5Retrieval` (Dense)
  * `HybridRetrieval` (combination of sparse + dense)
* **Usage**:

  * Always instantiate via `get_retriever(retrieval_type=..., ...)`.
  * Always call `.build()` before retrieval.
  * For evaluation, focus on Recall@k for multiple k values (1, 5, 10, 20, 50, 100).

### 3. Configuration Management

* **Source of truth**: YAML files in `configs/` (e.g., `configs/base.yaml`, `configs/exp/*.yaml`).
* **Schema**: Defined in `src/arguments.py`:

  * `ModelArguments`
  * `DataTrainingArguments`
  * `TrainingArguments`
* **Parsing**:

  * Uses `HfArgumentParser` to parse both CLI args and YAML config.
  * Typical usage:

    ```bash
    python train.py configs/your_experiment.yaml
    python inference.py configs/your_experiment.yaml
    ```

---

## Data Flow

1. **Loading**:

   * Data is loaded via HuggingFace `datasets` (Arrow format) from `./data`.
   * Train: 3,952 examples.
   * Validation: 240 examples.
   * Test: 600 examples (240 Public, 360 Private).
   * Test set does not contain context/answers → retrieval is mandatory.

2. **Preprocessing**:

   * `prepare_train_features`:

     * Tokenization
     * Sliding window with `doc_stride`
     * Label alignment (start/end positions)
   * `prepare_validation_features`:

     * Tokenization for inference
     * No labels, just mapping example IDs to tokenized contexts

3. **Retrieval Integration** (in `inference.py`):

   * `retrieve_and_build_dataset`:

     * Fetches top-k contexts for each question.
     * Builds a new dataset with `(question, retrieved_context)` pairs.
   * The Reader model then runs on these retrieved contexts.

---

## Critical Workflows

### Training

```bash
python train.py configs/your_experiment.yaml
```

* Ensure `output_dir` in the YAML is **unique per experiment** to avoid overwriting results.
* Use `TrainingArguments` to control:

  * `learning_rate`
  * `per_device_train_batch_size`
  * `gradient_accumulation_steps`
  * `num_train_epochs`
  * `save_strategy`, `evaluation_strategy`, etc.

### Inference / Submission

```bash
python inference.py configs/your_experiment.yaml
```

* `inference_split` in the YAML controls:

  * `validation`: run with labels, compute EM/F1.
  * `test`: run without labels, generate submission file.
* Before suggesting any modification, always preserve correct handling of:

  * `id`
  * `question`
  * `context`
  * `prediction_text`

---

## Coding Conventions

* **Type Hinting**:

  * Use `typing` (e.g., `List`, `Dict`, `Optional`, `Tuple`, `NoReturn`) extensively.
  * Prefer explicit return types.

* **HuggingFace Ecosystem**:

  * Prefer `transformers` and `datasets` APIs over raw PyTorch where possible.
  * Use `AutoModelForQuestionAnswering`, `AutoTokenizer`, etc.

* **Path Handling**:

  * Use absolute paths or paths relative to the project root.
  * Avoid hardcoding machine-specific paths.

* **Logging**:

  * Use `src.utils.get_logger` instead of `print`.
  * Provide informative log messages (stage names, dataset sizes, metrics).

* **Code Editing Behavior**:

  * Prefer **small, focused diffs** over large rewrites, unless the user explicitly asks.
  * Before making large refactors, briefly summarize the plan in comments or explanation text.

---

## Common Pitfalls

* **Token Type IDs**:

  * RoBERTa-based models do not use `token_type_ids`.
  * Check `model.config.type_vocab_size` before passing them.
  * When in doubt, avoid passing `token_type_ids` for RoBERTa.

* **Answer Alignment**:

  * When replacing context (e.g., retrieval-augmented training), `answer_start` indices must be recalculated.
  * The original indices refer to the original `context`.
  * Never assume answer offsets remain valid after context changes.

* **OOM Errors**:

  * If encountering CUDA OOM:

    * Reduce `per_device_train_batch_size`.
    * Increase `gradient_accumulation_steps` to keep effective batch size similar.
    * Optionally reduce `max_seq_length` if appropriate.

---

## Makefile Usage

The project includes a `Makefile` to simplify common tasks.

### Commands

* `make train CONFIG=configs/exp.yaml`
  Run training only.

* `make inference CONFIG=configs/exp.yaml`
  Run inference only.

* `make train-pipeline CONFIG=configs/exp.yaml`
  Run training followed by inference (recommended for full experiments).

* `make eval-val CONFIG=configs/exp.yaml`
  Run validation analysis (compares gold context vs retrieval).

* `make batch`
  Run all configs in `configs/active/` sequentially.

* `make check-config CONFIG=configs/exp.yaml`
  Validate a config file (schema-level checks).

* `make list-active`
  List active configs.

* `make gpu-status`
  Check GPU usage.

* `make clean-checkpoints`
  Remove intermediate checkpoints to save space.

* `make compare-results`
  Compare F1/EM scores of experiments.

### Example

```bash
make train-pipeline CONFIG=configs/active/my_experiment.yaml
```

---

## Language & Communication Guidelines (VERY IMPORTANT)

This is a **performance-focused, English-centric** configuration.

* **User Input Language**:

  * The user may write in **Korean or English**.
  * Treat Korean input as fully valid. Do NOT reject Korean questions.

* **Assistant Output Language (Default)**:

  * **Always respond in English by default**, regardless of input language.
  * Use clear, concise technical English, as if writing documentation or code review comments.

* **When the User Explicitly Requests Korean**:

  * If the user says phrases like:

    * “한국어로 답변해줘”
    * “한국어로 설명해줘”
    * “Please answer in Korean”
  * Then:

    * You MUST respond in Korean.
    * You may still use English for code, identifiers, and error messages.

* **Korean + English Mixed Mode (Optional)**:

  * When the user asks in Korean but does **not** explicitly ask for Korean output:

    * You may:

      * Explain the main reasoning in English.
      * Optionally add **short Korean summaries** for clarity.
  * Example structure:

    * English: Detailed explanation, bullet points, code.
    * Korean: 1–3 sentence summary of the key idea.

* **Code and Technical Terms**:

  * Code, function names, classes, error messages, CLI commands, and file paths should remain in English.
  * Technical terms like “Retriever”, “Reader”, “logits”, “Dataset” should stay in English.

* **Clarifying Questions**:

  * Before making large, irreversible changes, briefly ask clarifying questions in English.
  * Keep questions minimal and concrete.

---

## Reasoning & Answer Structure

For non-trivial tasks (new features, refactors, debugging), follow this structure:

1. **High-Level Plan (English)**

   * Bullet-point summary of what you will do.

2. **Code Changes**

   * Provide full code snippets or patches.

3. **Explanation**

   * Explain why this change is correct and how it fits the ODQA pipeline.

4. **(Optional) Short Korean Summary**

   * If the user asked in Korean, you may add a short Korean summary at the end.

This configuration is optimized for **maximum model performance and clarity in English**, while still allowing Korean input and optional Korean summaries.