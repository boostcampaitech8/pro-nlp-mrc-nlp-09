"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
from typing import Callable, Dict, List, NoReturn, Tuple

import evaluate
import numpy as np
from src.arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from src.retrieval import SparseRetrieval
from src.trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from src.utils import (
    check_no_error,
    postprocess_qa_predictions,
    wait_for_gpu_availability,
    get_config,
    get_logger,
    get_model_path,
    load_inference_dataset,
)

logger = get_logger(__name__, logging.INFO)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # gpu 사용 가능한지 체크
    wait_for_gpu_availability()

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = get_config(parser)

    # inference_split에 따라 do_eval/do_predict 자동 설정
    inference_split = data_args.inference_split
    if inference_split == "test":
        # test: 정답이 없으므로 predict만
        # test는 gold context가 없으므로 retrieval 필수
        if not data_args.eval_retrieval:
            raise ValueError(
                "❌ test split에는 gold context가 없으므로 eval_retrieval=True가 필수입니다.\n"
                "💡 config에서 eval_retrieval: true 설정 후 다시 실행하세요."
            )
        training_args.do_eval = False
        training_args.do_predict = True
        logger.info("🎯 Inference mode: TEST (do_predict only, retrieval required)")
    else:
        # train/validation: 정답이 있으므로 eval + predict 모두 수행
        training_args.do_eval = True
        training_args.do_predict = True
        logger.info(
            f"🎯 Inference mode: {inference_split.upper()} (do_eval + do_predict)"
        )

    # 모델 경로 자동 결정 (use_trained_model=True이면 best checkpoint 자동 탐색)
    model_path = get_model_path(model_args, training_args, for_inference=True)
    logger.info(f"📦 Model path: {model_path}")

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    # inference_split에 맞는 데이터셋 로드
    datasets = load_inference_dataset(data_args, inference_split)
    logger.info(f"📊 Dataset loaded: {datasets}")

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
    )

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize,
            datasets,
            training_args,
            data_args,
        )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model, inference_split)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # TODO: do_predict / do_eval 둘다 사용하는 경우 고려할 것
    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    inference_split: str,
) -> NoReturn:
    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 모델 타입에 따라 token_type_ids 지원 여부 자동 판별
    # 핵심: tokenizer가 만들 수 있는가가 아니라, 모델이 받을 수 있는가가 중요
    model_type = getattr(model.config, "model_type", "").lower()
    tokenizer_says_it_can = "token_type_ids" in getattr(
        tokenizer, "model_input_names", []
    )
    type_vocab_size = getattr(model.config, "type_vocab_size", 0)

    # RoBERTa/XLM-R은 type_vocab_size=1 이라 token_type_ids 넣으면 인덱스 에러 발생
    use_token_type_ids = bool(tokenizer_says_it_can and type_vocab_size > 1)

    print(
        f"model_type={model_type} | tokenizer_has_token_type_ids={tokenizer_says_it_can} "
        f"| type_vocab_size={type_vocab_size} | use_token_type_ids={use_token_type_ids}"
    )

    # 오류가 있는지 확인합니다.
    _, max_seq_length = check_no_error(data_args, training_args, datasets, tokenizer)

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples, _use_token_type_ids=use_token_type_ids):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=_use_token_type_ids,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 안전장치: 혹시 token_type_ids가 남아있으면 제거
        if not _use_token_type_ids and "token_type_ids" in tokenized_examples:
            tokenized_examples.pop("token_type_ids")

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
            prefix="test",  # inference.py는 test 예측이므로 test_pred.csv 생성
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    # eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        logger.info("=" * 80)
        logger.info("✅ INFERENCE COMPLETED - Results saved:")
        logger.info(
            f"   📄 predictions_test.json: {training_args.output_dir}/predictions_test.json"
        )
        logger.info(
            f"   📄 nbest_predictions_test.json: {training_args.output_dir}/nbest_predictions_test.json"
        )
        logger.info(f"   📊 test_pred.csv: {training_args.output_dir}/test_pred.csv")
        logger.info(f"      👉 Use this CSV file for test submission!")
        logger.info("=" * 80)

        # Validation set에서 gold vs retrieval 비교 (옵션)
        if (
            inference_split == "validation"
            and hasattr(data_args, "compare_retrieval")
            and data_args.compare_retrieval
        ):
            logger.info("")
            logger.info("=" * 80)
            logger.info("🔍 RETRIEVAL COMPARISON MODE")
            logger.info("=" * 80)

            import json
            import csv
            from src.retrieval.sparse import SparseRetrieval
            from src.utils.evaluator import (
                FinalEvaluator,
                save_predictions,
                save_detailed_results,
            )

            # 1. Gold context 예측 (이미 완료)
            logger.info("1️⃣  Gold context predictions (already done)")
            gold_pred_dict = {
                pred["id"]: pred["prediction_text"] for pred in predictions.predictions
            }

            # 정답 레이블 저장
            answer_column_name = (
                "answers" if "answers" in column_names else column_names[2]
            )
            val_ref_dict = {
                ex["id"]: ex[answer_column_name] for ex in datasets["validation"]
            }

            eval_labels_path = os.path.join(
                training_args.output_dir, "eval_labels.json"
            )
            with open(eval_labels_path, "w", encoding="utf-8") as f:
                json.dump(val_ref_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"   ✅ Labels saved: {eval_labels_path}")

            # Gold predictions CSV 저장
            eval_pred_gold_path = os.path.join(
                training_args.output_dir, "eval_pred_gold.csv"
            )
            with open(eval_pred_gold_path, "w", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                for key, value in gold_pred_dict.items():
                    writer.writerow([key, value])
            logger.info(f"   ✅ Gold predictions: {eval_pred_gold_path}")

            # 2. Retrieval context 예측
            logger.info("")
            logger.info("2️⃣  Running retrieval for validation set...")

            # Retrieval 초기화
            retriever = SparseRetrieval(
                tokenize_fn=tokenizer.tokenize,
                data_path=data_args.data_path
                if hasattr(data_args, "data_path")
                else "./data",
                context_path=data_args.context_path
                if hasattr(data_args, "context_path")
                else "wikipedia_documents.json",
            )
            retriever.get_sparse_embedding()

            # Retrieval 수행
            val_questions = datasets["validation"]["question"]
            if data_args.use_faiss:
                retrieved_contexts = retriever.retrieve_faiss(
                    val_questions, topk=data_args.top_k_retrieval
                )
            else:
                retrieved_contexts = retriever.retrieve(
                    val_questions, topk=data_args.top_k_retrieval
                )

            # Retrieved context로 새로운 dataset 생성
            val_with_retrieval = datasets["validation"].map(
                lambda example, idx: {"context": retrieved_contexts[idx]},
                with_indices=True,
                desc="Adding retrieved contexts",
            )

            # Feature 생성
            val_retrieval_dataset = val_with_retrieval.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Preparing validation features with retrieval",
            )

            # Retrieval 예측
            logger.info("   Running predictions with retrieved contexts...")
            val_retrieval_predictions = trainer.predict(
                test_dataset=val_retrieval_dataset, test_examples=val_with_retrieval
            )
            val_retrieval_pred_dict = {
                pred["id"]: pred["prediction_text"]
                for pred in val_retrieval_predictions.predictions
            }

            # Retrieval predictions CSV 저장
            eval_pred_retrieval_path = os.path.join(
                training_args.output_dir, "eval_pred_retrieval.csv"
            )
            with open(eval_pred_retrieval_path, "w", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                for key, value in val_retrieval_pred_dict.items():
                    writer.writerow([key, value])
            logger.info(f"   ✅ Retrieval predictions: {eval_pred_retrieval_path}")

            # 3. 자동 비교 실행
            logger.info("")
            logger.info("3️⃣  Comparing gold vs retrieval performance...")

            import subprocess

            comparison_script = "scripts/compare_retrieval.py"
            if os.path.exists(comparison_script):
                result = subprocess.run(
                    [sys.executable, comparison_script, training_args.output_dir],
                    capture_output=False,
                )
                if result.returncode == 0:
                    logger.info("   ✅ Comparison completed successfully!")
                else:
                    logger.warning(
                        f"   ⚠️  Comparison failed with code {result.returncode}"
                    )
            else:
                logger.warning(
                    f"   ⚠️  Comparison script not found: {comparison_script}"
                )
                logger.info(
                    f"   💡 Run manually: python {comparison_script} {training_args.output_dir}"
                )

            logger.info("=" * 80)
        else:
            print(
                "No metric can be presented because there is no correct answer given. Job done!"
            )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
