import os
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
import random

data_path = "./data/train_dataset"
neg_path = "./data/train_dataset/negative"
save_path = "./data/train_dataset_with_negatives"

# ==============================
# 1. 안전한 answer_start 재계산 함수
# ==============================
def merge_contexts_with_safe_offset(
    positive_context: str,
    positive_answer_start: int,
    positive_answer_text: str,
    negative_contexts: list,
    shuffle: bool = True,
):
    passages = []

    # positive passage
    passages.append(("positive", positive_context))

    # negative passages
    for neg in negative_contexts:
        passages.append(("negative", neg))

    # shuffle
    if shuffle:
        random.shuffle(passages)

    # merge passages
    merged_context = ""
    pos_prefix_offset = None

    for ptype, text in passages:
        if ptype == "positive":
            pos_prefix_offset = len(merged_context)
        merged_context += text + "\n\n"

    if pos_prefix_offset is None:
        raise RuntimeError("Positive passage not included!")

    # 최종 answer_start 계산
    new_answer_start = pos_prefix_offset + positive_answer_start

    # sanity check
    real_text = merged_context[new_answer_start:new_answer_start + len(positive_answer_text)]
    assert real_text == positive_answer_text, (
        f"Answer text mismatch!\nExpected: {positive_answer_text}\nGot: {real_text}"
    )

    return merged_context, new_answer_start

# ==============================
# 2. 데이터셋 로드
# ==============================
print("Loading original dataset...")
ds = load_from_disk(data_path)
ds_train = ds["train"]
ds_val = ds["validation"]
print("Original train size:", len(ds_train))
print("Original validation size:", len(ds_val))

print("Loading negative dataset (Arrow)...")
neg_ds = load_from_disk(neg_path)
print("Negative dataset size:", len(neg_ds))

# ==============================
# 3. 새로운 train 데이터 생성
# ==============================
new_rows = {
    "id": [], "question": [], "context": [], "answers": [], "title": [], "document_id": []
}

print("Processing train samples with negatives...")

for i, row in enumerate(tqdm(ds_train)):
    answer_text = row["answers"]["text"][0]
    orig_answer_start = row["answers"]["answer_start"][0]
    question = row["question"]

    # negative sample 가져오기
    neg_row = neg_ds[i]
    pos_context = neg_row["positive"]
    neg_contexts = neg_row["negatives"]

    # context 병합 및 answer_start 재계산
    merged_context, new_answer_start = merge_contexts_with_safe_offset(
        positive_context=pos_context,
        positive_answer_start=orig_answer_start,
        positive_answer_text=answer_text,
        negative_contexts=neg_contexts,
        shuffle=True
    )

    # 새로운 row 저장
    new_rows["id"].append(row["id"])
    new_rows["question"].append(question)
    new_rows["context"].append(merged_context)
    new_rows["answers"].append({
        "text": [answer_text],
        "answer_start": [new_answer_start]
    })
    new_rows["title"].append(row["title"])
    new_rows["document_id"].append(row["document_id"])

# train Dataset 생성
new_train_ds = Dataset.from_dict(new_rows)

# ==============================
# 4. DatasetDict 반환
# ==============================
result = DatasetDict({
    "train": new_train_ds,
    "validation": ds_val
})

result.save_to_disk(save_path)
