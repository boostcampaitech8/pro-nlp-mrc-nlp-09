import re
from datasets import load_from_disk, DatasetDict

# ===========================
def normalize_text(text: str) -> str:
    """개행 제거 + 다중 공백 정리"""
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def safe_normalize(text: str):
    if not isinstance(text, str):
        return text

    # 1) HTML 태그 제거 <...>
    text = re.sub(r"<[^>]+>", " ", text)

    # 2) 위키 reference 제거 [1], [2] 등
    text = re.sub(r"\[\d+\]", " ", text)

    # 3) 특수 기호 정리
    text = re.sub(r"[●★■◆▼▲…]", " ", text)

    # 4) 다중 공백 제거
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def apply_normalize(example):
    example["context"] = safe_normalize(normalize_text(example["context"]))
    example["question"] = safe_normalize(normalize_text(example["question"]))
    return example


# ===========================
def normalize_and_save(
    input_dataset_path="./data/train_dataset",
    output_dataset_path="./data/train_dataset_clean"
):
    print(f" Loading dataset from: {input_dataset_path}")
    datasets = load_from_disk(input_dataset_path)

    # train + validation 모두 존재하는 경우
    new_ds = {}

    for split in datasets.keys():
        print(f"Normalizing split: {split} ...")
        new_split = datasets[split].map(
            apply_normalize,
            num_proc=4,
            desc=f"Normalizing {split}"
        )
        new_ds[split] = new_split

    new_dataset = DatasetDict(new_ds)

    print(f" Saving cleaned dataset to: {output_dataset_path}")
    new_dataset.save_to_disk(output_dataset_path)
    


if __name__ == "__main__":
    # 기본 경로 → 원하면 아래 수정 가능
    normalize_and_save(
        input_dataset_path="./data/train_dataset",
        output_dataset_path="./data/train_dataset_clean"
    )
