from datasets import Dataset, load_from_disk

# 데이터로드
neg = load_from_disk("./data/train_dataset/negative")
orig = load_from_disk("./data/train_dataset_clean")["train"]

assert len(neg) == len(orig)

SEP = " [SEP] "

def build_concat_row(orig_row, neg_row):
    question = orig_row["question"]
    document_id = orig_row["document_id"]

    # --------------------------------------------------------
    # POSITIVE: 반드시 neg의 positive (정답 포함 문단)
    # --------------------------------------------------------
    positive = neg_row["positive"]

    # --------------------------------------------------------
    # Answer text 가져오기
    # --------------------------------------------------------
    ans_text = orig_row["answers"]["text"][0]

    # 1차 단순 매칭
    ans_start = positive.find(ans_text)

    # 2차 lower 매칭 fallback
    if ans_start == -1:
        ans_start = positive.lower().find(ans_text.lower())

    # 못 찾으면 스킵 (데이터 매우 적음)
    if ans_start == -1:
        print(f"[WARNING] answer not found in POSITIVE for id={orig_row['id']}")
        return None

    # --------------------------------------------------------
    # CONCAT: POS → EASY → MID → HARD3개
    # --------------------------------------------------------
    ctx_parts = [
        positive,
        neg_row["negatives"]["easy"],
        neg_row["negatives"]["mid"],
        neg_row["negatives"]["hard"][0],
        neg_row["negatives"]["hard"][1],
        neg_row["negatives"]["hard"][2],
    ]
    context = SEP.join(ctx_parts)

    # --------------------------------------------------------
    # 완성 row 반환
    # --------------------------------------------------------
    return {
        "id": orig_row["id"],
        "document_id": document_id,
        "question": question,
        "context": context,
        "answers": {
            "text": [ans_text],
            "answer_start": [ans_start],   # POSITIVE가 맨 앞이라 그대로 OK
        }
    }


# --------------------------------------------------------
# 전체 변환 실행
# --------------------------------------------------------
new_rows = []
for i in range(len(orig)):
    row = build_concat_row(orig[i], neg[i])
    if row is not None:
        new_rows.append(row)

concat_dataset = Dataset.from_list(new_rows)

save_path = "./data/train_dataset/concat_negatives"
concat_dataset.save_to_disk(save_path)

print("Saved concat dataset →", save_path)
print("num rows:", len(concat_dataset))

