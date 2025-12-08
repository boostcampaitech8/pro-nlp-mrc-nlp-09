import json
import re

INPUT_PATH = "./data/data/wikipedia_documents.json"
OUTPUT_PATH = "./data/data/wikipedia_documents_normalized.json"

def clean_wiki(text: str) -> str:
    if not isinstance(text, str):
        return text
    
    # 1) HTML 태그 제거
    text = re.sub(r"<[^>]+>", " ", text)

    # 2) 위키 citation 숫자 제거 [1], [주 2], [edit], [citation needed]
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # 3) URL 제거
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # 4) 특수기호 제거
    text = re.sub(r"[●★■◆▼▲▶▷◀◁…※]", " ", text)

    # 5) 다중 공백 정리
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_wikipedia():
    print("📂 Loading wikipedia_documents.json ...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        wiki = json.load(f)

    print("🔧 Cleaning documents... (may take 10~20 sec)")
    cleaned = {}

    for doc_id, content in wiki.items():
        cleaned[doc_id] = {
            "title": clean_wiki(content["title"]),
            "text": clean_wiki(content["text"])
        }

    print(f"💾 Saving cleaned file → {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print("🎉 Done! Wikipedia normalized version saved.")


if __name__ == "__main__":
    normalize_wikipedia()
