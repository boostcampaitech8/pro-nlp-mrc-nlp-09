"""
Retrieval 관련 파일 경로 중앙 관리 모듈

모든 임베딩, 인덱스, 캐시 파일의 경로를 이곳에서 통합 관리합니다.
각 모듈에서 하드코딩된 경로 대신 이 모듈의 상수를 사용하세요.

Usage:
    from src.retrieval.paths import PATHS, get_path

    # 경로 가져오기
    kure_emb_path = get_path("kure_corpus_emb")
    bm25_index_path = get_path("bm25_index")

    # 전체 경로 딕셔너리
    print(PATHS)
"""

import os
from pathlib import Path
from typing import Optional


# ============================================================
# 기본 경로 설정
# ============================================================

# 프로젝트 루트 (이 파일 기준으로 3단계 상위)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# 데이터 루트
DATA_ROOT = PROJECT_ROOT / "data"

# 하위 디렉토리들
EMBEDDINGS_DIR = DATA_ROOT / "embeddings"  # Dense 임베딩 저장
INDICES_DIR = DATA_ROOT / "indices"  # Sparse/Dense 인덱스
CACHE_DIR = DATA_ROOT / "cache"  # Retrieval 캐시 (JSONL)


# ============================================================
# 파일 경로 정의 (단일 진실의 원천)
# ============================================================

PATHS = {
    # === Wikipedia Corpus ===
    "wiki_corpus": str(DATA_ROOT / "wikipedia_documents.json"),
    # === Dense Embeddings (Sentence Transformers) ===
    # KoE5 (기존)
    "koe5_corpus_emb": str(EMBEDDINGS_DIR / "koe5_corpus_emb.npy"),
    # KURE-v1 (신규)
    "kure_corpus_emb": str(EMBEDDINGS_DIR / "kure_corpus_emb.npy"),
    "kure_passages_meta": str(EMBEDDINGS_DIR / "kure_passages_meta.jsonl"),
    # === Sparse Indices (BM25) ===
    "bm25_index_dir": str(INDICES_DIR / "bm25"),
    "bm25_model": str(INDICES_DIR / "bm25" / "bm25_model.pkl"),
    # === TF-IDF (Legacy) ===
    "tfidf_embedding": str(INDICES_DIR / "sparse" / "sparse_embedding.bin"),
    "tfidf_vectorizer": str(INDICES_DIR / "sparse" / "tfidv.bin"),
    # === Retrieval Cache (Dynamic Hard Negative용) ===
    "retrieval_cache_dir": str(CACHE_DIR / "retrieval"),
    "train_cache": str(CACHE_DIR / "retrieval" / "train_top50.jsonl"),
    "val_cache": str(CACHE_DIR / "retrieval" / "val_top50.jsonl"),
    "test_cache": str(CACHE_DIR / "retrieval" / "test_top50.jsonl"),
    # === Dataset Paths ===
    "train_dataset": str(DATA_ROOT / "train_dataset"),
    "test_dataset": str(DATA_ROOT / "test_dataset"),
}

# Output 디렉토리 (동적으로 결정됨)
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


# ============================================================
# 유틸리티 함수
# ============================================================


def get_path(key: str) -> str:
    """
    경로 키로 파일 경로를 가져옵니다.

    Args:
        key: PATHS 딕셔너리의 키

    Returns:
        해당 파일/디렉토리의 절대 경로

    Raises:
        KeyError: 존재하지 않는 키
    """
    if key not in PATHS:
        available = ", ".join(sorted(PATHS.keys()))
        raise KeyError(f"Unknown path key: '{key}'. Available: {available}")
    return PATHS[key]


def ensure_dir(key: str) -> str:
    """
    경로의 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.

    Args:
        key: PATHS 딕셔너리의 키

    Returns:
        해당 경로 (디렉토리가 생성된 상태)
    """
    path = get_path(key)

    # 파일 경로인 경우 상위 디렉토리 생성
    if "." in os.path.basename(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

    return path


def get_analysis_dir(output_dir: str, subdir: str = "val_analysis") -> Path:
    """
    분석 결과 저장 디렉토리를 반환합니다.

    Args:
        output_dir: 모델 출력 디렉토리 (예: ./outputs/dahyeong/model_name)
        subdir: 하위 디렉토리 이름 (기본: val_analysis)

    Returns:
        분석 결과 저장 디렉토리 Path (없으면 생성)
    """
    analysis_dir = Path(output_dir) / subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir


def exists(key: str) -> bool:
    """
    해당 경로의 파일/디렉토리가 존재하는지 확인합니다.

    Args:
        key: PATHS 딕셔너리의 키

    Returns:
        존재 여부
    """
    return os.path.exists(get_path(key))


def get_file_size(key: str) -> Optional[str]:
    """
    파일 크기를 사람이 읽기 좋은 형태로 반환합니다.

    Args:
        key: PATHS 딕셔너리의 키

    Returns:
        파일 크기 문자열 (예: "231.5 MB") 또는 None (파일 없음)
    """
    path = get_path(key)
    if not os.path.exists(path):
        return None

    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def print_status():
    """
    모든 관리되는 파일들의 존재 여부와 크기를 출력합니다.
    """
    print("\n" + "=" * 70)
    print("📁 Retrieval Files Status")
    print("=" * 70)

    categories = {
        "Dense Embeddings": [
            "koe5_corpus_emb",
            "kure_corpus_emb",
            "kure_passages_meta",
        ],
        "Sparse Indices": ["bm25_index_dir", "tfidf_embedding", "tfidf_vectorizer"],
        "Retrieval Cache": ["train_cache", "val_cache", "test_cache"],
        "Corpus": ["wiki_corpus"],
    }

    for category, keys in categories.items():
        print(f"\n📂 {category}")
        print("-" * 50)
        for key in keys:
            path = get_path(key)
            if os.path.exists(path):
                size = get_file_size(key) or "DIR"
                print(f"  ✅ {key}: {size}")
                print(f"     └─ {path}")
            else:
                print(f"  ❌ {key}: NOT FOUND")
                print(f"     └─ {path}")

    print("\n" + "=" * 70)


# ============================================================
# 마이그레이션 헬퍼 (기존 경로에서 새 경로로 이동)
# ============================================================

LEGACY_PATHS = {
    # 기존 경로 -> 새 경로 키 매핑
    "./data/koe5_corpus_emb.npy": "koe5_corpus_emb",
    "./data/kure_corpus_emb.npy": "kure_corpus_emb",
    "./data/kure_passages_meta.jsonl": "kure_passages_meta",
    "./data/indices/dense/koe5_corpus_emb.npy": "koe5_corpus_emb",
    "./data/retrieval_cache/train_top50.jsonl": "train_cache",
    "./data/retrieval_cache/val_top50.jsonl": "val_cache",
    "./data/retrieval_cache/test_top50.jsonl": "test_cache",
}


def migrate_legacy_files(dry_run: bool = True) -> None:
    """
    기존 경로의 파일들을 새 경로로 이동합니다.

    Args:
        dry_run: True면 실제 이동 없이 계획만 출력
    """
    import shutil

    print("\n" + "=" * 70)
    print("🔄 Legacy File Migration")
    print(f"   Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print("=" * 70)

    for legacy_path, new_key in LEGACY_PATHS.items():
        abs_legacy = str(PROJECT_ROOT / legacy_path.lstrip("./"))
        new_path = get_path(new_key)

        if os.path.exists(abs_legacy):
            if abs_legacy == new_path:
                print(f"  ⏭️  SAME: {legacy_path}")
                continue

            if os.path.exists(new_path):
                print(f"  ⚠️  CONFLICT: {legacy_path}")
                print(f"      Both exist! Manual resolution needed.")
                continue

            print(f"  📦 MOVE: {legacy_path}")
            print(f"      → {new_path}")

            if not dry_run:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.move(abs_legacy, new_path)
                print(f"      ✅ Done")
        else:
            print(f"  ⏭️  SKIP: {legacy_path} (not found)")

    print("\n" + "=" * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval paths management")
    parser.add_argument("--status", action="store_true", help="Show file status")
    parser.add_argument(
        "--migrate", action="store_true", help="Migrate legacy files (dry run)"
    )
    parser.add_argument(
        "--migrate-execute", action="store_true", help="Actually migrate files"
    )

    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.migrate:
        migrate_legacy_files(dry_run=True)
    elif args.migrate_execute:
        migrate_legacy_files(dry_run=False)
    else:
        print("Available paths:")
        for key, path in sorted(PATHS.items()):
            status = "✅" if os.path.exists(path) else "❌"
            print(f"  {status} {key}: {path}")
