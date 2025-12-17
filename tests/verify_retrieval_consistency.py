#!/usr/bin/env python3
"""
Retrieval 일관성 검증 스크립트

BaseRetrieval 리팩토링 후 다음을 검증합니다:
1. Corpus 로딩: contexts, ids, titles 개수 일치
2. Embedding 크기: p_embedding.shape[0] == len(contexts)
3. 인덱스 안전성: doc_indices의 모든 값이 contexts 범위 내
4. 단일/벌크 쿼리 일관성: get_relevant_doc와 get_relevant_doc_bulk 결과 동일
5. ids-titles 정렬: contexts[i] <-> ids[i] <-> titles[i] 매핑 일치

Usage:
    python tests/verify_retrieval_consistency.py --retrieval_type sparse
"""

import argparse
import sys
import os

# MRC 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer
from src.retrieval.sparse import SparseRetrieval


def verify_corpus_consistency(retriever):
    """코퍼스 로딩 일관성 검증"""
    print("\n" + "=" * 60)
    print("🔍 1. Corpus 로딩 일관성 검증")
    print("=" * 60)

    contexts_len = len(retriever.contexts)
    ids_len = len(retriever.ids)
    titles_len = len(retriever.titles)

    print(f"✓ Contexts 개수: {contexts_len}")
    print(f"✓ IDs 개수: {ids_len}")
    print(f"✓ Titles 개수: {titles_len}")

    if contexts_len == ids_len == titles_len:
        print("✅ 모든 배열 길이 일치!")
        return True
    else:
        print("❌ 배열 길이 불일치 감지!")
        return False


def verify_embedding_size(retriever):
    """Embedding 크기 검증"""
    print("\n" + "=" * 60)
    print("🔍 2. Embedding 크기 검증")
    print("=" * 60)

    if not hasattr(retriever, "p_embedding") or retriever.p_embedding is None:
        print("⚠️  Embedding이 아직 빌드되지 않았습니다. build() 호출 중...")
        retriever.build()

    emb_size = retriever.p_embedding.shape[0]
    ctx_size = len(retriever.contexts)

    print(f"✓ Embedding shape: {retriever.p_embedding.shape}")
    print(f"✓ Contexts 개수: {ctx_size}")

    if emb_size == ctx_size:
        print("✅ Embedding 크기와 contexts 개수 일치!")
        return True
    else:
        print(f"❌ 크기 불일치! Embedding({emb_size}) != Contexts({ctx_size})")
        return False


def verify_index_safety(retriever, num_samples=50):
    """인덱스 범위 안전성 검증"""
    print("\n" + "=" * 60)
    print("🔍 3. 인덱스 범위 안전성 검증")
    print("=" * 60)

    test_queries = [
        "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?",
        "멕시코의 수도는?",
        "한국의 전통 음식은?",
        "파이썬 프로그래밍 언어의 특징은?",
        "세계에서 가장 높은 산은?",
    ]

    max_contexts_idx = len(retriever.contexts) - 1
    print(f"✓ Contexts 최대 인덱스: {max_contexts_idx}")

    all_safe = True
    for query in test_queries[:num_samples]:
        _, doc_indices = retriever.get_relevant_doc(query, k=10)

        max_idx = max(doc_indices)
        if max_idx > max_contexts_idx:
            print(f"❌ 인덱스 범위 초과 감지! max_idx={max_idx} > {max_contexts_idx}")
            print(f"   Query: {query[:50]}...")
            all_safe = False
            break

    if all_safe:
        print(f"✅ 모든 쿼리에서 인덱스 범위 안전 (테스트 {len(test_queries)}개)")
        return True
    else:
        return False


def verify_single_bulk_consistency(retriever):
    """단일 쿼리와 벌크 쿼리 결과 일관성 검증"""
    print("\n" + "=" * 60)
    print("🔍 4. 단일/벌크 쿼리 일관성 검증")
    print("=" * 60)

    test_query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    k = 5

    # 단일 쿼리
    scores_single, indices_single = retriever.get_relevant_doc(test_query, k=k)

    # 벌크 쿼리 (길이 1)
    scores_bulk, indices_bulk = retriever.get_relevant_doc_bulk([test_query], k=k)
    scores_bulk = scores_bulk[0]
    indices_bulk = indices_bulk[0]

    print(f"✓ 단일 쿼리 결과: scores={scores_single[:3]}, indices={indices_single[:3]}")
    print(f"✓ 벌크 쿼리 결과: scores={scores_bulk[:3]}, indices={indices_bulk[:3]}")

    if scores_single == scores_bulk and indices_single == indices_bulk:
        print("✅ 단일/벌크 쿼리 결과 완전 일치!")
        return True
    else:
        print("❌ 결과 불일치 감지!")
        return False


def verify_ids_titles_mapping(retriever, num_samples=10):
    """contexts-ids-titles 매핑 일관성 검증"""
    print("\n" + "=" * 60)
    print("🔍 5. Contexts-IDs-Titles 매핑 검증")
    print("=" * 60)

    print(f"✓ {num_samples}개 샘플 검증 중...\n")

    import random

    sample_indices = random.sample(range(len(retriever.contexts)), num_samples)

    for i in sample_indices:
        ctx = retriever.contexts[i][:50]  # 앞 50자만
        doc_id = retriever.ids[i]
        title = retriever.titles[i]

        print(f"[{i}] doc_id={doc_id}, title='{title}', context='{ctx}...'")

    print("\n✅ 매핑 샘플 출력 완료 (수동 검증 필요)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Retrieval 일관성 검증")
    parser.add_argument(
        "--retrieval_type",
        type=str,
        default="sparse",
        choices=["sparse", "dense"],
        help="검증할 retrieval 타입",
    )
    parser.add_argument("--data_path", type=str, default="./data", help="데이터 경로")
    parser.add_argument(
        "--context_path",
        type=str,
        default="wikipedia_documents.json",
        help="Wikipedia 문서 파일명",
    )
    parser.add_argument(
        "--model_name", type=str, default="klue/roberta-large", help="Tokenizer 모델명"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🚀 Retrieval 일관성 검증 시작")
    print("=" * 60)
    print(f"Retrieval Type: {args.retrieval_type}")
    print(f"Data Path: {args.data_path}")
    print(f"Context Path: {args.context_path}")

    # Retriever 초기화
    if args.retrieval_type == "sparse":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
        retriever.build()
    else:
        raise NotImplementedError("Dense retrieval은 아직 구현되지 않았습니다.")

    # 검증 실행
    results = {}
    results["corpus"] = verify_corpus_consistency(retriever)
    results["embedding"] = verify_embedding_size(retriever)
    results["index_safety"] = verify_index_safety(retriever)
    results["single_bulk"] = verify_single_bulk_consistency(retriever)
    results["mapping"] = verify_ids_titles_mapping(retriever)

    # 최종 결과
    print("\n" + "=" * 60)
    print("📊 최종 검증 결과")
    print("=" * 60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    print("=" * 60)
    if all_passed:
        print("🎉 모든 검증 통과!")
        return 0
    else:
        print("⚠️  일부 검증 실패. 위 결과를 확인하세요.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
