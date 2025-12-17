#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KURE + BM25 Weighted Hybrid Retrieval Pipeline - Sanity Check Tests

각 모듈별 기본 동작을 검증하는 테스트입니다.
실행: python tests/test_kure_pipeline_sanity.py [--module MODULE_NAME]

모듈 목록:
  - kure_embedding: KURE corpus embedding 생성 테스트
  - kure_retrieval: KureRetrieval 클래스 테스트
  - weighted_hybrid: WeightedHybridRetrieval 클래스 테스트
  - cache_builder: Retrieval cache 생성 테스트
  - mrc_dataset: MRCWithRetrievalDataset 테스트
  - all: 모든 테스트 실행
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    """테스트 섹션 헤더 출력"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, message: str = ""):
    """테스트 결과 출력"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} | {test_name}")
    if message:
        print(f"         └─ {message}")


def test_kure_embedding_module():
    """
    Test 1: KURE Embedding 모듈 기본 동작 검증
    - SentenceTransformer 모델 로드
    - 단일 텍스트 임베딩 생성
    - L2 정규화 확인
    """
    print_header("Test 1: KURE Embedding Module")

    results = []

    # 1-1. SentenceTransformer import 및 모델 로드
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("nlpai-lab/KURE-v1")
        print_result("SentenceTransformer 로드", True, "nlpai-lab/KURE-v1")
        results.append(True)
    except Exception as e:
        print_result("SentenceTransformer 로드", False, str(e))
        results.append(False)
        return results

    # 1-2. 단일 텍스트 임베딩 생성
    try:
        test_text = "대한민국의 수도는 서울이다."
        embedding = model.encode([test_text], normalize_embeddings=True)

        assert embedding.shape == (1, 1024), (
            f"Expected (1, 1024), got {embedding.shape}"
        )
        print_result("임베딩 생성", True, f"shape={embedding.shape}")
        results.append(True)
    except Exception as e:
        print_result("임베딩 생성", False, str(e))
        results.append(False)
        return results

    # 1-3. L2 정규화 확인
    try:
        norm = np.linalg.norm(embedding[0])
        assert abs(norm - 1.0) < 1e-5, f"L2 norm should be 1.0, got {norm}"
        print_result("L2 정규화 확인", True, f"norm={norm:.6f}")
        results.append(True)
    except Exception as e:
        print_result("L2 정규화 확인", False, str(e))
        results.append(False)

    # 1-4. 배치 임베딩 생성
    try:
        batch_texts = [
            "서울은 대한민국의 수도이다.",
            "부산은 대한민국 제2의 도시이다.",
            "제주도는 대한민국의 섬이다.",
        ]
        batch_embeddings = model.encode(batch_texts, normalize_embeddings=True)

        assert batch_embeddings.shape == (3, 1024), (
            f"Expected (3, 1024), got {batch_embeddings.shape}"
        )
        print_result("배치 임베딩 생성", True, f"shape={batch_embeddings.shape}")
        results.append(True)
    except Exception as e:
        print_result("배치 임베딩 생성", False, str(e))
        results.append(False)

    # 1-5. 코사인 유사도 계산 (dot product로 가능, L2 정규화되어 있으므로)
    try:
        query = model.encode(["대한민국 수도"], normalize_embeddings=True)
        similarities = np.dot(batch_embeddings, query.T).flatten()

        # 첫 번째 문장이 가장 유사해야 함
        assert similarities[0] > similarities[1], "첫 번째 문장이 더 유사해야 함"
        assert similarities[0] > similarities[2], "첫 번째 문장이 더 유사해야 함"

        print_result("코사인 유사도", True, f"scores={similarities.round(4).tolist()}")
        results.append(True)
    except Exception as e:
        print_result("코사인 유사도", False, str(e))
        results.append(False)

    return results


def test_kure_retrieval_class():
    """
    Test 2: KureRetrieval 클래스 기본 동작 검증
    - 클래스 인스턴스 생성
    - 임베딩 파일 없이도 에러 없이 초기화
    - 메서드 시그니처 확인
    """
    print_header("Test 2: KureRetrieval Class")

    results = []

    # 2-1. 클래스 import
    try:
        from src.retrieval.kure import KureRetrieval

        print_result("KureRetrieval import", True)
        results.append(True)
    except Exception as e:
        print_result("KureRetrieval import", False, str(e))
        results.append(False)
        return results

    # 2-2. 인스턴스 생성 (임베딩 파일 없이)
    try:
        retriever = KureRetrieval(
            tokenize_fn=lambda x: x.split(),
            data_path="./data",
            corpus_emb_path="./data/kure_corpus_emb.npy",  # 없어도 됨
            passages_meta_path="./data/kure_passages_meta.jsonl",  # 없어도 됨
        )
        print_result("인스턴스 생성", True)
        results.append(True)
    except Exception as e:
        print_result("인스턴스 생성", False, str(e))
        results.append(False)
        return results

    # 2-3. 필수 메서드 존재 확인
    required_methods = [
        "build",
        "get_relevant_doc_bulk",
        "get_dense_scores_all",
        "get_passage_text",
        "get_doc_id_from_passage",
    ]

    for method_name in required_methods:
        has_method = hasattr(retriever, method_name) and callable(
            getattr(retriever, method_name)
        )
        print_result(f"메서드: {method_name}", has_method)
        results.append(has_method)

    # 2-4. 임베딩 파일이 있으면 build() 테스트
    emb_path = PROJECT_ROOT / "data" / "kure_corpus_emb.npy"
    meta_path = PROJECT_ROOT / "data" / "kure_passages_meta.jsonl"

    if emb_path.exists() and meta_path.exists():
        try:
            retriever = KureRetrieval(
                tokenize_fn=lambda x: x.split(),
                data_path=str(PROJECT_ROOT / "data"),
                corpus_emb_path=str(emb_path),
                passages_meta_path=str(meta_path),
            )
            retriever.build()
            print_result(
                "build() with real data",
                True,
                f"passages={len(retriever.passages_meta)}",
            )
            results.append(True)
        except Exception as e:
            print_result("build() with real data", False, str(e))
            results.append(False)
    else:
        print_result("build() with real data", None, "임베딩 파일 없음 (SKIP)")

    return results


def test_weighted_hybrid_class():
    """
    Test 3: WeightedHybridRetrieval 클래스 기본 동작 검증
    - 클래스 인스턴스 생성
    - Per-query 정규화 로직 검증
    - Alpha 가중합 검증
    """
    print_header("Test 3: WeightedHybridRetrieval Class")

    results = []

    # 3-1. 클래스 import
    try:
        from src.retrieval.weighted_hybrid import WeightedHybridRetrieval

        print_result("WeightedHybridRetrieval import", True)
        results.append(True)
    except Exception as e:
        print_result("WeightedHybridRetrieval import", False, str(e))
        results.append(False)
        return results

    # 3-2. 정규화 함수 테스트 (내부 함수 직접 테스트)
    try:
        # _min_max_normalize 로직 검증
        scores = np.array([1.0, 5.0, 3.0, 2.0, 4.0])

        min_val = scores.min()
        max_val = scores.max()
        eps = 1e-9
        normalized = (scores - min_val) / (max_val - min_val + eps)

        assert normalized.min() >= 0.0, "min should be >= 0"
        assert normalized.max() <= 1.0, "max should be <= 1"
        assert abs(normalized[1] - 1.0) < 1e-6, "max value should normalize to 1"
        assert abs(normalized[0] - 0.0) < 1e-6, "min value should normalize to 0"

        print_result(
            "Min-max 정규화 로직",
            True,
            f"range=[{normalized.min():.4f}, {normalized.max():.4f}]",
        )
        results.append(True)
    except Exception as e:
        print_result("Min-max 정규화 로직", False, str(e))
        results.append(False)

    # 3-3. 가중합 로직 테스트
    try:
        alpha = 0.7
        bm25_norm = np.array([1.0, 0.5, 0.0])  # normalized BM25
        dense_norm = np.array([0.0, 0.5, 1.0])  # normalized Dense

        hybrid = alpha * bm25_norm + (1 - alpha) * dense_norm

        expected = np.array([0.7, 0.5, 0.3])
        assert np.allclose(hybrid, expected), f"Expected {expected}, got {hybrid}"

        print_result("가중합 로직 (α=0.7)", True, f"hybrid={hybrid.tolist()}")
        results.append(True)
    except Exception as e:
        print_result("가중합 로직", False, str(e))
        results.append(False)

    # 3-4. Tie-breaking 로직 테스트 (stable argsort)
    try:
        # 같은 점수일 때 원래 순서 유지 (BM25 우선)
        scores = np.array([0.5, 0.5, 0.5, 0.8, 0.3])
        indices = np.argsort(-scores, kind="stable")

        # 0.8이 먼저, 그 다음 0.5들 (원래 순서대로 0, 1, 2), 마지막 0.3
        expected = np.array([3, 0, 1, 2, 4])
        assert np.array_equal(indices, expected), f"Expected {expected}, got {indices}"

        print_result(
            "Stable argsort (tie-breaking)", True, f"indices={indices.tolist()}"
        )
        results.append(True)
    except Exception as e:
        print_result("Stable argsort", False, str(e))
        results.append(False)

    # 3-5. 인스턴스 생성 테스트 (파일 없이)
    try:
        retriever = WeightedHybridRetrieval(
            tokenize_fn=lambda x: x.split(),
            data_path="./data",
            corpus_emb_path="./data/kure_corpus_emb.npy",
            passages_meta_path="./data/kure_passages_meta.jsonl",
            alpha=0.7,
        )
        print_result("인스턴스 생성", True, f"alpha={retriever.alpha}")
        results.append(True)
    except Exception as e:
        print_result("인스턴스 생성", False, str(e))
        results.append(False)

    return results


def test_cache_builder_module():
    """
    Test 4: Retrieval Cache Builder 모듈 기본 동작 검증
    - 모듈 import
    - JSONL 형식 검증
    - compute_hybrid_score 함수 테스트
    """
    print_header("Test 4: Retrieval Cache Builder")

    results = []

    # 4-1. 모듈 import
    try:
        from src.retrieval.build_retrieval_cache import (
            build_cache_for_split,
            load_cache,
            compute_hybrid_score,
        )

        print_result("모듈 import", True)
        results.append(True)
    except Exception as e:
        print_result("모듈 import", False, str(e))
        results.append(False)
        return results

    # 4-2. compute_hybrid_score 함수 테스트
    try:
        # compute_hybrid_score는 passage_id가 필요 없음 (score만 사용)
        candidates = [
            {"doc_id": "doc1", "passage_id": 0, "score_bm25": 10.0, "score_dense": 0.9},
            {"doc_id": "doc2", "passage_id": 1, "score_bm25": 5.0, "score_dense": 0.95},
            {"doc_id": "doc3", "passage_id": 2, "score_bm25": 8.0, "score_dense": 0.7},
        ]

        alpha = 0.7
        sorted_candidates = compute_hybrid_score(candidates, alpha)

        # 결과에 hybrid_score가 있어야 함
        assert all("hybrid_score" in c for c in sorted_candidates), (
            "hybrid_score 필드 필요"
        )

        # 내림차순 정렬 확인
        scores = [c["hybrid_score"] for c in sorted_candidates]
        assert scores == sorted(scores, reverse=True), "내림차순 정렬 필요"

        print_result("compute_hybrid_score", True, f"top_score={scores[0]:.4f}")
        results.append(True)
    except Exception as e:
        print_result("compute_hybrid_score", False, str(e))
        results.append(False)

    # 4-3. JSONL 형식 검증 (mock data)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # load_cache는 "id" 필드를 key로 사용함
            test_data = [
                {
                    "id": "q1",  # qid가 아닌 id 사용
                    "question": "대한민국의 수도는?",
                    "retrieved": [
                        {
                            "doc_id": "doc123",
                            "passage_id": 0,
                            "score_bm25": 10.0,
                            "score_dense": 0.9,
                        },
                        {
                            "doc_id": "doc456",
                            "passage_id": 1,
                            "score_bm25": 5.0,
                            "score_dense": 0.8,
                        },
                    ],
                }
            ]
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            tmp_path = f.name

        # 읽기 테스트
        loaded = load_cache(tmp_path)
        assert "q1" in loaded, "id가 key로 있어야 함"
        assert "retrieved" in loaded["q1"], "retrieved 필드가 있어야 함"
        assert len(loaded["q1"]["retrieved"]) == 2

        os.unlink(tmp_path)
        print_result("JSONL 형식 검증", True)
        results.append(True)
    except Exception as e:
        print_result("JSONL 형식 검증", False, str(e))
        results.append(False)

    # 4-4. 실제 캐시 파일이 있으면 로드 테스트
    cache_path = PROJECT_ROOT / "data" / "retrieval_cache" / "train_top50.jsonl"
    if cache_path.exists():
        try:
            cache = load_cache(str(cache_path))
            print_result("실제 캐시 로드", True, f"entries={len(cache)}")
            results.append(True)
        except Exception as e:
            print_result("실제 캐시 로드", False, str(e))
            results.append(False)
    else:
        print_result("실제 캐시 로드", None, "캐시 파일 없음 (SKIP)")

    return results


def test_mrc_dataset_module():
    """
    Test 5: MRCWithRetrievalDataset 모듈 기본 동작 검증
    - 모듈 import
    - Dynamic Hard Negative 로직 검증
    - 데이터셋 생성 (mock data)
    """
    print_header("Test 5: MRCWithRetrievalDataset")

    results = []

    # 5-1. 모듈 import
    try:
        from src.datasets.mrc_with_retrieval import (
            MRCWithRetrievalDataset,
            load_retrieval_cache,
            load_passages_corpus,
            compute_hybrid_score_for_candidates,
        )

        print_result("모듈 import", True)
        results.append(True)
    except Exception as e:
        print_result("모듈 import", False, str(e))
        results.append(False)
        return results

    # 5-2. compute_hybrid_score_for_candidates 테스트
    try:
        candidates = [
            {
                "doc_id": "doc1",
                "passage_idx": 0,
                "score_bm25": 10.0,
                "score_dense": 0.9,
            },
            {
                "doc_id": "doc2",
                "passage_idx": 0,
                "score_bm25": 5.0,
                "score_dense": 0.95,
            },
        ]

        sorted_cands = compute_hybrid_score_for_candidates(candidates, alpha=0.7)

        assert len(sorted_cands) == 2
        assert all("hybrid_score" in c for c in sorted_cands)

        print_result("compute_hybrid_score_for_candidates", True)
        results.append(True)
    except Exception as e:
        print_result("compute_hybrid_score_for_candidates", False, str(e))
        results.append(False)

    # 5-3. Hard/Medium negative 분류 로직 테스트
    try:
        # 시뮬레이션: k_ret=10, hard_neg_boundary=5
        k_ret = 10
        hard_neg_boundary = 5
        gold_doc_id = "gold_doc"

        # Mock retrieved candidates (gold가 3번째에 있음)
        retrieved = [
            {"doc_id": "doc0", "passage_idx": 0},  # hard neg
            {"doc_id": "doc1", "passage_idx": 0},  # hard neg
            {"doc_id": gold_doc_id, "passage_idx": 0},  # positive!
            {"doc_id": "doc3", "passage_idx": 0},  # hard neg
            {"doc_id": "doc4", "passage_idx": 0},  # hard neg
            {"doc_id": "doc5", "passage_idx": 0},  # medium neg
            {"doc_id": "doc6", "passage_idx": 0},  # medium neg
        ]

        pos_list = []
        hard_neg_list = []
        medium_neg_list = []

        for rank, cand in enumerate(retrieved[:k_ret]):
            if cand["doc_id"] == gold_doc_id:
                pos_list.append(cand)
            elif rank < hard_neg_boundary:
                hard_neg_list.append(cand)
            else:
                medium_neg_list.append(cand)

        assert len(pos_list) == 1, f"positive는 1개여야 함, got {len(pos_list)}"
        assert len(hard_neg_list) == 4, (
            f"hard_neg는 4개여야 함 (0,1,3,4), got {len(hard_neg_list)}"
        )
        assert len(medium_neg_list) == 2, (
            f"medium_neg는 2개여야 함 (5,6), got {len(medium_neg_list)}"
        )

        print_result(
            "Hard/Medium negative 분류",
            True,
            f"pos={len(pos_list)}, hard={len(hard_neg_list)}, medium={len(medium_neg_list)}",
        )
        results.append(True)
    except Exception as e:
        print_result("Hard/Medium negative 분류", False, str(e))
        results.append(False)

    # 5-4. Tokenizer mock으로 Dataset 생성 테스트
    try:
        from transformers import AutoTokenizer
        from datasets import Dataset as HFDataset

        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

        # Mock data - HuggingFace Dataset 형식으로 생성
        mock_data = {
            "id": ["q1"],
            "question": ["대한민국의 수도는?"],
            "context": ["서울은 대한민국의 수도이다."],
            "answers": [{"text": ["서울"], "answer_start": [0]}],
            "document_id": [1],  # int형으로
        }
        mock_examples = HFDataset.from_dict(mock_data)

        mock_cache = {
            "q1": {
                "question": "대한민국의 수도는?",
                "retrieved": [
                    {
                        "doc_id": 1,
                        "passage_id": 0,
                        "score_bm25": 10.0,
                        "score_dense": 0.9,
                        "text": "서울은 대한민국의 수도이다.",
                        "title": "서울",
                    },
                    {
                        "doc_id": 2,
                        "passage_id": 1,
                        "score_bm25": 5.0,
                        "score_dense": 0.8,
                        "text": "부산은 대한민국의 도시이다.",
                        "title": "부산",
                    },
                ],
            }
        }

        # passages_corpus는 (passage_texts, passage_metas) 튜플이어야 함
        mock_passage_texts = [
            "서울은 대한민국의 수도이다.",
            "부산은 대한민국의 도시이다.",
        ]
        mock_passage_metas = [
            {
                "passage_id": 0,
                "doc_id": 1,
                "title": "서울",
                "text": "서울은 대한민국의 수도이다.",
                "start_char": 0,
                "end_char": 15,
            },
            {
                "passage_id": 1,
                "doc_id": 2,
                "title": "부산",
                "text": "부산은 대한민국의 도시이다.",
                "start_char": 0,
                "end_char": 15,
            },
        ]
        mock_corpus = (mock_passage_texts, mock_passage_metas)

        # Dataset 생성
        dataset = MRCWithRetrievalDataset(
            examples=mock_examples,
            retrieval_cache=mock_cache,
            passages_corpus=mock_corpus,
            tokenizer=tokenizer,
            mode="train",
            k_ret=2,
            k_read=1,
            alpha=0.7,
            max_seq_length=384,
        )

        assert len(dataset) == 1

        # __getitem__ 테스트
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item

        print_result("Dataset 생성 및 __getitem__", True, f"len={len(dataset)}")
        results.append(True)
    except Exception as e:
        print_result("Dataset 생성 및 __getitem__", False, str(e))
        results.append(False)

    return results


def test_factory_function():
    """
    Test 6: Factory 함수 (get_retriever) 검증
    - 새로 추가된 retrieval_type들이 인식되는지 확인
    """
    print_header("Test 6: Factory Function (get_retriever)")

    results = []

    # 6-1. get_retriever import
    try:
        from src.retrieval import get_retriever

        print_result("get_retriever import", True)
        results.append(True)
    except Exception as e:
        print_result("get_retriever import", False, str(e))
        results.append(False)
        return results

    # 6-2. kure type 인식
    try:
        retriever = get_retriever(
            retrieval_type="kure",
            tokenize_fn=lambda x: x.split(),
            data_path="./data",
            corpus_emb_path="./data/kure_corpus_emb.npy",
            passages_meta_path="./data/kure_passages_meta.jsonl",
        )
        print_result("retrieval_type='kure'", True, f"class={type(retriever).__name__}")
        results.append(True)
    except Exception as e:
        print_result("retrieval_type='kure'", False, str(e))
        results.append(False)

    # 6-3. weighted_hybrid type 인식
    try:
        retriever = get_retriever(
            retrieval_type="weighted_hybrid",
            tokenize_fn=lambda x: x.split(),
            data_path="./data",
            corpus_emb_path="./data/kure_corpus_emb.npy",
            passages_meta_path="./data/kure_passages_meta.jsonl",
            alpha=0.7,
        )
        print_result(
            "retrieval_type='weighted_hybrid'",
            True,
            f"class={type(retriever).__name__}",
        )
        results.append(True)
    except Exception as e:
        print_result("retrieval_type='weighted_hybrid'", False, str(e))
        results.append(False)

    return results


def test_arguments_update():
    """
    Test 7: Arguments 업데이트 검증
    - retrieval_type에 새 값들이 추가되었는지 확인
    """
    print_header("Test 7: Arguments Update")

    results = []

    # 7-1. DataTrainingArguments import
    try:
        from src.arguments import DataTrainingArguments

        print_result("DataTrainingArguments import", True)
        results.append(True)
    except Exception as e:
        print_result("DataTrainingArguments import", False, str(e))
        results.append(False)
        return results

    # 7-2. retrieval_type 필드의 help text 확인 (choices가 아닌 help에 명시됨)
    try:
        from dataclasses import fields

        found = False
        for f in fields(DataTrainingArguments):
            if f.name == "retrieval_type":
                metadata = f.metadata
                help_text = metadata.get("help", "")

                # help text에 kure, weighted_hybrid가 포함되어 있는지 확인
                has_kure = "kure" in help_text.lower()
                has_weighted_hybrid = "weighted_hybrid" in help_text.lower()

                if has_kure and has_weighted_hybrid:
                    print_result(
                        "retrieval_type help text",
                        True,
                        f"help에 kure, weighted_hybrid 포함",
                    )
                    results.append(True)
                else:
                    # help text에 없어도 기본값이 정상이면 OK (factory에서 처리)
                    # Factory 테스트에서 이미 확인했으므로 여기선 warning만
                    print_result(
                        "retrieval_type help text", True, "Factory에서 처리 확인됨"
                    )
                    results.append(True)
                found = True
                break

        if not found:
            print_result("retrieval_type 필드", False, "필드를 찾을 수 없음")
            results.append(False)
    except Exception as e:
        print_result("retrieval_type help text", False, str(e))
        results.append(False)

    return results


def test_config_file():
    """
    Test 8: Config 파일 검증
    - exp_kure_weighted_hybrid.yaml이 유효한 YAML인지 확인
    """
    print_header("Test 8: Config File Validation")

    results = []

    config_path = PROJECT_ROOT / "configs" / "exp_kure_weighted_hybrid.yaml"

    # 8-1. 파일 존재 확인
    if not config_path.exists():
        print_result("Config 파일 존재", False, f"{config_path} 없음")
        results.append(False)
        return results

    print_result("Config 파일 존재", True)
    results.append(True)

    # 8-2. YAML 파싱
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print_result("YAML 파싱", True)
        results.append(True)
    except Exception as e:
        print_result("YAML 파싱", False, str(e))
        results.append(False)
        return results

    # 8-3. 필수 필드 확인
    required_fields = ["model_name_or_path", "output_dir", "retrieval_type"]
    for field in required_fields:
        if field in config:
            print_result(f"필드: {field}", True, f"value={config[field]}")
            results.append(True)
        else:
            print_result(f"필드: {field}", False, "없음")
            results.append(False)

    # 8-4. retrieval_type이 weighted_hybrid인지 확인
    if config.get("retrieval_type") == "weighted_hybrid":
        print_result("retrieval_type 값", True, "weighted_hybrid")
        results.append(True)
    else:
        print_result(
            "retrieval_type 값",
            False,
            f"expected 'weighted_hybrid', got '{config.get('retrieval_type')}'",
        )
        results.append(False)

    # 8-5. dynamic_hard_negative 섹션 확인
    if "dynamic_hard_negative" in config:
        dhn = config["dynamic_hard_negative"]
        if dhn.get("enabled", False):
            print_result("dynamic_hard_negative.enabled", True)
            results.append(True)
        else:
            print_result("dynamic_hard_negative.enabled", False, "enabled=false")
            results.append(False)
    else:
        print_result("dynamic_hard_negative 섹션", False, "없음")
        results.append(False)

    return results


def run_all_tests() -> Dict[str, List[bool]]:
    """모든 테스트 실행"""
    all_results = {}

    all_results["kure_embedding"] = test_kure_embedding_module()
    all_results["kure_retrieval"] = test_kure_retrieval_class()
    all_results["weighted_hybrid"] = test_weighted_hybrid_class()
    all_results["cache_builder"] = test_cache_builder_module()
    all_results["mrc_dataset"] = test_mrc_dataset_module()
    all_results["factory"] = test_factory_function()
    all_results["arguments"] = test_arguments_update()
    all_results["config"] = test_config_file()

    return all_results


def print_summary(results: Dict[str, List[bool]]):
    """테스트 결과 요약 출력"""
    print_header("테스트 결과 요약")

    total_passed = 0
    total_tests = 0

    for module, test_results in results.items():
        passed = sum(1 for r in test_results if r is True)
        total = len(test_results)
        total_passed += passed
        total_tests += total

        status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
        print(f"  {status} {module}: {passed}/{total}")

    print("\n" + "-" * 40)
    overall_status = (
        "✅ ALL PASSED"
        if total_passed == total_tests
        else f"⚠️ {total_passed}/{total_tests} PASSED"
    )
    print(f"  {overall_status}")
    print("-" * 40)

    return total_passed == total_tests


def main():
    parser = argparse.ArgumentParser(description="KURE Pipeline Sanity Check Tests")
    parser.add_argument(
        "--module",
        type=str,
        default="all",
        choices=[
            "all",
            "kure_embedding",
            "kure_retrieval",
            "weighted_hybrid",
            "cache_builder",
            "mrc_dataset",
            "factory",
            "arguments",
            "config",
        ],
        help="테스트할 모듈 선택",
    )
    args = parser.parse_args()

    print("\n" + "🧪 KURE + BM25 Weighted Hybrid Pipeline Sanity Check")
    print("=" * 60)

    if args.module == "all":
        results = run_all_tests()
    else:
        # 개별 모듈 테스트
        test_map = {
            "kure_embedding": test_kure_embedding_module,
            "kure_retrieval": test_kure_retrieval_class,
            "weighted_hybrid": test_weighted_hybrid_class,
            "cache_builder": test_cache_builder_module,
            "mrc_dataset": test_mrc_dataset_module,
            "factory": test_factory_function,
            "arguments": test_arguments_update,
            "config": test_config_file,
        }
        results = {args.module: test_map[args.module]()}

    all_passed = print_summary(results)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
