"""
KorQuAD 데이터셋과 기존 train_dataset을 결합하는 스크립트
"""
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict

def convert_korquad_format(example):
    """
    KorQuAD의 answer 형식을 train_dataset의 answers 형식으로 변환
    
    KorQuAD format:
    {
        "answer": {
            "text": "답변",
            "answer_start": 123
        }
    }
    
    Target format:
    {
        "answers": {
            "text": ["답변"],
            "answer_start": [123]
        }
    }
    """
    if "answer" in example:
        # KorQuAD의 answer를 answers 형식으로 변환
        example["answers"] = {
            "text": [example["answer"]["text"]],
            "answer_start": [example["answer"]["answer_start"]]
        }
        # 기존 answer 필드 제거
        del example["answer"]
    
    return example

def add_missing_columns(example, idx):
    """
    필요한 컬럼이 없으면 추가
    """
    if "__index_level_0__" not in example:
        example["__index_level_0__"] = idx
    
    if "document_id" not in example:
        example["document_id"] = 0  # 기본값 설정
    
    return example

def main():
    print("=" * 50)
    print("데이터셋 로딩 중...")
    print("=" * 50)
    
    # 1. 기존 train_dataset 로드
    train_dataset = load_from_disk("./data/train_dataset")
    print(f"\n기존 train_dataset 로드 완료")
    print(f"Train 샘플 수: {len(train_dataset['train'])}")
    if 'validation' in train_dataset:
        print(f"Validation 샘플 수: {len(train_dataset['validation'])}")
    
    # 2. KorQuAD 데이터셋 로드
    korquad = load_dataset("LGCNS/KorQuAD_1.0")
    print(f"\nKorQuAD 데이터셋 로드 완료")
    print(f"Train 샘플 수: {len(korquad['train'])}")
    print(f"Validation 샘플 수: {len(korquad['validation'])}")
    
    print("\n" + "=" * 50)
    print("KorQuAD 데이터셋 변환 중...")
    print("=" * 50)
    
    # 3. KorQuAD 형식 변환
    korquad_converted = korquad.map(
        convert_korquad_format,
        desc="Converting KorQuAD format"
    )
    
    # 4. 누락된 컬럼 추가
    korquad_converted = korquad_converted.map(
        add_missing_columns,
        with_indices=True,
        desc="Adding missing columns"
    )
    
    print("\n변환 완료!")
    print(f"KorQuAD train 컬럼: {korquad_converted['train'].column_names}")
    
    print("\n" + "=" * 50)
    print("데이터셋 결합 중...")
    print("=" * 50)
    
    # 5. 데이터셋 결합
    # Train 데이터셋 결합
    combined_train = concatenate_datasets([
        train_dataset['train'],
        korquad_converted['train']
    ])
    
    print(f"\n결합된 train 샘플 수: {len(combined_train)}")
    
    # Validation 데이터셋 결합 (존재하는 경우)
    if 'validation' in train_dataset:
        combined_validation = concatenate_datasets([
            train_dataset['validation'],
            korquad_converted['validation']
        ])
        print(f"결합된 validation 샘플 수: {len(combined_validation)}")
        
        # DatasetDict 생성
        combined_dataset = DatasetDict({
            'train': combined_train,
            'validation': combined_validation
        })
    else:
        combined_dataset = DatasetDict({
            'train': combined_train,
            'validation': korquad_converted['validation']
        })
    
    print("\n" + "=" * 50)
    print("결합된 데이터셋 저장 중...")
    print("=" * 50)
    
    # 6. 결합된 데이터셋 저장
    output_path = "./data/combined_train_dataset"
    combined_dataset.save_to_disk(output_path)
    
    print(f"\n✅ 결합된 데이터셋이 '{output_path}'에 저장되었습니다!")
    print("\n최종 데이터셋 정보:")
    print(f"  - Train: {len(combined_dataset['train'])} 샘플")
    print(f"  - Validation: {len(combined_dataset['validation'])} 샘플")
    print(f"  - 컬럼: {combined_dataset['train'].column_names}")
    
    # 샘플 확인
    print("\n" + "=" * 50)
    print("샘플 데이터 확인")
    print("=" * 50)
    sample = combined_dataset['train'][0]
    print(f"\nQuestion: {sample['question']}")
    print(f"Answer: {sample['answers']}")
    print(f"Context (처음 100자): {sample['context'][:100]}...")

if __name__ == "__main__":
    main()