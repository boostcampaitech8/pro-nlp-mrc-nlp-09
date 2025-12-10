"""
Cross-Encoder Re-ranker 학습
- BM25로 추출한 hard negative를 사용하여 cross-encoder 학습
- Query-Document pair를 함께 인코딩하여 relevance score 예측
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

# ============================
# Config
# ============================
BASE_MODEL = "klue/bert-base"  # 한국어 모델
# BASE_MODEL = "bert-base-multilingual-cased"  # Multilingual
BATCH_SIZE = 8  # Cross-encoder는 메모리 많이 씀
MAX_LEN = 512
LR = 2e-5
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PATH = "./outputs/reranker/cross_encoder"

print(f"Device: {DEVICE}")
print(f"Base Model: {BASE_MODEL}")

# ============================
# Dataset 로드
# ============================
print("\nLoading dataset with hard negatives...")
dataset = load_from_disk("./data/train_dataset/negative")

print(f"Dataset size: {len(dataset)}")
print(f"Sample keys: {dataset[0].keys()}")

# ============================
# Cross-Encoder Dataset
# ============================
class CrossEncoderDataset(Dataset):
    """
    Cross-encoder 학습용 데이터셋
    - Positive: (query, positive_doc) → label=1
    - Negative: (query, negative_doc) → label=0
    """
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 데이터를 (query, doc, label) 형태로 변환
        self.samples = []
        for item in hf_dataset:
            question = item["question"]
            positive = item["positive"]
            negatives = item["negatives"]
            
            # Positive pair
            self.samples.append({
                "query": question,
                "document": positive,
                "label": 1
            })
            
            # Negative pairs (모든 negative 사용)
            for neg in negatives:
                self.samples.append({
                    "query": question,
                    "document": str(neg),
                    "label": 0
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Query와 Document를 [SEP]로 결합
        encoding = self.tokenizer(
            sample["query"],
            sample["document"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.float)
        }

# ============================
# Model & Tokenizer
# ============================
print("\nInitializing model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Binary classification (relevant=1, irrelevant=0)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1  # Binary classification with sigmoid
).to(DEVICE)

# ============================
# DataLoader
# ============================
print("\nPreparing data loaders...")
train_dataset = CrossEncoderDataset(dataset, tokenizer, MAX_LEN)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

print(f"Total training samples: {len(train_dataset)}")
print(f"Positive samples: {len(dataset)}")
print(f"Negative samples: {len(train_dataset) - len(dataset)}")
print(f"Batches per epoch: {len(train_loader)}")

# ============================
# Training Setup
# ============================
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

# BCE Loss (Binary Cross Entropy)
criterion = nn.BCEWithLogitsLoss()

# ============================
# Training Loop
# ============================
print("\n" + "="*60)
print("Starting Training")
print("="*60)

model.train()
best_loss = float('inf')

for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits.squeeze(-1)  # [batch_size]
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    # Epoch summary
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"  → New best model! Saving...")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        model.save_pretrained(OUTPUT_PATH)
        tokenizer.save_pretrained(OUTPUT_PATH)

print("\n" + "="*60)
print("Training Completed!")
print("="*60)
print(f"Best Loss: {best_loss:.4f}")
print(f"Model saved at: {OUTPUT_PATH}")

# ============================
# Final Save
# ============================
final_path = OUTPUT_PATH + "_final"
os.makedirs(final_path, exist_ok=True)
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"Final model saved at: {final_path}")