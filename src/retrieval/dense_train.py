import torch
import os
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F

def dpr_train(model_name, output_dir, hard_sample_path):
    # ============================
    # Config
    # ============================
    MODEL_NAME = model_name
    BATCH_SIZE = 32
    MAX_LEN = 512
    LR = 2e-5
    EPOCHS = 3
    TEMPERATURE = 0.1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================
    # Dataset
    # ============================
    dataset = load_from_disk(hard_sample_path)

    class DPRDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self.data = hf_dataset

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return len(self.data)

    # ============================
    # Custom collate_fn
    # ============================
    def collate_fn(batch):
        questions = [item["question"] for item in batch]
        positives = [item["positive"] for item in batch]
        return {"question": questions, "positive": positives}

    train_loader = DataLoader(
        DPRDataset(dataset),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    # ============================
    # Model & Tokenizer
    # ============================
    q_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    p_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    for t in [q_tokenizer, p_tokenizer]:
        if t.pad_token is None:
            t.pad_token = t.eos_token

    q_encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    p_encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # ============================
    # Utilities
    # ============================
    def encode_batch(encoder, tokenizer, texts):
        inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        emb = encoder(**inputs).pooler_output
        return F.normalize(emb, p=2, dim=1)

    def inbatch_dpr_loss(q_vec, p_vec):
        """
        q_vec: [B, H], p_vec: [B, H]
        logits: [B, B], 각 q_i와 모든 p_j의 cosine similarity
        """
        logits = q_vec @ p_vec.T / TEMPERATURE
        labels = torch.arange(q_vec.size(0), device=DEVICE)
        return F.cross_entropy(logits, labels)

    # ============================
    # Optimizer & Scheduler
    # ============================
    optimizer = AdamW(list(q_encoder.parameters()) + list(p_encoder.parameters()), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # ============================
    # Training loop
    # ============================
    q_encoder.train()
    p_encoder.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            questions = batch["question"]
            positives = batch["positive"]

            # batch 내 모든 q와 p embedding
            q_vec = encode_batch(q_encoder, q_tokenizer, questions)
            p_vec = encode_batch(p_encoder, p_tokenizer, positives)

            loss = inbatch_dpr_loss(q_vec, p_vec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    # ============================
    # Save models
    # ============================
    q_save_path = os.path.join(output_dir, "question_encoder")
    q_encoder.save_pretrained(q_save_path)
    q_tokenizer.save_pretrained(q_save_path)

    p_save_path = os.path.join(output_dir, "context_encoder")
    p_encoder.save_pretrained(p_save_path)
    p_tokenizer.save_pretrained(p_save_path)

    print("DPR training completed and saved.")
