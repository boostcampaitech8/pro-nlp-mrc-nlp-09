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
    BATCH_SIZE = 4
    MAX_LEN = 512
    LR = 3e-5
    EPOCHS = 2
    TEMPERATURE = 0.05
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

    train_loader = DataLoader(DPRDataset(dataset), batch_size=BATCH_SIZE, shuffle=True)

    # ============================
    # Model & Tokenizer
    # ============================
    q_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    p_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    for t in [q_tokenizer, p_tokenizer]:
        if t.pad_token is None: t.pad_token = t.eos_token

    q_encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    p_encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # ============================
    # Utilities
    # ============================
    def encode_batch(encoder, tokenizer, texts):
        inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        emb = encoder(**inputs).pooler_output
        return F.normalize(emb, p=2, dim=1)

    def dpr_loss(q_vec, p_pos_vec, neg_embeds, neg_counts):
        pos_sim = F.cosine_similarity(q_vec, p_pos_vec, dim=-1) / TEMPERATURE
        max_count = max(neg_counts) if len(neg_counts) > 0 else 0
        neg_sims = torch.full((q_vec.size(0), max_count), -1e9, device=DEVICE)

        start = 0
        for i, count in enumerate(neg_counts):
            if i >= q_vec.size(0):
                break
            if count > 0:
                sim = F.cosine_similarity(q_vec[i].unsqueeze(0), neg_embeds[start:start+count], dim=-1)
                neg_sims[i, :count] = sim
                start += count

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1) / TEMPERATURE
        labels = torch.zeros(q_vec.size(0), dtype=torch.long, device=DEVICE)
        return F.cross_entropy(logits, labels)

    # ============================
    # Optimizer & Scheduler
    # ============================
    optimizer = AdamW(list(q_encoder.parameters()) + list(p_encoder.parameters()), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

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
            negatives = batch["negatives"]

            q_vec = encode_batch(q_encoder, q_tokenizer, questions)
            p_pos_vec = encode_batch(p_encoder, p_tokenizer, positives)

            # Flatten negatives safely
            flattened_neg = [str(n) for neg_list in negatives for n in neg_list]
            neg_counts = [len(neg_list) for neg_list in negatives]
            neg_embeds = encode_batch(p_encoder, p_tokenizer, flattened_neg) if len(flattened_neg) > 0 else torch.empty(0, q_vec.size(1), device=DEVICE)

            loss = dpr_loss(q_vec, p_pos_vec, neg_embeds, neg_counts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # ============================
    # Save models
    # ============================
    # Question encoder 저장
    q_save_path = os.path.join(output_dir, "question_encoder")
    q_encoder.save_pretrained(q_save_path)
    q_tokenizer.save_pretrained(q_save_path)

    # Context encoder 저장
    p_save_path = os.path.join(output_dir, "context_encoder")
    p_encoder.save_pretrained(p_save_path)
    p_tokenizer.save_pretrained(p_save_path)

    print("DPR training completed and saved.")
    pass
