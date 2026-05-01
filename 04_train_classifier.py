"""
=============================================================================
SCRIPT: 04_train_classifier.py
PURPOSE: Trains the fallacy classification model.
         Uses DeBERTa-v3-base (or xsmall for faster training) fine-tuned on
         the NER-normalized fallacy dataset from data/processed/train.csv.
         
         What DeBERTa does:
         - Reads the normalized argument text.
         - Produces a 768-dim representation of the full text.
         - A linear classification head maps this to one of the fallacy classes.
         - Trained by minimizing Cross-Entropy Loss between predictions and labels.
         
         Parallelism:
         - torch.set_num_threads uses all available CPU threads.
         - DataLoader uses multiple worker processes for parallel data loading.
         - Gradient accumulation simulates larger batch sizes without extra RAM.
         
         Training loop uses class weights to handle class imbalance
         (some fallacy types have fewer examples than others).
         
RUN:     python scripts/04_train_classifier.py
         Optional flags (set at top of CONFIG section):
         - Set USE_XSMALL=True for faster but slightly lower quality training.
=============================================================================
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
MODEL_OUT  = os.path.join(BASE_DIR, "models", "deberta_classifier")
LABEL_MAP  = os.path.join(PROCESSED, "label_map.json")
TRAIN_CSV  = os.path.join(PROCESSED, "train.csv")
VAL_CSV    = os.path.join(PROCESSED, "val.csv")

# ── Configuration ──────────────────────────────────────────────────────────
# Use deberta-v3-xsmall (22M params) for faster CPU training.
# Switch to deberta-v3-base (86M params) for better accuracy but 3–4x slower.
USE_XSMALL = True   # ← Set to False to use base model (slower, more accurate)

MODEL_NAME = (
    "microsoft/deberta-v3-xsmall" if USE_XSMALL
    else "microsoft/deberta-v3-base"
)

MAX_SEQ_LEN   = 256    # Maximum number of tokens per input
BATCH_SIZE    = 8      # Mini-batch size per gradient step
GRAD_ACCUM    = 4      # Gradient accumulation steps (effective batch = 32)
LEARNING_RATE = 2e-5
EPOCHS        = 5
WARMUP_RATIO  = 0.1    # 10% of total steps used for learning rate warm-up
SEED          = 42
NUM_WORKERS   = 4      # DataLoader worker processes for parallel data loading

# Maximize CPU threads on i5-1235U
torch.set_num_threads(min(torch.get_num_threads(), 10))
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset class ──────────────────────────────────────────────────────────
class FallacyDataset(Dataset):
    """
    PyTorch Dataset for fallacy classification.
    
    Wraps the CSV data and tokenizes on-the-fly during training.
    The __getitem__ method returns one tokenized example for each index.
    The DataLoader calls this in parallel across NUM_WORKERS processes.
    
    Input columns used: text_normalized (NER-normalized text), label_id (int)
    Falls back to 'text' if text_normalized is not present.
    """
    def __init__(self, csv_path: str, tokenizer, max_len: int):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Use normalized text if available, fall back to original
        self.text_col = "text_normalized" if "text_normalized" in self.df.columns else "text"
        
        # Drop rows with missing text or label
        self.df = self.df.dropna(subset=[self.text_col, "label_id"]).reset_index(drop=True)
        self.df["label_id"] = self.df["label_id"].astype(int)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        label = int(row["label_id"])
        
        # Tokenize: converts text → token IDs + attention mask
        # padding="max_length" pads all sequences to the same length for batching
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long)
        }


# ── Training utilities ─────────────────────────────────────────────────────

def compute_class_weights(csv_path: str, num_classes: int) -> torch.Tensor:
    """
    Computes inverse-frequency class weights to handle class imbalance.
    
    If fallacy type X has 200 examples and fallacy type Y has 20 examples,
    type Y gets 10x more weight in the loss function so the model doesn't
    just learn to predict type X for everything.
    
    Formula: weight[i] = total_samples / (num_classes × count[i])
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)
    
    counts = np.zeros(num_classes)
    for lbl in df["label_id"]:
        counts[int(lbl)] += 1
    
    total = counts.sum()
    weights = total / (num_classes * (counts + 1e-8))  # +epsilon avoids div-by-zero
    print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float)


def evaluate(model, loader, device, criterion):
    """
    Runs the model on the validation set and returns:
    - Average loss
    - Macro F1 score (treats all fallacy classes equally)
    - Per-class classification report
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits   # shape: (batch_size, num_classes)
            loss    = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_labels, all_preds


# ── Main training loop ─────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("SCRIPT 04: TRAIN DEBERTA FALLACY CLASSIFIER")
    print("=" * 60)
    
    if not os.path.exists(LABEL_MAP):
        print("ERROR: label_map.json not found. Run script 02 first.")
        sys.exit(1)
    
    # Load label map to determine number of classes
    with open(LABEL_MAP) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    id_to_label = {v: k for k, v in label_map.items()}
    print(f"\n  Number of fallacy classes: {num_classes}")
    print(f"  Labels: {list(label_map.keys())}")
    print(f"  Model:  {MODEL_NAME}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    # ── Load tokenizer and model ───────────────────────────────────────────
    print(f"\n  Loading tokenizer and model (first run downloads weights)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {total_params:.1f}M")
    
    # ── Datasets and DataLoaders ───────────────────────────────────────────
    print(f"\n  Loading datasets...")
    train_dataset = FallacyDataset(TRAIN_CSV, tokenizer, MAX_SEQ_LEN)
    val_dataset   = FallacyDataset(VAL_CSV,   tokenizer, MAX_SEQ_LEN)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    
    # num_workers > 0 enables parallel data loading (fetching next batch
    # while model is processing the current one)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    
    # ── Class weights and loss ─────────────────────────────────────────────
    print("\n  Computing class weights for imbalanced training...")
    class_weights = compute_class_weights(TRAIN_CSV, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # ── Optimizer and scheduler ───────────────────────────────────────────
    # AdamW is the standard optimizer for transformer fine-tuning.
    # It uses weight decay on all parameters except biases and layer norms.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    
    # Total training steps (accounting for gradient accumulation)
    total_steps   = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    
    # Linear warmup: LR increases from 0 → LR during warmup steps,
    # then linearly decreases from LR → 0 for the rest of training.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\n  Training config:")
    print(f"    Epochs:              {EPOCHS}")
    print(f"    Batch size:          {BATCH_SIZE}")
    print(f"    Gradient accum:      {GRAD_ACCUM}  (effective batch = {BATCH_SIZE * GRAD_ACCUM})")
    print(f"    Learning rate:       {LEARNING_RATE}")
    print(f"    Total steps:         {total_steps}")
    print(f"    Warmup steps:        {warmup_steps}")
    
    # ── Training loop ──────────────────────────────────────────────────────
    best_val_f1 = 0.0
    os.makedirs(MODEL_OUT, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"  Epoch {epoch}/{EPOCHS} [Train]")
        
        for step, batch in loop:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            
            # Forward pass: model predicts logits for each class
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits   # shape: (batch_size, num_classes)
            
            # Compute loss — cross-entropy between predicted logits and true labels
            loss = criterion(logits, labels)
            
            # Gradient accumulation: divide loss by GRAD_ACCUM so accumulated
            # gradients equal the true gradient for the effective batch
            loss = loss / GRAD_ACCUM
            
            # Backward pass: compute gradients
            loss.backward()
            
            epoch_loss += loss.item() * GRAD_ACCUM
            
            # Only update weights after accumulating GRAD_ACCUM mini-batches
            if (step + 1) % GRAD_ACCUM == 0:
                # Clip gradients: prevents exploding gradients, stabilizes training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            loop.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")
        
        # ── Validation ────────────────────────────────────────────────────
        print(f"\n  Running validation for epoch {epoch}...")
        val_loss, val_f1, all_labels, all_preds = evaluate(
            model, val_loader, device, criterion
        )
        
        print(f"  Epoch {epoch} | Train Loss: {epoch_loss/len(train_loader):.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        
        # Print per-class report
        label_names = [id_to_label[i] for i in range(num_classes)]
        report = classification_report(
            all_labels, all_preds, target_names=label_names, zero_division=0
        )
        print(f"\n  Classification Report (Epoch {epoch}):")
        print(report)
        
        # Save checkpoint if this is the best validation F1 so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(MODEL_OUT)
            tokenizer.save_pretrained(MODEL_OUT)
            print(f"  ✓ Best model saved (Val F1: {best_val_f1:.4f}) → {MODEL_OUT}")
        
        # Also save epoch checkpoint for safety
        epoch_dir = os.path.join(MODEL_OUT, f"epoch_{epoch}")
        model.save_pretrained(epoch_dir)
        print(f"  Checkpoint saved: {epoch_dir}")
    
    print("\n" + "=" * 60)
    print(f"Training complete. Best validation Macro F1: {best_val_f1:.4f}")
    print(f"Best model saved to: {MODEL_OUT}")
    print("Next: python scripts/05_train_generator.py")
    print("=" * 60)


if __name__ == "__main__":
    train()
