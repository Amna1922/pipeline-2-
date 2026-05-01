"""
=============================================================================
SCRIPT: 05_train_generator.py
PURPOSE: Trains the RAG-Token explanation generator.
         Uses T5-small (60M parameters) fine-tuned to produce natural language
         explanations of fallacy detections, grounded in retrieved passages.
         
         Training data comes from two sources (combined):
         1. Fallacy-specific template explanations — generated automatically
            from the training set (each fallacy type has a template that
            produces a contrastive explanation for that type).
         2. SNLI pretraining data (if snli_clean.csv was provided) — teaches
            the generator to produce explanation-style text in general.
         
         The generator is trained as a seq2seq model:
         INPUT:  "classify and explain: [ARGUMENT] context: [PASSAGE_1] [PASSAGE_2]..."
         OUTPUT: "Fallacy: [LABEL]. Explanation: [NATURAL LANGUAGE EXPLANATION]"
         
         Parallelism: DataLoader multi-worker prefetching, CPU multi-threading.
         
RUN:     python scripts/05_train_generator.py
=============================================================================
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
MODEL_OUT  = os.path.join(BASE_DIR, "models", "generator")
LABEL_MAP  = os.path.join(PROCESSED, "label_map.json")
TRAIN_CSV  = os.path.join(PROCESSED, "train.csv")
VAL_CSV    = os.path.join(PROCESSED, "val.csv")
SNLI_JSONL = os.path.join(PROCESSED, "snli_generator_train.jsonl")
CORPUS_DIR = os.path.join(BASE_DIR, "data", "retrieval_corpus")
KIALO_FILE = os.path.join(CORPUS_DIR, "kialo_passages.jsonl")
GEN_TRAIN_OUT = os.path.join(PROCESSED, "generator_train.jsonl")
GEN_VAL_OUT   = os.path.join(PROCESSED, "generator_val.jsonl")

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME     = "google/t5-small"   # 60M params; use t5-base (220M) if RAM allows
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 128
BATCH_SIZE     = 4
GRAD_ACCUM     = 8                   # Effective batch = 32
LEARNING_RATE  = 3e-4                # T5 uses higher LR than BERT-family
EPOCHS         = 3
WARMUP_RATIO   = 0.1
NUM_WORKERS    = 4
SEED           = 42

torch.set_num_threads(min(torch.get_num_threads(), 10))
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── Fallacy explanation templates ──────────────────────────────────────────
# These templates are filled with the actual argument text and fallacy label
# to create training pairs for the generator.
# Format: {argument} is replaced with the actual argument text.
# {alt} is replaced with another fallacy label to create contrast.

TEMPLATES = {
    "ad hominem": (
        "Fallacy: Ad Hominem. "
        "Explanation: This argument commits the Ad Hominem fallacy. "
        "Instead of addressing the logical content of the opposing claim, "
        "it attacks the character, credibility, or personal attributes of the person making the argument. "
        "This is distinct from {alt}, which involves a different form of faulty reasoning."
    ),
    "appeal to popularity": (
        "Fallacy: Appeal to Popularity. "
        "Explanation: This argument commits the Appeal to Popularity fallacy (also called Ad Populum). "
        "It argues that something is true or good simply because many people believe or do it, "
        "without providing logical evidence. Unlike {alt}, the error here lies in equating popularity with truth."
    ),
    "equivocation": (
        "Fallacy: Equivocation. "
        "Explanation: This argument commits the Equivocation fallacy. "
        "It uses a word or phrase with two different meanings interchangeably, creating a misleading impression. "
        "This is different from {alt}, where the error involves a different kind of ambiguity or misrepresentation."
    ),
    "fallacy of extension": (
        "Fallacy: Fallacy of Extension. "
        "Explanation: This argument commits the Fallacy of Extension (also called Straw Man). "
        "It misrepresents or exaggerates an opponent's position to make it easier to attack, "
        "rather than engaging with what was actually claimed. "
        "This differs from {alt} in that the opponent's position is being deliberately distorted."
    ),
    "false cause": (
        "Fallacy: False Cause. "
        "Explanation: This argument commits the False Cause fallacy (post hoc ergo propter hoc). "
        "It incorrectly assumes that because one event preceded another, the first caused the second. "
        "Correlation is being mistaken for causation. Unlike {alt}, the error here is in the causal inference."
    ),
    "false dilemma": (
        "Fallacy: False Dilemma. "
        "Explanation: This argument commits the False Dilemma fallacy (also called False Dichotomy). "
        "It presents only two possible options as if they are the only choices, when in fact other "
        "alternatives exist. This is different from {alt}, where the reasoning error takes a different form."
    ),
    "hasty generalization": (
        "Fallacy: Hasty Generalization. "
        "Explanation: This argument commits the Hasty Generalization fallacy. "
        "It draws a broad conclusion from an insufficient or unrepresentative sample of evidence. "
        "Unlike {alt}, the problem here is the over-extension of a limited observation to a general rule."
    ),
    "intentional": (
        "Fallacy: Intentional Fallacy. "
        "Explanation: This argument commits an intentional rhetorical fallacy, deliberately using "
        "misleading logic or deceptive reasoning to manipulate the audience rather than persuade through evidence. "
        "This is distinct from {alt}, which occurs as an unintentional reasoning error."
    ),
    "logical fallacy": (
        "Fallacy: Logical Fallacy. "
        "Explanation: This argument contains a general logical fallacy — an error in reasoning that "
        "undermines the validity of the argument's conclusion. The premise does not sufficiently support "
        "the claim being made. This is different from {alt}, which involves a more specific reasoning pattern."
    ),
    "relevance fallacy": (
        "Fallacy: Relevance Fallacy. "
        "Explanation: This argument commits a Relevance Fallacy. "
        "It introduces information or arguments that are logically irrelevant to the conclusion being drawn. "
        "The premises do not have any logical bearing on the truth of the claim. "
        "Unlike {alt}, the core issue here is that the evidence presented is simply beside the point."
    ),
}


def build_generator_training_data(label_map: dict) -> tuple:
    """
    Creates (input, output) training pairs for the T5 generator.
    
    For each training example in train.csv:
    - INPUT:  "classify and explain: [normalized argument text] context: [sample passage]"
    - OUTPUT: The filled template for that fallacy type, with a contrastive reference.
    
    We sample random Kialo passages as "retrieved context" to simulate what
    the RAG retriever would provide during actual inference.
    This teaches the generator to use context when available.
    
    Returns (train_records, val_records) — lists of {input, output} dicts.
    """
    id_to_label = {v: k for k, v in label_map.items()}
    all_labels  = list(label_map.keys())
    
    # Load Kialo passages for fake "retrieval" during training
    kialo_passages = []
    if os.path.exists(KIALO_FILE):
        with open(KIALO_FILE) as f:
            for line in f:
                if line.strip():
                    kialo_passages.append(json.loads(line.strip())["text"])
    
    def make_record(row, use_normalized=True):
        """Creates one (input, output) record from a dataset row."""
        text_col = "text_normalized" if (use_normalized and "text_normalized" in row.index) else "text"
        arg_text = str(row[text_col]) if pd.notna(row[text_col]) else str(row["text"])
        label_id = int(row["label_id"])
        label    = id_to_label[label_id]
        
        # Pick a random alternative label for contrastive template
        alternatives = [l for l in all_labels if l != label]
        alt = random.choice(alternatives) if alternatives else "another fallacy type"
        
        # Pick up to 2 random Kialo passages as simulated retrieval context
        context_parts = []
        if kialo_passages:
            sampled = random.sample(kialo_passages, min(2, len(kialo_passages)))
            context_parts = sampled
        
        # Build input string
        context_str = " ".join(context_parts) if context_parts else ""
        if context_str:
            input_text = f"classify and explain: {arg_text} context: {context_str}"
        else:
            input_text = f"classify and explain: {arg_text}"
        
        # Build output string from template
        template = TEMPLATES.get(label, TEMPLATES["logical fallacy"])
        output_text = template.format(alt=alt)
        
        return {"input": input_text, "output": output_text, "label": label, "source": "fallacy"}
    
    # Build records for train and val splits
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = train_df.dropna(subset=["label_id"])
    val_df   = pd.read_csv(VAL_CSV)
    val_df   = val_df.dropna(subset=["label_id"])
    
    train_records = [make_record(row) for _, row in train_df.iterrows()]
    val_records   = [make_record(row) for _, row in val_df.iterrows()]
    
    # Append SNLI data to training set if available
    if os.path.exists(SNLI_JSONL):
        print(f"  Loading SNLI pretraining data from {SNLI_JSONL}...")
        snli_records = []
        with open(SNLI_JSONL) as f:
            for line in f:
                if line.strip():
                    snli_records.append(json.loads(line.strip()))
        # Limit SNLI to 3× the fallacy data to avoid overwhelming the model
        max_snli = min(len(snli_records), 3 * len(train_records))
        snli_records = random.sample(snli_records, max_snli)
        train_records.extend(snli_records)
        print(f"  Added {max_snli} SNLI records. Total train: {len(train_records)}")
    
    random.shuffle(train_records)
    return train_records, val_records


def save_records(records, path):
    """Saves a list of records to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── Dataset class ──────────────────────────────────────────────────────────
class GeneratorDataset(Dataset):
    """
    PyTorch Dataset for the seq2seq generator.
    
    Each example has:
    - input_ids: tokenized version of the input (argument + context)
    - attention_mask: which tokens are real vs padding
    - labels: tokenized version of the target output (explanation)
      (padded positions are set to -100 so they don't contribute to loss)
    """
    def __init__(self, jsonl_path: str, tokenizer, max_input: int, max_target: int):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line.strip()))
        self.tokenizer  = tokenizer
        self.max_input  = max_input
        self.max_target = max_target
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        
        # Tokenize input
        enc = self.tokenizer(
            rec["input"],
            max_length=self.max_input,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target output
        # T5 uses -100 as the ignore_index in CrossEntropyLoss for padding
        with self.tokenizer.as_target_tokenizer():
            dec = self.tokenizer(
                rec["output"],
                max_length=self.max_target,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        labels = dec["input_ids"].squeeze(0)
        # Replace padding token ids with -100 (ignored in loss computation)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels
        }


# ── Training loop ──────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("SCRIPT 05: TRAIN T5 EXPLANATION GENERATOR")
    print("=" * 60)
    
    if not os.path.exists(LABEL_MAP):
        print("ERROR: label_map.json not found. Run script 02 first.")
        sys.exit(1)
    
    with open(LABEL_MAP) as f:
        label_map = json.load(f)
    
    print(f"\n  Model: {MODEL_NAME}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    # ── Build training data ────────────────────────────────────────────────
    print("\n  Building generator training data from templates...")
    train_records, val_records = build_generator_training_data(label_map)
    print(f"  Train records: {len(train_records)}")
    print(f"  Val records:   {len(val_records)}")
    
    save_records(train_records, GEN_TRAIN_OUT)
    save_records(val_records,   GEN_VAL_OUT)
    print(f"  Saved to {GEN_TRAIN_OUT} and {GEN_VAL_OUT}")
    
    # ── Load tokenizer and model ───────────────────────────────────────────
    print(f"\n  Loading T5 tokenizer and model (first run downloads ~230MB)...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {total_params:.1f}M")
    
    # ── DataLoaders ────────────────────────────────────────────────────────
    train_dataset = GeneratorDataset(GEN_TRAIN_OUT, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    val_dataset   = GeneratorDataset(GEN_VAL_OUT,   tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    
    # ── Optimizer and scheduler ────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    print(f"\n  Training config:")
    print(f"    Epochs:              {EPOCHS}")
    print(f"    Batch size:          {BATCH_SIZE}")
    print(f"    Gradient accum:      {GRAD_ACCUM}  (effective batch = {BATCH_SIZE * GRAD_ACCUM})")
    print(f"    Learning rate:       {LEARNING_RATE}")
    print(f"    Total steps:         {total_steps}")
    
    best_val_loss = float("inf")
    os.makedirs(MODEL_OUT, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"  Epoch {epoch}/{EPOCHS} [Train]")
        
        for step, batch in loop:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            
            # T5 forward pass: computes the language modelling loss internally
            # when labels are provided (teacher forcing)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            
            epoch_loss += loss.item() * GRAD_ACCUM
            
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            loop.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")
        
        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Validating", leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"\n  Epoch {epoch} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Generate a sample explanation to inspect quality
        model.eval()
        with torch.no_grad():
            sample_input = "classify and explain: you cannot trust politicians because they all lie."
            sample_tokens = tokenizer(sample_input, return_tensors="pt", truncation=True, max_length=256)
            sample_tokens = {k: v.to(device) for k, v in sample_tokens.items()}
            generated = model.generate(
                **sample_tokens,
                max_new_tokens=80,
                num_beams=4,           # Beam search: considers 4 candidate sequences
                early_stopping=True
            )
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"  Sample output: {decoded[:200]}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(MODEL_OUT)
            tokenizer.save_pretrained(MODEL_OUT)
            print(f"  ✓ Best generator saved (Val Loss: {best_val_loss:.4f}) → {MODEL_OUT}")
    
    print("\n" + "=" * 60)
    print(f"Generator training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_OUT}")
    print("Next: python scripts/06_cpace_module.py  (run in background)")
    print("Then: python scripts/07_inference.py")
    print("=" * 60)


if __name__ == "__main__":
    train()
