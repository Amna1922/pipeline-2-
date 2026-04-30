"""
=============================================================================
SCRIPT: 02_preprocess.py
PURPOSE: Layer 1 of Pipeline II.
         1. Loads final_cleaned_dataset.csv (fallacy classifier training data)
         2. Loads kialo_flat_clean.csv (retrieval corpus for RAG)
         3. Optionally loads snli_clean.csv (generator pretraining data)
         4. Applies Named Entity Recognition (NER) normalization using spaCy:
            - Replaces named entities (PERSON, ORG, GPE, etc.) with [ENTITY_TYPE]
            - Focuses the model on rhetorical structure, not specific names
         5. Splits fallacy data into train/val/test (80/10/10)
         6. Saves all processed files to data/processed/ and data/retrieval_corpus/
         Parallelism: NER runs in parallel across CPU cores using multiprocessing.
RUN:     python scripts/02_preprocess.py
=============================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import spacy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
CORPUS    = os.path.join(BASE_DIR, "data", "retrieval_corpus")

FALLACY_CSV  = os.path.join(RAW, "final_cleaned_dataset.csv")
KIALO_CSV    = os.path.join(RAW, "kialo_flat_clean.csv")
SNLI_CSV     = os.path.join(RAW, "snli_clean.csv")   # optional

# ── Configuration ──────────────────────────────────────────────────────────
RANDOM_SEED  = 42          # for reproducible splits
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10
# Number of parallel workers for NER. On i5-1235U with 10 cores, use up to 8.
N_WORKERS    = min(cpu_count(), 8)

# ── Entity replacement map (spaCy label → placeholder) ─────────────────────
ENTITY_MAP = {
    "PERSON":   "[PERSON]",
    "ORG":      "[ORG]",
    "GPE":      "[GPE]",
    "LOC":      "[LOC]",
    "NORP":     "[NORP]",
    "FAC":      "[FAC]",
    "EVENT":    "[EVENT]",
    "DATE":     "[DATE]",
    "TIME":     "[TIME]",
    "MONEY":    "[MONEY]",
    "PERCENT":  "[PERCENT]",
    "PRODUCT":  "[PRODUCT]",
    "WORK_OF_ART": "[WORK_OF_ART]",
    "LAW":      "[LAW]",
    "LANGUAGE": "[LANGUAGE]",
}

# ── NER normalization functions ────────────────────────────────────────────

def basic_clean(text: str) -> str:
    """
    Basic text cleaning before NER:
    - Remove HTML tags
    - Collapse multiple whitespace into single space
    - Strip leading/trailing whitespace
    Note: we do NOT lowercase yet — spaCy NER works better on cased text.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    text = text.strip()
    return text


def ner_normalize(text: str, nlp) -> str:
    """
    Applies NER to 'text' using the provided spaCy model and replaces all
    detected entity spans with their canonical placeholder tags.
    
    How it works:
    - spaCy reads the text left-to-right and identifies entity spans.
    - We replace each span (by character position) with the tag.
    - Replacement is done right-to-left to preserve character indices.
    - Finally, we lowercase the result (placeholders remain uppercase).

    Example:
        Input:  "You cannot trust Joe Biden because NATO said so."
        Output: "you cannot trust [person] because [org] said so."
    """
    text = basic_clean(text)
    if not text:
        return ""
    
    doc = nlp(text)
    
    # Collect replacements: (start_char, end_char, replacement_tag)
    replacements = []
    for ent in doc.ents:
        tag = ENTITY_MAP.get(ent.label_, None)
        if tag:
            replacements.append((ent.start_char, ent.end_char, tag))
    
    # Apply replacements right-to-left to preserve earlier indices
    chars = list(text)
    for start, end, tag in sorted(replacements, key=lambda x: x[0], reverse=True):
        chars[start:end] = list(tag)
    
    normalized = "".join(chars).lower()
    # Clean up extra spaces that may appear after replacements
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# ── Multiprocessing worker ─────────────────────────────────────────────────
# spaCy models cannot be pickled, so we load one per worker process.
_nlp = None

def _init_worker():
    """Initializer for each multiprocessing worker: loads spaCy model once."""
    global _nlp
    _nlp = spacy.load("en_core_web_lg", disable=["parser", "senter"])
    # Disabling parser and senter speeds up NER-only inference significantly.


def _normalize_one(text: str) -> str:
    """Worker function: normalize a single text string."""
    global _nlp
    return ner_normalize(text, _nlp)


def parallel_normalize(texts: list, desc: str = "NER normalization") -> list:
    """
    Normalizes a list of texts in parallel using N_WORKERS processes.
    Each worker loads its own spaCy model and processes a chunk of texts.
    Returns a list of normalized strings in the same order as input.
    """
    print(f"  Using {N_WORKERS} parallel workers for NER...")
    with Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        # imap preserves order; chunksize=16 reduces inter-process overhead
        results = list(tqdm(
            pool.imap(_normalize_one, texts, chunksize=16),
            total=len(texts),
            desc=desc
        ))
    return results


# ── Main processing functions ──────────────────────────────────────────────

def process_fallacy_dataset():
    """
    Loads final_cleaned_dataset.csv, applies NER normalization to the 'text'
    column, and splits into train/val/test sets.
    
    Columns: id, text, label, type
    Output files:
        data/processed/train.csv
        data/processed/val.csv
        data/processed/test.csv
        data/processed/label_map.json   ← maps label string → integer id
    """
    print("\n[1/3] Processing fallacy dataset (final_cleaned_dataset.csv)...")
    
    df = pd.read_csv(FALLACY_CSV)
    print(f"  Loaded {len(df)} rows.")
    print(f"  Columns: {list(df.columns)}")
    
    # Drop rows with missing text or label
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    
    # ── Build label map (label string → integer for training) ──────────────
    # Sorted alphabetically for reproducibility across runs
    unique_labels = sorted(df["label"].unique())
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    label_map_path = os.path.join(PROCESSED, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Label map ({len(label_map)} classes): {label_map}")
    df["label_id"] = df["label"].map(label_map)
    
    # ── NER normalization ──────────────────────────────────────────────────
    print("  Applying NER normalization to fallacy texts...")
    df["text_normalized"] = parallel_normalize(
        df["text"].tolist(), desc="  Normalizing fallacy texts"
    )
    
    # ── Train / Val / Test split ───────────────────────────────────────────
    # Shuffle with fixed seed for reproducibility
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train : n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]
    
    train_df.to_csv(os.path.join(PROCESSED, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED, "val.csv"),   index=False)
    test_df.to_csv(os.path.join(PROCESSED, "test.csv"),  index=False)
    
    print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"  Saved to data/processed/")


def process_kialo_corpus():
    """
    Loads kialo_flat_clean.csv, combines question + argument into a single
    passage string, applies NER normalization, and saves as JSONL for indexing.
    
    Columns: question, argument, type, id
    Output: data/retrieval_corpus/kialo_passages.jsonl
            Each line: {"passage_id": int, "text": str, "source": "kialo",
                        "question": str, "original": str}
    """
    print("\n[2/3] Processing Kialo retrieval corpus...")
    
    df = pd.read_csv(KIALO_CSV)
    print(f"  Loaded {len(df)} rows.")
    
    df = df.dropna(subset=["argument"])
    df["argument"] = df["argument"].astype(str)
    df["question"] = df["question"].fillna("").astype(str)
    
    # Combine question and argument into a single passage for richer retrieval
    # Format: "Topic: [question] Argument: [argument]"
    df["combined"] = df.apply(
        lambda r: f"Topic: {r['question'].strip()} Argument: {r['argument'].strip()}"
        if r["question"].strip() else r["argument"].strip(),
        axis=1
    )
    
    print("  Applying NER normalization to Kialo passages...")
    normalized = parallel_normalize(
        df["combined"].tolist(), desc="  Normalizing Kialo"
    )
    
    # Save as JSONL (one JSON object per line — memory-efficient for large corpora)
    out_path = os.path.join(CORPUS, "kialo_passages.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (orig_row, norm_text) in enumerate(zip(df.itertuples(), normalized)):
            record = {
                "passage_id": idx,
                "text": norm_text,          # NER-normalized version (used for indexing)
                "original": orig_row.combined,  # original text (shown in explanations)
                "source": "kialo",
                "question": orig_row.question,
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"  Saved {len(df)} passages to data/retrieval_corpus/kialo_passages.jsonl")


def process_snli_for_generator():
    """
    Loads snli_clean.csv and prepares it for generator pretraining.
    The generator learns to produce explanations by mapping:
        (premise + hypothesis) → label + explanation
    Since e-SNLI explanations are for entailment/contradiction/neutral,
    we create templates that mimic the format we need for fallacy explanations.
    
    Columns: label, premise, hypothesis, label_id
    Output: data/processed/snli_generator_train.jsonl
    
    This is OPTIONAL. If snli_clean.csv is not found, this step is skipped.
    """
    if not os.path.exists(SNLI_CSV):
        print("\n[3/3] snli_clean.csv not found — skipping SNLI generator pretraining data.")
        print("  The generator will train on fallacy-specific templates only (see script 05).")
        return
    
    print("\n[3/3] Processing SNLI data for generator pretraining...")
    
    df = pd.read_csv(SNLI_CSV)
    print(f"  Loaded {len(df)} rows.")
    df = df.dropna(subset=["premise", "hypothesis", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    
    # Map SNLI labels to explanation-friendly descriptions
    label_desc = {
        "entailment":    "logically follows from",
        "contradiction": "directly contradicts",
        "neutral":       "is neither confirmed nor denied by",
    }
    
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Building SNLI training pairs"):
        desc = label_desc.get(row["label"], row["label"])
        input_text = (
            f"Given the statement: \"{row['premise'].strip()}\" "
            f"determine the relationship to: \"{row['hypothesis'].strip()}\""
        )
        # Target: the explanation the generator should produce
        output_text = (
            f"The hypothesis {desc} the premise. "
            f"Specifically, \"{row['hypothesis'].strip()}\" "
            f"{desc} what is stated in \"{row['premise'].strip()}\"."
        )
        records.append({"input": input_text, "output": output_text, "source": "snli"})
    
    # Save only a 20K sample to avoid overwhelming the generator with SNLI
    # (fallacy-specific templates should dominate the fine-tuning signal)
    import random
    random.seed(RANDOM_SEED)
    if len(records) > 20000:
        records = random.sample(records, 20000)
    
    out_path = os.path.join(PROCESSED, "snli_generator_train.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(records)} SNLI training pairs to data/processed/snli_generator_train.jsonl")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SCRIPT 02: PREPROCESSING & NER NORMALIZATION")
    print("=" * 60)
    
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(CORPUS, exist_ok=True)
    
    process_fallacy_dataset()
    process_kialo_corpus()
    process_snli_for_generator()
    
    print("\n" + "=" * 60)
    print("Preprocessing complete. Output files:")
    print("  data/processed/train.csv")
    print("  data/processed/val.csv")
    print("  data/processed/test.csv")
    print("  data/processed/label_map.json")
    print("  data/retrieval_corpus/kialo_passages.jsonl")
    print("  data/processed/snli_generator_train.jsonl  (if SNLI available)")
    print("Next: python scripts/03_build_index.py")
    print("=" * 60)
