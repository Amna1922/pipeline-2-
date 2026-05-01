"""
=============================================================================
SCRIPT: 03_build_index.py
PURPOSE: Layer 2 of Pipeline II — Retrieval Index Construction.
         1. Loads all passages from data/retrieval_corpus/kialo_passages.jsonl
         2. Encodes each passage into a 768-dimensional dense vector using
            the DPR (Dense Passage Retrieval) document encoder from HuggingFace.
            Think of this as converting each passage into a unique numerical
            "fingerprint" that captures its meaning.
         3. Builds a FAISS IVF (Inverted File) index from these vectors.
            FAISS allows searching millions of vectors in milliseconds.
         4. Saves the index and passage metadata to disk.
         Parallelism: Passage encoding uses batched inference with all CPU
         threads. torch.set_num_threads() maximizes core utilization.
RUN:     python scripts/03_build_index.py
=============================================================================
"""

import os
import sys
import json
import numpy as np
import torch
import faiss
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR   = os.path.join(BASE_DIR, "data", "retrieval_corpus")
INDEX_DIR    = os.path.join(BASE_DIR, "index")
PASSAGES_IN  = os.path.join(CORPUS_DIR, "kialo_passages.jsonl")
INDEX_OUT    = os.path.join(INDEX_DIR, "kialo.index")
META_OUT     = os.path.join(INDEX_DIR, "kialo_meta.json")

# ── Configuration ──────────────────────────────────────────────────────────
# DPR document encoder — pre-trained on question-answer pairs, produces
# 768-dimensional sentence embeddings well-suited for semantic retrieval.
DPR_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
BATCH_SIZE = 16      # Reduce to 8 if you run out of RAM
# FAISS IVF parameters:
# nlist = number of clusters. Rule of thumb: sqrt(N) ≤ nlist ≤ 4*sqrt(N)
# For ~650 passages: nlist=32. For >10K: nlist=100.
NLIST = 32
# nprobe = how many clusters to search per query. Higher = more accurate, slower.
NPROBE = 8
VECTOR_DIM = 768     # DPR embedding dimension

# ── Maximize CPU thread usage on i5-1235U ──────────────────────────────────
torch.set_num_threads(min(torch.get_num_threads(), 10))
faiss.omp_set_num_threads(min(os.cpu_count() or 4, 8))


def load_passages(path: str):
    """
    Loads all passages from the JSONL file created by script 02.
    Returns a list of dicts: {passage_id, text, original, source, question}
    """
    passages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
    print(f"  Loaded {len(passages)} passages from {os.path.basename(path)}")
    return passages


def encode_passages(passages: list, model, tokenizer, device: str) -> np.ndarray:
    """
    Encodes all passages into dense vectors using the DPR Context Encoder.
    
    The DPR encoder is a BERT-based model. It reads each passage and produces
    a single 768-dimensional vector that summarizes the passage's meaning.
    
    Process:
    1. Tokenize each passage into numerical token IDs (input the model can read).
    2. Run the model to get the [CLS] token embedding — a 768-dim summary vector.
    3. Collect all vectors into a numpy matrix of shape (N_passages, 768).
    
    Batching: we process BATCH_SIZE passages at once to use RAM and CPU
    cache efficiently. Each passage is padded/truncated to 256 tokens.
    """
    all_embeddings = []
    
    texts = [p["text"] for p in passages]
    
    model.eval()  # Set to eval mode — no gradient computation needed
    with torch.no_grad():  # Don't compute gradients (saves memory and time)
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Encoding passages"):
            batch_texts = texts[i : i + BATCH_SIZE]
            
            # Tokenize: convert text strings → token id tensors
            inputs = tokenizer(
                batch_texts,
                max_length=256,
                padding=True,        # Pad shorter sequences to same length
                truncation=True,     # Cut sequences longer than max_length
                return_tensors="pt"  # Return PyTorch tensors
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass through DPR encoder
            # pooler_output is the [CLS] embedding — shape: (batch_size, 768)
            outputs = model(**inputs)
            embeddings = outputs.pooler_output  # shape: (B, 768)
            
            # Move to CPU and convert to numpy (FAISS needs numpy arrays)
            all_embeddings.append(embeddings.cpu().numpy())
    
    # Stack all batches into one matrix: shape (N, 768)
    embedding_matrix = np.vstack(all_embeddings).astype("float32")
    print(f"  Encoded {embedding_matrix.shape[0]} passages → shape {embedding_matrix.shape}")
    return embedding_matrix


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS IVF (Inverted File Index) with Inner Product (dot product)
    similarity. 
    
    Why IVF?  
    With N passages, comparing a query against every passage takes O(N) time.
    IVF first clusters all passage vectors into `nlist` groups (centroids).
    At query time, only the `nprobe` nearest clusters are searched — much faster.
    
    Why Inner Product (IP)?  
    DPR was trained with dot product as the similarity measure, so we use IP.
    (Cosine similarity would require L2-normalizing vectors first.)
    
    Step-by-step:
    1. Create a quantizer: a flat IP index used to find nearest centroids.
    2. Create IVF index: clusters passage vectors using k-means.
    3. Train the index on our embeddings (learns cluster centroids).
    4. Add all embeddings to the index (assigns each vector to a cluster).
    """
    n, d = embeddings.shape
    print(f"  Building FAISS IVF index: {n} vectors, dim={d}, nlist={NLIST}, nprobe={NPROBE}")
    
    # IVF with IP (Inner Product / dot product) similarity
    quantizer = faiss.IndexFlatIP(d)        # Flat index for centroid search
    index = faiss.IndexIVFFlat(quantizer, d, NLIST, faiss.METRIC_INNER_PRODUCT)
    
    # Train: clusters the embeddings using k-means
    # Requires at least nlist training vectors
    assert n >= NLIST, f"Need at least {NLIST} passages to build index, got {n}"
    print("  Training FAISS index (k-means clustering)...")
    index.train(embeddings)
    
    # Add all passage vectors to the index
    index.add(embeddings)
    
    # nprobe: how many clusters to search per query
    index.nprobe = NPROBE
    
    print(f"  Index built. Total vectors indexed: {index.ntotal}")
    return index


def save_index_and_metadata(index, passages):
    """
    Saves the FAISS index to disk and the passage metadata (original text,
    source, etc.) as a JSON file.
    The index stores only the vectors; the metadata maps vector positions
    back to human-readable passage text.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, INDEX_OUT)
    print(f"  FAISS index saved to {INDEX_OUT}")
    
    # Save passage metadata (maps index position → original passage info)
    # We store both the normalized text (for re-encoding) and the original
    # text (for displaying in explanations)
    meta = [
        {
            "passage_id": p["passage_id"],
            "text": p["text"],
            "original": p.get("original", p["text"]),
            "source": p.get("source", "unknown"),
            "question": p.get("question", ""),
        }
        for p in passages
    ]
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Passage metadata saved to {META_OUT}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SCRIPT 03: BUILD FAISS RETRIEVAL INDEX")
    print("=" * 60)
    
    if not os.path.exists(PASSAGES_IN):
        print(f"ERROR: {PASSAGES_IN} not found. Run script 02 first.")
        sys.exit(1)
    
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    # Load pre-trained DPR context (document) encoder
    print(f"\n  Loading DPR document encoder: {DPR_MODEL}")
    print("  (First run downloads ~400MB from HuggingFace — will be cached)")
    tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(DPR_MODEL)
    model = DPRContextEncoder.from_pretrained(DPR_MODEL).to(device)
    
    # Load passages
    print(f"\n  Loading passages from {PASSAGES_IN}...")
    passages = load_passages(PASSAGES_IN)
    
    # Encode all passages to dense vectors
    print(f"\n  Encoding {len(passages)} passages (this takes several minutes on CPU)...")
    embeddings = encode_passages(passages, model, tokenizer, device)
    
    # Build FAISS index
    print("\n  Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    # Save to disk
    print("\n  Saving index and metadata...")
    save_index_and_metadata(index, passages)
    
    print("\n" + "=" * 60)
    print("Index building complete.")
    print(f"  Index file:    index/kialo.index  ({os.path.getsize(INDEX_OUT)/1e6:.1f} MB)")
    print(f"  Metadata file: index/kialo_meta.json")
    print("Next: python scripts/04_train_classifier.py")
    print("=" * 60)
