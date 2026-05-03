"""
=============================================================================
SCRIPT: 07_inference.py
PURPOSE: Runs the complete Pipeline II inference chain on new argument text.
         Given an argument, the system:
         
         LAYER 1 — Preprocessing:
           • NER normalization (entity replacement)
         
         LAYER 2 — Retrieval:
           • DPR Query Encoder converts argument → 768-dim query vector
           • FAISS index searched for top-K most similar debate passages
         
         LAYER 3A — RAG-Token Generator:
           • T5 generator reads argument + retrieved passages
           • Produces: fallacy label + grounded explanation
           (RAG-Token = model can draw on different passages for different words)
         
         LAYER 3B — CPACE Module:
           • Runs in parallel (using ThreadPoolExecutor) with Layer 3A
           • Generates a contrastive explanation using concept extraction
         
         LAYER 3C — Fusion Module:
           • Embeds both explanations using MiniLM sentence encoder
           • Computes cosine similarity of each against retrieved passages
           • Selects the explanation with higher semantic alignment
         
         LAYER 4 — Output:
           • Writes structured JSON results to outputs/predictions/
         
         Parallelism:
           • NER uses multiprocessing (same as script 02)
           • RAG generation and CPACE run in parallel threads
           • Batched FAISS queries for efficiency
           • DataLoader-style batched preprocessing
         
         INPUT:   Can run on test.csv (held-out test set) or any CSV/text input.
         
RUN:     python scripts/07_inference.py                     (uses test.csv)
         python scripts/07_inference.py --input path/to/file.csv
         python scripts/07_inference.py --text "Your argument here"
=============================================================================
"""

import os
import sys
import json
import re
import time
import argparse
import numpy as np
import pandas as pd
import torch
import faiss
import spacy
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast,
    T5ForConditionalGeneration, T5Tokenizer
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict, Optional

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED    = os.path.join(BASE_DIR, "data", "processed")
MODEL_CLS    = os.path.join(BASE_DIR, "models", "deberta_classifier")
MODEL_GEN    = os.path.join(BASE_DIR, "models", "generator")
INDEX_FILE   = os.path.join(BASE_DIR, "index", "kialo.index")
META_FILE    = os.path.join(BASE_DIR, "index", "kialo_meta.json")
LABEL_MAP    = os.path.join(PROCESSED, "label_map.json")
TEST_CSV     = os.path.join(PROCESSED, "test.csv")
OUTPUT_FILE  = os.path.join(BASE_DIR, "outputs", "predictions", "results.jsonl")

# ── Add scripts directory to path for CPACE import ─────────────────────────
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
from cpace_module import generate_contrastive_explanation   # noqa

# ── Configuration ──────────────────────────────────────────────────────────
TOP_K          = 5         # Number of passages to retrieve per argument
BATCH_SIZE     = 16        # Arguments processed per batch during inference
# DPR query encoder model (same family as document encoder used in index build)
DPR_Q_MODEL   = "facebook/dpr-question_encoder-single-nq-base"
# Fusion: MiniLM produces 384-dim embeddings extremely fast on CPU
FUSION_MODEL   = "all-MiniLM-L6-v2"
# Confidence threshold below which a prediction is flagged for review
CONFIDENCE_THR = 0.60
# Max new tokens for generator output
MAX_NEW_TOKENS = 100

# Maximize CPU threads
torch.set_num_threads(min(torch.get_num_threads(), 10))
faiss.omp_set_num_threads(min(os.cpu_count() or 4, 8))


# ── NER normalization (reuse from script 02) ────────────────────────────────
ENTITY_MAP = {
    "PERSON": "[PERSON]", "ORG": "[ORG]", "GPE": "[GPE]", "LOC": "[LOC]",
    "NORP": "[NORP]", "FAC": "[FAC]", "EVENT": "[EVENT]", "DATE": "[DATE]",
    "TIME": "[TIME]", "MONEY": "[MONEY]", "PERCENT": "[PERCENT]",
    "PRODUCT": "[PRODUCT]", "WORK_OF_ART": "[WORK_OF_ART]",
    "LAW": "[LAW]", "LANGUAGE": "[LANGUAGE]",
}

_ner_nlp = None

def get_ner_nlp():
    global _ner_nlp
    if _ner_nlp is None:
        _ner_nlp = spacy.load("en_core_web_lg", disable=["parser", "senter"])
    return _ner_nlp

def normalize_text(text: str) -> str:
    """Applies NER normalization to a single text string."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    nlp = get_ner_nlp()
    doc = nlp(text)
    replacements = []
    for ent in doc.ents:
        tag = ENTITY_MAP.get(ent.label_)
        if tag:
            replacements.append((ent.start_char, ent.end_char, tag))
    
    chars = list(text)
    for start, end, tag in sorted(replacements, key=lambda x: x[0], reverse=True):
        chars[start:end] = list(tag)
    
    return re.sub(r"\s+", " ", "".join(chars).lower()).strip()


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

class PipelineModels:
    """
    Container that loads and holds all model components.
    Loading is done once and reused across all batches.
    
    Components:
    - classifier:     DeBERTa trained on fallacy dataset
    - cls_tokenizer:  DeBERTa tokenizer
    - dpr_encoder:    DPR query encoder (converts argument → 768-dim query vector)
    - dpr_tokenizer:  DPR query tokenizer
    - generator:      T5 trained to produce explanations
    - gen_tokenizer:  T5 tokenizer
    - fusion_model:   MiniLM sentence encoder for explanation scoring
    - faiss_index:    FAISS vector index for passage retrieval
    - passages:       List of passage metadata dicts
    - label_map:      {label_string: int_id}
    - id_to_label:    {int_id: label_string}
    """
    
    def __init__(self, device: str):
        self.device = device
        
        # Load label map
        with open(LABEL_MAP) as f:
            self.label_map = json.load(f)
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        self.num_classes  = len(self.label_map)
        print(f"  Labels: {list(self.label_map.keys())}")
        
        self._load_classifier()
        self._load_dpr()
        self._load_generator()
        self._load_fusion()
        self._load_faiss()
    
    def _load_classifier(self):
        print("  Loading DeBERTa classifier...")
        self.cls_tokenizer = AutoTokenizer.from_pretrained(MODEL_CLS)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_CLS)
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
    
    def _load_dpr(self):
        print("  Loading DPR query encoder...")
        self.dpr_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(DPR_Q_MODEL)
        self.dpr_encoder   = DPRQuestionEncoder.from_pretrained(DPR_Q_MODEL).to(self.device)
        self.dpr_encoder.eval()
    
    def _load_generator(self):
        print("  Loading T5 generator...")
        self.gen_tokenizer = T5Tokenizer.from_pretrained(MODEL_GEN)
        self.generator = T5ForConditionalGeneration.from_pretrained(MODEL_GEN).to(self.device)
        self.generator.eval()
    
    def _load_fusion(self):
        print(f"  Loading Fusion model ({FUSION_MODEL})...")
        # SentenceTransformer wraps the MiniLM model; produces 384-dim embeddings
        self.fusion_model = SentenceTransformer(FUSION_MODEL)
    
    def _load_faiss(self):
        print("  Loading FAISS index...")
        self.faiss_index = faiss.read_index(INDEX_FILE)
        self.faiss_index.nprobe = 8  # Search 8 clusters per query
        
        with open(META_FILE) as f:
            self.passages = json.load(f)
        print(f"  Index loaded: {self.faiss_index.ntotal} passages indexed")


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def classify_batch(texts: List[str], models: PipelineModels) -> List[Dict]:
    """
    LAYER 1 (partial) + LAYER 3A (classification part):
    Runs the DeBERTa classifier on a batch of normalized texts.
    
    Returns for each text:
    - predicted_label: string label (e.g., "ad hominem")
    - label_id: integer id
    - confidence: softmax probability of the predicted class (0–1)
    - all_probs: full probability distribution over all classes
    """
    inputs = models.cls_tokenizer(
        texts,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(models.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = models.classifier(**inputs).logits  # (B, num_classes)
    
    # Softmax converts raw logit scores → probabilities summing to 1
    # Equation: P(class_i) = exp(logit_i) / sum_j(exp(logit_j))
    probs     = torch.softmax(logits, dim=-1).cpu().numpy()
    label_ids = np.argmax(probs, axis=-1)
    
    results = []
    for i, (lid, prob_dist) in enumerate(zip(label_ids, probs)):
        results.append({
            "predicted_label": models.id_to_label[int(lid)],
            "label_id":        int(lid),
            "confidence":      float(prob_dist[lid]),
            "all_probs":       {models.id_to_label[j]: float(p) for j, p in enumerate(prob_dist)}
        })
    return results


def encode_queries(texts: List[str], models: PipelineModels) -> np.ndarray:
    """
    LAYER 2 (encoding):
    Converts a batch of argument texts into 768-dim DPR query vectors.
    
    The DPR query encoder is a BERT-based model that reads the text and
    produces a single vector capturing the argument's semantic meaning.
    This vector is then compared against the indexed passage vectors in FAISS.
    """
    inputs = models.dpr_tokenizer(
        texts,
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(models.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # pooler_output = [CLS] token embedding = 768-dim query vector
        query_vecs = models.dpr_encoder(**inputs).pooler_output
    
    return query_vecs.cpu().numpy().astype("float32")


def retrieve_passages(query_vecs: np.ndarray, models: PipelineModels, top_k: int) -> List[List[Dict]]:
    """
    LAYER 2 (retrieval):
    Searches the FAISS index for the top-K passages most similar to each query.
    
    FAISS returns:
    - scores: dot product similarity (higher = more similar)
    - indices: integer positions in the index (maps to passage metadata)
    
    Returns a list (one per query) of top-K passage dicts.
    """
    # scores shape: (B, top_k), indices shape: (B, top_k)
    scores, indices = models.faiss_index.search(query_vecs, top_k)
    
    all_retrieved = []
    for q_idx in range(len(query_vecs)):
        retrieved = []
        for rank, (score, pos) in enumerate(zip(scores[q_idx], indices[q_idx])):
            if pos < 0 or pos >= len(models.passages):
                continue  # FAISS returns -1 for unfilled slots
            passage = models.passages[pos]
            retrieved.append({
                "rank":       rank + 1,
                "score":      float(score),
                "text":       passage["text"],
                "original":   passage.get("original", passage["text"]),
                "source":     passage.get("source", "unknown"),
                "question":   passage.get("question", ""),
            })
        all_retrieved.append(retrieved)
    
    return all_retrieved


def generate_rag_explanation(
    argument: str,
    passages: List[Dict],
    predicted_label: str,
    models: PipelineModels
) -> str:
    """
    LAYER 3A (RAG-Token generation):
    Generates a natural language explanation using T5 with retrieved passages as context.
    
    RAG-Token means the generator can attend to DIFFERENT passages for different
    parts of the generated explanation, unlike RAG-Sequence which picks one passage.
    In practice with T5, we concatenate all passages into the input and let the
    cross-attention mechanism decide which parts of the context to use for each
    generated token.
    
    Input format:
        "classify and explain: [ARGUMENT] context: [P1_TEXT] [P2_TEXT] ..."
    Output format:
        "Fallacy: [LABEL]. Explanation: [NATURAL LANGUAGE EXPLANATION]"
    """
    # Build context string from top-K passages (truncated for input length)
    context_parts = [p["text"][:150] for p in passages[:TOP_K]]  # limit each passage
    context_str   = " ".join(context_parts)
    
    input_text = f"classify and explain: {argument} context: {context_str}"
    
    inputs = models.gen_tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(models.device)
    
    with torch.no_grad():
        output_ids = models.generator.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,            # Beam search: considers 4 candidate sequences
            early_stopping=True,
            no_repeat_ngram_size=3  # Prevents repetitive phrasing
        )
    
    explanation = models.gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # If the generator didn't produce the expected format, wrap it
    if "Fallacy:" not in explanation:
        explanation = f"Fallacy: {predicted_label}. Explanation: {explanation}"
    
    return explanation


def fusion_select(
    rag_explanation: str,
    cpace_explanation: str,
    passages: List[Dict],
    models: PipelineModels
) -> tuple:
    """
    LAYER 3C (Fusion Module):
    Selects the better explanation between RAG and CPACE outputs.
    
    Method:
    1. Embed both explanations and the average passage text using MiniLM.
    2. Compute cosine similarity of each explanation to the retrieved passages.
       Cosine similarity = (A · B) / (|A| × |B|)
       A value of 1.0 = same direction = highly similar meaning.
    3. Return the explanation with higher similarity to the retrieved passages.
       Higher similarity means the explanation is more grounded in the evidence.
    
    Returns: (best_explanation, winner, rag_score, cpace_score)
    """
    # Combine top passages into a single reference text
    passage_text = " ".join([p["text"][:200] for p in passages[:3]])
    
    # Encode all three texts using the sentence encoder
    # SentenceTransformer.encode() returns normalized vectors by default
    embeddings = models.fusion_model.encode(
        [rag_explanation, cpace_explanation, passage_text],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    rag_emb    = embeddings[0]
    cpace_emb  = embeddings[1]
    passage_emb = embeddings[2]
    
    # Cosine similarity (dot product of normalized vectors = cosine similarity)
    rag_score   = float(np.dot(rag_emb, passage_emb))
    cpace_score = float(np.dot(cpace_emb, passage_emb))
    
    if rag_score >= cpace_score:
        return rag_explanation, "rag", rag_score, cpace_score
    else:
        return cpace_explanation, "cpace", rag_score, cpace_score


# ═══════════════════════════════════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(texts: List[str], original_texts: List[str], models: PipelineModels) -> List[Dict]:
    """
    Runs the full Pipeline II inference on a list of argument texts.
    
    For each batch:
    1. NER normalization (already done if input comes from test.csv)
    2. DeBERTa classification → label + confidence
    3. DPR encoding → query vectors
    4. FAISS retrieval → top-K passages
    5. In parallel threads:
       - Thread A: T5 RAG-Token generation
       - Thread B: CPACE contrastive generation
    6. Fusion Module → best explanation selected
    7. Assemble result record
    
    Parallelism: RAG generation and CPACE run in parallel using ThreadPoolExecutor.
    Each batch of 16 arguments has both threads running simultaneously,
    cutting total generation time roughly in half versus serial execution.
    """
    all_results = []
    
    for batch_start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Running inference"):
        batch_texts    = texts[batch_start : batch_start + BATCH_SIZE]
        batch_originals = original_texts[batch_start : batch_start + BATCH_SIZE]
        
        # ── Step 1: Classify ────────────────────────────────────────────────
        cls_results = classify_batch(batch_texts, models)
        
        # ── Step 2: Encode queries ──────────────────────────────────────────
        query_vecs = encode_queries(batch_texts, models)
        
        # ── Step 3: Retrieve passages ───────────────────────────────────────
        retrieved_batched = retrieve_passages(query_vecs, models, TOP_K)
        
        # ── Steps 4A + 4B: Parallel RAG & CPACE ────────────────────────────
        # We use ThreadPoolExecutor for thread-level parallelism.
        # The GIL doesn't block us here because T5 generation and CPACE
        # concept extraction are both I/O and compute operations that release
        # the GIL at the C-extension level (PyTorch C++, spaCy Cython).
        
        def run_rag(i):
            """Thread A: RAG-Token generation for argument i in the batch."""
            return generate_rag_explanation(
                argument      = batch_texts[i],
                passages      = retrieved_batched[i],
                predicted_label = cls_results[i]["predicted_label"],
                models        = models
            )
        
        def run_cpace(i):
            """Thread B: CPACE contrastive generation for argument i in the batch."""
            return generate_contrastive_explanation(
                argument_text   = batch_texts[i],
                predicted_label = cls_results[i]["predicted_label"],
                label_map       = models.label_map,
                retrieved_passages = [p["text"] for p in retrieved_batched[i]]
            )
        
        # Submit all RAG and CPACE jobs to the thread pool simultaneously
        rag_explanations   = [None] * len(batch_texts)
        cpace_explanations = [None] * len(batch_texts)
        
        with ThreadPoolExecutor(max_workers=min(len(batch_texts) * 2, 8)) as executor:
            # Submit all RAG jobs
            rag_futures   = {executor.submit(run_rag, i): i for i in range(len(batch_texts))}
            # Submit all CPACE jobs (runs in parallel with RAG jobs)
            cpace_futures = {executor.submit(run_cpace, i): i for i in range(len(batch_texts))}
            
            # Collect RAG results
            for future in as_completed(rag_futures):
                i = rag_futures[future]
                try:
                    rag_explanations[i] = future.result()
                except Exception as e:
                    rag_explanations[i] = f"Fallacy: {cls_results[i]['predicted_label']}. [Generation error: {e}]"
            
            # Collect CPACE results
            for future in as_completed(cpace_futures):
                i = cpace_futures[future]
                try:
                    cpace_explanations[i] = future.result()
                except Exception as e:
                    cpace_explanations[i] = f"[CPACE error: {e}]"
        
        # ── Step 5: Fusion ──────────────────────────────────────────────────
        for i in range(len(batch_texts)):
            best_expl, winner, rag_score, cpace_score = fusion_select(
                rag_explanation   = rag_explanations[i],
                cpace_explanation = cpace_explanations[i],
                passages          = retrieved_batched[i],
                models            = models
            )
            
            confidence     = cls_results[i]["confidence"]
            human_review   = confidence < CONFIDENCE_THR
            
            result = {
                "argument_original":   batch_originals[i],
                "argument_normalized": batch_texts[i],
                "predicted_label":     cls_results[i]["predicted_label"],
                "confidence":          round(confidence, 4),
                "all_class_probs":     {k: round(v, 4) for k, v in cls_results[i]["all_probs"].items()},
                "retrieved_passages":  retrieved_batched[i],
                "rag_explanation":     rag_explanations[i],
                "cpace_explanation":   cpace_explanations[i],
                "final_explanation":   best_expl,
                "explanation_source":  winner,
                "rag_score":           round(rag_score, 4),
                "cpace_score":         round(cpace_score, 4),
                "human_review_flag":   human_review,
            }
            all_results.append(result)
    
    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pipeline II Inference")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to CSV file with 'text' column (default: test.csv)")
    parser.add_argument("--text", type=str, default=None,
                        help="Single argument text to analyze")
    parser.add_argument("--top_k", type=int, default=TOP_K,
                        help=f"Number of passages to retrieve (default: {TOP_K})")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SCRIPT 07: PIPELINE II INFERENCE")
    print("=" * 60)
    
    # Check that required model files exist
    for path, name in [
        (MODEL_CLS,   "DeBERTa classifier"), (MODEL_GEN, "T5 generator"),
        (INDEX_FILE,  "FAISS index"),         (LABEL_MAP, "Label map"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            print("Run scripts 02–05 before running inference.")
            sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    
    # ── Load all models ────────────────────────────────────────────────────
    print("\n  Loading all pipeline models...")
    t0 = time.time()
    models = PipelineModels(device)
    print(f"  All models loaded in {time.time()-t0:.1f}s")
    
    # ── Prepare input data ─────────────────────────────────────────────────
    if args.text:
        # Single-argument mode
        original_texts = [args.text]
        print(f"\n  Single argument mode: '{args.text[:80]}...'")
    elif args.input:
        df = pd.read_csv(args.input)
        text_col = "text_normalized" if "text_normalized" in df.columns else "text"
        original_texts = df[text_col].fillna("").astype(str).tolist()
        print(f"\n  Loaded {len(original_texts)} arguments from {args.input}")
    else:
        # Default: use test.csv
        if not os.path.exists(TEST_CSV):
            print("ERROR: test.csv not found. Run script 02 first.")
            sys.exit(1)
        df = pd.read_csv(TEST_CSV)
        text_col = "text_normalized" if "text_normalized" in df.columns else "text"
        original_texts = df[text_col].fillna("").astype(str).tolist()
        print(f"\n  Using test set: {len(original_texts)} arguments from test.csv")
    
    # ── NER normalize all texts ────────────────────────────────────────────
    print(f"\n  Normalizing {len(original_texts)} texts...")
    normalized_texts = [normalize_text(t) for t in tqdm(original_texts, desc="  NER normalize")]
    
    # ── Run inference ──────────────────────────────────────────────────────
    print(f"\n  Running full Pipeline II inference...")
    t_start = time.time()
    results = run_inference(normalized_texts, original_texts, models)
    elapsed = time.time() - t_start
    
    # ── Save results ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # ── Print summary ──────────────────────────────────────────────────────
    total = len(results)
    flagged = sum(1 for r in results if r["human_review_flag"])
    rag_wins = sum(1 for r in results if r["explanation_source"] == "rag")
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Total arguments processed:  {total}")
    print(f"  Total time:                 {elapsed:.1f}s")
    print(f"  Avg. time per argument:     {elapsed/max(total,1):.2f}s")
    print(f"  Flagged for human review:   {flagged} ({100*flagged/max(total,1):.1f}%)")
    print(f"  RAG explanation selected:   {rag_wins} ({100*rag_wins/max(total,1):.1f}%)")
    print(f"  CPACE explanation selected: {total-rag_wins} ({100*(total-rag_wins)/max(total,1):.1f}%)")
    print(f"\n  Results saved to: {OUTPUT_FILE}")
    
    # ── Print first 3 results for inspection ───────────────────────────────
    print("\n  === SAMPLE RESULTS (first 3) ===")
    for r in results[:3]:
        print(f"\n  Argument:  {r['argument_original'][:100]}...")
        print(f"  Predicted: {r['predicted_label']}  (confidence: {r['confidence']:.2%})")
        print(f"  Source:    {r['explanation_source'].upper()} (RAG={r['rag_score']:.3f}, CPACE={r['cpace_score']:.3f})")
        print(f"  Explanation: {r['final_explanation'][:200]}...")
        if r["human_review_flag"]:
            print(f"  ⚠ FLAGGED FOR HUMAN REVIEW (confidence below {CONFIDENCE_THR:.0%})")
    
    print("\nNext: python scripts/08_evaluate.py")


if __name__ == "__main__":
    main()
