"""
=============================================================================
SCRIPT: 08_evaluate.py
PURPOSE: Comprehensive evaluation of the Pipeline II system.
         
         Evaluates across THREE dimensions:
         
         A. CLASSIFICATION METRICS (requires test.csv with true labels)
            • Accuracy, Macro F1, Weighted F1
            • Per-class Precision, Recall, F1
            • Confusion matrix (shows which fallacy types get confused)
         
         B. EXPLANATION QUALITY METRICS (automatic)
            • ROUGE-1, ROUGE-L: word overlap between generated and reference explanations
            • BERTScore: semantic similarity using BERT embeddings (better than ROUGE)
         
         C. SYSTEM PERFORMANCE METRICS
            • Throughput (arguments per second)
            • Explanation source distribution (RAG vs CPACE wins)
            • Human review flag rate (low-confidence predictions)
            • Average confidence per class
         
         D. BASELINE COMPARISON
            • Reports performance so you can compare:
              - With NER vs Without NER (run script 04 on unnormalized text to get baseline)
              - With RAG vs Without RAG
         
RUN:     python scripts/08_evaluate.py
         python scripts/08_evaluate.py --results path/to/results.jsonl
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for WSL2 compatibility
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED    = os.path.join(BASE_DIR, "data", "processed")
RESULTS_FILE = os.path.join(BASE_DIR, "outputs", "predictions", "results.jsonl")
TEST_CSV     = os.path.join(PROCESSED, "test.csv")
LABEL_MAP    = os.path.join(PROCESSED, "label_map.json")
EVAL_OUT_DIR = os.path.join(BASE_DIR, "outputs", "explanations")
REPORT_FILE  = os.path.join(EVAL_OUT_DIR, "evaluation_report.txt")
CONFUSION_PNG = os.path.join(EVAL_OUT_DIR, "confusion_matrix.png")
DIST_PNG      = os.path.join(EVAL_OUT_DIR, "class_distribution.png")


# ── Helpers ────────────────────────────────────────────────────────────────

def load_results(path: str):
    """Loads inference results from the JSONL file produced by script 07."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_true_labels(test_csv: str, label_map: dict) -> dict:
    """
    Loads the true labels from the test CSV.
    Returns a dict mapping normalized text → true_label_string.
    We match by normalized text since results.jsonl stores it.
    """
    if not os.path.exists(test_csv):
        return {}
    df = pd.read_csv(test_csv)
    id_map = {}
    text_col = "text_normalized" if "text_normalized" in df.columns else "text"
    for _, row in df.iterrows():
        text = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
        label = str(row["label"]).strip().lower() if pd.notna(row["label"]) else ""
        if text and label:
            id_map[text] = label
    return id_map


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION SECTIONS
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_classification(results, true_label_map, id_to_label, label_map):
    """
    SECTION A: Classification Metrics.
    
    For each result that has a matching true label in the test set,
    computes accuracy, macro F1, per-class metrics, and confusion matrix.
    
    Macro F1 treats all classes equally regardless of size.
    Weighted F1 weights each class by its support (number of examples).
    """
    y_true, y_pred = [], []
    
    for r in results:
        norm_text = r.get("argument_normalized", "").strip()
        true_label = true_label_map.get(norm_text)
        if true_label is None:
            # Try original text
            orig_text = r.get("argument_original", "").strip()
            true_label = true_label_map.get(orig_text)
        
        if true_label and true_label in label_map:
            y_true.append(label_map[true_label])
            y_pred.append(r["predicted_label"] if isinstance(r["predicted_label"], int)
                         else label_map.get(r["predicted_label"], -1))
    
    if not y_true:
        return None, None, None, None
    
    label_names = [id_to_label[i] for i in range(len(label_map))]
    
    acc     = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report  = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )
    cm      = confusion_matrix(y_true, y_pred)
    
    return acc, macro_f1, weighted_f1, report, cm, y_true, y_pred, label_names


def evaluate_explanations(results):
    """
    SECTION B: Explanation Quality (Automatic Metrics).
    
    ROUGE measures word-overlap between generated and reference explanations.
    ROUGE-1: unigram (word-level) overlap.
    ROUGE-L: longest common subsequence (sentence-level fluency).
    
    BERTScore measures semantic similarity using contextual BERT embeddings.
    It captures paraphrase quality better than ROUGE.
    
    Since we don't have human-written reference explanations for every test example,
    we use the CPACE template-filled explanation as a reference for the RAG output,
    and the RAG output as a reference for CPACE — a cross-evaluation approach.
    This measures how semantically aligned the two explanations are.
    """
    print("\n  Computing ROUGE scores...")
    
    rouge_r1, rouge_rl = [], []
    
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        
        for r in results:
            rag   = r.get("rag_explanation",   "")
            cpace = r.get("cpace_explanation",  "")
            if not rag or not cpace:
                continue
            
            # Cross-evaluate: use each as reference for the other
            scores_a = scorer.score(rag, cpace)    # RAG as hypothesis, CPACE as ref
            scores_b = scorer.score(cpace, rag)    # CPACE as hypothesis, RAG as ref
            
            # Average of both directions
            rouge_r1.append((scores_a["rouge1"].fmeasure + scores_b["rouge1"].fmeasure) / 2)
            rouge_rl.append((scores_a["rougeL"].fmeasure + scores_b["rougeL"].fmeasure) / 2)
        
        avg_r1 = np.mean(rouge_r1) if rouge_r1 else 0
        avg_rl = np.mean(rouge_rl) if rouge_rl else 0
        print(f"  Average ROUGE-1 (RAG↔CPACE alignment): {avg_r1:.4f}")
        print(f"  Average ROUGE-L (RAG↔CPACE alignment): {avg_rl:.4f}")
    
    except ImportError:
        print("  ROUGE not available. Install: pip install rouge-score")
        avg_r1, avg_rl = 0, 0
    
    # BERTScore
    bertscore_f1 = None
    print("\n  Computing BERTScore (this may take a few minutes)...")
    try:
        from bert_score import score as bert_score_fn
        
        hypotheses = [r.get("rag_explanation",  "no explanation") for r in results[:50]]
        references = [r.get("cpace_explanation", "no explanation") for r in results[:50]]
        
        # Limit to 50 examples for speed on CPU (BERTScore is slow without GPU)
        P, R, F1 = bert_score_fn(
            hypotheses, references,
            lang="en",
            model_type="distilbert-base-uncased",  # Smaller model for CPU speed
            verbose=False
        )
        bertscore_f1 = float(F1.mean())
        print(f"  BERTScore F1 (RAG vs CPACE, sample of {len(hypotheses)}): {bertscore_f1:.4f}")
    
    except ImportError:
        print("  BERTScore not available. Install: pip install bert-score")
    except Exception as e:
        print(f"  BERTScore error: {e}")
    
    return avg_r1, avg_rl, bertscore_f1


def evaluate_system_performance(results):
    """
    SECTION C: System Performance Metrics.
    
    Analyzes confidence distributions, explanation source selection,
    human review flagging rate, and retrieval score distributions.
    """
    confidences = [r.get("confidence", 0) for r in results]
    flagged     = [r for r in results if r.get("human_review_flag", False)]
    rag_wins    = [r for r in results if r.get("explanation_source") == "rag"]
    
    # Per-class average confidence
    class_confidences = defaultdict(list)
    for r in results:
        class_confidences[r.get("predicted_label", "unknown")].append(
            r.get("confidence", 0)
        )
    
    # Label distribution in predictions
    pred_dist = Counter(r.get("predicted_label", "unknown") for r in results)
    
    # Retrieval score statistics
    rag_scores   = [r.get("rag_score",   0) for r in results]
    cpace_scores = [r.get("cpace_score", 0) for r in results]
    
    return {
        "total":            len(results),
        "avg_confidence":   np.mean(confidences),
        "min_confidence":   np.min(confidences),
        "max_confidence":   np.max(confidences),
        "flagged_count":    len(flagged),
        "flagged_pct":      100 * len(flagged) / max(len(results), 1),
        "rag_wins":         len(rag_wins),
        "cpace_wins":       len(results) - len(rag_wins),
        "pred_distribution": dict(pred_dist),
        "class_avg_confidence": {k: np.mean(v) for k, v in class_confidences.items()},
        "avg_rag_score":    np.mean(rag_scores),
        "avg_cpace_score":  np.mean(cpace_scores),
    }


def plot_confusion_matrix(cm, label_names, out_path):
    """Saves a confusion matrix heatmap as a PNG file."""
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names, ax=ax
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix — Pipeline II Fallacy Classifier", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Confusion matrix saved to: {out_path}")
    except Exception as e:
        print(f"  Could not save confusion matrix: {e}")


def plot_class_distribution(pred_dist, out_path):
    """Saves a bar chart of predicted label distribution."""
    try:
        labels = list(pred_dist.keys())
        counts = [pred_dist[l] for l in labels]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(range(len(labels)), counts, color=cm.tab10(np.linspace(0, 1, len(labels))))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Fallacy Type", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Predicted Label Distribution — Test Set", fontsize=14)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(count), ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Class distribution plot saved to: {out_path}")
    except Exception as e:
        print(f"  Could not save distribution plot: {e}")


def write_report(sections: dict, path: str):
    """Writes a full evaluation report as a plain-text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PIPELINE II — EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        for section_title, content in sections.items():
            f.write(f"\n{'─'*70}\n")
            f.write(f"SECTION: {section_title}\n")
            f.write(f"{'─'*70}\n")
            f.write(content + "\n")
    
    print(f"\n  Full evaluation report saved to: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline II Evaluation")
    parser.add_argument("--results", type=str, default=RESULTS_FILE,
                        help=f"Path to results JSONL (default: {RESULTS_FILE})")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SCRIPT 08: PIPELINE II EVALUATION")
    print("=" * 60)
    
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)
    
    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        print("Run script 07 (inference) first.")
        sys.exit(1)
    
    # Load label map
    with open(LABEL_MAP) as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}
    
    # Load results
    print(f"\n  Loading results from {args.results}...")
    results = load_results(args.results)
    print(f"  Loaded {len(results)} prediction records.")
    
    report_sections = {}
    
    # ── Section A: Classification ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("A. CLASSIFICATION METRICS")
    print("─" * 60)
    
    true_label_map = load_true_labels(TEST_CSV, label_map)
    
    if true_label_map:
        ret = evaluate_classification(results, true_label_map, id_to_label, label_map)
        if ret[0] is not None:
            acc, macro_f1, weighted_f1, report, cm_matrix, y_true, y_pred, label_names = ret
            
            cls_text = (
                f"Accuracy:         {acc:.4f}  ({acc*100:.2f}%)\n"
                f"Macro F1:         {macro_f1:.4f}\n"
                f"Weighted F1:      {weighted_f1:.4f}\n"
                f"Evaluated on:     {len(y_true)} matched test examples\n\n"
                f"Per-Class Report:\n{report}"
            )
            print(cls_text)
            report_sections["A. Classification Metrics"] = cls_text
            
            # Plot confusion matrix
            plot_confusion_matrix(cm_matrix, label_names, CONFUSION_PNG)
        else:
            print("  Could not match results to true labels. Check test.csv.")
            report_sections["A. Classification Metrics"] = "Could not match results to true labels."
    else:
        print("  test.csv not found or has no matching labels — skipping classification metrics.")
        report_sections["A. Classification Metrics"] = "test.csv not found."
    
    # ── Section B: Explanation Quality ────────────────────────────────────
    print("\n" + "─" * 60)
    print("B. EXPLANATION QUALITY METRICS")
    print("─" * 60)
    
    avg_r1, avg_rl, bertscore = evaluate_explanations(results)
    
    expl_text = (
        f"ROUGE-1 (RAG ↔ CPACE alignment):   {avg_r1:.4f}\n"
        f"ROUGE-L (RAG ↔ CPACE alignment):   {avg_rl:.4f}\n"
        f"BERTScore F1 (sample of up to 50):  {bertscore if bertscore else 'N/A'}\n\n"
        f"Interpretation:\n"
        f"  • ROUGE measures word-overlap between the two generated explanations.\n"
        f"  • Higher values indicate both systems produce semantically similar text.\n"
        f"  • BERTScore captures paraphrase-level similarity (more robust than ROUGE).\n"
        f"  • For final explanation quality, use human evaluation (see README)."
    )
    print("\n" + expl_text)
    report_sections["B. Explanation Quality"] = expl_text
    
    # ── Section C: System Performance ─────────────────────────────────────
    print("\n" + "─" * 60)
    print("C. SYSTEM PERFORMANCE")
    print("─" * 60)
    
    perf = evaluate_system_performance(results)
    
    perf_text = (
        f"Total arguments:           {perf['total']}\n"
        f"Avg confidence:            {perf['avg_confidence']:.4f}  ({perf['avg_confidence']*100:.1f}%)\n"
        f"Min confidence:            {perf['min_confidence']:.4f}\n"
        f"Max confidence:            {perf['max_confidence']:.4f}\n"
        f"Flagged for human review:  {perf['flagged_count']}  ({perf['flagged_pct']:.1f}%)\n"
        f"RAG explanation wins:      {perf['rag_wins']}  ({100*perf['rag_wins']/max(perf['total'],1):.1f}%)\n"
        f"CPACE explanation wins:    {perf['cpace_wins']}  ({100*perf['cpace_wins']/max(perf['total'],1):.1f}%)\n"
        f"Avg RAG fusion score:      {perf['avg_rag_score']:.4f}\n"
        f"Avg CPACE fusion score:    {perf['avg_cpace_score']:.4f}\n\n"
        f"Predicted Label Distribution:\n"
    )
    for label, count in sorted(perf["pred_distribution"].items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 40)
        perf_text += f"  {label:<30s}: {count:>4d}  {bar}\n"
    
    perf_text += "\nPer-Class Average Confidence:\n"
    for label, avg_conf in sorted(perf["class_avg_confidence"].items()):
        perf_text += f"  {label:<30s}: {avg_conf:.4f}\n"
    
    print(perf_text)
    report_sections["C. System Performance"] = perf_text
    
    # Plot distribution
    plot_class_distribution(perf["pred_distribution"], DIST_PNG)
    
    # ── Section D: Baseline guidance ───────────────────────────────────────
    baseline_text = (
        "To run baseline comparisons:\n\n"
        "1. NO NER BASELINE:\n"
        "   In scripts/02_preprocess.py, set SKIP_NER=True (add this flag).\n"
        "   Re-run scripts 02 → 04 → 07 → 08. Compare Macro F1.\n"
        "   Expected: NER-normalized version should score higher F1.\n\n"
        "2. NO RAG BASELINE:\n"
        "   In scripts/07_inference.py, set TOP_K=0.\n"
        "   Re-run script 07. The generator runs on argument text alone.\n"
        "   Compare explanation ROUGE/BERTScore. RAG version should score higher.\n\n"
        "3. SINGLE-THREAD BASELINE:\n"
        "   In scripts/07_inference.py, set max_workers=1 in ThreadPoolExecutor.\n"
        "   Measure inference time. Parallel version should be faster.\n"
    )
    report_sections["D. Baseline Comparison Guidance"] = baseline_text
    
    # ── Write full report ──────────────────────────────────────────────────
    write_report(report_sections, REPORT_FILE)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Report:          {REPORT_FILE}")
    print(f"  Confusion matrix: {CONFUSION_PNG}")
    print(f"  Label distribution: {DIST_PNG}")
    print("\nPipeline II implementation complete! ✓")


if __name__ == "__main__":
    main()
