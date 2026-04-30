"""
=============================================================================
SCRIPT: 01_setup_check.py
PURPOSE: Verifies that all required packages are installed, the spaCy model
         is available, and all dataset files exist in data/raw/.
         Run this FIRST before any other script.
RUN:     python scripts/01_setup_check.py
=============================================================================
"""

import sys
import os

# ── Paths ──────────────────────────────────────────────────────────────────
# All paths are relative to the project root (pipeline2/).
# If you run this script from inside scripts/, adjust BASE_DIR accordingly.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")

REQUIRED_FILES = [
    "final_cleaned_dataset.csv",
    "kialo_flat_clean.csv",
    # "snli_clean.csv",  # uncomment if you have this file
]

REQUIRED_PACKAGES = [
    "torch", "transformers", "datasets", "sentence_transformers",
    "faiss", "spacy", "sklearn", "pandas", "numpy", "tqdm",
    "flask", "requests", "evaluate",
]

print("=" * 60)
print("PIPELINE II — SETUP CHECK")
print("=" * 60)

# ── 1. Python version ──────────────────────────────────────────────────────
print(f"\n[1] Python version: {sys.version}")
if sys.version_info < (3, 10):
    print("    WARNING: Python 3.10+ recommended.")
else:
    print("    ✓ Python version OK")

# ── 2. Check packages ─────────────────────────────────────────────────────
print("\n[2] Checking required packages...")
missing = []
for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg)
        print(f"    ✓ {pkg}")
    except ImportError:
        print(f"    ✗ MISSING: {pkg}")
        missing.append(pkg)

if missing:
    print(f"\n    Install missing packages with:")
    print(f"    pip install {' '.join(missing)}")

# ── 3. Check spaCy model ───────────────────────────────────────────────────
print("\n[3] Checking spaCy NER model (en_core_web_lg)...")
try:
    import spacy
    nlp = spacy.load("en_core_web_lg")
    print("    ✓ en_core_web_lg loaded OK")
except Exception as e:
    print(f"    ✗ spaCy model not found. Run: python -m spacy download en_core_web_lg")

# ── 4. Check dataset files ─────────────────────────────────────────────────
print("\n[4] Checking dataset files in data/raw/...")
for fname in REQUIRED_FILES:
    fpath = os.path.join(DATA_RAW, fname)
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"    ✓ {fname}  ({size_mb:.2f} MB)")
    else:
        print(f"    ✗ MISSING: {fpath}")

# ── 5. Check PyTorch device ────────────────────────────────────────────────
print("\n[5] Checking PyTorch device...")
try:
    import torch
    print(f"    PyTorch version: {torch.__version__}")
    print(f"    CUDA available:  {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Will use device: {device}")
    if device == "cpu":
        # Try to use as many threads as possible on the i5-1235U (12 threads)
        torch.set_num_threads(min(torch.get_num_threads(), 10))
        print(f"    CPU threads set: {torch.get_num_threads()}")
except Exception as e:
    print(f"    ✗ PyTorch error: {e}")

# ── 6. Create output directories ───────────────────────────────────────────
print("\n[6] Creating output directories if missing...")
dirs = [
    "data/processed", "data/retrieval_corpus",
    "models/deberta_classifier", "models/generator",
    "index", "outputs/predictions", "outputs/explanations"
]
for d in dirs:
    path = os.path.join(BASE_DIR, d)
    os.makedirs(path, exist_ok=True)
    print(f"    ✓ {d}")

print("\n" + "=" * 60)
print("Setup check complete. Fix any ✗ items above before proceeding.")
print("=" * 60)
