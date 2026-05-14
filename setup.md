# Distributed Fallacy Detection & Explanation System
## Single-Machine Implementation (WSL2 / Ubuntu on Windows)

---

## Project Structure

```
pipeline2/
├── data/
│   ├── raw/                        ← Put your CSV files here
│   │   ├── final_cleaned_dataset.csv
│   │   ├── kialo_flat_clean.csv
│   │   └── snli_clean.csv          ← optional, for generator pretraining
│   ├── processed/                  ← Split & NER-normalized datasets
│   └── retrieval_corpus/           ← Kialo passages ready for indexing
├── models/
│   ├── deberta_classifier/         ← Trained DeBERTa checkpoint
│   └── generator/                  ← Trained T5 generator checkpoint
├── index/                          ← FAISS vector index
├── outputs/
│   ├── predictions/                ← JSON prediction results
│   └── explanations/               ← Generated explanation files
├── scripts/                        ← All runnable Python scripts
└── cpace_service/                  ← Contrastive explanation module
```

---

## Script Run Order

Run scripts in this exact order:

```
Step 1:  python scripts/01_setup_check.py
Step 2:  python scripts/02_preprocess.py
Step 3:  python scripts/03_build_index.py
Step 4:  python scripts/04_train_classifier.py
Step 5:  python scripts/05_train_generator.py
Step 6:  python scripts/06_cpace_module.py          (keep running in background)
Step 7:  python scripts/07_inference.py
Step 8:  python scripts/08_evaluate.py
```

---

## Setup (WSL2 Ubuntu)

```bash
# 1. Install Python 3.10 if not installed
sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip -y

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Install CPU-only PyTorch first
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Download spaCy NER model
python -m spacy download en_core_web_lg

# 6. Place your dataset files in data/raw/
#    - final_cleaned_dataset.csv
#    - kialo_flat_clean.csv
#    - snli_clean.csv (if available)
```

---

## Dataset Columns (Reference)

| File | Columns | Used For |
|------|---------|----------|
| final_cleaned_dataset.csv | id, text, label, type | Fallacy classifier training |
| kialo_flat_clean.csv | question, argument, type, id | RAG retrieval corpus |
| snli_clean.csv | label, premise, hypothesis, label_id | Generator pretraining |

## Fallacy Labels in Dataset
- ad hominem, appeal to popularity, equivocation, fallacy of extension,
  false cause, false dilemma, hasty generalization, intentional,
  logical fallacy, relevance fallacy

---

## Time Estimates (i5-1235U, CPU only)

| Step | Estimated Time |
|------|---------------|
| Preprocessing | 5–15 min |
| Index building | 10–20 min |
| Classifier training (5 epochs) | 1–3 hours |
| Generator training (3 epochs) | 1–2 hours |
| Inference (full test set) | 10–30 min |
| Evaluation | 2–5 min |

---
