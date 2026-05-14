## Distributed Fallacy Detection & Explanation System

---
# Logical Fallacy Detection System

This system automatically detects logical fallacies in text (like ad hominem attacks or false dilemmas) and explains why something is a fallacy.

## What Makes This Different

Most existing systems are slow and don't scale well. This system runs tasks in parallel (doing many things at once) instead of one after another, making it much faster.

## How It Works

The system has several steps:
1. **Normalization** – cleans and prepares the text
2. **Retrieval** – searches a database of 150,000 examples for similar arguments
3. **Classification** – identifies which of 13 fallacy types the text contains
4. **Explanation** – generates two types of explanations why it's a fallacy
5. **Parallel processing** – runs multiple tasks simultaneously using GPU batching and threading

## Performance

- **Throughput:** 48.1 texts per second (vs 10.3 in sequential systems) – **4.66x faster**
- **Speed per text:** 7.10 milliseconds
- **Accuracy:** 47.46% across 13 fallacy types
- **F1 score:** 0.4504

## Training Data

- 2,300 examples
- 13 fallacy classes

## Key Technologies

- DistilBERT (lightweight transformer model)
- FAISS (fast similarity search)
- RAG (retrieval-augmented generation for explanations)
- CPACE (contrastive symbolic reasoning)

## Main Contributions

- Complete pipeline from raw text to fallacy detection + explanation
- 4.66x speed improvement over traditional approaches
- Dual explanation system (evidence-based + contrastive reasoning)
- Scalable retrieval from 150,000 passages
- Open-source benchmarking of parallel vs sequential inference

---
- The setup instructions are in setup.md
- All CSV files can be found in dataset.zip

