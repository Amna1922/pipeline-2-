## Distributed Fallacy Detection & Explanation System

---

Automated logical fallacy detection in natural language is a challenging task requiring 
robust argument understanding, large-scale retrieval, and interpretable explanation 
generation. Existing approaches predominantly rely on sequence-to-sequence (Seq2Seq) 
pipelines that suffer from poor scalability and high inference latency. This paper 
proposes a parallel and distributed NLP framework that integrates named entity 
recognition (NER)-based normalization, dense passage retrieval (DPR) over 
FAISS-indexed corpus, a distilBERT-based multi-class classifier, a retrieval-augmented 
generation (RAG) explanation module, and a contrastive parallel argumentation and 
contrastive explanation (CPACE) module. The entire inference pipeline is built using 
GPU batch processing and thread-level concurrency, achieving a 4.66x throughput 
speed-up over a sequential baseline. Trained on a 2300-sample logical fallacy dataset 
across 13 fallacy classes. The system attains a micro-average F1 equal to 0.4504 and 
accuracy equal to 0.4746 on the test set, along with inference latency of 7.10ms per 
argument. Experimental results demonstrate that parallel execution not only accelerates 
processing to 48.1 tasks per second, but also supports scalable deployment over a large 
retrieval corpora of 150,000 passages. This work advances the state of the art in 
argument analysis by demonstrating that parallel and distributed computing strategies 
directly and measurably improve NLP pipeline efficiency, throughput, and scalability 
without compromising classification quality. 

---
The setup instructions are in setup.md

