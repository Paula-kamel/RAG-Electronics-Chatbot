# RAG-Based Electronics Chatbot

An intelligent Retrieval-Augmented Generation (RAG) chatbot for electronics product search, comparison, and recommendation. Built as a Final Year Project for BSc Computer Science at the University of Hertfordshire (2025-2026).

## Project Overview

This system allows users to search, compare, and receive recommendations for electronics products through natural language interaction. All responses are grounded in retrieved records from the Amazon Electronics metadata dataset, ensuring factual accuracy and eliminating hallucination.

The system supports four query types:
- Direct product lookup by ASIN identifier
- Natural language product search
- Side-by-side product comparison
- Budget-constrained product recommendation

## System Architecture

The system is built around two pipelines:

**Offline Pipeline** — runs once before deployment:
1. Dataset cleaning and preprocessing
2. Dense vector embedding generation
3. FAISS index construction

**Online Pipeline** — runs at query time:
1. Query classification and analysis
2. Hybrid semantic and lexical retrieval
3. Multi-signal reranking and filtering
4. Grounded response generation via LLM
5. Gradio web interface display

## Repository Structure

├── script1_data_prep.py — Dataset cleaning and text preparation
├── script2_indexing.py — Embedding generation and FAISS indexing
├── amazon_electronics_real_rag_rewritten.py — Main chatbot application
└── README.md

## Requirements

### Python Libraries
pip install pandas numpy faiss-cpu requests gradio tqdm

### Models via Ollama
Install Ollama from https://ollama.com then pull the required models:

ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull qwen2.5:7b

## How to Run

### Step 1 — Prepare the Dataset
Download the Amazon Electronics metadata dataset and update the input path in script1_data_prep.py then run:

python script1_data_prep.py

This produces records_sample.jsonl

### Step 2 — Generate Embeddings and Build Index
Update the paths in script2_indexing.py to match your system then run:

python script2_indexing.py

This produces electronics_index1.faiss and metadata_map1.jsonl

### Step 3 — Run the Chatbot
Make sure Ollama is running with both models loaded then run:

python amazon_electronics_real_rag_rewritten.py

The Gradio interface will open in your browser automatically.

## Dataset

This project uses the Amazon Reviews 2023 dataset Electronics category:

McAuley, J., Shao, Z., Deng, W., Bhatt, R. and Ramchandran, K. (2023) Amazon Reviews 2023. Hugging Face. Available at: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

## Author

Paula Hany Joseph Kamel
BSc Computer Science
University of Hertfordshire
Academic Year 2025-2026
Supervisor: Rasha Shoitan
