#!/bin/bash
#SBATCH --job-name=stark_faiss_build
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/faiss_build_%j.out

# On GWDG HPC: use faiss-gpu instead of faiss-cpu
# pip install faiss-gpu  (do NOT use faiss-cpu here)

# TODO: implement
# Steps:
# 1. Load STARK-AMAZON corpus from data/stark_amazon/corpus.jsonl
# 2. Encode all product nodes with dense_model_name (e5-mistral-7b-instruct
#    or Qwen3-Embedding-4B — verify availability first)
# 3. Build faiss.IndexFlatL2 over the embeddings
# 4. Save with faiss.write_index to data/faiss_index/stark_amazon.index
