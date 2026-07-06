#!/bin/bash
# Start a vLLM OpenAI-compatible server on aixsrv1.
# Run this in Terminal 1 before running any pipeline script.
#
# Usage:
#   bash scripts/start_vllm_server.sh <model_name> [gpu_id]
#
# Examples:
#   bash scripts/start_vllm_server.sh Qwen/Qwen3-30B-A3B-Instruct
#   bash scripts/start_vllm_server.sh deepseek-ai/DeepSeek-R1-Distill-Llama-70B 1
#
# The server listens on http://localhost:8000/v1 (OpenAI-compatible).
# Your pipeline's api_client.py points to this URL automatically via base.yaml.
# Press Ctrl+C to stop the server when done.

set -e

MODEL=${1:?"Error: model name required. Usage: $0 <model_name> [gpu_id]"}
GPU_ID=${2:-0}  # default to GPU 0

echo "Starting vLLM server..."
echo "  Model    : $MODEL"
echo "  GPU      : $GPU_ID"
echo "  Endpoint : http://localhost:8000/v1"
echo ""
echo "Waiting for server to be ready — run your pipeline in a second terminal."
echo "Press Ctrl+C to stop."
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --dtype auto
