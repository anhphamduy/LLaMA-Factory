#!/bin/bash
#
# Run the Genlio LLaMA-Factory Worker (handles both training and inference)
#
# Required environment variables:
#   DATABASE_URL       - PostgreSQL connection string
#   SUPABASE_URL       - Supabase project URL
#   SUPABASE_KEY       - Supabase service key
#
# Optional environment variables:
#   WORKER_ID          - Unique worker identifier (auto-generated if not set)
#   WORKER_NAME        - Human-readable worker name
#   POLL_INTERVAL      - Seconds between job polls (default: 5)
#   DATA_DIR           - Directory for LLaMA-Factory datasets (default: ./data)
#   OUTPUT_DIR         - Directory for training outputs (default: ./outputs)
#   MODEL_CACHE_DIR    - Directory to cache adapters (default: ./model_cache)
#   TRAINING_MIN_GPU_GB  - Minimum GPU memory for training (default: 12)
#   INFERENCE_MIN_GPU_GB - Minimum GPU memory for inference (default: 6)
#
# Usage:
#   ./run_worker.sh
#   WORKER_NAME="gpu-server-1" ./run_worker.sh
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check required environment variables
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL environment variable is required"
    echo "Example: export DATABASE_URL='postgresql://user:pass@host:5432/db'"
    exit 1
fi

if [ -z "$SUPABASE_URL" ]; then
    echo "ERROR: SUPABASE_URL environment variable is required"
    echo "Example: export SUPABASE_URL='https://xxx.supabase.co'"
    exit 1
fi

if [ -z "$SUPABASE_KEY" ]; then
    echo "ERROR: SUPABASE_KEY environment variable is required"
    exit 1
fi

# Set defaults
export WORKER_ID="${WORKER_ID:-worker-$(hostname)-$(date +%s | tail -c 5)}"
export POLL_INTERVAL="${POLL_INTERVAL:-5}"
export DATA_DIR="${DATA_DIR:-./data}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-./model_cache}"
export TRAINING_MIN_GPU_GB="${TRAINING_MIN_GPU_GB:-12}"
export INFERENCE_MIN_GPU_GB="${INFERENCE_MIN_GPU_GB:-6}"

echo "=========================================="
echo "  Genlio LLaMA-Factory Worker"
echo "=========================================="
echo "Worker ID:     $WORKER_ID"
echo "Worker Name:   ${WORKER_NAME:-auto}"
echo "Poll Interval: ${POLL_INTERVAL}s"
echo "Data Dir:      $DATA_DIR"
echo "Output Dir:    $OUTPUT_DIR"
echo "Model Cache:   $MODEL_CACHE_DIR"
echo "Training GPU:  ${TRAINING_MIN_GPU_GB}GB min"
echo "Inference GPU: ${INFERENCE_MIN_GPU_GB}GB min"
echo "=========================================="

# Create directories
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$MODEL_CACHE_DIR"

# Run the worker
exec python genlio_worker.py \
    --database-url "$DATABASE_URL" \
    --worker-id "$WORKER_ID" \
    ${WORKER_NAME:+--worker-name "$WORKER_NAME"} \
    --poll-interval "$POLL_INTERVAL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --supabase-url "$SUPABASE_URL" \
    --supabase-key "$SUPABASE_KEY" \
    --model-cache-dir "$MODEL_CACHE_DIR" \
    --training-min-gpu-gb "$TRAINING_MIN_GPU_GB" \
    --inference-min-gpu-gb "$INFERENCE_MIN_GPU_GB"
