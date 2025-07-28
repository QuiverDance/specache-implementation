#!/bin/bash
set -e

cd ~/specache-project
PROJECT_ROOT=$PWD

INPUT_TEXT_PATH="$PROJECT_ROOT/pg19_firstbook.txt"

export TOKENIZERS_PARALLELISM=false

if [ ! -f "$INPUT_TEXT_PATH" ]; then
    echo "Error: pg19_firstbook.txt not found in project root."
    exit 1
fi


ln -sfn speedup/src/mistral_baseline flexgen

echo "Running Mistral Baseline..."

python -m flexgen.flex_mistral \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --path ~/flexgen_weights \
    --offload-dir ~/flexgen_offload_dir \
    --prompt-len 512 \
    --gen-len 128 \
    --gpu-batch-size 4 \
    --num-gpu-batches 1 \
    --percent 100 00 100 0 100 0 \
    --warmup-input-path "$INPUT_TEXT_PATH" \
    --test-input-path "$INPUT_TEXT_PATH" \
    --verbose 2

rm flexgen
