#!/bin/bash

export BASE_MODEL="Qwen/Qwen3-VL-4B-Instruct"
export INITIALIZED_MODEL=""

export PYTHONPATH="$PWD/src:$PWD"

python src/initialize_ptd_tokens.py \
  --model-id "$BASE_MODEL" \
  --output-dir "$INITIALIZED_MODEL" \
  --dtype bfloat16
