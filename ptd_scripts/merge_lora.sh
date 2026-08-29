#!/bin/bash

export ADAPTER=""
export BASE_MODEL=""
export MERGED_MODEL=""

export PYTHONPATH="$PWD/src:$PWD"

python src/merge_lora_weights.py \
  --model-path "$ADAPTER" \
  --model-base "$BASE_MODEL" \
  --save-model-path "$MERGED_MODEL" \
  --safe-serialization
