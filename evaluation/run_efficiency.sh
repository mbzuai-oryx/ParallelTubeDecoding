#!/bin/bash
set -euo pipefail

: "${MODEL:?Set MODEL to the merged PTD checkpoint}"
: "${LMMS_EVAL_REPO:?Set LMMS_EVAL_REPO to lmms-eval commit 88b23e2bfa16a1edbc16e9e238ed82130b3a4f56}"

PTD_REPO="${PTD_REPO:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PTD_REPO}/efficiency_results}"
LIMIT="${LIMIT:-}"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="${PTD_REPO}/src:${LMMS_EVAL_REPO}:${PYTHONPATH:-}"
COMMON="pretrained=${MODEL},ptd_root=${PTD_REPO},generation_format=spatio_temporal_grounding,measure_efficiency=true,fps=2,max_num_frames=64,min_pixels=131072,max_pixels=786432,temporal_patch_size=1,system_prompt=none"
LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

cd "$LMMS_EVAL_REPO"

# Quantized (NTP) uses SDPA for the complete Qwen3-VL model.
python -m lmms_eval \
  --model ptd_qwen3_vl \
  --model_args "${COMMON},decoding=quantized,attn_implementation=sdpa,efficiency_log=${OUTPUT_DIR}/quantized.jsonl" \
  --tasks ptd_vidstg \
  --batch_size 1 \
  --output_path "${OUTPUT_DIR}/quantized_lmms" \
  "${LIMIT_ARGS[@]}"

# PTD uses text SDPA, vision FlashAttention-2, and PTD FlashAttention-2.
python -m lmms_eval \
  --model ptd_qwen3_vl \
  --model_args "${COMMON},decoding=ptd,attn_implementation=sdpa,vision_attn_implementation=flash_attention_2,ptd_attn_implementation=flash_attention_2,efficiency_log=${OUTPUT_DIR}/ptd.jsonl" \
  --tasks ptd_vidstg \
  --batch_size 1 \
  --output_path "${OUTPUT_DIR}/ptd_lmms" \
  "${LIMIT_ARGS[@]}"

python "${PTD_REPO}/evaluation/report_efficiency.py" \
  --quantized "${OUTPUT_DIR}/quantized.jsonl" \
  --ptd "${OUTPUT_DIR}/ptd.jsonl"
