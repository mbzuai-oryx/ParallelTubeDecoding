#!/bin/bash

export INITIALIZED_MODEL=""
export ANNOTATIONS=""
export VIDEO_ROOT=""
export OUTPUT_DIR=""
export GPUS=8
export GRADIENT_ACCUMULATION_STEPS=8

export PYTHONPATH="$PWD/src:$PWD"

torchrun --standalone --nproc-per-node="$GPUS" src/train/train_sft.py \
  --deepspeed ptd_scripts/zero3.json \
  --model_id "$INITIALIZED_MODEL" \
  --data_path "$ANNOTATIONS" \
  --image_folder "$VIDEO_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --remove_unused_columns false \
  --bf16 true \
  --lora_enable true \
  --freeze_llm true \
  --freeze_vision_tower true \
  --freeze_merger true \
  --disable_flash_attn2 true \
  --use_liger_kernel false \
  --bits 16 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate 2e-5 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --max_grad_norm 0.3 \
  --gradient_checkpointing true \
  --fps 2 \
  --max_frames 64 \
  --temporal_patch_size 1 \
  --video_max_pixels $((360 * 420)) \
  --max_seq_length 32768 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 3 \
  --dataloader_num_workers 4 \
  --report_to none
