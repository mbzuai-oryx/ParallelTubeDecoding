#!/bin/bash

export SFT_MODEL=""
export ANNOTATIONS=""
export VIDEO_ROOT=""
export OUTPUT_DIR=""
export GPUS=8

export PYTHONPATH="$PWD/src:$PWD"

torchrun --standalone --nproc-per-node="$GPUS" src/train/train_grpo.py \
  --deepspeed ptd_scripts/zero3.json \
  --model_id "$SFT_MODEL" \
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
  --num_train_epochs 1 \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --steps_per_generation 1 \
  --num_iterations 1 \
  --max_completion_length 1024 \
  --max_prompt_length 32768 \
  --temperature 0.9 \
  --top_p 1.0 \
  --top_k 50 \
  --beta 0.04 \
  --reward_names "temporal_iou_reward,spatial_reward" \
  --loss_type grpo \
  --use_vllm false \
  --learning_rate 5e-6 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --max_grad_norm 0.3 \
  --gradient_checkpointing true \
  --fps 2 \
  --max_frames 64 \
  --temporal_patch_size 1 \
  --video_max_pixels $((360 * 420)) \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 3 \
  --dataloader_num_workers 4 \
  --report_to none
