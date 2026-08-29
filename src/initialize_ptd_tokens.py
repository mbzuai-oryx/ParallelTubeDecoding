#!/usr/bin/env python3
"""Add and initialize PTD tokens, then save a training-ready checkpoint."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from model.ptd_tokens import NUM_TIME_TOKENS, add_and_initialize_ptd_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if model.config.model_type != "qwen3_vl":
        raise ValueError("PTD token initialization requires Qwen3-VL.")
    token_ids = add_and_initialize_ptd_tokens(
        model, processor, num_time_tokens=NUM_TIME_TOKENS
    )
    model.config.ptd_num_time_tokens = NUM_TIME_TOKENS
    model.config.ptd_block_size = 6
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    print(
        f"Saved {len(token_ids)} PTD token rows with the initialized checkpoint "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
