# PTD tasks for lmms-eval

We used lmms-eval v0.7.1 at upstream commit
`88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`. Check out that exact revision
before applying the PTD files below:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
git checkout 88b23e2bfa16a1edbc16e9e238ed82130b3a4f56
pip install -e .
cd ..
```

Use the PTD environment from `INSTALL.md`; in particular, evaluation uses the
pinned `qwen-vl-utils==0.0.14` video sampler.

The exact evaluation preprocessing used for the paper is:

```text
fps=2,max_num_frames=64,min_pixels=131072,max_pixels=786432,temporal_patch_size=1
```

The adapter validates these values instead of silently accepting a different
evaluation setup.

The files under `lmms_eval/` mirror their destination in an lmms-eval
checkout. Copy the four task folders and model adapter:

```bash
export PTD_REPO="/path/to/ParallelTubeDecoding"
export LMMS_EVAL_REPO="/path/to/lmms-eval"

cp -r \
  "$PTD_REPO/evaluation/lmms_eval/lmms_eval/tasks/vidstg" \
  "$PTD_REPO/evaluation/lmms_eval/lmms_eval/tasks/hcstvg" \
  "$PTD_REPO/evaluation/lmms_eval/lmms_eval/tasks/charades_temp_loc" \
  "$PTD_REPO/evaluation/lmms_eval/lmms_eval/tasks/activitynet_temp_loc" \
  "$LMMS_EVAL_REPO/lmms_eval/tasks/"
cp "$PTD_REPO/evaluation/lmms_eval/lmms_eval/models/simple/ptd_qwen3_vl.py" \
  "$LMMS_EVAL_REPO/lmms_eval/models/simple/"
```

Add the following entry to `AVAILABLE_SIMPLE_MODELS` in
`lmms_eval/models/__init__.py`:

```python
"ptd_qwen3_vl": "PTDQwen3VL",
```

Edit the `data_files.test` path in each PTD task YAML. Relative video paths are
resolved with these variables:

```bash
export VIDSTG_VIDEO_ROOT="/path/to/VidSTG"
export HCSTVG_VIDEO_ROOT="/path/to/HC-STVG"
export CHARADES_STA_VIDEO_ROOT="/path/to/Charades_v1_480"
export ACTIVITYNET_VIDEO_ROOT="/path/to/ActivityNet/videos"
```

VidSTG and HC-STVG annotations are JSONL files with these fields:

- `video_path`, `caption`, and `gt_sampled_frame_boxes`
- `qtype` (`declarative` or `interrogative`) for VidSTG
- `version` (`v1` or `v2`) for HC-STVG
- optional `video_start_sec` and `video_end_sec`

Each item in `gt_sampled_frame_boxes` contains a one-based `time_index` and a
0-1000 `bbox`. Charades-STA and ActivityNet JSONL annotations contain
`video_path`, `caption`, and `timestamp`. They can also include `frame_count`
and `fps`; otherwise the task reads this metadata from the video.

Run the spatio-temporal tasks with the PTD tube format:

```bash
export MODEL="/path/to/model"

cd "$LMMS_EVAL_REPO"
python -m accelerate.commands.launch -m lmms_eval \
  --model ptd_qwen3_vl \
  --model_args pretrained="$MODEL",ptd_root="$PTD_REPO",decoding=ptd,generation_format=spatio_temporal_grounding,fps=2,max_num_frames=64,min_pixels=131072,max_pixels=786432,temporal_patch_size=1,system_prompt=none,attn_implementation=sdpa,vision_attn_implementation=flash_attention_2,ptd_attn_implementation=flash_attention_2 \
  --tasks ptd_vidstg,ptd_hcstvg \
  --batch_size 1 \
  --log_samples \
  --output_path ./ptd_stvg_results
```

Run the temporal-localization tasks with the temporal segment format:

```bash
python -m accelerate.commands.launch -m lmms_eval \
  --model ptd_qwen3_vl \
  --model_args pretrained="$MODEL",ptd_root="$PTD_REPO",decoding=ptd,generation_format=temporal_localization,fps=2,max_num_frames=64,min_pixels=131072,max_pixels=786432,temporal_patch_size=1,system_prompt=none,attn_implementation=sdpa,vision_attn_implementation=flash_attention_2,ptd_attn_implementation=flash_attention_2 \
  --tasks ptd_charades_sta,ptd_activitynet \
  --batch_size 1 \
  --log_samples \
  --output_path ./ptd_temporal_results
```

lmms-eval computes and aggregates the task metrics during these runs. VidSTG
and HC-STVG report m_tIoU, m_vIoU, vIoU@0.3, and vIoU@0.5. Charades-STA and
ActivityNet report R@0.3, R@0.5, R@0.7, and mIoU.

## TCL and BPS

The efficiency code intentionally exposes only the two inference modes of the
released PTD checkpoint:

- **Quantized (NTP):** `attn_implementation=sdpa`
- **PTD:** `attn_implementation=sdpa`,
  `vision_attn_implementation=flash_attention_2`, and
  `ptd_attn_implementation=flash_attention_2`

TCL follows the paper definition `TCL = T_full - T_TTFT`: timing begins after
multimodal prefill produces the first output logits and ends after the final
GPU decode operation. CPU decoding, parsing, metrics, and file I/O are
excluded. BPS is `num_predicted_boxes / TCL`. The paper protocol uses BF16,
batch size 1, and one GPU. Quantized and PTD must use the same merged PTD
checkpoint.

After installing the task/model files above, run:

```bash
export MODEL="/path/to/merged-ptd-checkpoint"
export PTD_REPO="/path/to/ParallelTubeDecoding"
export LMMS_EVAL_REPO="/path/to/lmms-eval"

bash "$PTD_REPO/evaluation/run_efficiency.sh"
```

Set `LIMIT=10` for a smoke test. The runner writes one JSONL file per decoding
mode and prints mean/median TCL and BPS with
`evaluation/report_efficiency.py`. FlashAttention-2 is required for the PTD
run; an unavailable requested backend raises an error rather than changing the
benchmark configuration.
