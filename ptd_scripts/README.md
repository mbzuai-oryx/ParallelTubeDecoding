# PTD training

Prepare the annotations with the scripts described in
[`data/README.md`](../data/README.md), then fill in the exported paths at the
top of each script. Run the commands below from the repository root.

First add and initialize the PTD time/location tokens in the released Qwen3-VL
checkpoint. Set `INITIALIZED_MODEL` in the script to a new output directory:

```bash
bash ptd_scripts/initialize_tokens.sh
```

Set `INITIALIZED_MODEL` in `train_sft.sh` to the checkpoint produced above, then run SFT:

```bash
bash ptd_scripts/train_sft.sh
```

Both released training launchers use the paper's video preprocessing:

```bash
--video_max_pixels $((360 * 420)) \
--fps 2 \
--max_frames 64 \
--temporal_patch_size 1
```

Both SFT and GRPO optimize LoRA adapters and only the new PTD embedding/output
rows. The embedding modules are stored with each PEFT adapter so disabling the
GRPO adapter restores the frozen SFT reference weights for KL computation.

Merge the resulting adapter by setting `BASE_MODEL` to the initialized model:

```bash
bash ptd_scripts/merge_lora.sh
```

Set `SFT_MODEL` in `train_grpo.sh` to that merged checkpoint and run GRPO:

```bash
bash ptd_scripts/train_grpo.sh
```

The GRPO launcher names the two released rewards explicitly:

```bash
--reward_names "temporal_iou_reward,spatial_reward"
```

`temporal_iou_reward` scores the predicted temporal interval. `spatial_reward`
scores the PTD boxes with the spatial GIoU/L1 objective. The training entry
point rejects other reward lists so the launcher and implementation cannot
silently diverge.

`train_grpo.sh` consumes an already prepared annotation JSON. The data and the
GRPO data-selection procedure are intentionally not part of this release.

The GRPO adapter can be merged with the same merge script by setting
`BASE_MODEL` to the merged SFT checkpoint.
