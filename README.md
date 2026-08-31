<div align="center">

# Locate Anything in Videos: Rethinking Efficient Generative Spatio-Temporal Video Grounding

<p align="center">
  <img src="assets/branding/colored_line.png" width="100%" alt="">
</p>

[Hanoona Rasheed](https://github.com/hanoonaR)<sup>1</sup> · [Haania Siddiqui](https://pk.linkedin.com/in/haaniasiddiqui)<sup>1</sup> · [Ming-Hsuan Yang](https://scholar.google.com.pk/citations?user=p9-ohHsAAAAJ&hl=en)<sup>2</sup> · [Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home)<sup>1,3</sup> · [Salman Khan](https://salman-h-khan.github.io/)<sup>1,3</sup>

<sup>1</sup> Mohamed bin Zayed University of Artificial Intelligence · <sup>2</sup> University of California, Merced · <sup>3</sup> Apertix

[![Paper](https://img.shields.io/badge/📄_Paper-PDF-blue)](https://arxiv.org/pdf/2608.28192)
[![Project Page](https://img.shields.io/badge/🌐_Project-Page-blue)](https://mbzuai-oryx.github.io/ParallelTubeDecoding/)
[![Model](https://img.shields.io/badge/🤗_Model-Hugging_Face-yellow)](https://huggingface.co/MBZUAI/ParallelTubeDecoding-Qwen3-VL-4B)

</div>

## 📝 Abstract

Spatio-temporal video grounding requires identifying when a queried event occurs and localizing the referred entity throughout that interval. We introduce **Parallel Tube Decoding (PTD)**, which predicts the temporal interval first and then decodes all time-conditioned spatial blocks in parallel. PTD reduces tube generation to two decoding rounds, independent of tube length, while improving both localization accuracy and inference efficiency.

<p align="center">
  <img src="assets/figures/intro_strategies.png" width="100%" alt="Comparison of autoregressive localization strategies with Parallel Tube Decoding">
</p>

**Autoregressive localization vs. Parallel Tube Decoding.** Given a video and a referring expression, STVG predicts when the event occurs and the bounding box of the referred entity throughout that interval. PTD generates all time-conditioned spatial blocks in parallel after temporal localization, reducing the sequential decoding depth to $1 + 1$. Compared with standard Unquantized Token Decoding, PTD achieves $79\times$ lower Tube Completion Latency and $92\times$ higher spatial decoding throughput while improving spatio-temporal grounding accuracy.

## 🔥 Contributions

- **Parallel tube generation.** PTD removes token-level and trajectory-level dependencies by generating the temporal block in one round and all spatial blocks in a second round.
- **Decoupled Block Attention.** Each spatial block uses the shared video-query context and predicted temporal interval without depending on other generated boxes.
- **Localization-aware optimization.** Complementary temporal and spatial rewards improve event boundaries, bounding-box geometry, and target consistency.
- **Efficiency and accuracy.** PTD achieves $79\times$ lower Tube Completion Latency and $92\times$ higher spatial decoding throughput than standard autoregressive decoding while improving grounding performance.

## 🛠️ Using the Code

| Component | Location |
| --- | --- |
| Load the released PTD checkpoint | `MBZUAI/ParallelTubeDecoding-Qwen3-VL-4B` on [Hugging Face](https://huggingface.co/MBZUAI/ParallelTubeDecoding-Qwen3-VL-4B) |
| Install the PTD environment | [Installation guide](INSTALL.md) |
| Prepare training and evaluation annotations | [Data preparation guide](data/README.md), including SFT preparation for VidSTG/HC-STVG and evaluation preparation for all four released benchmarks |
| Run SFT, merge the adapter, and run GRPO | [Training guide](ptd_scripts/README.md), [SFT launcher](ptd_scripts/train_sft.sh), [merge script](ptd_scripts/merge_lora.sh), and [GRPO launcher](ptd_scripts/train_grpo.sh) |
| Evaluate VidSTG, HC-STVG, Charades-STA, and ActivityNet | [Evaluation guide](evaluation/README.md) and [lmms-eval task definitions](evaluation/lmms_eval/lmms_eval/tasks) |
| Measure Quantized/PTD TCL and BPS | [Efficiency runner](evaluation/run_efficiency.sh), [reporter](evaluation/report_efficiency.py), and the [evaluation guide](evaluation/README.md#tcl-and-bps) |

### 🚀 Quick Start: Local Inference

After [installing the environment](INSTALL.md), launch the Gradio demo from the
repository root. It loads the released merged PTD checkpoint from Hugging Face
by default; `--disable_flash_attention` uses the broadly supported SDPA backend.

```bash
PYTHONPATH=src python src/serve/app.py --disable_flash_attention
```

Upload a video and use the grounding prompt below, replacing the text in angle
brackets with the target event:

```text
Given the query: '<description of the target event>' Localize the described object throughout the video. Use object reference tokens, time tokens, and box tokens. Return the object reference, event time segment, and per-time bbox coordinates.
```

Pass `--model-path /path/to/merged-ptd-checkpoint` to use a local merged
checkpoint. For benchmark inference with the paper's preprocessing and
evaluation settings, follow the [evaluation guide](evaluation/README.md).

## 💡 Parallel Tube Decoding

<p align="center">
  <img src="assets/figures/sbd_vs_ptd_attention.png" width="100%" alt="Attention masks for Sequential Block Decoding and Parallel Tube Decoding">
</p>

**Attention masks for Sequential Block Decoding and PTD.** Sequential Block Decoding retains causal attention across spatial blocks. PTD replaces this cross-box dependency with Decoupled Block Attention: every spatial block accesses the shared multimodal prefix and temporal block while all other spatial blocks remain masked.

## 📊 Main Results

### ⚡ Decoding strategies

<p align="center">
  <img src="assets/tables/table1.png" width="100%" alt="Comparison of decoding strategies on VidSTG">
</p>

**Table 1: Comparison of decoding strategies on VidSTG.** We report temporal and video IoU for declarative and interrogative queries, together with Tube Completion Latency (TCL) and Boxes Per Second (BPS). Parallel Tube Decoding achieves the strongest grounding performance, lowest latency, and highest throughput.

<p align="center">
  <img src="assets/figures/ptd_plots.png" width="100%" alt="Analysis of decoding efficiency and trajectory-level dependency">
</p>

**Analysis of decoding efficiency and trajectory-level dependency.** (a) Tube completion latency as the number of grounded frames increases. PTD maintains nearly constant latency, while token-based and block decoding scale with tube length. (b) Attention distribution across tube decoding for Sequential Block Decoding (dotted) and PTD (solid). Sequential decoding progressively shifts attention from the video toward prior text. (c) History-correction analysis for Sequential Block Decoding. Replacing an erroneous box $B_i$ with its ground-truth box improves subsequent predictions, with the effect gradually decreasing as the decoding distance $j-i$ increases.

### 🎯 Spatio-temporal video grounding

<p align="center">
  <img src="assets/tables/table2.png" width="100%" alt="Results on VidSTG">
</p>

**Table 2: Results on VidSTG.** Comparison with backbone baselines and prior multimodal large language models for declarative and interrogative spatio-temporal grounding. Our compact Qwen3-VL-4B model with GRPO and PTD delivers the strongest overall results.

<p align="center">
  <img src="assets/tables/table3.png" width="100%" alt="Results on HC-STVG version 1 and version 2">
</p>

**Table 3: Results on HC-STVG.** Comparison with backbone baselines and prior methods on HC-STVG v1 and v2. PTD achieves strong temporal localization and the best video IoU across both benchmark versions.

### 🌐 Generalization beyond STVG

<p align="center">
  <img src="assets/tables/table4.png" width="100%" alt="Zero-shot temporal grounding results on Charades-STA and ActivityNet Captions">
</p>

**Table 4: Zero-shot temporal grounding.** Results on Charades-STA and ActivityNet Captions without task-specific training. Our model improves over the strongest prior zero-shot methods, including under stricter temporal-overlap thresholds.

### 🔬 Ablation study

<p align="center">
  <img src="assets/tables/table6.png" width="100%" alt="Ablation of temporal and spatial localization rewards">
</p>

**Table 6: Ablation of localization-aware rewards.** Temporal and spatial rewards provide complementary improvements for declarative and interrogative grounding. Combining both produces the strongest overall spatio-temporal localization performance.

## 👀 Qualitative Results

<p align="center">
  <img src="assets/figures/qualitative_mainfig.png" width="100%" alt="Qualitative comparison with prior spatio-temporal video grounding methods">
</p>

**Comparison with prior STVG methods.** PTD directly generates both the temporal interval and complete spatial tube. The examples highlight fine-grained target identification among visually similar distractors, large changes in object scale, and event-specific temporal localization. Bounding boxes show spatial predictions, while the horizontal lines indicate the predicted temporal intervals.

<p align="center">
  <img src="assets/figures/sft_vs_temporal_reward.png" width="100%" alt="Effect of the temporal reward">
</p>

**Effect of the temporal reward.** The first example shows how the reward recovers the complete interaction when SFT localizes only its most salient portion. The second shows how it prevents the prediction from extending beyond the queried event while the target remains visible.

<p align="center">
  <img src="assets/figures/sft_vs_spatial_reward.png" width="100%" alt="Effect of the spatial reward">
</p>

**Effect of the spatial reward.** The examples highlight tighter localization under rapid motion and more consistent target identity despite substantial scale changes and interference from nearby entities.

<p align="center">
  <img src="assets/figures/sbd_vs_ptd.png" width="100%" alt="Qualitative comparison of Sequential Block Decoding and Parallel Tube Decoding">
</p>

**Sequential Block Decoding vs. PTD.** Abrupt scale changes and partial occlusion expose how Sequential Block Decoding propagates localization errors across subsequent boxes. PTD maintains tighter target localization by removing cross-box dependencies.

<p align="center">
  <img src="assets/figures/qualitative_suppfig.png" width="100%" alt="Additional qualitative results of Parallel Tube Decoding">
</p>

**Additional qualitative results.** PTD accurately localizes the queried entity under visually similar distractors, target motion and scale changes, partial occlusion, progressive object reveal, and multi-instance interactions.

<p align="center">
  <img src="assets/figures/failure_cases.png" width="100%" alt="Representative failure cases">
</p>

**Representative failure cases.** Temporally subtle state changes can produce ambiguous event boundaries, while spatial localization becomes difficult for small, rapidly moving, or occluded targets.

## 🙏 Acknowledgements

This codebase is built on [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune). We thank its authors for releasing their fine-tuning framework. We also thank the [Qwen team](https://github.com/QwenLM/Qwen3-VL) for releasing Qwen3-VL and acknowledge [Locate Anything](https://github.com/NVlabs/Eagle/tree/main/Embodied) for making its work and implementation publicly available.

## 📜 Citation

```bibtex
@misc{rasheed2026locatevideosrethinkingefficient,
  title         = {Locate Anything in Videos: Rethinking Efficient Generative Spatio-Temporal Video Grounding},
  author        = {Hanoona Rasheed and Haania Siddiqui and Ming-Hsuan Yang and Fahad Shahbaz Khan and Salman Khan},
  year          = {2026},
  eprint        = {2608.28192},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.28192}
}
```

---

<p align="center">
  <a href="https://www.ival-mbzuai.com"><img src="assets/branding/IVAL_logo.png" width="200" alt="Intelligent Visual Analytics Lab"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/mbzuai-oryx"><img src="assets/branding/Oryx_logo.png" width="100" alt="Oryx"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://mbzuai.ac.ae"><img src="assets/branding/MBZUAI_logo.png" width="360" alt="Mohamed bin Zayed University of Artificial Intelligence"></a>
</p>
