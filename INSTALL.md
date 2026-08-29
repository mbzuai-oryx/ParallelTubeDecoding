# Installation

## Environment

PTD uses Python 3.11. The provided requirements install the training stack used
by this repository, including PyTorch 2.8, Transformers 5.12.1, DeepSpeed, TRL,
and the video-processing dependencies.

```bash
conda create -n ptd python=3.11 -y
conda activate ptd

pip install --upgrade pip
pip install -r requirements.txt
```

The requirements include the CUDA 12.8 PyTorch packages. If your system uses a
different CUDA or ROCm version, install the matching PyTorch build for your
system and adjust the PyTorch package pins in `requirements.txt` before
installing the remaining dependencies.

The PTD TCL/BPS benchmark additionally requires a FlashAttention-2 build that
matches your CUDA or ROCm/PyTorch environment. Training and Quantized (NTP)
inference use SDPA and do not require that optional backend.

## Check the installation

Run this command from the repository root:

```bash
python -c "import torch, transformers, qwen_vl_utils; print(torch.__version__, transformers.__version__)"
```
