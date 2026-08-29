"""FlashAttention-2 backend for cached Parallel Tube Decoding.

PTD probes share a committed prefix but must not attend across spatial blocks.
This backend evaluates the shared-prefix and block-local partitions separately
and combines them with their FlashAttention log-sum-exp normalizers.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch


PTD_FLASH_ATTENTION_BACKEND = "ptd_flash_attention_2"

_FLASH_ATTN_FUNCS: Optional[tuple[Any, Any]] = None
_FLASH_ATTN_ERROR: Optional[BaseException] = None
_CU_SEQLENS_CACHE: dict[tuple[str, Optional[int], int, int], torch.Tensor] = {}


class PTDFlashAttentionPlanError(ValueError):
    """Raised when a PTD mask cannot use the shared-prefix Flash path."""


@dataclass(frozen=True)
class PTDFlashAttentionPlan:
    past_len: int
    catchup_len: int
    num_blocks: int
    block_size: int
    context_limit: int
    cu_seqlens: torch.Tensor

    @property
    def committed_len(self) -> int:
        return self.past_len + self.catchup_len

    @property
    def probe_len(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def query_len(self) -> int:
        return self.catchup_len + self.probe_len

    @property
    def key_len(self) -> int:
        return self.past_len + self.query_len


def _load_flash_attention_functions() -> tuple[Any, Any]:
    global _FLASH_ATTN_FUNCS, _FLASH_ATTN_ERROR
    if _FLASH_ATTN_FUNCS is not None:
        return _FLASH_ATTN_FUNCS
    if _FLASH_ATTN_ERROR is not None:
        raise _FLASH_ATTN_ERROR
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        _FLASH_ATTN_FUNCS = (flash_attn_func, flash_attn_varlen_func)
        return _FLASH_ATTN_FUNCS
    except BaseException as exc:
        _FLASH_ATTN_ERROR = exc
        raise


def resolve_ptd_attn_implementation(requested: str) -> str:
    requested = str(requested).strip().lower()
    valid = ("auto", "flash_attention_2", "sdpa")
    if requested not in valid:
        raise ValueError(
            f"ptd_attn_implementation must be one of {valid}, got {requested!r}"
        )
    if requested == "sdpa":
        return requested

    available = torch.cuda.is_available()
    if available:
        try:
            _load_flash_attention_functions()
        except Exception:
            available = False
    if requested == "flash_attention_2" and not available:
        message = (
            "ptd_attn_implementation=flash_attention_2 requires the "
            "FlashAttention package and a CUDA device"
        )
        if _FLASH_ATTN_ERROR is not None:
            raise RuntimeError(message) from _FLASH_ATTN_ERROR
        raise RuntimeError(message)
    return "flash_attention_2" if available else "sdpa"


def build_ptd_flash_attention_plan(
    *,
    past_len: int,
    catchup_len: int,
    num_blocks: int,
    context_limits: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> PTDFlashAttentionPlan:
    if past_len < 0 or catchup_len < 0:
        raise PTDFlashAttentionPlanError("PTD cache lengths must be non-negative.")
    if num_blocks <= 0 or block_size <= 0:
        raise PTDFlashAttentionPlanError("PTD block counts and sizes must be positive.")

    limits = torch.as_tensor(context_limits, dtype=torch.long, device=device).flatten()
    if limits.numel() != num_blocks:
        raise PTDFlashAttentionPlanError("PTD requires one context limit per block.")
    context_limit = int(limits[0].item())
    if not bool(limits.eq(context_limit).all().item()):
        raise PTDFlashAttentionPlanError(
            "The PTD Flash path requires uniform context limits."
        )
    if not 0 <= context_limit <= past_len + catchup_len:
        raise PTDFlashAttentionPlanError(
            "PTD context limits must lie inside the committed prefix."
        )

    cache_key = (device.type, device.index, int(num_blocks), int(block_size))
    cu_seqlens = _CU_SEQLENS_CACHE.get(cache_key)
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0,
            num_blocks * block_size + 1,
            block_size,
            dtype=torch.int32,
            device=device,
        )
        _CU_SEQLENS_CACHE[cache_key] = cu_seqlens
    return PTDFlashAttentionPlan(
        past_len=int(past_len),
        catchup_len=int(catchup_len),
        num_blocks=int(num_blocks),
        block_size=int(block_size),
        context_limit=context_limit,
        cu_seqlens=cu_seqlens,
    )


def _unpack_flash_result(result, total_queries: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("FlashAttention did not return log-sum-exp values.")
    output, lse = result[0], result[1]
    if lse.ndim == 3 and lse.shape[0] == 1:
        lse = lse[0].transpose(0, 1)
    elif lse.ndim == 2 and lse.shape[1] == total_queries:
        lse = lse.transpose(0, 1)
    elif lse.ndim != 2 or lse.shape[0] != total_queries:
        raise RuntimeError(
            f"Unexpected FlashAttention LSE shape {tuple(lse.shape)}."
        )
    return output, lse.float()


def ptd_flash_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    ptd_attention_plan: Optional[PTDFlashAttentionPlan] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Run batch-one cached PTD attention without materializing a dense mask."""
    del kwargs
    if ptd_attention_plan is None:
        raise ValueError("PTD FlashAttention requires ptd_attention_plan.")
    if attention_mask is not None:
        raise ValueError("The PTD FlashAttention mask is supplied by its plan.")
    if float(dropout) != 0.0:
        raise ValueError("PTD FlashAttention is inference-only and requires dropout=0.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("PTD FlashAttention expects rank-four Q/K/V tensors.")
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("PTD FlashAttention supports batch size one.")
    if not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("PTD FlashAttention requires CUDA FP16/BF16 tensors.")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("PTD FlashAttention requires matching Q/K/V dtypes.")

    plan = ptd_attention_plan
    if query.shape[2] != plan.query_len:
        raise ValueError("PTD query length does not match its attention plan.")
    if key.shape[2] < plan.key_len or value.shape[2] < plan.key_len:
        raise ValueError("The PTD KV cache is shorter than its attention plan.")

    flash_attn_func, flash_attn_varlen_func = _load_flash_attention_functions()
    softmax_scale = float(scaling if scaling is not None else module.scaling)
    outputs = []

    if plan.catchup_len:
        catchup_q = query[:, :, : plan.catchup_len, :].transpose(1, 2).contiguous()
        catchup_k = key[:, :, : plan.committed_len, :].transpose(1, 2).contiguous()
        catchup_v = value[:, :, : plan.committed_len, :].transpose(1, 2).contiguous()
        catchup_output = flash_attn_func(
            catchup_q,
            catchup_k,
            catchup_v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
        )
        outputs.append(catchup_output[0] if isinstance(catchup_output, tuple) else catchup_output)

    probe_q_batched = query[:, :, plan.catchup_len : plan.query_len, :].transpose(1, 2).contiguous()
    probe_q = probe_q_batched[0]
    local_k = key[0, :, plan.committed_len : plan.key_len, :].transpose(0, 1).contiguous()
    local_v = value[0, :, plan.committed_len : plan.key_len, :].transpose(0, 1).contiguous()
    local_output, local_lse = _unpack_flash_result(
        flash_attn_varlen_func(
            probe_q,
            local_k,
            local_v,
            plan.cu_seqlens,
            plan.cu_seqlens,
            plan.block_size,
            plan.block_size,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            return_attn_probs=True,
        ),
        plan.probe_len,
    )

    if plan.context_limit:
        prefix_k = key[:, :, : plan.context_limit, :].transpose(1, 2).contiguous()
        prefix_v = value[:, :, : plan.context_limit, :].transpose(1, 2).contiguous()
        prefix_output, prefix_lse = _unpack_flash_result(
            flash_attn_func(
                probe_q_batched,
                prefix_k,
                prefix_v,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
                return_attn_probs=True,
            ),
            plan.probe_len,
        )
        prefix_output = prefix_output[0]
        normalizer = torch.logaddexp(prefix_lse, local_lse)
        probe_output = (
            prefix_output.float() * torch.exp(prefix_lse - normalizer).unsqueeze(-1)
            + local_output.float() * torch.exp(local_lse - normalizer).unsqueeze(-1)
        ).to(dtype=query.dtype)
    else:
        probe_output = local_output

    outputs.append(probe_output.unsqueeze(0))
    return torch.cat(outputs, dim=1), None


@contextmanager
def use_ptd_flash_attention(language_model) -> Iterator[None]:
    """Select the PTD backend for one language-model probe."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    registered = ALL_ATTENTION_FUNCTIONS.get(PTD_FLASH_ATTENTION_BACKEND)
    if registered is None:
        ALL_ATTENTION_FUNCTIONS.register(
            PTD_FLASH_ATTENTION_BACKEND, ptd_flash_attention_forward
        )
    elif registered is not ptd_flash_attention_forward:
        raise RuntimeError(
            f"Attention backend {PTD_FLASH_ATTENTION_BACKEND!r} is already registered."
        )

    config = language_model.config
    previous = getattr(config, "_attn_implementation_internal", None)
    config._attn_implementation_internal = PTD_FLASH_ATTENTION_BACKEND
    try:
        yield
    finally:
        config._attn_implementation_internal = previous
