"""Reusable KV storage for cached Parallel Tube Decoding probes."""
from __future__ import annotations

from typing import Any, Optional

import torch


_REUSABLE_MARKER = "_ptd_reusable_cache"
_CAPACITY_ATTR = "_ptd_cache_capacity"


def ptd_cache_capacity(cache: Any) -> Optional[int]:
    if not bool(getattr(cache, _REUSABLE_MARKER, False)):
        return None
    capacity = getattr(cache, _CAPACITY_ATTR, None)
    return int(capacity) if capacity is not None else None


def _set_layer_length(layer: Any, length: int) -> None:
    cumulative_length = getattr(layer, "cumulative_length", None)
    if isinstance(cumulative_length, torch.Tensor):
        cumulative_length.fill_(int(length))
    elif cumulative_length is not None:
        layer.cumulative_length = int(length)
    else:
        raise TypeError(f"Static cache layer {type(layer)!r} has no cumulative_length.")


@torch.no_grad()
def make_reusable_ptd_cache(past_key_values: Any, *, config: Any, max_cache_len: int) -> Any:
    """Convert a completed dynamic prefill to bounded in-place KV storage."""
    if past_key_values is None or ptd_cache_capacity(past_key_values) is not None:
        return past_key_values
    layers = getattr(past_key_values, "layers", None)
    if config is None or not isinstance(layers, list) or not layers:
        return past_key_values
    if any(bool(getattr(layer, "is_sliding", False)) for layer in layers):
        return past_key_values

    source_length = int(past_key_values.get_seq_length())
    max_cache_len = int(max_cache_len)
    if max_cache_len < source_length:
        raise ValueError("PTD cache capacity is smaller than the multimodal prefill.")
    source_tensors = []
    for layer in layers:
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if (
            not bool(getattr(layer, "is_initialized", False))
            or not isinstance(keys, torch.Tensor)
            or not isinstance(values, torch.Tensor)
            or keys.ndim != 4
            or values.ndim != 4
            or keys.shape[-2] != source_length
            or values.shape[-2] != source_length
        ):
            return past_key_values
        source_tensors.append((keys, values))

    try:
        from transformers.cache_utils import StaticCache

        reusable = StaticCache(config=config, max_cache_len=max_cache_len)
        target_layers = reusable.layers
        if len(target_layers) != len(layers):
            return past_key_values
        for target, (keys, values) in zip(target_layers, source_tensors):
            target.lazy_initialization(keys, values)
            if target.keys.shape[-2] < max_cache_len:
                return past_key_values
            target.keys[..., :source_length, :].copy_(keys)
            target.values[..., :source_length, :].copy_(values)
            _set_layer_length(target, source_length)
    except torch.cuda.OutOfMemoryError:
        raise
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return past_key_values

    setattr(reusable, _REUSABLE_MARKER, True)
    setattr(reusable, _CAPACITY_ATTR, max_cache_len)
    return reusable


@torch.no_grad()
def rewind_reusable_ptd_cache(cache: Any, length: int) -> bool:
    capacity = ptd_cache_capacity(cache)
    if capacity is None:
        return False
    length = int(length)
    if not 0 <= length <= capacity:
        raise ValueError(f"PTD cache length must lie in [0, {capacity}], got {length}.")
    for layer in cache.layers:
        if bool(getattr(layer, "is_initialized", False)):
            _set_layer_length(layer, length)
    return True


def pad_mask_to_ptd_cache_capacity(attention_mask: torch.Tensor, cache: Any) -> torch.Tensor:
    """Mask unused StaticCache slots when a probe falls back to SDPA."""
    capacity = ptd_cache_capacity(cache)
    if capacity is None or attention_mask.shape[-1] == capacity:
        return attention_mask
    if attention_mask.shape[-1] > capacity:
        raise ValueError("The PTD attention mask exceeds the reusable cache capacity.")
    padded = attention_mask.new_full(
        (*attention_mask.shape[:-1], capacity), torch.finfo(attention_mask.dtype).min
    )
    padded[..., : attention_mask.shape[-1]] = attention_mask
    return padded
