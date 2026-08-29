from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F

from model.ptd_mask_utils import build_cached_ptd_attention_mask_4d
from model.ptd_flash_attention import (
    build_ptd_flash_attention_plan,
    resolve_ptd_attn_implementation,
    use_ptd_flash_attention,
)
from model.ptd_reusable_cache import (
    make_reusable_ptd_cache,
    pad_mask_to_ptd_cache_capacity,
    ptd_cache_capacity,
    rewind_reusable_ptd_cache,
)


@dataclass
class PTDGenerationResult:
    steps: int
    generated_tokens: int
    stopped: bool
    decode_latency_s: Optional[float] = None
    trace: Optional["PTDRolloutTrace"] = None


@dataclass
class PTDRolloutTrace:
    """Compact record of the probe actions sampled during one PTD rollout."""

    prefix_input_ids: torch.Tensor
    prefix_attention_mask: torch.Tensor
    query_token_ids: torch.Tensor
    target_token_ids: torch.Tensor
    target_mask: torch.Tensor
    probe_position_starts: torch.Tensor
    context_limits: torch.Tensor


def get_token_id(tokenizer, token: str, *, required: bool = True) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_id:
        if required:
            raise ValueError(f"Token {token!r} is missing from tokenizer.")
        return None
    return int(token_id)


def build_ptd_token_ids(
    tokenizer,
    *,
    max_time_tokens: Optional[int] = None,
) -> dict[str, Any]:
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise ValueError(f"Expected newline to encode as one token, got {newline_ids}.")

    if max_time_tokens is not None and max_time_tokens <= 0:
        raise ValueError(
            f"max_time_tokens must be positive when provided, got {max_time_tokens}."
        )

    ordered_time_token_ids = []
    scan_limit = max_time_tokens if max_time_tokens is not None else 1000
    for idx in range(1, scan_limit + 1):
        token_id = get_token_id(
            tokenizer,
            f"<t{idx}>",
            required=max_time_tokens is not None,
        )
        if token_id is None:
            break
        ordered_time_token_ids.append(token_id)
    if not ordered_time_token_ids:
        raise ValueError("No <tN> tokens were found in the tokenizer.")

    time_token_ids = tuple(ordered_time_token_ids)
    coord_id_to_value: dict[int, int] = {}
    for value in range(1001):
        token_id = get_token_id(tokenizer, f"<{value}>")
        if token_id in coord_id_to_value:
            raise ValueError(
                f"Coordinate tokens <{coord_id_to_value[token_id]}> and <{value}> "
                f"share tokenizer id {token_id}."
            )
        coord_id_to_value[token_id] = value

    return {
        "eos": get_token_id(tokenizer, "<|im_end|>"),
        "text_mask": get_token_id(tokenizer, "<text_mask>"),
        "null": get_token_id(tokenizer, "<null>"),
        "box_start": get_token_id(tokenizer, "<|box_start|>"),
        "box_end": get_token_id(tokenizer, "<|box_end|>"),
        "coord_start": get_token_id(tokenizer, "<0>"),
        "coord_end": get_token_id(tokenizer, "<1000>"),
        "ref_start": get_token_id(tokenizer, "<|object_ref_start|>"),
        "coord_id_to_value": coord_id_to_value,
        "ref_end": get_token_id(tokenizer, "<|object_ref_end|>"),
        "time_start": get_token_id(tokenizer, "<|time_start|>"),
        "time_end": get_token_id(tokenizer, "<|time_end|>"),
        "newline": int(newline_ids[0]),
        "time_tokens": set(time_token_ids),
        "ordered_time_tokens": time_token_ids,
        "time_token_indices": {
            token_id: time_idx for time_idx, token_id in enumerate(time_token_ids)
        },
    }


def configure_ptd_model(model, block_size: int = 6) -> None:
    if block_size != 6:
        raise ValueError(f"PTD uses six-token blocks, got {block_size}.")
    configured = set()
    for candidate in (
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    ):
        config = getattr(candidate, "config", None)
        if config is None or id(config) in configured:
            continue
        configured.add(id(config))

        if isinstance(config, MutableMapping):
            saved_block_size = config.get("ptd_block_size")
        else:
            saved_block_size = getattr(config, "ptd_block_size", None)
        if saved_block_size is not None and int(saved_block_size) != block_size:
            raise ValueError(
                "Checkpoint PTD block size is incompatible with time-anchored "
                f"parallel PTD: saved {saved_block_size}, requested {block_size}."
            )
        if isinstance(config, MutableMapping):
            config["ptd_block_size"] = block_size
        else:
            config.ptd_block_size = block_size


def make_ptd_probe_inputs(
    generated_ids: torch.Tensor,
    generated_attention_mask: torch.Tensor,
    query_token_ids: torch.Tensor,
    token_ids: dict[str, Any],
    *,
    probe_position_starts: torch.Tensor,
    context_limits: torch.Tensor,
    block_size: int = 6,
) -> dict[str, torch.Tensor]:
    """Append independent query-plus-mask blocks to one generated prefix.

    A physical block contains one known query token followed by five mask
    tokens. Its six logits predict a six-token target. The explicit metadata
    keeps physical suffix placement, allowed prefix context, and RoPE positions
    independent.

    Batch size one is intentional: generated temporal spans can contain a
    different number of box probes for each sample. Batched callers should run
    generation once per sample.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if block_size != 6:
        raise ValueError(f"Time-anchored PTD requires block_size=6, got {block_size}.")
    if generated_ids.ndim != 2 or generated_ids.shape[0] != 1:
        raise ValueError(
            "Time-anchored PTD generation requires input_ids with batch size 1, "
            f"got {tuple(generated_ids.shape)}."
        )
    if generated_attention_mask.shape != generated_ids.shape:
        raise ValueError(
            "generated_attention_mask must match generated_ids, got "
            f"{tuple(generated_attention_mask.shape)} and {tuple(generated_ids.shape)}."
        )

    query_token_ids = torch.as_tensor(
        query_token_ids,
        dtype=generated_ids.dtype,
        device=generated_ids.device,
    ).flatten()
    probe_position_starts = torch.as_tensor(
        probe_position_starts,
        dtype=torch.long,
        device=generated_ids.device,
    ).flatten()
    context_limits = torch.as_tensor(
        context_limits,
        dtype=torch.long,
        device=generated_ids.device,
    ).flatten()
    num_blocks = query_token_ids.numel()
    if num_blocks == 0:
        raise ValueError("At least one PTD query token is required.")
    if probe_position_starts.numel() != num_blocks or context_limits.numel() != num_blocks:
        raise ValueError(
            "query_token_ids, probe_position_starts, and context_limits must have the "
            f"same length, got {num_blocks}, {probe_position_starts.numel()}, and "
            f"{context_limits.numel()}."
        )

    masks = torch.full(
        (num_blocks, block_size - 1),
        int(token_ids["text_mask"]),
        dtype=generated_ids.dtype,
        device=generated_ids.device,
    )
    suffix_ids = torch.cat([query_token_ids[:, None], masks], dim=1).reshape(1, -1)
    probe_ids = torch.cat([generated_ids, suffix_ids], dim=1)

    suffix_attention = torch.ones(
        (1, suffix_ids.shape[1]),
        dtype=generated_attention_mask.dtype,
        device=generated_attention_mask.device,
    )
    attention_mask = torch.cat([generated_attention_mask, suffix_attention], dim=1)

    prefix_len = generated_ids.shape[1]
    prefix_positions = torch.arange(
        prefix_len,
        device=generated_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    position_offsets = torch.arange(block_size, device=generated_ids.device, dtype=torch.long)
    suffix_positions = (probe_position_starts[:, None] + position_offsets).reshape(1, -1)
    ptd_position_ids = torch.cat([prefix_positions, suffix_positions], dim=1)

    suffix_context_limits = context_limits[:, None].expand(-1, block_size).reshape(1, -1)
    ptd_context_limits = torch.cat(
        [torch.zeros_like(prefix_positions), suffix_context_limits],
        dim=1,
    )
    ptd_prefix_lengths = torch.tensor(
        [prefix_len],
        dtype=torch.long,
        device=generated_ids.device,
    )

    return {
        "input_ids": probe_ids,
        "attention_mask": attention_mask,
        "ptd_position_ids": ptd_position_ids,
        "ptd_prefix_lengths": ptd_prefix_lengths,
        "ptd_context_limits": ptd_context_limits,
    }


def make_ptd_replay_inputs(
    trace: PTDRolloutTrace,
    token_ids: dict[str, Any],
    *,
    block_size: int = 6,
) -> dict[str, torch.Tensor]:
    """Pack a rollout trace for differentiable PTD policy scoring."""
    if trace.target_token_ids.ndim != 2:
        raise ValueError(
            "PTD replay targets must be [num_blocks, block_size], got "
            f"{tuple(trace.target_token_ids.shape)}."
        )
    num_blocks, target_block_size = trace.target_token_ids.shape
    if num_blocks <= 0 or target_block_size != block_size:
        raise ValueError(
            "PTD replay requires at least one complete target block; got "
            f"{tuple(trace.target_token_ids.shape)} for block_size={block_size}."
        )
    if trace.target_mask.shape != trace.target_token_ids.shape:
        raise ValueError(
            "PTD replay target_mask must match target_token_ids, got "
            f"{tuple(trace.target_mask.shape)} and "
            f"{tuple(trace.target_token_ids.shape)}."
        )

    replay = make_ptd_probe_inputs(
        trace.prefix_input_ids,
        trace.prefix_attention_mask,
        trace.query_token_ids,
        token_ids,
        probe_position_starts=trace.probe_position_starts,
        context_limits=trace.context_limits,
        block_size=block_size,
    )
    # generate_ptd runs under torch.inference_mode(), so views of tensors
    # captured in its trace retain the inference-tensor flag. Autograd needs to
    # save target indices and masks while scoring the differentiable replay;
    # cloning here, outside inference mode, creates ordinary tensors.
    replay["ptd_target_ids"] = trace.target_token_ids.reshape(1, -1).clone()
    replay["ptd_action_mask"] = (
        trace.target_mask.reshape(1, -1).to(torch.long).clone()
    )
    return replay


def sample_token_ids(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if logits.ndim != 3:
        raise ValueError(f"logits must be [batch, seq, vocab], got {tuple(logits.shape)}")

    if not temperature or temperature <= 0:
        return logits.argmax(dim=-1)

    # Categorical validates a BF16 probability simplex at FP32-level tolerance.
    # Large parallel PTD probes can therefore be rejected on ROCm even when the
    # distribution is valid. Filter and normalize in FP32, then sample from the
    # flattened batch with multinomial.
    logits = logits.float() / temperature
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_sorted = cumulative_probs > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(logits, dtype=torch.bool)
        remove.scatter_(-1, sorted_indices, remove_sorted)
        logits = logits.masked_fill(remove, torch.finfo(logits.dtype).min)
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)

    probs = F.softmax(logits, dim=-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    row_sums = flat_probs.sum(dim=-1, keepdim=True)
    valid_rows = (
        torch.isfinite(flat_probs).all(dim=-1)
        & torch.isfinite(row_sums.squeeze(-1))
        & row_sums.squeeze(-1).gt(0)
    )

    fallback_logits = torch.nan_to_num(
        logits,
        nan=torch.finfo(logits.dtype).min,
        posinf=torch.finfo(logits.dtype).max,
        neginf=torch.finfo(logits.dtype).min,
    )
    sampled = fallback_logits.argmax(dim=-1).reshape(-1)
    if valid_rows.any():
        normalized = flat_probs[valid_rows] / row_sums[valid_rows]
        sampled[valid_rows] = torch.multinomial(normalized, num_samples=1).squeeze(-1)
    return sampled.reshape(logits.shape[:-1])


def _block_as_ints(block: list[int] | torch.Tensor, block_size: int) -> list[int]:
    if isinstance(block, torch.Tensor):
        block = block.detach().tolist()
    result = [int(token_id) for token_id in block]
    if len(result) != block_size:
        raise RuntimeError(
            f"PTD returned {len(result)} target logits for a {block_size}-token block."
        )
    return result


def _parse_semantic_block(
    block: list[int] | torch.Tensor,
    token_ids: dict[str, Any],
    *,
    first_block: bool,
    block_size: int,
) -> tuple[list[int], bool]:
    block = _block_as_ints(block, block_size)
    ref_start_id = int(token_ids["ref_start"])
    ref_end_id = int(token_ids["ref_end"])
    null_id = int(token_ids["null"])

    if first_block and block[0] != ref_start_id:
        raise RuntimeError("The first semantic PTD block must begin with <|object_ref_start|>.")
    if not first_block and ref_start_id in block:
        raise RuntimeError("A later semantic PTD block repeated <|object_ref_start|>.")

    forbidden = {
        int(token_ids["newline"]),
        int(token_ids["eos"]),
        int(token_ids["text_mask"]),
        int(token_ids["box_start"]),
        int(token_ids["box_end"]),
        int(token_ids["time_start"]),
        int(token_ids["time_end"]),
    }
    for token_idx, token_id in enumerate(block):
        if token_id == ref_end_id:
            # Slots after the closing marker were predicted simultaneously and
            # are not part of the semantic response.
            return block[: token_idx + 1], True
        if token_id == null_id:
            raise RuntimeError("Semantic PTD predicted <null> before the closing marker.")
        if token_id in forbidden:
            raise RuntimeError(
                "Semantic PTD predicted a structural token before the closing marker."
            )

    return block, False


def _parse_temporal_block(
    block: list[int] | torch.Tensor,
    token_ids: dict[str, Any],
    *,
    block_size: int,
) -> tuple[list[int], list[int]]:
    block = _block_as_ints(block, block_size)
    if not (
        block[0] == int(token_ids["time_start"])
        and block[1] in token_ids["time_tokens"]
        and block[2] in token_ids["time_tokens"]
        and block[3] == int(token_ids["time_end"])
    ):
        raise RuntimeError(
            "Temporal PTD must begin with "
            "[<|time_start|>, <t_start>, <t_end>, <|time_end|>]."
        )

    time_indices = token_ids["time_token_indices"]
    start_idx = int(time_indices[block[1]])
    end_idx = int(time_indices[block[2]])
    if end_idx < start_idx:
        raise RuntimeError("Temporal PTD predicted a reversed time span.")
    anchors = list(
        token_ids["ordered_time_tokens"][start_idx : end_idx + 1]
    )
    if not anchors:
        raise RuntimeError("Temporal PTD predicted an empty time span.")
    return block[:4], anchors


def _validate_box_blocks(
    blocks: torch.Tensor,
    token_ids: dict[str, Any],
    *,
    expected_blocks: int,
    block_size: int,
) -> list[list[int]]:
    if blocks.ndim != 2 or tuple(blocks.shape) != (expected_blocks, block_size):
        raise RuntimeError(
            "PTD returned the wrong box shape: expected "
            f"({expected_blocks}, {block_size}), got {tuple(blocks.shape)}."
        )

    box_start_id = int(token_ids["box_start"])
    box_end_id = int(token_ids["box_end"])
    coord_id_to_value = token_ids["coord_id_to_value"]
    validated = []
    for block_idx, raw_block in enumerate(blocks):
        block = _block_as_ints(raw_block, block_size)
        valid = (
            block[0] == box_start_id
            and all(token_id in coord_id_to_value for token_id in block[1:5])
            and block[5] == box_end_id
        )
        if not valid:
            raise RuntimeError(
                f"PTD box block {block_idx} violated the six-token grammar."
            )
        validated.append(block)
    return validated


def _cache_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        first_layer = past_key_values[0]
        key_states = (
            first_layer[0]
            if isinstance(first_layer, (tuple, list))
            else first_layer
        )
        return int(key_states.shape[-2])
    return 0


def _synchronize_for_timing(device: torch.device) -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _crop_past_key_values(past_key_values: Any, max_length: int) -> Any:
    if past_key_values is None:
        return None
    if rewind_reusable_ptd_cache(past_key_values, max_length):
        return past_key_values
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(max_length)
        return past_key_values
    if isinstance(past_key_values, tuple):
        return tuple(
            (
                key_states[:, :, :max_length, :],
                value_states[:, :, :max_length, :],
            )
            for key_states, value_states in past_key_values
        )
    raise TypeError(
        f"Cannot crop cache type {type(past_key_values)!r} after PTD probe pass."
    )


def _core_model(model):
    model = getattr(model, "module", model)
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return getattr(model, "model", model)


def _compute_cached_position_ids(
    model,
    input_ids: torch.Tensor,
    position_offsets: torch.Tensor,
) -> torch.Tensor:
    if position_offsets.ndim != 1 or position_offsets.shape[0] != input_ids.shape[1]:
        raise ValueError(
            "position_offsets must be a 1D tensor matching input length, "
            f"got {tuple(position_offsets.shape)} for input length {input_ids.shape[1]}"
        )

    core = _core_model(model)
    batch_size = input_ids.shape[0]
    position_ids = position_offsets.to(device=input_ids.device, dtype=torch.long)
    position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
    rope_deltas = getattr(core, "rope_deltas", None)
    if rope_deltas is not None:
        if batch_size % rope_deltas.shape[0] != 0:
            raise ValueError(
                "Cached PTD batch size must be divisible by the stored rope-delta "
                f"batch size, got {batch_size} and {rope_deltas.shape[0]}."
            )
        delta = rope_deltas.repeat_interleave(
            batch_size // rope_deltas.shape[0], dim=0
        )
        position_ids = position_ids + delta.to(device=input_ids.device)
    return position_ids


def _run_language_model(
    model,
    input_ids: torch.Tensor,
    past_key_values: Any,
    *,
    attention_mask: Optional[torch.Tensor],
    position_offsets: torch.Tensor,
    logits_to_keep: int,
    ptd_attention_plan: Any = None,
):
    core = _core_model(model)
    wrapped_model = getattr(model, "module", model)
    if hasattr(wrapped_model, "get_base_model"):
        wrapped_model = wrapped_model.get_base_model()
    language_model = getattr(core, "language_model", None)
    lm_head = getattr(wrapped_model, "lm_head", None)
    if language_model is None or lm_head is None or not hasattr(core, "get_input_embeddings"):
        raise AttributeError(
            "Cached PTD generation requires the Qwen3-VL language model, embeddings, and LM head."
        )

    inputs_embeds = core.get_input_embeddings()(input_ids)
    position_ids = _compute_cached_position_ids(
        model,
        input_ids=input_ids,
        position_offsets=position_offsets,
    )
    attention_context = nullcontext()
    extra_forward_kwargs = {}
    if ptd_attention_plan is not None:
        attention_context = use_ptd_flash_attention(language_model)
        extra_forward_kwargs["ptd_attention_plan"] = ptd_attention_plan
    with attention_context:
        outputs = language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **extra_forward_kwargs,
        )
    hidden_states = outputs.last_hidden_state[:, -logits_to_keep:, :]
    return outputs, lm_head(hidden_states)


def _make_cached_parallel_probe_inputs(
    generated: torch.Tensor,
    token_ids: dict[str, Any],
    *,
    query_token_ids: torch.Tensor,
    probe_position_starts: torch.Tensor,
    prefix_kv_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    committed_len = generated.shape[1]
    if prefix_kv_len < 0 or prefix_kv_len > committed_len:
        raise ValueError(
            f"prefix_kv_len must be in [0, {committed_len}], got {prefix_kv_len}."
        )

    query_token_ids = torch.as_tensor(
        query_token_ids,
        dtype=generated.dtype,
        device=generated.device,
    ).flatten()
    probe_position_starts = torch.as_tensor(
        probe_position_starts,
        dtype=torch.long,
        device=generated.device,
    ).flatten()
    num_blocks = query_token_ids.numel()
    if num_blocks == 0 or probe_position_starts.numel() != num_blocks:
        raise ValueError(
            "Cached query tokens and position starts must have the same positive "
            f"length, got {num_blocks} and {probe_position_starts.numel()}."
        )

    catchup_ids = generated[:, prefix_kv_len:committed_len]
    catchup_len = catchup_ids.shape[1]
    masks = torch.full(
        (num_blocks, block_size - 1),
        int(token_ids["text_mask"]),
        dtype=generated.dtype,
        device=generated.device,
    )
    probe_ids = torch.cat([query_token_ids[:, None], masks], dim=1).reshape(1, -1)
    query_ids = torch.cat([catchup_ids, probe_ids], dim=1)

    catchup_positions = torch.arange(
        prefix_kv_len,
        committed_len,
        dtype=torch.long,
        device=generated.device,
    )
    block_offsets = torch.arange(
        block_size,
        dtype=torch.long,
        device=generated.device,
    )
    probe_positions = (
        probe_position_starts[:, None] + block_offsets
    ).reshape(-1)
    position_offsets = torch.cat([catchup_positions, probe_positions], dim=0)
    return query_ids, position_offsets, catchup_len


def _run_cached_ptd_probe(
    model,
    generated: torch.Tensor,
    token_ids: dict[str, Any],
    past_key_values: Any,
    *,
    query_token_ids: torch.Tensor,
    probe_position_starts: torch.Tensor,
    context_limits: torch.Tensor,
    block_size: int,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    ptd_attn_implementation: str,
) -> tuple[torch.Tensor, Any]:
    committed_len = generated.shape[1]
    prefix_kv_len = _cache_length(past_key_values)
    if prefix_kv_len > committed_len:
        past_key_values = _crop_past_key_values(past_key_values, committed_len)
        prefix_kv_len = committed_len

    query_ids, position_offsets, catchup_len = _make_cached_parallel_probe_inputs(
        generated,
        token_ids,
        query_token_ids=query_token_ids,
        probe_position_starts=probe_position_starts,
        prefix_kv_len=prefix_kv_len,
        block_size=block_size,
    )
    num_blocks = int(torch.as_tensor(query_token_ids).numel())
    suffix_len = num_blocks * block_size
    capacity = ptd_cache_capacity(past_key_values)
    required_length = prefix_kv_len + query_ids.shape[1]
    if capacity is not None and required_length > capacity:
        raise RuntimeError(
            f"PTD probe exceeds reusable KV capacity: {required_length} > {capacity}."
        )

    ptd_attention_plan = None
    if ptd_attn_implementation == "flash_attention_2":
        ptd_attention_plan = build_ptd_flash_attention_plan(
            past_len=prefix_kv_len,
            catchup_len=catchup_len,
            num_blocks=num_blocks,
            context_limits=context_limits,
            block_size=block_size,
            device=query_ids.device,
        )

    probe_attention_mask = None
    if ptd_attention_plan is None:
        probe_attention_mask = build_cached_ptd_attention_mask_4d(
            past_len=prefix_kv_len,
            catchup_len=catchup_len,
            num_blocks=num_blocks,
            context_limits=context_limits,
            block_size=block_size,
            device=query_ids.device,
            dtype=next(_core_model(model).language_model.parameters()).dtype,
        )
        probe_attention_mask = pad_mask_to_ptd_cache_capacity(
            probe_attention_mask, past_key_values
        )
    outputs, block_logits = _run_language_model(
        model,
        query_ids,
        past_key_values,
        attention_mask=probe_attention_mask,
        position_offsets=position_offsets,
        logits_to_keep=suffix_len,
        ptd_attention_plan=ptd_attention_plan,
    )
    sampled = sample_token_ids(
        block_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    # Commit real catch-up tokens and discard every temporary probe block.
    past_key_values = _crop_past_key_values(
        outputs.past_key_values, committed_len
    )
    return sampled.reshape(num_blocks, block_size), past_key_values


@torch.inference_mode()
def generate_ptd(
    model,
    tokenizer,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int,
    block_size: int = 6,
    max_time_tokens: Optional[int] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    generation_format: str = "spatio_temporal_grounding",
    ptd_attn_implementation: str = "sdpa",
    measure_decode: bool = False,
    return_trace: bool = False,
) -> tuple[torch.Tensor, PTDGenerationResult]:
    """Generate a semantic reference and temporal segment with optional boxes.

    One multimodal prefill is followed by cached language-model probes. All
    time-conditioned box probes remain flattened
    into one independent multi-block forward for ``spatio_temporal_grounding``.
    ``temporal_localization`` matches temporal-localization training and
    stops immediately after the four-token segment.
    """

    generation_format = str(generation_format).strip().lower()
    if generation_format not in {
        "spatio_temporal_grounding",
        "temporal_localization",
    }:
        raise ValueError(
            "Time-anchored parallel PTD only supports generation_format="
            "spatio_temporal_grounding or temporal_localization, "
            f"got {generation_format!r}."
        )
    if block_size != 6:
        raise ValueError(f"Time-anchored parallel PTD requires block_size=6, got {block_size}.")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}.")
    if "input_ids" not in inputs:
        raise ValueError("inputs must contain input_ids.")
    ptd_attn_implementation = resolve_ptd_attn_implementation(ptd_attn_implementation)

    generated = inputs["input_ids"]
    if generated.ndim != 2 or generated.shape[0] != 1 or generated.shape[1] == 0:
        raise ValueError(
            "Time-anchored parallel PTD currently requires non-empty batch-size-one input_ids, "
            f"got {tuple(generated.shape)}."
        )
    prompt_len = generated.shape[1]
    generated_attention_mask = inputs.get("attention_mask")
    if generated_attention_mask is None:
        generated_attention_mask = torch.ones_like(generated, dtype=torch.long)
    if generated_attention_mask.shape != generated.shape:
        raise ValueError(
            f"attention_mask shape {tuple(generated_attention_mask.shape)} does not match "
            f"input_ids shape {tuple(generated.shape)}."
        )
    if not bool(generated_attention_mask[0, -1].item()):
        raise ValueError("The last prompt token must be valid; right-padded generation is unsupported.")

    token_ids = build_ptd_token_ids(
        tokenizer, max_time_tokens=max_time_tokens
    )
    configure_ptd_model(model, block_size)
    newline_id = int(token_ids["newline"])
    eos_id = int(token_ids["eos"])
    step = 0
    stopped = False
    trace_prefix_end = prompt_len
    trace_queries: list[torch.Tensor] = []
    trace_targets: list[torch.Tensor] = []
    trace_target_masks: list[torch.Tensor] = []
    trace_position_starts: list[torch.Tensor] = []
    trace_context_limits: list[torch.Tensor] = []

    prefill_inputs = dict(inputs)
    for stale_key in (
        "position_ids",
        "cache_position",
        "past_key_values",
        "labels",
        "ptd_position_ids",
        "ptd_prefix_lengths",
        "ptd_context_limits",
    ):
        prefill_inputs.pop(stale_key, None)
    prefill_inputs["use_cache"] = True
    prefill_inputs["return_dict"] = True
    prefill_inputs["logits_to_keep"] = 1
    prefill_outputs = model(**prefill_inputs)
    past_key_values = getattr(prefill_outputs, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("The model did not return past_key_values for PTD prefill.")
    if ptd_attn_implementation == "flash_attention_2" and max_new_tokens > 0:
        # Reserve the largest possible temporary probe suffix once. Rewinding
        # this cache after each probe avoids repeatedly copying the prefix.
        max_parallel_blocks = max(
            1,
            min(len(token_ids["ordered_time_tokens"]), max_new_tokens // 8),
        )
        cache_capacity = (
            _cache_length(past_key_values)
            + max_new_tokens
            + block_size * max_parallel_blocks
        )
        past_key_values = make_reusable_ptd_cache(
            past_key_values,
            config=getattr(_core_model(model).language_model, "config", None),
            max_cache_len=cache_capacity,
        )
    decode_start_time = None
    decode_end_time = None
    if measure_decode:
        _synchronize_for_timing(generated.device)
        decode_start_time = time.perf_counter()

    def stop_decode_timer() -> None:
        nonlocal decode_end_time
        if decode_start_time is not None and decode_end_time is None:
            _synchronize_for_timing(generated.device)
            decode_end_time = time.perf_counter()

    def build_trace() -> Optional[PTDRolloutTrace]:
        if not return_trace:
            return None
        if not trace_queries:
            return PTDRolloutTrace(
                prefix_input_ids=generated[:, :trace_prefix_end].detach().clone(),
                prefix_attention_mask=(
                    generated_attention_mask[:, :trace_prefix_end].detach().clone()
                ),
                query_token_ids=generated.new_empty((0,)),
                target_token_ids=generated.new_empty((0, block_size)),
                target_mask=generated.new_empty((0, block_size), dtype=torch.long),
                probe_position_starts=generated.new_empty((0,), dtype=torch.long),
                context_limits=generated.new_empty((0,), dtype=torch.long),
            )
        return PTDRolloutTrace(
            prefix_input_ids=generated[:, :trace_prefix_end].detach().clone(),
            prefix_attention_mask=(
                generated_attention_mask[:, :trace_prefix_end].detach().clone()
            ),
            query_token_ids=torch.cat(trace_queries, dim=0),
            target_token_ids=torch.cat(trace_targets, dim=0),
            target_mask=torch.cat(trace_target_masks, dim=0),
            probe_position_starts=torch.cat(trace_position_starts, dim=0),
            context_limits=torch.cat(trace_context_limits, dim=0),
        )

    def record_probe(
        blocks: torch.Tensor,
        *,
        query_token_ids: torch.Tensor,
        probe_position_starts: torch.Tensor,
        context_limits: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> None:
        if not return_trace:
            return
        blocks = blocks.detach().reshape(-1, block_size)
        target_mask = target_mask.to(
            device=blocks.device,
            dtype=torch.long,
        ).reshape(blocks.shape)
        trace_queries.append(query_token_ids.detach().flatten().clone())
        trace_targets.append(blocks.clone())
        trace_target_masks.append(target_mask.clone())
        trace_position_starts.append(
            probe_position_starts.detach().to(torch.long).flatten().clone()
        )
        trace_context_limits.append(
            context_limits.detach().to(torch.long).flatten().clone()
        )

    def finish(is_stopped: bool) -> tuple[torch.Tensor, PTDGenerationResult]:
        generated_ids = generated[:, prompt_len:]
        if generated_ids.shape[1] == 0 and max_new_tokens > 0:
            # GRPO completion masking requires a non-empty sequence. An EOS-only
            # fallback keeps a rejected first PTD probe at zero task reward
            # without terminating the distributed job.
            generated_ids = generated.new_full((generated.shape[0], 1), eos_id)
        decode_latency_s = None
        if decode_start_time is not None:
            stop_decode_timer()
            decode_latency_s = max(decode_end_time - decode_start_time, 0.0)
        result = PTDGenerationResult(
            steps=step,
            generated_tokens=int(generated_ids.shape[1]),
            stopped=is_stopped,
            decode_latency_s=decode_latency_s,
            trace=build_trace(),
        )
        return generated_ids, result

    def run_probe(
        *,
        query_token_ids: torch.Tensor,
        probe_position_starts: torch.Tensor,
        context_limits: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal past_key_values
        probe_kwargs = {
            "query_token_ids": query_token_ids,
            "probe_position_starts": probe_position_starts,
            "context_limits": context_limits,
            "block_size": block_size,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "ptd_attn_implementation": ptd_attn_implementation,
        }
        blocks, past_key_values = _run_cached_ptd_probe(
            model,
            generated,
            token_ids,
            past_key_values,
            **probe_kwargs,
        )
        return blocks

    def append_visible(tokens: list[int]) -> None:
        nonlocal generated, generated_attention_mask
        if not tokens:
            return
        token_tensor = torch.tensor([tokens], dtype=generated.dtype, device=generated.device)
        token_attention = torch.ones(
            token_tensor.shape,
            dtype=generated_attention_mask.dtype,
            device=generated_attention_mask.device,
        )
        generated = torch.cat([generated, token_tensor], dim=1)
        generated_attention_mask = torch.cat(
            [generated_attention_mask, token_attention], dim=1
        )

    def remaining_budget() -> int:
        return max_new_tokens - (generated.shape[1] - prompt_len)

    # Semantic response: each forward predicts exactly six target tokens. The
    # first probe skips a trailing chat-template newline while retaining it as
    # prefix context. Later probes copy the previous semantic token and exclude
    # its original occurrence from prefix attention.
    semantic_complete = False
    first_semantic_block = True
    while remaining_budget() > 0:
        prefix_len = generated.shape[1]
        if first_semantic_block:
            semantic_query_pos = prefix_len - 1
            while (
                semantic_query_pos >= 0
                and int(generated[0, semantic_query_pos].item()) == newline_id
            ):
                semantic_query_pos -= 1
            if semantic_query_pos < 0:
                raise RuntimeError(
                    "No non-newline query token precedes the semantic response."
                )
            semantic_query = generated[0, semantic_query_pos : semantic_query_pos + 1]
            semantic_context_limit = (
                prefix_len
                if semantic_query_pos != prefix_len - 1
                else prefix_len - 1
            )
        else:
            semantic_query = generated[0, -1:]
            semantic_context_limit = prefix_len - 1
        semantic_position_starts = torch.tensor([prefix_len - 1], device=generated.device)
        semantic_context_limits = torch.tensor([semantic_context_limit], device=generated.device)
        raw_semantic = run_probe(
            query_token_ids=semantic_query,
            probe_position_starts=semantic_position_starts,
            context_limits=semantic_context_limits,
        )[0]
        step += 1
        try:
            semantic_tokens, semantic_complete = _parse_semantic_block(
                raw_semantic,
                token_ids,
                first_block=first_semantic_block,
                block_size=block_size,
            )
        except RuntimeError:
            record_probe(
                raw_semantic.unsqueeze(0),
                query_token_ids=semantic_query,
                probe_position_starts=semantic_position_starts,
                context_limits=semantic_context_limits,
                target_mask=torch.ones((1, block_size), dtype=torch.long, device=generated.device),
            )
            return finish(False)
        semantic_target_mask = torch.zeros((1, block_size), dtype=torch.long, device=generated.device)
        semantic_target_mask[:, : len(semantic_tokens)] = 1
        record_probe(
            raw_semantic.unsqueeze(0),
            query_token_ids=semantic_query,
            probe_position_starts=semantic_position_starts,
            context_limits=semantic_context_limits,
            target_mask=semantic_target_mask,
        )
        visible_semantic = semantic_tokens + ([newline_id] if semantic_complete else [])
        if len(visible_semantic) > remaining_budget():
            break
        append_visible(visible_semantic)
        trace_prefix_end = generated.shape[1]
        first_semantic_block = False
        if semantic_complete:
            break

    if not semantic_complete or remaining_budget() < 5:
        return finish(False)

    # Keep the visible newline in the prefix and use a copied ref_end token as
    # the probe query at that delimiter's RoPE position.
    prefix_len = generated.shape[1]
    temporal_query = torch.tensor([token_ids["ref_end"]], device=generated.device)
    temporal_position_starts = torch.tensor([prefix_len - 1], device=generated.device)
    temporal_context_limits = torch.tensor([prefix_len], device=generated.device)
    raw_temporal = run_probe(
        query_token_ids=temporal_query,
        probe_position_starts=temporal_position_starts,
        context_limits=temporal_context_limits,
    )[0]
    step += 1
    temporal_target_mask = torch.zeros((1, block_size), dtype=torch.long, device=generated.device)
    temporal_target_mask[:, :4] = 1
    record_probe(
        raw_temporal.unsqueeze(0),
        query_token_ids=temporal_query,
        probe_position_starts=temporal_position_starts,
        context_limits=temporal_context_limits,
        target_mask=temporal_target_mask,
    )
    try:
        temporal_tokens, time_anchors = _parse_temporal_block(
            raw_temporal,
            token_ids,
            block_size=block_size,
        )
    except RuntimeError:
        # A sampled rollout may violate the learned PTD structure, especially
        # early in GRPO. Treat it as an incomplete completion (and therefore a
        # zero temporal/spatial reward) instead of terminating the whole
        # distributed job because one rank produced an invalid sample.
        return finish(False)
    temporal_delimiter = (
        eos_id
        if generation_format == "temporal_localization"
        else newline_id
    )
    visible_temporal = temporal_tokens + [temporal_delimiter]
    append_visible(visible_temporal)
    trace_prefix_end = generated.shape[1]

    if generation_format == "temporal_localization":
        stopped = True
        return finish(stopped)

    # Every visible row contributes seven time-box tokens plus either a newline
    # or the final im_end. Never emit a partial row when the budget is too small.
    num_boxes = len(time_anchors)
    required_box_tokens = 8 * num_boxes
    if required_box_tokens > remaining_budget():
        return finish(False)

    prefix_len = generated.shape[1]
    time_anchor_tensor = torch.tensor(
        time_anchors,
        dtype=generated.dtype,
        device=generated.device,
    )
    block_indices = torch.arange(num_boxes, dtype=torch.long, device=generated.device)
    box_position_starts = prefix_len + 8 * block_indices
    box_context_limits = torch.full((num_boxes,), prefix_len, dtype=torch.long, device=generated.device)
    raw_boxes = run_probe(
        query_token_ids=time_anchor_tensor,
        probe_position_starts=box_position_starts,
        context_limits=box_context_limits,
    )
    # For a complete tube, TCL ends with the final spatial GPU operation.
    # Format validation, text assembly, decoding, scoring, and I/O are outside
    # the timed region.
    stop_decode_timer()
    step += 1
    record_probe(
        raw_boxes,
        query_token_ids=time_anchor_tensor,
        probe_position_starts=box_position_starts,
        context_limits=box_context_limits,
        target_mask=torch.ones_like(raw_boxes, dtype=torch.long),
    )
    try:
        box_blocks = _validate_box_blocks(
            raw_boxes,
            token_ids,
            expected_blocks=num_boxes,
            block_size=block_size,
        )
    except RuntimeError:
        # Keep the valid temporal segment in the completion, but omit all box
        # rows so the spatial scorer assigns zero instead of crashing training.
        return finish(False)
    visible_boxes = []
    for box_idx, (time_anchor, box_block) in enumerate(zip(time_anchors, box_blocks)):
        visible_boxes.extend([time_anchor, *box_block])
        visible_boxes.append(eos_id if box_idx == num_boxes - 1 else newline_id)
    append_visible(visible_boxes)
    stopped = True

    return finish(stopped)
