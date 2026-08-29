import torch


@torch.no_grad()
def prepare_ptd_attention(
    position_ids,
    attention_mask,
    ptd_position_ids,
    ptd_prefix_lengths,
    ptd_context_limits,
    dtype,
    block_size=6,
):
    """Apply PTD logical positions and block attention to a Qwen3 sequence."""
    metadata = (ptd_position_ids, ptd_prefix_lengths, ptd_context_limits)
    if all(value is None for value in metadata):
        return position_ids, attention_mask
    if any(value is None for value in metadata):
        raise ValueError("PTD requires position ids, prefix lengths, and context limits together.")
    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError("PTD requires a two-dimensional right-padded attention mask.")
    if ptd_position_ids.ndim != 2:
        raise ValueError("ptd_position_ids must have shape [batch, sequence].")

    batch_size, sequence_length = ptd_position_ids.shape
    expected_shape = (batch_size, sequence_length)
    if tuple(attention_mask.shape) != expected_shape:
        raise ValueError("attention_mask and ptd_position_ids must have the same shape.")
    if tuple(ptd_context_limits.shape) != expected_shape:
        raise ValueError("ptd_context_limits and ptd_position_ids must have the same shape.")
    if ptd_prefix_lengths.ndim != 1 or ptd_prefix_lengths.shape[0] != batch_size:
        raise ValueError("ptd_prefix_lengths must have shape [batch].")
    if tuple(position_ids.shape[-2:]) != expected_shape:
        raise ValueError("Qwen position ids and PTD position ids must have matching shapes.")

    device = position_ids.device
    logical_positions = ptd_position_ids.to(device=device, dtype=torch.long)
    prefix_lengths = ptd_prefix_lengths.to(device=device, dtype=torch.long)
    context_limits = ptd_context_limits.to(device=device, dtype=torch.long)
    valid_tokens = attention_mask.to(device=device).bool()
    valid_lengths = valid_tokens.sum(dim=1)
    physical_positions = torch.arange(sequence_length, device=device).view(1, -1)

    if not torch.equal(valid_tokens, physical_positions < valid_lengths.view(batch_size, 1)):
        raise ValueError("PTD batches must use contiguous right padding.")
    if bool(((prefix_lengths <= 0) | (prefix_lengths > valid_lengths)).any()):
        raise ValueError("Each PTD prefix length must be within the valid sequence.")

    suffix_lengths = valid_lengths - prefix_lengths
    if bool(suffix_lengths.remainder(block_size).ne(0).any()):
        raise ValueError(f"Each PTD suffix must contain complete {block_size}-token blocks.")

    suffix_tokens = (
        (physical_positions >= prefix_lengths.view(batch_size, 1))
        & (physical_positions < valid_lengths.view(batch_size, 1))
    )
    suffix_offsets = (physical_positions - prefix_lengths.view(batch_size, 1)).clamp(min=0)
    block_starts = (
        prefix_lengths.view(batch_size, 1)
        + torch.div(suffix_offsets, block_size, rounding_mode="floor") * block_size).clamp(max=sequence_length - 1)
    block_context_limits = context_limits.gather(1, block_starts)
    if bool((suffix_tokens & context_limits.ne(block_context_limits)).any()):
        raise ValueError("Each PTD block must use one context limit.")
    if bool(
        (
            suffix_tokens
            & (
                context_limits.lt(0)
                | context_limits.gt(prefix_lengths.view(batch_size, 1))
            )
        ).any()
    ):
        raise ValueError("PTD context limits must lie inside the original prefix.")

    suffix_mask = suffix_tokens & valid_tokens
    last_prefix_position = prefix_lengths.sub(1).view(batch_size, 1)
    source_positions = torch.minimum(
        logical_positions.clamp(min=0), last_prefix_position
    )
    overflow = (logical_positions - last_prefix_position).clamp(min=0)
    if position_ids.ndim == 3:
        gather_positions = source_positions.unsqueeze(0).expand(
            position_ids.shape[0], -1, -1
        )
        mapped_positions = position_ids.gather(-1, gather_positions)
        mapped_positions = mapped_positions + overflow.unsqueeze(0)
        position_ids = torch.where(
            suffix_mask.unsqueeze(0), mapped_positions, position_ids
        )
    else:
        mapped_positions = position_ids.gather(-1, source_positions) + overflow
        position_ids = torch.where(suffix_mask, mapped_positions, position_ids)

    query = torch.arange(sequence_length, device=device).view(1, -1, 1)
    key = torch.arange(sequence_length, device=device).view(1, 1, -1)
    prefix = prefix_lengths.view(batch_size, 1, 1)
    prefix_query = query < prefix
    prefix_key = key < prefix
    prefix_causal = prefix_query & prefix_key & (key <= query)
    query_block = torch.div(query - prefix, block_size, rounding_mode="floor")
    key_block = torch.div(key - prefix, block_size, rounding_mode="floor")
    same_block = (~prefix_query) & (~prefix_key) & (query_block == key_block)
    ptd_to_prefix = (
        (~prefix_query)
        & prefix_key
        & (key < context_limits.view(batch_size, sequence_length, 1))
    )
    valid_query = valid_tokens.view(batch_size, sequence_length, 1)
    valid_key = valid_tokens.view(batch_size, 1, sequence_length)
    visible = (prefix_causal | same_block | ptd_to_prefix) & valid_query & valid_key
    visible |= (~valid_query) & (query == key)

    ptd_mask = torch.full(
        (batch_size, 1, sequence_length, sequence_length),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    ptd_mask.masked_fill_(visible.unsqueeze(1), 0)
    return position_ids, ptd_mask


@torch.no_grad()
def build_cached_ptd_attention_mask_4d(
    past_len,
    catchup_len,
    num_blocks,
    context_limits,
    *,
    block_size=6,
    device,
    dtype=torch.bfloat16,
):
    """Build the attention mask used by cached PTD generation."""
    if past_len < 0 or catchup_len < 0:
        raise ValueError("Cached PTD lengths must be non-negative.")
    if num_blocks <= 0 or block_size <= 0:
        raise ValueError("Cached PTD requires positive block counts and sizes.")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("The cached PTD attention mask requires a floating dtype.")

    context_limits = torch.as_tensor(
        context_limits, dtype=torch.long, device=device
    ).flatten()
    if context_limits.numel() != num_blocks:
        raise ValueError("PTD requires one context limit per cached block.")

    committed_len = past_len + catchup_len
    if bool(((context_limits < 0) | (context_limits > committed_len)).any()):
        raise ValueError("Cached PTD context limits must lie in the committed prefix.")

    query_len = catchup_len + num_blocks * block_size
    key_len = past_len + query_len
    visible = torch.zeros((query_len, key_len), dtype=torch.bool, device=device)

    if catchup_len:
        queries = torch.arange(catchup_len, device=device).view(-1, 1)
        keys = torch.arange(key_len, device=device).view(1, -1)
        visible[:catchup_len] = keys <= past_len + queries

    for block_idx, context_limit in enumerate(context_limits.tolist()):
        query_start = catchup_len + block_idx * block_size
        query_end = query_start + block_size
        visible[query_start:query_end, :context_limit] = True

        key_start = committed_len + block_idx * block_size
        key_end = key_start + block_size
        visible[query_start:query_end, key_start:key_end] = True

    attention_mask = torch.full(
        (1, 1, query_len, key_len),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    attention_mask.masked_fill_(visible.view(1, 1, query_len, key_len), 0)
    return attention_mask
