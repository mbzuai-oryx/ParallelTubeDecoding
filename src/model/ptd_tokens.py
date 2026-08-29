from __future__ import annotations

import torch
from transformers import AddedToken


NUM_COORD_TOKENS = 1001
NUM_TIME_TOKENS = 100
STRUCTURAL_TOKENS = ("<null>", "<text_mask>", "<|time_start|>", "<|time_end|>")


def ptd_token_strings(num_time_tokens: int = NUM_TIME_TOKENS) -> list[str]:
    return (
        list(STRUCTURAL_TOKENS)
        + [f"<{index}>" for index in range(NUM_COORD_TOKENS)]
        + [f"<t{index}>" for index in range(1, num_time_tokens + 1)]
    )


def ptd_token_ids(tokenizer, num_time_tokens: int = NUM_TIME_TOKENS) -> list[int]:
    token_ids = []
    for token in ptd_token_strings(num_time_tokens):
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if (
            token_id is None
            or token_id == tokenizer.unk_token_id
            or encoded != [token_id]
        ):
            raise ValueError(
                f"PTD token {token!r} is missing or is not a single token: {encoded}. "
                "Run src/initialize_ptd_tokens.py before training."
            )
        token_ids.append(int(token_id))
    return token_ids


@torch.no_grad()
def add_and_initialize_ptd_tokens(
    model, processor, num_time_tokens: int = NUM_TIME_TOKENS
) -> list[int]:
    """Add PTD tokens and initialize only the newly added embedding rows."""
    tokenizer = processor.tokenizer
    old_vocab_size = len(tokenizer)
    tokenizer.add_tokens(
        [
            AddedToken(token, normalized=False, special=True)
            for token in STRUCTURAL_TOKENS
        ],
        special_tokens=True,
    )
    tokenizer.add_tokens(
        [
            AddedToken(token, normalized=False, special=False)
            for token in ptd_token_strings(num_time_tokens)[len(STRUCTURAL_TOKENS) :]
        ],
        special_tokens=False,
    )
    if len(tokenizer) == old_vocab_size:
        return ptd_token_ids(tokenizer, num_time_tokens)

    model.resize_token_embeddings(len(tokenizer))
    input_weight = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings()
    output_weight = None if output_embeddings is None else output_embeddings.weight
    sources = {
        "<null>": "null",
        "<text_mask>": "text mask",
        "<|time_start|>": "time start",
        "<|time_end|>": "time end",
    }
    sources.update({f"<{index}>": str(index) for index in range(NUM_COORD_TOKENS)})
    sources.update(
        {
            f"<t{index}>": f"time segment {index}"
            for index in range(1, num_time_tokens + 1)
        }
    )

    for token, source in sources.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id < old_vocab_size:
            continue
        source_ids = tokenizer.encode(source, add_special_tokens=False)
        if not source_ids:
            raise ValueError(f"Cannot initialize {token!r} from {source!r}.")
        input_weight[token_id] = input_weight[source_ids].float().mean(dim=0).to(
            input_weight.dtype
        )
        if output_weight is not None and output_weight.data_ptr() != input_weight.data_ptr():
            output_weight[token_id] = output_weight[source_ids].float().mean(dim=0).to(
                output_weight.dtype
            )

    return ptd_token_ids(tokenizer, num_time_tokens)


def train_only_ptd_token_rows(model, token_ids: list[int]) -> None:
    """Train PTD vocabulary rows while masking gradients for the base vocabulary."""
    selected = torch.tensor(sorted(set(token_ids)), dtype=torch.long)

    def register(weight):
        weight.requires_grad = True

        def mask_gradient(gradient):
            mask = torch.zeros(
                gradient.shape[0], dtype=torch.bool, device=gradient.device
            )
            mask[selected.to(gradient.device)] = True
            return gradient * mask.view(-1, *([1] * (gradient.ndim - 1)))

        weight.register_hook(mask_gradient)

    input_weight = model.get_input_embeddings().weight
    register(input_weight)
    output_embeddings = model.get_output_embeddings()
    if (
        output_embeddings is not None
        and output_embeddings.weight.data_ptr() != input_weight.data_ptr()
    ):
        register(output_embeddings.weight)
