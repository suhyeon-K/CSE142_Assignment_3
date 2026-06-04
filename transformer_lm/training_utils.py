"""Training utilities: batching and text generation."""

from __future__ import annotations

import torch

from transformer_lm.nn_utils import softmax


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of input-target pairs from a 1-D token array.

    Args:
        data: 1-D tensor of token IDs.
        batch_size: Number of examples per batch.
        context_length: Number of tokens in each sequence.
        device: Device to place tensors on.

    Returns:
        ``(x, y)`` both of shape ``(batch_size, context_length)``.
    """
    if data.shape[0] <= context_length:
        raise ValueError("data must be longer than context_length")

    data = data.to(device=device, dtype=torch.long)
    starts = torch.randint(
        low=0,
        high=data.shape[0] - context_length,
        size=(batch_size,),
        device=data.device,
    )
    offsets = torch.arange(context_length, device=data.device)
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)

    x = data[indices]
    y = data[indices + 1]
    return x, y


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    context_length: int | None = None,
) -> list[int]:
    """Autoregressively generate tokens from a language model.

    Args:
        model: Maps ``(B, T)`` integer input to ``(B, T, vocab_size)`` logits.
        prompt_ids: Starting token IDs.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        context_length: Maximum context window (defaults to ``model.context_length``).

    Returns:
        List of token IDs (prompt + generated).
    """
    if context_length is None:
        context_length = model.context_length
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    device = next(model.parameters()).device
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        context = generated[-context_length:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids)
        next_logits = logits[:, -1, :] / temperature
        probs = softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)

    return generated
