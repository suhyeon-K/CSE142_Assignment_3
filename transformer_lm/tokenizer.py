"""Byte-level BPE tokenizer (no regex pre-tokenization)."""

from __future__ import annotations

from collections import Counter


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Train a byte-level BPE tokenizer.

    Merge order: at each step, merge the most frequent adjacent pair.
    Break ties by selecting the pair with the smallest ``(id1, id2)``
    in lexicographic (tuple) order.

    IDs 0–255 are single bytes. Merge tokens get IDs starting from 256.
    Special tokens get the highest IDs in the vocab.

    Args:
        input_path: Path to a UTF-8 text file.
        vocab_size: Target vocabulary size (>= 256 + len(special_tokens)).
        special_tokens: Optional special token strings.

    Returns:
        vocab: ``dict[int, bytes]`` mapping token ID to byte string.
        merges: ``list[tuple[int, int]]`` merge pairs in order.
    """
    if special_tokens is None:
        special_tokens = []
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small for base bytes and special tokens")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    vocab = {i: bytes([i]) for i in range(256)}
    ids = list(text.encode("utf-8"))
    merges: list[tuple[int, int]] = []
    num_merges = vocab_size - 256 - len(special_tokens)

    for merge_idx in range(num_merges):
        pair_counts = Counter(zip(ids, ids[1:]))
        if not pair_counts:
            break

        max_count = max(pair_counts.values())
        pair = min(p for p, count in pair_counts.items() if count == max_count)
        new_id = 256 + merge_idx
        merges.append(pair)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

        merged_ids = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                merged_ids.append(new_id)
                i += 2
            else:
                merged_ids.append(ids[i])
                i += 1
        ids = merged_ids

    special_start = vocab_size - len(special_tokens)
    for i, token in enumerate(special_tokens):
        vocab[special_start + i] = token.encode("utf-8")

    return vocab, merges


class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:
        if special_tokens is None:
            special_tokens = []

        self.vocab = vocab
        self.merges = merges
        self.merge_to_id = {pair: 256 + i for i, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.special_token_to_id: dict[str, int] = {}

        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            matching_ids = [tid for tid, value in vocab.items() if value == token_bytes]
            if matching_ids:
                self.special_token_to_id[token] = max(matching_ids)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        ids: list[int] = []
        i = 0

        while i < len(text):
            matched_special = None
            for token in self.special_tokens:
                if text.startswith(token, i):
                    matched_special = token
                    break

            if matched_special is not None:
                ids.append(self.special_token_to_id[matched_special])
                i += len(matched_special)
                continue

            next_special_pos = len(text)
            for token in self.special_tokens:
                pos = text.find(token, i + 1)
                if pos != -1 and pos < next_special_pos:
                    next_special_pos = pos

            segment = text[i:next_special_pos]
            ids.extend(self._encode_bytes(segment.encode("utf-8")))
            i = next_special_pos

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        data = b"".join(self.vocab[token_id] for token_id in ids)
        return data.decode("utf-8", errors="replace")

    def _encode_bytes(self, data: bytes) -> list[int]:
        ids = list(data)
        for pair, new_id in self.merge_to_id.items():
            merged_ids = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    merged_ids.append(new_id)
                    i += 2
                else:
                    merged_ids.append(ids[i])
                    i += 1
            ids = merged_ids
        return ids
