"""A token stream that can be seeked to any training step in O(1).

Resumability is the whole design constraint. A run that is killed at 12h and
restarted has to see exactly the batch it would have seen, and it cannot get
there by replaying the dataloader -- at 2.26B tokens that is hours of work to
recover a position. So sequence order is a pure function of (seed, step): the
trainer asks for step N's sequences and gets them without having touched steps
0..N-1.

numpy only, no torch: the ordering is arithmetic, and keeping it torch-free
means the resume logic is testable without a GPU or a model.
"""

from pathlib import Path
from typing import Union

import numpy as np

_M64 = (1 << 64) - 1

# Tokens are stored as uint16, which holds the 32k vocab of spec 7 with room to
# spare and halves the bytes read against int32. Widen this if the vocab ever
# exceeds 65535.
TOKEN_DTYPE = np.uint16


def _mix(x: int) -> int:
    """splitmix64 finalizer: an integer hash with good avalanche."""
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _M64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def permute(index: int, n: int, key: int, rounds: int = 4) -> int:
    """Map [0, n) onto itself bijectively, in O(1) time and memory.

    A materialised np.random.permutation would be simpler, but it is O(n)
    memory: 2.2M sequences at the dev budget is tolerable, 48M at the target
    budget is 390 MB held for the whole run just to know what comes next. A
    Feistel network over the next power of four, with cycle-walking back into
    range, gives the same guarantee (every sequence once per epoch, no
    repeats) while storing nothing.

    Termination: the network is a bijection on [0, 2**2h), so iterating it from
    a point inside [0, n) traverses a finite cycle that must return to a point
    inside [0, n).
    """
    if n <= 1:
        return 0
    half = max(1, ((n - 1).bit_length() + 1) // 2)
    mask = (1 << half) - 1
    subkeys = [_mix(key + rnd) for rnd in range(rounds)]

    x = index
    while True:
        left, right = x >> half, x & mask
        for subkey in subkeys:
            left, right = right, left ^ (_mix(right ^ subkey) & mask)
        x = (left << half) | right
        if x < n:
            return x


class TokenStream:
    """A flat token array, read as a deterministic stream of training sequences.

    Sequences do not overlap and every one is visited once per epoch. Epoch
    boundaries reshuffle: the ordering key is derived from (seed, epoch), so a
    run longer than the corpus does not repeat the same order.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, seed: int):
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        # -1 because targets are inputs shifted by one: the last sequence needs
        # a token after its end to predict.
        n_sequences = (len(tokens) - 1) // seq_len
        if n_sequences < 1:
            raise ValueError(
                f"corpus of {len(tokens)} tokens holds no sequence of "
                f"{seq_len} (+1 for the shifted target)"
            )
        self.tokens = tokens
        self.seq_len = seq_len
        self.seed = seed
        self.n_sequences = n_sequences

    @classmethod
    def from_path(cls, path: Union[str, Path], seq_len: int, seed: int) -> "TokenStream":
        """Memory-map a .bin of uint16 tokens, as written by scripts/prepare_data.py."""
        return cls(np.memmap(path, dtype=TOKEN_DTYPE, mode="r"), seq_len, seed)

    @property
    def tokens_per_epoch(self) -> int:
        return self.n_sequences * self.seq_len

    def start_of(self, global_index: int) -> int:
        """Token offset of the global_index'th sequence the run will ever see."""
        epoch, within = divmod(global_index, self.n_sequences)
        return permute(within, self.n_sequences, _mix(self.seed + epoch)) * self.seq_len

    def indices_for_step(self, step: int, sequences_per_step: int) -> np.ndarray:
        """Global sequence indices making up one optimizer step.

        Indexed by step rather than by micro-batch on purpose: this is what
        makes micro_batch a pure VRAM knob. Re-chunking a step into a different
        number of micro-batches reorders nothing, so the same config trains
        identically on a 16 GB T4 and an 80 GB H100.
        """
        base = step * sequences_per_step
        return np.arange(base, base + sequences_per_step, dtype=np.int64)

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gather (inputs, targets) for the given global sequence indices."""
        inputs = np.empty((len(indices), self.seq_len), dtype=np.int64)
        targets = np.empty_like(inputs)
        for row, index in enumerate(indices):
            start = self.start_of(int(index))
            window = self.tokens[start : start + self.seq_len + 1]
            inputs[row] = window[:-1]
            targets[row] = window[1:]
        return inputs, targets
