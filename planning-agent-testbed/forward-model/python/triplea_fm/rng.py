"""Explicit game randomness (spec §11).

Two interchangeable sources:
  * SeededRng  - counter-based SplitMix64 stream ("splitmix64-ctr", version 1). The value
                 (seed, stream, counter) fully determines future draws, so cloning is copying.
  * TapeRng    - strict replay of recorded requests; mismatched or missing requests fail.

Draws are zero-based integers in [0, upper_bound) using unbiased rejection sampling.
Never use these for agent/policy randomness; keep that in a separate generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

_MASK = (1 << 64) - 1
ALGORITHM_ID = "splitmix64-ctr"
ALGORITHM_VERSION = 1


class RngError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


@dataclass(frozen=True)
class RandomRequest:
    purpose: str  # e.g. "combat:attacker", "combat:defender", "aa", "raid"
    upper_bound: int
    count: int


@dataclass
class SeededRng:
    seed: int
    stream: int = 0
    counter: int = 0

    algorithm_id = ALGORITHM_ID
    algorithm_version = ALGORITHM_VERSION

    def copy(self) -> "SeededRng":
        return SeededRng(self.seed, self.stream, self.counter)

    def _next64(self) -> int:
        base = (self.seed * 0x100000001B3 + self.stream * 0xC2B2AE3D27D4EB4F) & _MASK
        value = _splitmix64((base + self.counter * 0x9E3779B97F4A7C15) & _MASK)
        self.counter += 1
        return value

    def draw(self, request: RandomRequest) -> List[int]:
        bound = request.upper_bound
        if bound <= 0:
            raise RngError("RNG_BAD_REQUEST", f"upper bound {bound}")
        limit = (1 << 64) - ((1 << 64) % bound)
        out = []
        for _ in range(request.count):
            while True:
                v = self._next64()
                if v < limit:
                    out.append(v % bound)
                    break
        return out

    def to_json(self):
        return {"algorithm": ALGORITHM_ID, "version": ALGORITHM_VERSION, "seed": str(self.seed),
                "stream": self.stream, "counter": str(self.counter)}


@dataclass
class TapeRng:
    """Replays recorded (purpose, upper_bound, results) entries in order."""

    entries: Sequence[Tuple[Optional[str], int, Tuple[int, ...]]]
    position: int = 0
    check_purpose: bool = False

    def copy(self) -> "TapeRng":
        return TapeRng(self.entries, self.position, self.check_purpose)

    def draw(self, request: RandomRequest) -> List[int]:
        if self.position >= len(self.entries):
            raise RngError("RNG_EXHAUSTED", f"no tape entry for {request}")
        purpose, bound, results = self.entries[self.position]
        if bound != request.upper_bound or len(results) != request.count or \
                (self.check_purpose and purpose is not None and purpose != request.purpose):
            raise RngError("RNG_TAPE_MISMATCH", f"tape has {(purpose, bound, len(results))}, got {request}")
        self.position += 1
        return list(results)


class RecordingRng:
    """Wraps another source and records requests/outcomes (for chance transcripts)."""

    def __init__(self, inner):
        self.inner = inner
        self.log: List[Tuple[str, int, Tuple[int, ...]]] = []

    def copy(self):
        r = RecordingRng(self.inner.copy())
        r.log = list(self.log)
        return r

    def draw(self, request: RandomRequest) -> List[int]:
        out = self.inner.draw(request)
        self.log.append((request.purpose, request.upper_bound, tuple(out)))
        return out
