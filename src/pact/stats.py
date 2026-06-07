"""Paired statistical tests using stdlib fallbacks."""

from __future__ import annotations

import math
import random
from collections import defaultdict

from pact.schema import Episode, Prediction
from pact.scoring import binary_successes


def cluster_bootstrap_diff(episodes: list[Episode], a: list[Prediction], b: list[Prediction], *, iters: int = 1000, seed: int = 0) -> dict[str, float]:
    rng = random.Random(seed)
    fams = sorted({ep.family for ep in episodes})
    by_fam = defaultdict(list)
    a_bits = dict(zip([ep.episode_id for ep in episodes], binary_successes(episodes, a)))
    b_bits = dict(zip([ep.episode_id for ep in episodes], binary_successes(episodes, b)))
    for ep in episodes:
        by_fam[ep.family].append(ep.episode_id)
    diffs = []
    for _ in range(iters):
        ids = []
        for _ in fams:
            ids.extend(by_fam[rng.choice(fams)])
        diffs.append(sum(a_bits[i] - b_bits[i] for i in ids) / len(ids))
    diffs.sort()
    return {"mean_diff": sum(diffs) / len(diffs), "ci_low": diffs[int(0.025 * len(diffs))], "ci_high": diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]}


def mcnemar(a_bits: list[int], b_bits: list[int]) -> dict[str, float]:
    b01 = sum(1 for a, b in zip(a_bits, b_bits) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(a_bits, b_bits) if a == 1 and b == 0)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p": 1.0}
    k = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1)))
    return {"b01": b01, "b10": b10, "p": p}


def permutation_test(a_bits: list[int], b_bits: list[int], *, iters: int = 1000, seed: int = 0) -> dict[str, float]:
    rng = random.Random(seed)
    observed = sum(a - b for a, b in zip(a_bits, b_bits)) / len(a_bits)
    count = 0
    for _ in range(iters):
        diff = 0
        for a, b in zip(a_bits, b_bits):
            if rng.random() < 0.5:
                a, b = b, a
            diff += a - b
        if abs(diff / len(a_bits)) >= abs(observed):
            count += 1
    return {"observed_diff": observed, "p": (count + 1) / (iters + 1)}


def holm(pairs: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pairs.items(), key=lambda item: item[1])
    out: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, p) in enumerate(ordered):
        adjusted = min(1.0, p * (m - rank))
        running = max(running, adjusted)
        out[name] = running
    return out

