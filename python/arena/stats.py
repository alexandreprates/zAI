"""Interval estimates shared by the trainers and the arena."""

import math


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion `p` observed over `n` trials."""
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half
