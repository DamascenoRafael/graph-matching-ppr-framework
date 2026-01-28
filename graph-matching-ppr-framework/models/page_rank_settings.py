from dataclasses import dataclass


@dataclass
class PageRankSettings:
    is_sparse: bool
    max_horizon: int
    return_alpha: float
