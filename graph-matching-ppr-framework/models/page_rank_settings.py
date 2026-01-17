from dataclasses import dataclass


@dataclass
class PageRankSettings:
    is_sparse: bool = True
    max_horizon: int = 5
    return_alpha: float = 0.2
