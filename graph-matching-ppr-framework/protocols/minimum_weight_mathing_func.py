from typing import Protocol

from models import WeightedEdge, NodePair


class MinimumWeightMatchingFunc(Protocol):
    def __call__(
        self,
        edges: list[WeightedEdge],
        necessary_edges: int
    ) -> list[NodePair]:
        ...
