from dataclasses import dataclass


@dataclass(frozen=True)
class NodePair:
    node_a: int
    node_b: int
