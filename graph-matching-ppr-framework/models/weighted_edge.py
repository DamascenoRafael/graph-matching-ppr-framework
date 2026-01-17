from dataclasses import dataclass, field
import numpy as np

from .node_pair import NodePair


@dataclass(order=True)
class WeightedEdge:
    node_pair: NodePair = field(compare=False)
    weight: np.float64
