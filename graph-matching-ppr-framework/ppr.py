from __future__ import annotations
from random import shuffle
import networkx as nx
import numpy as np
from scipy import sparse

from models import PageRankSettings
from ppr_calculator import SparsePPRCalculator, DensePPRCalculator


class PPR:
    def __init__(self, graph: nx.Graph[object], settings: PageRankSettings):
        self.settings = settings
        self.number_of_nodes = graph.number_of_nodes()
        self.node_to_id, self.id_to_node = self._create_node_mapping(list(graph.nodes))
        self.ppr_values = self._calculate_ppr_values(graph)

    def _create_node_mapping(self, nodes: list[object]) -> tuple[dict[object, int], dict[int, object]]:
        node_to_id: dict[object, int] = {}
        id_to_node: dict[int, object] = {}
        shuffle(nodes) # prevents the framework from being biased by the ordered mapping of nodes
        for node in nodes:
            node_id = len(node_to_id)
            node_to_id[node] = node_id
            id_to_node[node_id] = node
        return node_to_id, id_to_node

    def _calculate_ppr_values(self, graph: nx.Graph[object]) -> list[np.ndarray] | list[sparse.csr_matrix]:
        calculator_class = SparsePPRCalculator if self.settings.is_sparse else DensePPRCalculator
        return calculator_class(
            graph=graph,
            node_to_id=self.node_to_id,
            id_to_node=self.id_to_node,
            settings=self.settings
        ).calculate()

    def ppr_value(self, point_of_view_node: int, node_of_interest: int, horizon_t: int) -> np.float64:
        if horizon_t == 0:
            return np.float64(1) if point_of_view_node == node_of_interest else np.float64(0)
        ppr_horizon_t = self.ppr_values[horizon_t - 1]
        return ppr_horizon_t[point_of_view_node, node_of_interest]

    def ppr_sequence(self, point_of_view_node: int, node_of_interest: int) -> list[np.float64]:
        sequence: list[np.float64] = []
        for t in range(self.settings.max_horizon + 1):
            ppr_value = self.ppr_value(
                point_of_view_node=point_of_view_node,
                node_of_interest=node_of_interest,
                horizon_t=t
            )
            sequence.append(ppr_value)
        return sequence
