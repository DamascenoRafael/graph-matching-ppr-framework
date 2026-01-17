import numpy as np

from ppr import PPR
from models import DistanceAmounts, DistanceMetricsSettings, NodePair
from utils.first_nonzero_index import first_nonzero_index


class DistanceMetrics:
    def __init__(
        self,
        ppr_graph_a: PPR,
        ppr_graph_b: PPR,
        settings: DistanceMetricsSettings,
    ):
        self.ppr_graph_a = ppr_graph_a
        self.ppr_graph_b = ppr_graph_b

        self.settings = settings
        self.max_horizon = ppr_graph_a.settings.max_horizon

        self._ppr_distance_cache_values: dict[NodePair, np.float64] = {}
        self._anchors_confidence_cache_values: dict[int, list[np.float64]] = {}

    def _base_ppr_value_distance_metric(
        self,
        ppr_node_a: np.float64,
        ppr_node_b: np.float64,
        amortization_power: int,
    ) -> np.float64:
        amortization_factor = np.power(self.settings.amortization_sigma, amortization_power)
        return (np.absolute(ppr_node_a - ppr_node_b) / np.maximum(ppr_node_a, ppr_node_b)) * amortization_factor

    def _base_ppr_sequence_distance_metric(
        self,
        ppr_sequence_a: list[np.float64],
        ppr_sequence_b: list[np.float64],
        initial_execution_t: int,
        amortization_t_offset: int,
    ) -> np.float64:
        sum_distance = np.float64(0.0)
        for execution_t in range(initial_execution_t, self.max_horizon + 1):
            sum_distance += self._base_ppr_value_distance_metric(
                ppr_node_a=ppr_sequence_a[execution_t],
                ppr_node_b=ppr_sequence_b[execution_t],
                amortization_power=execution_t - amortization_t_offset,
            )
        return sum_distance

    def _ppr_distance_cached(self, node_pair: NodePair) -> np.float64:
        if node_pair not in self._ppr_distance_cache_values:
            self._ppr_distance_cache_values[node_pair] = self._ppr_distance(node_pair)
        return self._ppr_distance_cache_values[node_pair]

    def _ppr_distance(self, node_pair: NodePair) -> np.float64:
        ppr_sequence_a = self.ppr_graph_a.ppr_sequence(
            point_of_view_node=node_pair.node_a,
            node_of_interest=node_pair.node_a,
        )
        ppr_sequence_b = self.ppr_graph_b.ppr_sequence(
            point_of_view_node=node_pair.node_b,
            node_of_interest=node_pair.node_b,
        )
        return self._base_ppr_sequence_distance_metric(
            ppr_sequence_a=ppr_sequence_a,
            ppr_sequence_b=ppr_sequence_b,
            initial_execution_t=2,
            amortization_t_offset=2,
        )

    def _ppr_distance_with_anchor(
        self,
        node_pair: NodePair,
        anchor_pair: NodePair,
    ) -> np.float64 | None:
        ppr_sequence_a = self.ppr_graph_a.ppr_sequence(
            point_of_view_node=node_pair.node_a,
            node_of_interest=anchor_pair.node_a,
        )
        ppr_sequence_b = self.ppr_graph_b.ppr_sequence(
            point_of_view_node=node_pair.node_b,
            node_of_interest=anchor_pair.node_b,
        )
        initial_execution_t = first_nonzero_index(ppr_sequence_a, ppr_sequence_b)
        if initial_execution_t is None:
            return None
        return self._base_ppr_sequence_distance_metric(
            ppr_sequence_a=ppr_sequence_a,
            ppr_sequence_b=ppr_sequence_b,
            initial_execution_t=initial_execution_t,
            amortization_t_offset=1,
        )

    def anchors_confidence_cache(
            self,
            anchors_a: list[int],
            anchors_b: list[int],
    ) -> list[np.float64]:
        anchors_size = len(anchors_a)
        if anchors_size not in self._anchors_confidence_cache_values:
            self._anchors_confidence_cache_values[anchors_size] = self._anchors_confidence(
                anchors_a=anchors_a,
                anchors_b=anchors_b,
            )
        return self._anchors_confidence_cache_values[anchors_size]

    def _anchors_confidence(
        self,
        anchors_a: list[int],
        anchors_b: list[int],
    ) -> list[np.float64]:
        epsilon = np.finfo(np.float64).eps  # pylint: disable=no-member # using epsilon to avoid division by zero
        inverse_anchor_ppr_distances: list[np.float64] = []
        sum_inverse_anchor_ppr_distances = np.float64(0.0)

        for anchor_a, anchor_b in zip(anchors_a, anchors_b):
            distance_epsilon = self._ppr_distance_cached(NodePair(node_a=anchor_a, node_b=anchor_b)) + epsilon
            inverse_distance = 1 / distance_epsilon
            inverse_anchor_ppr_distances.append(inverse_distance)
            sum_inverse_anchor_ppr_distances += inverse_distance

        anchors_confidence: list[np.float64] = []
        for inverse_distance in inverse_anchor_ppr_distances:
            confidence = inverse_distance / sum_inverse_anchor_ppr_distances
            anchors_confidence.append(confidence)
        return anchors_confidence

    def _ppr_distance_with_all_anchors(
        self,
        node_pair: NodePair,
        anchors_a: list[int],
        anchors_b: list[int],
    ) -> np.float64:
        ppr_distance_with_anchors = np.float64(0.0)
        anchors_confidence = self.anchors_confidence_cache(anchors_a, anchors_b)
        for anchor_a, anchor_b, anchor_confidence in zip(anchors_a, anchors_b, anchors_confidence):
            distance = self._ppr_distance_with_anchor(
                node_pair=node_pair, anchor_pair=NodePair(anchor_a, anchor_b)
            )
            if distance is not None:
                # ignore distance with anchor if distance is None (both sequences are zero)
                weighted_distance = distance * anchor_confidence
                ppr_distance_with_anchors += weighted_distance
        return ppr_distance_with_anchors

    def node_distance_amounts(
        self,
        node_pair: NodePair,
        anchors_a: list[int],
        anchors_b: list[int]
    ) -> DistanceAmounts:
        ppr_distance_a_b = self._ppr_distance_cached(node_pair)
        ppr_distance_with_anchors_a_b = self._ppr_distance_with_all_anchors(node_pair, anchors_a, anchors_b)
        return DistanceAmounts(main_distance=ppr_distance_a_b, anchor_distance=ppr_distance_with_anchors_a_b)

    def node_distance_combine_amounts(self, distance_amounts: DistanceAmounts) -> np.float64:
        return (
            distance_amounts.main_distance
            * self.settings.main_distance_importance_gamma
            + distance_amounts.anchor_distance
            * (1 - self.settings.main_distance_importance_gamma)
        )
