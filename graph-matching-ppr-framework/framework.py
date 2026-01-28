from __future__ import annotations
import concurrent.futures
import multiprocessing
from pathlib import Path
from typing import Callable, Tuple
import networkx as nx
import numpy as np

from enums import SortMethod
from models import (
    DistanceMetricsSettings,
    PageRankSettings,
    MatchingProgress,
    WeightedEdge,
    DistanceAmounts,
    NodePair,
)
from ppr import PPR
from distance_metrics import DistanceMetrics
from sort_methods.sort_nodes_by_degree import sort_nodes_by_degree
from sort_methods.sort_nodes_by_ppr_t2 import sort_nodes_by_ppr_t2
from sort_methods.sort_nodes_by_ego_net_1 import sort_nodes_by_ego_net_1
from protocols.minimum_weight_mathing_func import MinimumWeightMatchingFunc


class Framework:
    def __init__(
        self,
        graph_a: nx.Graph[object],
        graph_b: nx.Graph[object],
        sort_method: SortMethod,
        minimum_weight_matching: MinimumWeightMatchingFunc,
        page_rank_settings: PageRankSettings,
        distance_metrics_settings: DistanceMetricsSettings,
        oracle_order: list[object] | None = None,
    ):
        ppr_graph_a = PPR(graph=graph_a, settings=page_rank_settings)
        ppr_graph_b = PPR(graph=graph_b, settings=page_rank_settings)

        self.distance_metrics = DistanceMetrics(
            ppr_graph_a=ppr_graph_a, ppr_graph_b=ppr_graph_b, settings=distance_metrics_settings
        )

        self.final_matching_size = min(ppr_graph_a.number_of_nodes, ppr_graph_b.number_of_nodes)

        self.order_nodes_a = self._resolve_node_order(graph_a, ppr_graph_a, sort_method, oracle_order)
        self.order_nodes_b = self._resolve_node_order(graph_b, ppr_graph_b, sort_method, oracle_order)

        self.minimum_weight_matching = minimum_weight_matching

        self.anchors_a: list[int] = []
        self.anchors_b: list[int] = []

        self.round_h = 1
        self.number_of_candidates: int = 0
        self.candidates_a: list[int] = []
        self.candidates_b: list[int] = []

        self.cpu_count = multiprocessing.cpu_count()

    def _resolve_node_order(
        self, graph: nx.Graph[object], ppr: PPR, sort_method: SortMethod, oracle_order: list[object] | None
    ) -> list[int]:
        # validation regarding oracle order
        if sort_method == SortMethod.ORACLE and oracle_order is None:
            raise ValueError("oracle_order must be provided when using ORACLE sort method.")
        if sort_method != SortMethod.ORACLE and oracle_order is not None:
            raise ValueError("oracle_order should not be provided when not using ORACLE sort method.")

        sort_strategies: dict[SortMethod, Callable[[], list[int]]] = {
            SortMethod.ORACLE: lambda: [ppr.node_to_id[node] for node in (oracle_order or [])],
            SortMethod.PPR_T2: lambda: sort_nodes_by_ppr_t2(ppr_horizon_2=ppr.ppr_values[1]),
            SortMethod.EGO_NET_1: lambda: sort_nodes_by_ego_net_1(graph=graph, id_to_node=ppr.id_to_node),
            SortMethod.DEGREE: lambda: sort_nodes_by_degree(graph=graph, id_to_node=ppr.id_to_node),
        }

        if sort_method not in sort_strategies:
            raise ValueError(f"Unsupported sort method: {sort_method}")

        sort_func = sort_strategies[sort_method]
        return sort_func()

    def _update_number_of_candidates(self):
        self.number_of_candidates = min(2 ** (self.round_h), self.final_matching_size)

    def _update_nodes_candidates(self):
        self.candidates_a = self.order_nodes_a[: self.number_of_candidates]
        self.candidates_b = self.order_nodes_b[: self.number_of_candidates]

    def _update_anchors(self, matching: list[NodePair]) -> None:
        self.anchors_a = [pair.node_a for pair in matching]
        self.anchors_b = [pair.node_b for pair in matching]

    def _compute_distances_for_node_a(self, node_a: int) -> list[DistanceAmounts]:
        return [
            self.distance_metrics.node_distance_amounts(
                node_pair=NodePair(node_a, node_b),
                anchors_a=self.anchors_a,
                anchors_b=self.anchors_b,
            )
            for node_b in self.candidates_b
        ]

    def _compute_distances_for_all_nodes(self) -> list[list[DistanceAmounts]]:
        self.distance_metrics.anchors_confidence_cache( # precompute anchors confidence to avoid recomputing
            anchors_a=self.anchors_a,
            anchors_b=self.anchors_b
        )
        distance_results: list[list[DistanceAmounts]] = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            distance_results = list(
                executor.map(
                    self._compute_distances_for_node_a,
                    self.candidates_a,
                    chunksize=len(self.candidates_a) // self.cpu_count or 1,
                )
            )
        return distance_results

    def _normalize_distances(self, distance_results: list[list[DistanceAmounts]]) -> Tuple[np.ndarray, np.ndarray]:
        main_distances = np.array(
            [[distance.main_distance for distance in node_distances] for node_distances in distance_results],
            dtype=np.float64
        )
        anchor_distances = np.array(
            [[distance.anchor_distance for distance in node_distances] for node_distances in distance_results],
            dtype=np.float64
        )

        max_main_distance = np.max(main_distances)
        max_anchor_distance = np.max(anchor_distances)

        main_distances = main_distances / (np.float64(1.0) if max_main_distance == 0 else max_main_distance)
        anchor_distances = anchor_distances / (np.float64(1.0) if max_anchor_distance == 0 else max_anchor_distance)

        return main_distances, anchor_distances

    def _make_weighted_edge(
        self,
        node_a: int,
        node_b: int,
        main_distance: np.float64,
        anchor_distance: np.float64
    ) -> WeightedEdge:
        node_pair = NodePair(node_a, node_b)
        combined_distance = self.distance_metrics.node_distance_combine_amounts(
            DistanceAmounts(main_distance, anchor_distance)
        )
        return WeightedEdge(node_pair=node_pair, weight=combined_distance)

    def _create_bipartite_graph_edges(self) -> list[WeightedEdge]:
        edges: list[WeightedEdge] = []
        distance_results = self._compute_distances_for_all_nodes()
        main_distances, anchor_distances = self._normalize_distances(distance_results)
        for node_a, main_row, anchor_row in zip(self.candidates_a, main_distances, anchor_distances):
            for node_b, normalized_main, normalized_anchor in zip(self.candidates_b, main_row, anchor_row):
                edges.append(self._make_weighted_edge(node_a, node_b, normalized_main, normalized_anchor))
        return edges

    def _evaluate_matching(self, matching: list[NodePair]) -> int:
        def is_match(pair: NodePair) -> bool:
            node_a = self.distance_metrics.ppr_graph_a.id_to_node[pair.node_a]
            node_b = self.distance_metrics.ppr_graph_b.id_to_node[pair.node_b]
            return node_a == node_b

        return sum(is_match(pair) for pair in matching)

    def _max_possible_matches(self, expected_matching_size: int) -> int:
        nodes_candidates_a = {
            self.distance_metrics.ppr_graph_a.id_to_node[node_id]
            for node_id in self.candidates_a
        }
        nodes_candidates_b = {
            self.distance_metrics.ppr_graph_b.id_to_node[node_id]
            for node_id in self.candidates_b
        }
        common_nodes = nodes_candidates_a.intersection(nodes_candidates_b)
        return min(len(common_nodes), expected_matching_size)

    def _matching_progress(self, matching: list[NodePair], expected_matching_size: int) -> MatchingProgress:
        correct_matches = self._evaluate_matching(matching)
        max_possible_matches = self._max_possible_matches(expected_matching_size)
        match_ratio = np.float64(correct_matches) / len(matching)
        return MatchingProgress(
            correct_matches=correct_matches,
            matching_size=len(matching),
            max_possible_matches=max_possible_matches,
            match_ratio=match_ratio,
        )

    def _save_progress_file(self, filename: Path, progress: list[MatchingProgress]):
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"{MatchingProgress.header()}\n")
            progress_lines = [f"{p.to_csv()}\n" for p in progress]
            file.writelines(progress_lines)

    def _save_mathing_file(self, filename: Path, matching: list[NodePair]):
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as file:
            for pair in matching:
                node_a = self.distance_metrics.ppr_graph_a.id_to_node[pair.node_a]
                node_b = self.distance_metrics.ppr_graph_b.id_to_node[pair.node_b]
                file.write(f"{node_a} {node_b}\n")

    def execute(
        self,
        progress_filename: Path | None = None,
        matching_filename: Path | None = None,
        max_rounds: int | None = None,
        is_verbose: bool = False,
    ) -> list[MatchingProgress]:
        progress: list[MatchingProgress] = []

        if is_verbose:
            print(MatchingProgress.header())

        while True:
            self._update_number_of_candidates()
            self._update_nodes_candidates()

            is_last_round = self.number_of_candidates == self.final_matching_size
            expected_matching_size = self.number_of_candidates if is_last_round else self.number_of_candidates // 2

            bipartite_graph_edges = self._create_bipartite_graph_edges()
            matching = self.minimum_weight_matching(bipartite_graph_edges, expected_matching_size)

            progress.append(self._matching_progress(matching, expected_matching_size))

            if is_verbose:
                print(f"{progress[-1].to_csv()}")

            if is_last_round or (max_rounds is not None and self.round_h >= max_rounds):
                if progress_filename is not None:
                    self._save_progress_file(progress_filename, progress)
                if matching_filename is not None:
                    self._save_mathing_file(matching_filename, matching)
                return progress

            self._update_anchors(matching)
            self.round_h += 1
