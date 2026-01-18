from __future__ import annotations
import networkx as nx

from framework import Framework
from models import PageRankSettings, DistanceMetricsSettings
from enums import SortMethod
from utils.create_argparse import create_argparse
from weight_matching.minimum_weight_matching_greedy import minimum_weight_matching_greedy


def main() -> None:
    parser = create_argparse()
    args = parser.parse_args()

    graph_a: nx.Graph[object] = nx.read_edgelist(args.edgelist_a)  # type: ignore
    graph_b: nx.Graph[object] = nx.read_edgelist(args.edgelist_b)  # type: ignore

    page_rank_settings = PageRankSettings(is_sparse=False, max_horizon=5, return_alpha=0.05)
    distance_metrics_settings = DistanceMetricsSettings(amortization_sigma=0.8, main_distance_importance_gamma=0.25)

    instance = Framework(
        graph_a=graph_a,  # type: ignore
        graph_b=graph_b,  # type: ignore
        sort_method=SortMethod.EGO_NET_1,
        minimum_weight_matching=minimum_weight_matching_greedy,
        page_rank_settings=page_rank_settings,
        distance_metrics_settings=distance_metrics_settings,
    )

    progress = instance.execute(
        progress_filename=args.progress_output,
        matching_filename=args.matching_output,
        is_verbose=args.is_verbose,
    )

    correct = progress[-1].correct_matches
    total = progress[-1].matching_size
    accuracy = progress[-1].match_ratio
    print(f"Final result: {correct}/{total} correct matches ({accuracy:.2f} accuracy)")


if __name__ == "__main__":
    main()
