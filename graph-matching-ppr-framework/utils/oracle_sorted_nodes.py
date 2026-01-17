from __future__ import annotations
from typing import Callable
import networkx as nx

from ppr import PPR
from models import PageRankSettings
from enums import SortMethod
from sort_methods.sort_nodes_by_degree import sort_nodes_by_degree
from sort_methods.sort_nodes_by_ppr_t2 import sort_nodes_by_ppr_t2
from sort_methods.sort_nodes_by_ego_net_1 import sort_nodes_by_ego_net_1


def oracle_sorted_nodes(
    original_graph: nx.Graph[object],
    sort_method: SortMethod,
    ppr_settings: PageRankSettings
) -> list[object]:
    ppr = PPR(graph=original_graph, settings=ppr_settings)

    sort_strategies: dict[SortMethod, Callable[[], list[int]]] = {
        SortMethod.PPR_T2: lambda: sort_nodes_by_ppr_t2(ppr_horizon_2=ppr.ppr_values[1]),
        SortMethod.EGO_NET_1: lambda: sort_nodes_by_ego_net_1(graph=original_graph, id_to_node=ppr.id_to_node),
        SortMethod.DEGREE: lambda: sort_nodes_by_degree(graph=original_graph, id_to_node=ppr.id_to_node),
    }

    if sort_method not in sort_strategies:
        raise ValueError(f"Unsupported sort method: {sort_method}")

    sort_func = sort_strategies[sort_method]
    sorted_node_ids = sort_func()
    sorted_nodes = [ppr.id_to_node[node_id] for node_id in sorted_node_ids]

    return sorted_nodes
