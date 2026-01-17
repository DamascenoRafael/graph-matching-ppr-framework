from __future__ import annotations
import networkx as nx


def sort_nodes_by_ego_net_1(
    graph: nx.Graph[object],
    id_to_node: dict[int, object]
) -> list[int]:
    number_of_nodes = graph.number_of_nodes()
    ego_net_1_values: list[int] = []

    for node_id in range(number_of_nodes):
        node = id_to_node[node_id]
        neighbors = list(graph.neighbors(node))
        neighbors.append(node)
        subgraph = graph.subgraph(neighbors) # faster than using nx.ego_graph
        number_of_edges = len(subgraph.edges)
        ego_net_1_values.append(number_of_edges)

    return sorted(range(number_of_nodes), key=lambda node_id: ego_net_1_values[node_id], reverse=True)
