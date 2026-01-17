from __future__ import annotations
import networkx as nx


def sort_nodes_by_degree(
    graph: nx.Graph[object],
    id_to_node: dict[int, object]
) -> list[int]:
    number_of_nodes = graph.number_of_nodes()
    degree_values: list[int] = []

    for node_id in range(number_of_nodes):
        node = id_to_node[node_id]
        value = graph.degree[node]
        degree_values.append(value)

    return sorted(range(number_of_nodes), key=lambda node_id: degree_values[node_id], reverse=True)
