from heapq import heapify, heappop

from models import WeightedEdge, NodePair


def minimum_weight_matching_greedy(edges: list[WeightedEdge], necessary_edges: int) -> list[NodePair]:
    heapify(edges)

    match: list[NodePair] = []
    marking_a: set[int] = set()
    marking_b: set[int] = set()

    while len(match) < necessary_edges:
        edge = heappop(edges)
        if (
            edge.node_pair.node_a not in marking_a
            and edge.node_pair.node_b not in marking_b
        ):
            marking_a.add(edge.node_pair.node_a)
            marking_b.add(edge.node_pair.node_b)
            match.append(edge.node_pair)
    return match
