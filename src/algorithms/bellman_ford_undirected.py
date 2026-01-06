from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import networkx as nx


def _best_edge_weight(G: nx.Graph, u: Any, v: Any, weight: str = "length") -> Optional[float]:
    data = G.get_edge_data(u, v)
    if data is None:
        return None

    if isinstance(data, dict) and any(isinstance(k, int) for k in data.keys()):
        best = None
        for _, attr in data.items():
            w = attr.get(weight)
            if w is None:
                continue
            w = float(w)
            if best is None or w < best:
                best = w
        return best

    w = data.get(weight)
    return float(w) if w is not None else None


def bellman_ford_shortest_path(
    G: nx.Graph,
    source: Any,
    target: Any,
    weight: str = "length",
) -> Tuple[float, List[Any]]:
    nodes = list(G.nodes)
    dist: Dict[Any, float] = {n: float("inf") for n in nodes}
    parent: Dict[Any, Any] = {n: None for n in nodes}
    dist[source] = 0.0

    edges = list(G.edges())

    for _ in range(len(nodes) - 1):
        changed = False
        for u, v in edges:
            w = _best_edge_weight(G, u, v, weight=weight)
            if w is None:
                continue

            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                changed = True

            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                parent[u] = v
                changed = True

        if not changed:
            break

    if dist[target] == float("inf"):
        return float("inf"), []

    path = []
    cur = target
    seen = set()
    while cur is not None:
        if cur in seen:
            break
        seen.add(cur)
        path.append(cur)
        if cur == source:
            break
        cur = parent[cur]
    path.reverse()

    if not path or path[0] != source:
        return float("inf"), []

    return float(dist[target]), path
