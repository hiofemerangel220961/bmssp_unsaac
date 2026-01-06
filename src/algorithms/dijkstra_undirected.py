from __future__ import annotations
import heapq
from typing import Any, Dict, List, Tuple, Optional
import networkx as nx


def _best_edge_weight(G: nx.Graph, u: Any, v: Any, weight: str = "length") -> Optional[float]:
    data = G.get_edge_data(u, v)
    if data is None:
        return None

    # MultiGraph: dict key->attr
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

    # Graph simple
    w = data.get(weight)
    return float(w) if w is not None else None


def dijkstra_shortest_path(
    G: nx.Graph,
    source: Any,
    target: Any,
    weight: str = "length",
) -> Tuple[float, List[Any]]:
    dist: Dict[Any, float] = {source: 0.0}
    parent: Dict[Any, Any] = {source: None}
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for v in G.neighbors(u):
            w = _best_edge_weight(G, u, v, weight=weight)
            if w is None:
                continue
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if target not in dist:
        return float("inf"), []

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return float(dist[target]), path
