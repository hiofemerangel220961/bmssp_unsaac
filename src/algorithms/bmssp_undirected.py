from __future__ import annotations
import math
import heapq
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import networkx as nx


INFINITY = float("inf")


# ----------------- Estructuras internas -----------------

@dataclass
class Edge:
    to: int
    weight: float


class Graph:
    def __init__(self, vertices: int):
        self.vertices = vertices
        self.adj: List[List[Edge]] = [[] for _ in range(vertices)]

    def add_edge(self, u: int, v: int, weight: float):
        self.adj[u].append(Edge(v, weight))


def dijkstra_graph(graph: Graph, source: int, goal: int) -> Optional[Tuple[float, List[int]]]:
    n = graph.vertices
    dist = [INFINITY] * n
    parent = [None] * n
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == goal:
            break

        for e in graph.adj[u]:
            nd = d + e.weight
            if nd < dist[e.to]:
                dist[e.to] = nd
                parent[e.to] = u
                heapq.heappush(pq, (nd, e.to))

    if dist[goal] == INFINITY:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        if cur == source:
            break
        cur = parent[cur]
    path.reverse()

    if not path or path[0] != source:
        return None
    return dist[goal], path


class EfficientDataStructure:
    def __init__(self, block_size: int, bound: float):
        self.batch_blocks = deque()
        self.sorted_blocks: List[List[Tuple[int, float]]] = []
        self.block_size = block_size
        self.bound = bound

    def insert(self, vertex: int, distance: float):
        if distance < self.bound:
            if not self.sorted_blocks or len(self.sorted_blocks[-1]) >= self.block_size:
                self.sorted_blocks.append([])
            self.sorted_blocks[-1].append((vertex, distance))

    def batch_prepend(self, items: List[Tuple[int, float]]):
        if items:
            self.batch_blocks.appendleft(list(items))

    def peek_min(self) -> float:
        min_val = self.bound
        for block in list(self.batch_blocks) + self.sorted_blocks:
            if block:
                block_min = min(d for _, d in block)
                if block_min < min_val:
                    min_val = block_min
        return min_val

    def pull(self) -> Tuple[float, List[int]]:
        block_to_process = None

        if self.batch_blocks:
            block_to_process = self.batch_blocks.popleft()
        elif self.sorted_blocks:
            min_dist_in_blocks = [
                (min(d for _, d in block) if block else INFINITY)
                for block in self.sorted_blocks
            ]
            min_block_idx = min(range(len(min_dist_in_blocks)), key=min_dist_in_blocks.__getitem__)
            block_to_process = self.sorted_blocks.pop(min_block_idx)

        if block_to_process:
            block_to_process.sort(key=lambda p: p[1])
            vertices = [v for v, _ in block_to_process]
            next_bound = self.peek_min()
            return next_bound, vertices

        return self.bound, []

    def is_empty(self) -> bool:
        return not any(self.batch_blocks) and not any(self.sorted_blocks)


class BmsspSolver:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.n = graph.vertices

        self.k = int(math.log2(self.n) ** (1 / 3) * 2) if self.n > 1 else 1
        self.t = int(math.log2(self.n) ** (2 / 3)) if self.n > 1 else 1
        self.k = max(self.k, 3)
        self.t = max(self.t, 2)

        self.distances = [INFINITY] * self.n
        self.predecessors = [None] * self.n
        self.complete = [False] * self.n
        self.best_goal = INFINITY

    def solve(self, source: int, goal: int) -> Optional[Tuple[float, List[int]]]:
        self.distances = [INFINITY] * self.n
        self.predecessors = [None] * self.n
        self.complete = [False] * self.n
        self.best_goal = INFINITY
        self.distances[source] = 0.0

        if self.n < 1000:
            return dijkstra_graph(self.graph, source, goal)

        max_level = math.ceil(math.log2(self.n) / self.t) if self.n > 1 else 0
        self._bmssp(max_level, INFINITY, [source], goal)

        if self.distances[goal] == INFINITY:
            return None

        path = self._reconstruct_path(source, goal)
        return self.distances[goal], path

    def _reconstruct_path(self, source: int, goal: int) -> List[int]:
        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            if curr == source:
                break
            curr = self.predecessors[curr]
        return path[::-1]

    def _bmssp(self, level: int, bound: float, pivots: List[int], goal: Optional[int]) -> List[int]:
        if not pivots or (goal is not None and self.complete[goal]):
            return []

        if level == 0:
            return self._base_case(bound, pivots, goal)

        pivots, _ = self._find_pivots(bound, pivots)

        block_size = 2 ** max(0, (level - 1) * self.t)
        ds = EfficientDataStructure(block_size, bound)

        for pivot in pivots:
            if not self.complete[pivot] and self.distances[pivot] < bound:
                ds.insert(pivot, self.distances[pivot])

        result_set: List[int] = []

        while not ds.is_empty():
            if goal is not None and self.complete[goal]:
                break

            subset_bound, subset = ds.pull()
            if not subset:
                continue

            sub_result = self._bmssp(level - 1, subset_bound, subset, goal)
            result_set.extend(sub_result)
            self._edge_relaxation(sub_result, subset_bound, bound, ds)

        return result_set

    def _base_case(self, bound: float, frontier: List[int], goal: Optional[int]) -> List[int]:
        pq = []
        for start_node in frontier:
            if not self.complete[start_node] and self.distances[start_node] < bound:
                heapq.heappush(pq, (self.distances[start_node], start_node))

        completed_nodes: List[int] = []

        while pq:
            dist, u = heapq.heappop(pq)
            if self.complete[u] or dist > self.distances[u]:
                continue

            self.complete[u] = True
            completed_nodes.append(u)

            if u == goal:
                if dist < self.best_goal:
                    self.best_goal = dist
                break

            for edge in self.graph.adj[u]:
                new_dist = dist + edge.weight
                if (
                    (not self.complete[edge.to])
                    and new_dist <= self.distances[edge.to]
                    and new_dist < bound
                    and new_dist < self.best_goal
                ):
                    self.distances[edge.to] = new_dist
                    self.predecessors[edge.to] = u
                    heapq.heappush(pq, (new_dist, edge.to))

        return completed_nodes

    def _find_pivots(self, bound: float, frontier: List[int]) -> Tuple[List[int], List[int]]:
        working_set = set(frontier)
        current_layer = {node for node in frontier if not self.complete[node]}

        for _ in range(self.k):
            next_layer = set()
            for u in current_layer:
                if self.distances[u] >= bound:
                    continue

                for edge in self.graph.adj[u]:
                    v = edge.to
                    if self.complete[v]:
                        continue

                    new_dist = self.distances[u] + edge.weight
                    if new_dist <= self.distances[v] and new_dist < bound and new_dist < self.best_goal:
                        self.distances[v] = new_dist
                        self.predecessors[v] = u
                        if v not in working_set:
                            next_layer.add(v)

            if not next_layer:
                break

            working_set.update(next_layer)
            current_layer = next_layer

            if len(working_set) > self.k * len(frontier):
                return frontier, list(working_set)

        children: Dict[int, List[int]] = {node: [] for node in working_set}
        for node in working_set:
            pred = self.predecessors[node]
            if pred is not None and pred in working_set:
                children.setdefault(pred, []).append(node)

        subtree_sizes = {node: len(ch) for node, ch in children.items()}
        pivots = [root for root in frontier if subtree_sizes.get(root, 0) >= self.k]

        if not pivots:
            return frontier, list(working_set)

        return pivots, list(working_set)

    def _edge_relaxation(self, completed_vertices: List[int], lower_bound: float, upper_bound: float, ds: EfficientDataStructure):
        batch_prepend_list: List[Tuple[int, float]] = []

        for u in completed_vertices:
            for edge in self.graph.adj[u]:
                v = edge.to
                if self.complete[v]:
                    continue

                new_dist = self.distances[u] + edge.weight
                if new_dist <= self.distances[v] and new_dist < self.best_goal:
                    self.distances[v] = new_dist
                    self.predecessors[v] = u

                    if new_dist < lower_bound:
                        batch_prepend_list.append((v, new_dist))
                    elif new_dist < upper_bound:
                        ds.insert(v, new_dist)

        if batch_prepend_list:
            ds.batch_prepend(batch_prepend_list)


# ----------------- Conversión OSMnx -> Graph UNDIRECTED -----------------

def osm_to_custom_graph_min_edges_undirected(G_osm: nx.Graph, weight_attr: str = "length"):
    osm_nodes = list(G_osm.nodes())
    osm2i = {nid: i for i, nid in enumerate(osm_nodes)}
    i2osm = osm_nodes

    g = Graph(len(osm_nodes))
    best_uv: Dict[Tuple[int, int], float] = {}

    # MultiGraph o MultiDiGraph: edges(keys=True,data=True)
    for u, v, k, data in G_osm.edges(keys=True, data=True):
        w = data.get(weight_attr)
        if w is None:
            continue
        w = float(w)

        ui = osm2i[u]
        vi = osm2i[v]

        # u->v
        key1 = (ui, vi)
        prev = best_uv.get(key1)
        if prev is None or w < prev:
            best_uv[key1] = w

        # v->u
        key2 = (vi, ui)
        prev2 = best_uv.get(key2)
        if prev2 is None or w < prev2:
            best_uv[key2] = w

    for (ui, vi), w in best_uv.items():
        g.add_edge(ui, vi, w)

    return g, osm2i, i2osm


class BmsspCache:
    """
    Crea 1 vez (graph interno + solver) y se reutiliza en todas las pruebas.
    """
    def __init__(self, G_osm: nx.Graph, weight_attr: str = "length"):
        self.weight_attr = weight_attr
        self.custom_graph, self.osm2i, self.i2osm = osm_to_custom_graph_min_edges_undirected(G_osm, weight_attr)
        self.solver = BmsspSolver(self.custom_graph)


def bmssp_shortest_path(
    G_osm: nx.Graph,
    source_osm: Any,
    target_osm: Any,
    cache: Optional[BmsspCache] = None,
    weight: str = "length",
) -> Tuple[float, List[Any]]:
    """
    Firma estándar para el benchmark:
      - devuelve (dist_m, path_nodes_osm)
      - si no hay ruta: (inf, [])
    """
    if cache is None:
        cache = BmsspCache(G_osm, weight_attr=weight)

    s = cache.osm2i.get(source_osm)
    t = cache.osm2i.get(target_osm)
    if s is None or t is None:
        return float("inf"), []

    res = cache.solver.solve(s, t)
    if res is None:
        return float("inf"), []

    dist_m, path_idx = res
    path_osm = [cache.i2osm[i] for i in path_idx]
    return float(dist_m), path_osm