import os
import time
import math
import heapq
from collections import deque
from typing import Optional, Tuple, List, Dict

import osmnx as ox
import folium

INFINITY = float("inf")


# ============================================================
# 1) Estructuras (tu repo) - 1 sola carpeta
# ============================================================

class Edge:
    def __init__(self, to: int, weight: float):
        self.to = to
        self.weight = weight


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


# ============================================================
# 2) OSMnx -> tu Graph + nearest node rápido (numpy si existe)
# ============================================================

def build_node_index(G_osm):
    """
    Prepara arrays para buscar nearest node rápido (sin scipy/sklearn).
    Usa numpy si está disponible; si no, igual devuelve listas.
    """
    node_ids = []
    lats = []
    lons = []
    for nid, data in G_osm.nodes(data=True):
        node_ids.append(nid)
        lats.append(float(data["y"]))
        lons.append(float(data["x"]))

    try:
        import numpy as np
        return {"node_ids": node_ids, "lats": np.array(lats), "lons": np.array(lons), "np": np}
    except Exception:
        return {"node_ids": node_ids, "lats": lats, "lons": lons, "np": None}


def nearest_node_fast(index, lat, lon):
    """
    Nearest node usando numpy (rápido) o fallback brute force (lento).
    """
    np = index["np"]
    if np is not None:
        dy = index["lats"] - lat
        dx = index["lons"] - lon
        i = int(np.argmin(dx * dx + dy * dy))
        return index["node_ids"][i]

    best = None
    best_d2 = float("inf")
    for nid, y, x in zip(index["node_ids"], index["lats"], index["lons"]):
        dy = y - lat
        dx = x - lon
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = nid
    return best


def osm_to_custom_graph_min_edges(G_osm):
    osm_nodes = list(G_osm.nodes())
    osm2i = {nid: i for i, nid in enumerate(osm_nodes)}
    i2osm = osm_nodes

    g = Graph(len(osm_nodes))

    best_uv: Dict[Tuple[int, int], float] = {}

    for u, v, k, data in G_osm.edges(keys=True, data=True):
        w = data.get("length", None)
        if w is None:
            continue
        ui = osm2i[u]
        vi = osm2i[v]
        key = (ui, vi)
        w = float(w)
        prev = best_uv.get(key)
        if prev is None or w < prev:
            best_uv[key] = w

    for (ui, vi), w in best_uv.items():
        g.add_edge(ui, vi, w)

    return g, osm2i, i2osm


def benchmark_once(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000


# ============================================================
# 3) MAIN: 10 rutas + un mapa + métricas
# ============================================================

def main():
    graph_path = os.path.join("data", "cusco_city_drive.graphml")
    out_html = "cusco_10_routes_bmssp.html"
    out_csv = "cusco_10_routes_metrics.csv"

    if not os.path.exists(graph_path):
        raise FileNotFoundError("No encontré data/cusco_city_drive.graphml")

    G_osm = ox.load_graphml(graph_path)

    # --------- DEFINE AQUÍ TUS 10 CAMINOS (lat, lon) ----------
    # Formato: ("NombreRuta", (lat_ini, lon_ini), (lat_fin, lon_fin))
    ROUTES = [
        ("R1", (-13.5167684, -71.9788000), (-13.5046639, -71.9759972)),
        ("R2", (-13.5155000, -71.9805000), (-13.5312000, -71.9448000)),
        ("R3", (-13.5220000, -71.9760000), (-13.5160000, -71.9900000)),
        ("R4", (-13.5195000, -71.9720000), (-13.5085000, -71.9820000)),
        ("R5", (-13.5312000, -71.9448000), (-13.5046639, -71.9759972)),
        ("R6", (-13.5150000, -71.9850000), (-13.5100000, -71.9600000)),
        ("R7", (-13.5200000, -71.9650000), (-13.5350000, -71.9800000)),
        ("R8", (-13.5250000, -71.9800000), (-13.5120000, -71.9850000)),
        ("R9", (-13.5080000, -71.9750000), (-13.5210000, -71.9900000)),
        ("R10", (-13.5140000, -71.9700000), (-13.5300000, -71.9750000)),
    ]
    # ----------------------------------------------------------

    # Índice rápido de nodos (para nearest node)
    node_index = build_node_index(G_osm)

    # Conversión OSM->Graph (una sola vez)
    (custom_graph, osm2i, i2osm), conv_ms = benchmark_once(lambda: osm_to_custom_graph_min_edges(G_osm))
    solver = BmsspSolver(custom_graph)

    # Mapa base con calles
    _, edges_gdf = ox.graph_to_gdfs(G_osm, nodes=True, edges=True, fill_edge_geometry=True)

    MAX_EDGES_TO_DRAW = 25000
    if len(edges_gdf) > MAX_EDGES_TO_DRAW:
        edges_gdf_draw = edges_gdf.sample(n=MAX_EDGES_TO_DRAW, random_state=1)
    else:
        edges_gdf_draw = edges_gdf

    minx, miny, maxx, maxy = edges_gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    folium.GeoJson(
        edges_gdf_draw.to_json(),
        name="Calles (grafo)",
        style_function=lambda feature: {"weight": 1, "opacity": 0.5},
    ).add_to(m)

    # 10 colores (Folium acepta strings)
    COLORS = ["red", "blue", "green", "purple", "orange", "black", "darkred", "cadetblue", "darkgreen", "pink"]

    results = []

    for idx, (name, (s_lat, s_lon), (e_lat, e_lon)) in enumerate(ROUTES, start=1):
        # nearest osm nodes
        orig_osm = nearest_node_fast(node_index, s_lat, s_lon)
        dest_osm = nearest_node_fast(node_index, e_lat, e_lon)

        source = osm2i[orig_osm]
        goal = osm2i[dest_osm]

        # tiempo SOLO del algoritmo
        res, alg_ms = benchmark_once(lambda: solver.solve(source, goal))
        if res is None:
            print(f"{name}: SIN RUTA")
            continue

        dist_m, path_idx = res
        path_osm = [i2osm[i] for i in path_idx]
        nodes_count = len(path_idx)
        dist_km = dist_m / 1000.0

        # dibujar ruta
        color = COLORS[(idx - 1) % len(COLORS)]
        route_latlon = [(G_osm.nodes[n]["y"], G_osm.nodes[n]["x"]) for n in path_osm]

        fg = folium.FeatureGroup(name=f"{name} ({color})", show=True)

        # marcadores inicio/fin con numerito
        folium.CircleMarker(
            location=(s_lat, s_lon),
            radius=6,
            popup=f"{name} Inicio",
            tooltip=f"{name} Inicio",
            fill=True,
            fill_opacity=1.0,
        ).add_to(fg)

        folium.CircleMarker(
            location=(e_lat, e_lon),
            radius=6,
            popup=f"{name} Fin",
            tooltip=f"{name} Fin",
            fill=True,
            fill_opacity=1.0,
        ).add_to(fg)

        folium.PolyLine(
            route_latlon,
            weight=6,
            opacity=0.9,
            color=color,
            tooltip=f"{name}: {dist_km:.3f} km | {nodes_count} nodos | {alg_ms:.2f} ms",
        ).add_to(fg)

        fg.add_to(m)

        results.append({
            "route": name,
            "time_ms": round(alg_ms, 3),
            "nodes": nodes_count,
            "distance_km": round(dist_km, 5),
        })

        print(f"{name}: {alg_ms:.2f} ms | nodos={nodes_count} | dist={dist_km:.3f} km")

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    m.save(out_html)

    # Guardar CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("route,time_ms,nodes,distance_km\n")
        for r in results:
            f.write(f"{r['route']},{r['time_ms']},{r['nodes']},{r['distance_km']}\n")

    print("\n--- RESUMEN ---")
    print(f"Conversión OSM->Graph: {conv_ms:.2f} ms (1 vez)")
    print(f"HTML: {out_html}")
    print(f"CSV : {out_csv}")
    print("Si el HTML sale blanco: baja MAX_EDGES_TO_DRAW y/o abre con: py -m http.server 8000")


if __name__ == "__main__":
    main()