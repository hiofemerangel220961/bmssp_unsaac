import os
import random
from typing import Any, Dict, List, Tuple

import osmnx as ox
import folium


GRAPH_PATH = os.path.join("data", "cusco_city_undirected.graphml")
RESULTS_LONG = os.path.join("data", "results_long.csv")

OUT_DIR = "maps"
SAMPLE_PER_TYPE = 10   # 10 short + 10 medium + 10 long = 30 rutas
SEED = 7

# Colores (folium acepta strings)
COLORS = [
    "red", "blue", "green", "purple", "orange",
    "black", "darkred", "cadetblue", "darkgreen", "pink",
    "gray", "darkblue", "lightblue", "lightgreen", "beige"
]


def read_results_long(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            r = dict(zip(header, parts))

            # parse básicos
            r["algo"] = r["algo"]
            r["type"] = r["type"]
            r["u"] = int(r["u"])
            r["v"] = int(r["v"])
            r["ok"] = int(r["ok"])

            # algunos pueden venir vacíos
            r["time_ms"] = float(r["time_ms"]) if r["time_ms"] else None
            r["dist_km"] = float(r["dist_km"]) if r["dist_km"] else None
            r["path_nodes"] = int(r["path_nodes"]) if r["path_nodes"] else 0

            rows.append(r)
    return rows


def stratified_sample(rows: List[Dict[str, Any]], per_type: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)

    out = []
    for t, items in by_type.items():
        if len(items) <= per_type:
            out.extend(items)
        else:
            out.extend(random.sample(items, per_type))
    return out


def build_base_map(G) -> Tuple[folium.Map, Tuple[float, float, float, float]]:
    _, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True, fill_edge_geometry=True)

    minx, miny, maxx, maxy = edges_gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Dibujar calles (muestreo de edges para no pesar)
    MAX_EDGES_TO_DRAW = 25000
    edges_draw = edges_gdf.sample(n=min(len(edges_gdf), MAX_EDGES_TO_DRAW), random_state=1)

    folium.GeoJson(
        edges_draw.to_json(),
        name="Calles (grafo)",
        style_function=lambda feature: {"weight": 1, "opacity": 0.45},
    ).add_to(m)

    return m, (miny, minx, maxy, maxx)


def reconstruct_route_nodes(G, u: Any, v: Any, weight: str = "length") -> List[Any]:
    """
    Recalcula el camino para dibujarlo (así no dependes de haber guardado el path).
    """
    import networkx as nx
    try:
        return nx.shortest_path(G, u, v, weight=weight, method="dijkstra")
    except Exception:
        return []


def make_map_for_algo(G, items: List[Dict[str, Any]], algo: str, out_html: str):
    m, (miny, minx, maxy, maxx) = build_base_map(G)

    for idx, r in enumerate(items, start=1):
        u, v = r["u"], r["v"]
        ttype = r["type"]
        dist_km = r.get("dist_km")
        time_ms = r.get("time_ms")
        nodes_count = r.get("path_nodes", 0)

        path = reconstruct_route_nodes(G, u, v, weight="length")
        if not path:
            continue

        color = COLORS[(idx - 1) % len(COLORS)]
        route_latlon = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

        fg = folium.FeatureGroup(name=f"{ttype.upper()} #{idx}", show=True)

        # Inicio / fin
        folium.CircleMarker(
            location=(G.nodes[u]["y"], G.nodes[u]["x"]),
            radius=5,
            popup=f"{algo} | {ttype} | Inicio",
            tooltip=f"Inicio {ttype} #{idx}",
            fill=True,
            fill_opacity=1.0,
        ).add_to(fg)

        folium.CircleMarker(
            location=(G.nodes[v]["y"], G.nodes[v]["x"]),
            radius=5,
            popup=f"{algo} | {ttype} | Fin",
            tooltip=f"Fin {ttype} #{idx}",
            fill=True,
            fill_opacity=1.0,
        ).add_to(fg)

        tip = f"{algo} | {ttype} | "
        if dist_km is not None:
            tip += f"{dist_km:.3f} km | "
        tip += f"{nodes_count} nodos | "
        if time_ms is not None:
            tip += f"{time_ms:.2f} ms"

        folium.PolyLine(
            route_latlon,
            weight=6,
            opacity=0.9,
            color=color,
            tooltip=tip,
        ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    print("OK ->", out_html)


def main():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"No existe {GRAPH_PATH}")
    if not os.path.exists(RESULTS_LONG):
        raise FileNotFoundError(f"No existe {RESULTS_LONG}. Ejecuta 04_run_benchmarks primero.")

    print("Cargando grafo:", GRAPH_PATH)
    G = ox.load_graphml(GRAPH_PATH)

    rows = read_results_long(RESULTS_LONG)

    # Solo rutas OK
    rows_ok = [r for r in rows if r["ok"] == 1]

    # Un mapa por algoritmo
    algos = sorted(set(r["algo"] for r in rows_ok))

    for algo in algos:
        algo_rows = [r for r in rows_ok if r["algo"] == algo]

        # Muestreo estratificado por tipo (short/medium/long)
        sample = stratified_sample(algo_rows, per_type=SAMPLE_PER_TYPE, seed=SEED)

        out_html = os.path.join(OUT_DIR, f"map_{algo}.html")
        make_map_for_algo(G, sample, algo=algo, out_html=out_html)


if __name__ == "__main__":
    main()