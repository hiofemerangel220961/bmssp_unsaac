import os
import time
import heapq
import osmnx as ox
import folium


def nearest_node_bruteforce(G, lat, lon):
    best = None
    best_d2 = float("inf")
    for n, data in G.nodes(data=True):
        dy = data["y"] - lat
        dx = data["x"] - lon
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = n
    return best


def dijkstra(G, source, target, weight="length"):
    dist = {source: 0.0}
    parent = {source: None}
    pq = [(0.0, source)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for v in G.successors(u):
            edges_dict = G.get_edge_data(u, v)
            if not edges_dict:
                continue

            best_w = None
            for _, attr in edges_dict.items():
                w = attr.get(weight, None)
                if w is None:
                    continue
                w = float(w)
                if best_w is None or w < best_w:
                    best_w = w

            if best_w is None:
                continue

            nd = d + best_w
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
    return dist[target], path


def benchmark_once(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000


def main():
    graph_path = os.path.join("data", "cusco_city_drive.graphml")
    out_html = "cusco_10_routes_dijkstra.html"
    out_csv = "cusco_10_routes_dijkstra_metrics.csv"

    if not os.path.exists(graph_path):
        raise FileNotFoundError("No encontré el grafo")

    G = ox.load_graphml(graph_path)

    # --------- DEFINE AQUÍ TUS 10 CAMINOS (lat, lon) ----------
    # ("NombreRuta", (lat_ini, lon_ini), (lat_fin, lon_fin))
    ROUTES = [
        ("R1", (-13.5352850, -71.9274490), (-13.5167684, -71.9788000)),
        ("R2", (-13.5167684, -71.9788000), (-13.5046639, -71.9759972)),
        ("R3", (-13.5155000, -71.9805000), (-13.5312000, -71.9448000)),
        ("R4", (-13.5220000, -71.9760000), (-13.5160000, -71.9900000)),
        ("R5", (-13.5195000, -71.9720000), (-13.5085000, -71.9820000)),
        ("R6", (-13.5312000, -71.9448000), (-13.5046639, -71.9759972)),
        ("R7", (-13.5150000, -71.9850000), (-13.5100000, -71.9600000)),
        ("R8", (-13.5200000, -71.9650000), (-13.5350000, -71.9800000)),
        ("R9", (-13.5250000, -71.9800000), (-13.5120000, -71.9850000)),
        ("R10", (-13.5080000, -71.9750000), (-13.5210000, -71.9900000)),
    ]
    # ----------------------------------------------------------

    # Overlay del grafo
    _, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True, fill_edge_geometry=True)

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

    COLORS = ["red", "blue", "green", "purple", "orange", "black", "darkred", "cadetblue", "darkgreen", "pink"]

    results = []

    for idx, (name, (s_lat, s_lon), (e_lat, e_lon)) in enumerate(ROUTES, start=1):
        orig = nearest_node_bruteforce(G, s_lat, s_lon)
        dest = nearest_node_bruteforce(G, e_lat, e_lon)

        (total_meters, route_nodes), alg_ms = benchmark_once(
            lambda: dijkstra(G, orig, dest, weight="length")
        )

        if not route_nodes:
            print(f"{name}: SIN RUTA")
            continue

        nodes_count = len(route_nodes)
        dist_km = total_meters / 1000.0
        color = COLORS[(idx - 1) % len(COLORS)]

        route_latlon = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route_nodes]

        fg = folium.FeatureGroup(name=f"{name} ({color})", show=True)

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

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("route,time_ms,nodes,distance_km\n")
        for r in results:
            f.write(f"{r['route']},{r['time_ms']},{r['nodes']},{r['distance_km']}\n")

    print("\nOK ->", out_html)
    print("CSV ->", out_csv)
    print("Si el HTML sale blanco: baja MAX_EDGES_TO_DRAW y abre con: py -m http.server 8000")


if __name__ == "__main__":
    main()
