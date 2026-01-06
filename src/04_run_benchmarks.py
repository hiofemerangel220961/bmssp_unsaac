import os
import time
import statistics
from typing import Dict, Any, List, Tuple, Optional

import osmnx as ox

from src.algorithms.dijkstra_undirected import dijkstra_shortest_path
from src.algorithms.bellman_ford_undirected import bellman_ford_shortest_path
from src.algorithms.bmssp_undirected import bmssp_shortest_path, BmsspCache


GRAPH_PATH = os.path.join("data", "cusco_city_undirected.graphml")
PAIRS_CSV = os.path.join("data", "pairs_generated.csv")
OUT_LONG = os.path.join("data", "results_long.csv")
OUT_SUMMARY = os.path.join("data", "summary.csv")


# ---------------------------------------------------------
# Medición de memoria (opcional)
# ---------------------------------------------------------
def get_rss_mb() -> Optional[float]:
    try:
        import psutil
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


# ---------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------
def read_pairs_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            d = dict(zip(header, parts))
            d["type"] = d["type"]
            d["u"] = int(d["u"])
            d["v"] = int(d["v"])
            d["distance_km"] = float(d["distance_km"])
            rows.append(d)
    return rows


def write_long_csv(rows: List[Dict[str, Any]], path: str):
    def cell(x):
        return "" if x is None else str(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [
        "algo", "type", "u", "v",
        "time_ms", "cpu_ms", "rss_mb",
        "dist_km", "path_nodes", "ok"
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(cell(r.get(c, None)) for c in cols) + "\n")


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def write_summary_csv(rows: List[Dict[str, Any]], path: str):
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["algo"], r["type"])
        groups.setdefault(key, []).append(r)

    cols = [
        "algo", "type", "n",
        "time_ms_mean", "time_ms_p50", "time_ms_p95",
        "dist_km_mean", "nodes_mean",
        "rss_mb_mean"
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for (algo, t), items in sorted(groups.items()):
            ok_items = [x for x in items if x["ok"] == 1]
            times = [x["time_ms"] for x in ok_items]
            dists = [x["dist_km"] for x in ok_items]
            nodes = [x["path_nodes"] for x in ok_items]

            # ✅ SOLO valores numéricos
            rss = [x["rss_mb"] for x in ok_items if isinstance(x.get("rss_mb"), (int, float))]

            row = {
                "algo": algo,
                "type": t,
                "n": len(items),
                "time_ms_mean": round(statistics.mean(times), 3) if times else "",
                "time_ms_p50": round(percentile(times, 0.50), 3) if times else "",
                "time_ms_p95": round(percentile(times, 0.95), 3) if times else "",
                "dist_km_mean": round(statistics.mean(dists), 5) if dists else "",
                "nodes_mean": round(statistics.mean(nodes), 2) if nodes else "",
                "rss_mb_mean": round(statistics.mean(rss), 2) if rss else "",
            }

            f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")


# ---------------------------------------------------------
# Ejecutores por algoritmo
# ---------------------------------------------------------
def run_algo(G, algo_name: str, u: int, v: int, bmssp_cache: Optional[BmsspCache]) -> Tuple[float, List[int]]:
    if algo_name == "dijkstra":
        return dijkstra_shortest_path(G, u, v, weight="length")
    if algo_name == "bellman_ford":
        return bellman_ford_shortest_path(G, u, v, weight="length")
    if algo_name == "bmssp":
        return bmssp_shortest_path(G, u, v, cache=bmssp_cache, weight="length")
    raise ValueError("algo_name inválido")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"No existe {GRAPH_PATH}")
    if not os.path.exists(PAIRS_CSV):
        raise FileNotFoundError(f"No existe {PAIRS_CSV}. Ejecuta 03_generate_pairs primero.")

    print("Cargando grafo:", GRAPH_PATH)
    G = ox.load_graphml(GRAPH_PATH)

    pairs = read_pairs_csv(PAIRS_CSV)
    print("Pares cargados:", len(pairs))

    algos = ["dijkstra", "bellman_ford", "bmssp"]

    bmssp_cache = BmsspCache(G) if "bmssp" in algos else None

    results_long: List[Dict[str, Any]] = []

    for i, p in enumerate(pairs, start=1):
        ttype = p["type"]
        u = p["u"]
        v = p["v"]

        for algo in algos:
            cpu0 = time.process_time()
            t0 = time.perf_counter()

            dist_m, path = run_algo(G, algo, u, v, bmssp_cache)

            t1 = time.perf_counter()
            cpu1 = time.process_time()
            rss1 = get_rss_mb()  # ✅ tomar después (simple)

            ok = 1 if path and dist_m != float("inf") else 0
            dist_km = (dist_m / 1000.0) if ok else None
            path_nodes = len(path) if ok else 0

            results_long.append({
                "algo": algo,
                "type": ttype,
                "u": u,
                "v": v,
                "time_ms": round((t1 - t0) * 1000.0, 3),
                "cpu_ms": round((cpu1 - cpu0) * 1000.0, 3),
                "rss_mb": round(rss1, 2) if isinstance(rss1, (int, float)) else None,
                "dist_km": round(dist_km, 5) if isinstance(dist_km, (int, float)) else None,
                "path_nodes": path_nodes,
                "ok": ok,
            })

        if i % max(1, len(pairs)//10) == 0:
            print(f"Progreso: {i}/{len(pairs)}")

    write_long_csv(results_long, OUT_LONG)
    write_summary_csv(results_long, OUT_SUMMARY)

    print("\n✅ Listo:")
    print("  -", OUT_LONG)
    print("  -", OUT_SUMMARY)
    print("\nEjecuta:")
    print("py -3.13 -m src.04_run_benchmarks")


if __name__ == "__main__":
    main()
