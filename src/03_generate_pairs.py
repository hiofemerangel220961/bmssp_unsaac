import os
import random
import time
from typing import Any, Dict, List, Tuple, Optional

import osmnx as ox

# Importa tu módulo Dijkstra (el de src/algorithms)
from src.algorithms.dijkstra_undirected import dijkstra_shortest_path


# =========================
# Config (ajusta a gusto)
# =========================

GRAPH_PATH = os.path.join("data", "cusco_city_undirected.graphml")
OUT_CSV = os.path.join("data", "pairs_generated.csv")

# Rangos por tipo (kilómetros)
DIST_RANGES_KM = {
    "short":  (0.2, 1.0),
    "medium": (1.0, 3.5),
    "long":   (3.5, 8.0),
}

# Prefiltro euclidiano (km) para no calcular rutas de más
EUCLID_FACTOR = 1.6  # ruta suele ser >= euclidiana (factor seguro)
MAX_TRIES_MULTIPLIER = 80  # intentos máximos = N * este factor (por tipo)


# =========================
# Utilidades
# =========================

def _euclid_km(G, u: Any, v: Any) -> float:
    uy, ux = G.nodes[u]["y"], G.nodes[u]["x"]
    vy, vx = G.nodes[v]["y"], G.nodes[v]["x"]

    # Aproximación rápida: grados -> km (bien para distancias locales)
    # lat: ~111 km/deg, lon: ~111*cos(lat)
    lat_km = 111.0
    lon_km = 111.0 * max(0.1, abs(__import__("math").cos(__import__("math").radians((uy + vy) / 2))))

    dy = (vy - uy) * lat_km
    dx = (vx - ux) * lon_km
    return (dx * dx + dy * dy) ** 0.5


def _route_km(G, u: Any, v: Any) -> Tuple[float, List[Any]]:
    dist_m, path = dijkstra_shortest_path(G, u, v, weight="length")
    if not path or dist_m == float("inf"):
        return float("inf"), []
    return dist_m / 1000.0, path


def _pick_pair_in_range(
    G,
    nodes: List[Any],
    km_min: float,
    km_max: float,
    max_tries: int,
) -> Optional[Tuple[Any, Any, float]]:
    """
    Retorna (u, v, dist_km) con distancia de ruta dentro del rango.
    """
    for _ in range(max_tries):
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u == v:
            continue

        # Prefiltro euclidiano: si ya supera el máximo (aprox), descartamos
        e_km = _euclid_km(G, u, v)
        if e_km > km_max:
            continue
        if e_km * EUCLID_FACTOR < km_min:
            continue

        d_km, path = _route_km(G, u, v)
        if not path:
            continue

        if km_min <= d_km <= km_max:
            return u, v, d_km

    return None


def generate_pairs(
    G,
    n_per_type: int,
    selected_types: List[str],
    seed: int = 7,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    nodes = list(G.nodes)

    rows: List[Dict[str, Any]] = []
    for t in selected_types:
        if t not in DIST_RANGES_KM:
            raise ValueError(f"Tipo inválido: {t}. Usa: {list(DIST_RANGES_KM.keys())}")

        km_min, km_max = DIST_RANGES_KM[t]
        max_tries = n_per_type * MAX_TRIES_MULTIPLIER

        print(f"\nGenerando {n_per_type} pares tipo={t} rango={km_min}-{km_max} km ...")
        created = 0
        tries_used = 0

        while created < n_per_type and tries_used < max_tries:
            tries_used += 1
            got = _pick_pair_in_range(G, nodes, km_min, km_max, max_tries=1)
            if got is None:
                continue

            u, v, d_km = got
            rows.append({
                "type": t,
                "u": u,
                "v": v,
                "distance_km": round(d_km, 5),
            })
            created += 1

            if created % max(1, n_per_type // 10) == 0:
                print(f"  {t}: {created}/{n_per_type}")

        if created < n_per_type:
            print(f"⚠️  Solo se generaron {created}/{n_per_type} para tipo={t}. "
                  f"Prueba ampliar rangos o aumentar MAX_TRIES_MULTIPLIER.")

    return rows


def save_csv(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("type,u,v,distance_km\n")
        for r in rows:
            f.write(f"{r['type']},{r['u']},{r['v']},{r['distance_km']}\n")


# =========================
# Main
# =========================

def main():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(
            f"No existe {GRAPH_PATH}. Primero genera el grafo undirected."
        )

    print("Cargando grafo:", GRAPH_PATH)
    G = ox.load_graphml(GRAPH_PATH)

    # ✅ Edita aquí lo que quieras:
    n_per_type = 50
    selected_types = ["short", "medium", "long"]
    seed = 7

    t0 = time.perf_counter()
    rows = generate_pairs(G, n_per_type=n_per_type, selected_types=selected_types, seed=seed)
    t1 = time.perf_counter()

    save_csv(rows, OUT_CSV)
    print("\nOK ->", OUT_CSV)
    print("Total pares:", len(rows))
    print("Tiempo:", round((t1 - t0), 2), "s")


if __name__ == "__main__":
    main()