import os
import osmnx as ox
import networkx as nx


def main():
    in_file = os.path.join("data", "cusco_city_drive.graphml")
    out_file = os.path.join("data", "cusco_city_undirected.graphml")

    if not os.path.exists(in_file):
        raise FileNotFoundError(
            f"No existe {in_file}. Primero ejecuta: py src/01_download_cusco_city.py"
        )

    # Si ya existe, no reconvertir
    if os.path.exists(out_file):
        print(f"✅ Ya existe: {out_file}")
        print("Si quieres reconvertir, bórralo primero.")
        return

    print(f"Cargando grafo dirigido: {in_file}")
    G = ox.load_graphml(in_file)

    print("Convirtiendo a grafo NO dirigido (NetworkX)...")
    # MultiDiGraph -> MultiGraph (no dirigido)
    Gu = G.to_undirected(as_view=False)

    # Quedarse con la componente conexa más grande
    print("Extrayendo la componente conexa más grande...")
    largest_cc = max(nx.connected_components(Gu), key=len)
    Gu = Gu.subgraph(largest_cc).copy()

    print(f"Guardando: {out_file}")
    ox.save_graphml(Gu, out_file)

    print(f"\n✅ OK -> Guardado: {out_file}")
    print(f"Nodos: {len(Gu.nodes):,} | Aristas: {len(Gu.edges):,}")


if __name__ == "__main__":
    main()