import os
import osmnx as ox


def main():
    # Lugar: Cusco ciudad
    place = "Cusco, Cusco, Peru"

    # Para empezar es más estable (menos data que walk)
    network_type = "drive"  # cambia a "walk" si luego quieres más detalle peatonal

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, "cusco_city_drive.graphml")

    # Configuración útil
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.timeout = 600  # tolera servidores lentos

    # Si ya existe el archivo, no descargar de nuevo
    if os.path.exists(out_file):
        print(f"✅ Ya existe: {out_file}")
        print("Si quieres re-descargar, bórralo primero.")
        return

    print(f"Descargando grafo para: {place} | network_type={network_type}")

    G = ox.graph_from_place(
        place,
        network_type=network_type,
        simplify=True,
        retain_all=False,
        truncate_by_edge=True
    )

    ox.save_graphml(G, out_file)

    print(f"\n✅ OK -> Guardado en: {out_file}")
    print(f"Nodos: {len(G.nodes):,} | Aristas: {len(G.edges):,}")


if __name__ == "__main__":
    main()
