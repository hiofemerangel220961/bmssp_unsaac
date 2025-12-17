import os
import osmnx as ox

def main():
    # Prueba primero este "place". Si no te da el límite correcto,
    # cambia a: "Cusco District, Cusco, Peru"
    place = "Cusco, Cusco, Peru"
    network_type = "drive"  # "walk" si quieres calles peatonales también

    out_dir = "data"
    out_file = os.path.join(out_dir, "cusco_city_drive.graphml")
    os.makedirs(out_dir, exist_ok=True)

    ox.settings.use_cache = True
    ox.settings.log_console = True

    print(f"Descargando grafo para: {place} (network_type={network_type}) ...")

    # truncate_by_edge=True ayuda a recortar mejor al borde del polígono
    G = ox.graph_from_place(
        place,
        network_type=network_type,
        simplify=True,
        retain_all=False,
        truncate_by_edge=True
    )

    ox.save_graphml(G, out_file)
    print(f"OK -> Guardado en: {out_file}")
    print(f"Nodos: {len(G.nodes):,} | Aristas: {len(G.edges):,}")

if __name__ == "__main__":
    main()
