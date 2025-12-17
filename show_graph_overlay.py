import os
import osmnx as ox
import folium

def main():
    graph_path = os.path.join("data", "cusco_city_drive.graphml")
    out_html = "cusco_city_graph_overlay.html"

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"No encontré {graph_path}. Ejecuta primero 01_download_cusco_city_graph.py")

    G = ox.load_graphml(graph_path)

    # GeoDataFrame de aristas (calles)
    _, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True, fill_edge_geometry=True)

    # bbox para centrar y encuadrar el mapa
    minx, miny, maxx, maxy = edges_gdf.total_bounds  # lon_min, lat_min, lon_max, lat_max
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Dibujar TODAS las calles del grafo
    folium.GeoJson(
        edges_gdf.to_json(),
        name="Grafo (calles de Cusco)",
        style_function=lambda feature: {
            "weight": 1,
            "opacity": 0.6,
        },
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    m.save(out_html)

    print(f"OK -> Mapa con grafo: {out_html}")

if __name__ == "__main__":
    main()
