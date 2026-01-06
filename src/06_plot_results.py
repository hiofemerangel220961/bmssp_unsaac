import os
import csv
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt


SUMMARY_CSV = os.path.join("data", "summary.csv")
LONG_CSV = os.path.join("data", "results_long.csv")
OUT_DIR = "reports"


def read_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def inum(x):
    try:
        return int(float(x))
    except Exception:
        return None


def save_bar_chart(x_labels, series_dict, title, ylabel, out_png):
    plt.figure()
    for name, vals in series_dict.items():
        plt.plot(range(len(x_labels)), vals, marker="o", label=name)
    plt.xticks(range(len(x_labels)), x_labels, rotation=0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_boxplot(groups: Dict[str, List[float]], title: str, ylabel: str, out_png: str):
    labels = list(groups.keys())
    data = [groups[k] for k in labels]
    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"No existe {SUMMARY_CSV}. Ejecuta 04_run_benchmarks primero.")
    if not os.path.exists(LONG_CSV):
        raise FileNotFoundError(f"No existe {LONG_CSV}. Ejecuta 04_run_benchmarks primero.")

    os.makedirs(OUT_DIR, exist_ok=True)

    summary = read_csv(SUMMARY_CSV)
    long_rows = read_csv(LONG_CSV)

    # ----------------------------
    # 1) Tabla resumen ordenada
    # ----------------------------
    out_table = os.path.join(OUT_DIR, "summary_table.csv")
    with open(out_table, "w", encoding="utf-8", newline="") as f:
        cols = [
            "algo","type","n",
            "time_ms_mean","time_ms_p50","time_ms_p95",
            "dist_km_mean","nodes_mean","rss_mb_mean"
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(summary, key=lambda x: (x["type"], x["algo"])):
            w.writerow({c: r.get(c, "") for c in cols})
    print("OK ->", out_table)

    # ----------------------------
    # 2) Gráfica p50/p95 por tipo
    # ----------------------------
    # Formato: para cada tipo, una curva por algoritmo
    types_order = ["short", "medium", "long"]
    algos = sorted(set(r["algo"] for r in summary))

    # p50
    series_p50 = {}
    for algo in algos:
        vals = []
        for t in types_order:
            row = next((r for r in summary if r["algo"] == algo and r["type"] == t), None)
            vals.append(fnum(row["time_ms_p50"]) if row else None)
        series_p50[algo] = vals

    out_p50 = os.path.join(OUT_DIR, "time_p50.png")
    save_bar_chart(types_order, series_p50, "Tiempo p50 (ms) por tipo de ruta", "ms", out_p50)
    print("OK ->", out_p50)

    # p95
    series_p95 = {}
    for algo in algos:
        vals = []
        for t in types_order:
            row = next((r for r in summary if r["algo"] == algo and r["type"] == t), None)
            vals.append(fnum(row["time_ms_p95"]) if row else None)
        series_p95[algo] = vals

    out_p95 = os.path.join(OUT_DIR, "time_p95.png")
    save_bar_chart(types_order, series_p95, "Tiempo p95 (ms) por tipo de ruta", "ms", out_p95)
    print("OK ->", out_p95)

    # ----------------------------
    # 3) Boxplot de tiempos (global)
    # ----------------------------
    # grupos: "algo|type"
    box_groups: Dict[str, List[float]] = defaultdict(list)
    for r in long_rows:
        if inum(r.get("ok")) != 1:
            continue
        t = r.get("type")
        a = r.get("algo")
        tm = fnum(r.get("time_ms"))
        if tm is None:
            continue
        box_groups[f"{a}-{t}"].append(tm)

    out_box = os.path.join(OUT_DIR, "time_boxplot.png")
    save_boxplot(box_groups, "Distribución de tiempos (ms) por algoritmo y tipo", "ms", out_box)
    print("OK ->", out_box)

    # ----------------------------
    # 4) Tasa de éxito (ok %)
    # ----------------------------
    success: Dict[Tuple[str, str], Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in long_rows:
        key = (r.get("algo"), r.get("type"))
        ok = inum(r.get("ok")) or 0
        a, b = success[key]
        success[key] = (a + ok, b + 1)

    # armar curvas por algoritmo
    series_ok = {}
    for algo in algos:
        vals = []
        for t in types_order:
            ok_count, total = success.get((algo, t), (0, 0))
            vals.append((ok_count / total) * 100.0 if total else 0.0)
        series_ok[algo] = vals

    out_ok = os.path.join(OUT_DIR, "success_rate.png")
    save_bar_chart(types_order, series_ok, "Tasa de éxito (%) por tipo de ruta", "%", out_ok)
    print("OK ->", out_ok)

    # ----------------------------
    # 5) Mini reporte markdown
    # ----------------------------
    out_md = os.path.join(OUT_DIR, "report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Reporte de Benchmark\n\n")
        f.write("## Archivos generados\n")
        f.write("- summary_table.csv\n- time_p50.png\n- time_p95.png\n- time_boxplot.png\n- success_rate.png\n\n")
        f.write("## Interpretación rápida\n")
        f.write("- **p50**: tiempo típico\n- **p95**: casos difíciles (cola)\n- **boxplot**: dispersión\n- **success rate**: % de rutas resueltas\n")
    print("OK ->", out_md)

    print("\n✅ Listo. Revisa la carpeta:", OUT_DIR)


if __name__ == "__main__":
    main()
