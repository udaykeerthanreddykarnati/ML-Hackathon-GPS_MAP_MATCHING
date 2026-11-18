"""Stage 1: Exploratory Data Analysis, plots, and initial Folium maps."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.config import OUTPUTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from pathsnap.data_loader import list_trip_ids, load_trip
from pathsnap.geo import LocalProjection, haversine_m


def analyze_partition(split_name: str, split_dir: Path) -> tuple[pd.DataFrame, list[float], list[float]]:
    """Collect per-trip inventory, coordinate bounds, sampling gaps, and connectivity stats."""
    rows = []
    all_dts: list[float] = []
    all_speeds: list[float] = []

    for tid in list_trip_ids(split_dir):
        trip = load_trip(split_dir, tid, include_route=True)
        proj = LocalProjection.from_lon_lat(trip.track_lon, trip.track_lat)
        nodes_xy = proj.to_xy(trip.nodes_lon, trip.nodes_lat)

        # Sampling intervals and GPS step speeds
        dts = np.diff(trip.track_ts)
        step_m = haversine_m(
            trip.track_lon[:-1], trip.track_lat[:-1],
            trip.track_lon[1:], trip.track_lat[1:],
        )
        valid_dt = np.maximum(dts, 1e-3)
        speeds = step_m / valid_dt
        all_dts.extend(dts.tolist())
        all_speeds.extend(speeds.tolist())

        # Arc lengths and isolated nodes
        ab = nodes_xy[trip.arcs_to] - nodes_xy[trip.arcs_from]
        arc_lens = np.sqrt(np.sum(ab * ab, axis=1))
        used_nodes = np.unique(np.concatenate([trip.arcs_from, trip.arcs_to]))
        isolated_nodes = len(trip.nodes_lon) - len(used_nodes)

        # Reference route continuity check
        disjoint_route = 0
        route_len_m = 0.0
        if trip.route_arcs is not None and len(trip.route_arcs) > 0:
            route_len_m = float(np.sum(arc_lens[trip.route_arcs]))
            for a1, a2 in zip(trip.route_arcs[:-1], trip.route_arcs[1:]):
                if trip.arcs_to[a1] != trip.arcs_from[a2]:
                    disjoint_route += 1

        rows.append(
            {
                "partition": split_name,
                "trip_id": tid,
                "n_gps": len(trip.track_lon),
                "n_nodes": len(trip.nodes_lon),
                "n_arcs": len(trip.arcs_from),
                "n_route_arcs": len(trip.route_arcs) if trip.route_arcs is not None else 0,
                "duration_s": float(trip.track_ts[-1] - trip.track_ts[0]),
                "frac_dt_1s": float(np.mean(np.isclose(dts, 1.0))) if len(dts) else 1.0,
                "max_dt_s": float(np.max(dts)) if len(dts) else 0.0,
                "median_speed_mps": float(np.median(speeds)) if len(speeds) else 0.0,
                "max_speed_mps": float(np.max(speeds)) if len(speeds) else 0.0,
                "route_len_km": route_len_m / 1000.0,
                "median_arc_m": float(np.median(arc_lens)),
                "max_arc_m": float(np.max(arc_lens)),
                "isolated_nodes": int(isolated_nodes),
                "route_disjoint_jumps": int(disjoint_route),
                "center_lon": float(np.median(trip.track_lon)),
                "center_lat": float(np.median(trip.track_lat)),
            }
        )

    return pd.DataFrame(rows), all_dts, all_speeds


def create_eda_plots(df: pd.DataFrame, dts: list[float], out_dir: Path) -> None:
    """Save clean publication-ready EDA summary figures."""
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sns.histplot(data=df, x="n_gps", hue="partition", bins=25, ax=axes[0, 0], kde=True)
    axes[0, 0].set_title("GPS Observations per Trip")
    axes[0, 0].set_xlabel("Number of GPS points")

    sns.histplot(data=df, x="route_len_km", hue="partition", bins=25, ax=axes[0, 1], kde=True)
    axes[0, 1].set_title("Reference Route Length (km)")
    axes[0, 1].set_xlabel("Route length (km)")

    sns.scatterplot(
        data=df, x="n_nodes", y="n_arcs", hue="partition", style="partition", ax=axes[1, 0], s=60
    )
    axes[1, 0].set_title("Road Network Size: Nodes vs. Directed Arcs")
    axes[1, 0].set_xlabel("Nodes")
    axes[1, 0].set_ylabel("Directed Arcs")

    dt_clipped = np.clip(np.asarray(dts), 1.0, 60.0)
    gt1 = dt_clipped[dt_clipped > 1.0]
    axes[1, 1].hist(gt1, bins=30, color="#d97706", edgecolor="white")
    axes[1, 1].set_title("Irregular GPS Gaps (dt > 1s, clipped at 60s)")
    axes[1, 1].set_xlabel("Time gap dt (s)")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(out_dir / "eda_distributions.png", dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df, x="center_lon", y="center_lat", hue="partition", style="partition", s=80, ax=ax2
    )
    ax2.set_title("Global Geographical Distribution of Trips (Train vs. Student Test)")
    ax2.set_xlabel("Longitude (deg)")
    ax2.set_ylabel("Latitude (deg)")
    plt.tight_layout()
    fig2.savefig(out_dir / "eda_geo_locations.png", dpi=160)
    plt.close(fig2)


def create_sample_folium_map(split_dir: Path, trip_id: str, out_html: Path) -> None:
    """Create an interactive Folium EDA map showing road network, GPS points, and reference route."""
    trip = load_trip(split_dir, trip_id, include_route=True)
    center = [float(np.median(trip.track_lat)), float(np.median(trip.track_lon))]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    # Filter road arcs near the GPS bounding box so HTML stays crisp and lightweight.
    lon_pad = 0.006
    lat_pad = 0.004
    min_lon, max_lon = trip.track_lon.min() - lon_pad, trip.track_lon.max() + lon_pad
    min_lat, max_lat = trip.track_lat.min() - lat_pad, trip.track_lat.max() + lat_pad

    net_group = folium.FeatureGroup(name="Road Network (Local)", show=True)
    for u, v in zip(trip.arcs_from, trip.arcs_to):
        la1, lo1 = trip.nodes_lat[u], trip.nodes_lon[u]
        la2, lo2 = trip.nodes_lat[v], trip.nodes_lon[v]
        if (min_lat <= la1 <= max_lat and min_lon <= lo1 <= max_lon) or (
            min_lat <= la2 <= max_lat and min_lon <= lo2 <= max_lon
        ):
            folium.PolyLine([[la1, lo1], [la2, lo2]], color="#94a3b8", weight=1.5, opacity=0.55).add_to(net_group)
    net_group.add_to(m)

    if trip.route_arcs is not None:
        ref_group = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        for aid in trip.route_arcs:
            u, v = trip.arcs_from[aid], trip.arcs_to[aid]
            folium.PolyLine(
                [[trip.nodes_lat[u], trip.nodes_lon[u]], [trip.nodes_lat[v], trip.nodes_lon[v]]],
                color="#16a34a",
                weight=5,
                opacity=0.85,
            ).add_to(ref_group)
        ref_group.add_to(m)

    gps_group = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    step = max(1, len(trip.track_lat) // 600)
    for idx in range(0, len(trip.track_lat), step):
        folium.CircleMarker(
            location=[float(trip.track_lat[idx]), float(trip.track_lon[idx])],
            radius=2.5,
            color="#2563eb",
            fill=True,
            fill_opacity=0.85,
            popup=f"t={trip.track_ts[idx]:.0f}s (idx={idx})",
        ).add_to(gps_group)
    gps_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))


def main() -> None:
    out_dir = OUTPUTS_DIR / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train, dts_tr, sp_tr = analyze_partition("train", TRAIN_DIR)
    df_test, dts_te, sp_te = analyze_partition("student_test", STUDENT_TEST_DIR)
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all.to_csv(out_dir / "dataset_inventory.csv", index=False)

    summary = {}
    for name, sub in [("train", df_train), ("student_test", df_test)]:
        summary[name] = {
            "n_trips": int(len(sub)),
            "total_gps": int(sub["n_gps"].sum()),
            "total_nodes": int(sub["n_nodes"].sum()),
            "total_arcs": int(sub["n_arcs"].sum()),
            "total_route_arcs": int(sub["n_route_arcs"].sum()),
            "median_gps_per_trip": float(sub["n_gps"].median()),
            "median_route_km": float(sub["route_len_km"].median()),
            "trips_with_dt_gt_10s": int((sub["max_dt_s"] > 10.0).sum()),
            "total_isolated_nodes": int(sub["isolated_nodes"].sum()),
            "total_route_disjoint_jumps": int(sub["route_disjoint_jumps"].sum()),
        }
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    create_eda_plots(df_all, dts_tr + dts_te, out_dir)
    create_sample_folium_map(TRAIN_DIR, "00000006", out_dir / "folium_train_00000006.html")
    create_sample_folium_map(STUDENT_TEST_DIR, "00000007", out_dir / "folium_student_test_00000007.html")

    print("EDA complete ->", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
