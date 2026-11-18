"""Generate supplemental EDA diagnostics (weak connectivity, integrity, reference alignment
outlier summary, 0-30s sampling histogram, trip 00000002 Folium map), feature dictionary,
and extra comparison charts required for the technical report."""

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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.candidates import densify_arcs
from pathsnap.config import OUTPUTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from pathsnap.data_loader import list_trip_ids, load_trip
from pathsnap.features import FEATURE_NAMES
from pathsnap.geo import LocalProjection, point_to_segments_geometry

FEATURE_DESCRIPTIONS = {
    "dist_to_arc_m": "Perpendicular Euclidean distance (m) from GPS observation to arc segment in local metric projection",
    "signed_lateral_m": "Signed lateral offset (m): positive left of directed arc, negative right",
    "abs_lateral_m": "Absolute lateral offset (m) from directed arc line",
    "proj_frac": "Normalized projection position s in [0, 1] along arc (0=from_node, 1=to_node)",
    "dist_to_from_m": "Distance (m) from GPS observation to arc start node (from_node)",
    "dist_to_to_m": "Distance (m) from GPS observation to arc end node (to_node)",
    "arc_len_m": "Physical length of directed road arc (m)",
    "log_arc_len": "log1p(arc_len_m) compressed segment length",
    "dist_rank": "Zero-based rank (0..K-1) of candidate arc by distance at GPS step t",
    "dist_diff_best": "Distance margin (m) over closest candidate at step t",
    "dist_ratio_best": "Ratio (dist + 1) / (min_dist + 1) relative to closest candidate at step t",
    "cos_heading_diff": "Cosine similarity between smoothed GPS movement vector and directed arc vector",
    "sin_heading_diff": "Sine of signed turn angle between GPS movement vector and directed arc vector",
    "heading_conf": "Confidence in [0, 1] that local GPS displacement exceeds stationary jitter (>= 4m)",
    "cos_heading_weighted": "cos_heading_diff * heading_conf (suppresses stationary heading noise)",
    "gps_speed_mps": "Smoothed local GPS speed (m/s), clipped to [0, 60]",
    "log_dt_prev": "log1p(dt_prev) seconds since previous GPS observation",
    "log_dt_next": "log1p(dt_next) seconds to next GPS observation",
    "prev_obs_dist_m": "Distance (m) from previous GPS observation (t-1) to this candidate arc",
    "next_obs_dist_m": "Distance (m) from next GPS observation (t+1) to this candidate arc",
    "prev2_obs_dist_m": "Distance (m) from GPS observation (t-2) to this candidate arc",
    "next2_obs_dist_m": "Distance (m) from GPS observation (t+2) to this candidate arc",
    "mean_5pt_dist_m": "Mean distance (m) from local 5-point window [t-2..t+2] to this candidate arc",
    "proj_delta_m": "Signed forward progress (m) along directed candidate arc between t-1 and t",
    "from_in_deg": "Graph in-degree of arc's from_node",
    "to_out_deg": "Graph out-degree of arc's to_node",
    "connects_from_prev": "1 if same arc or prev_top5.to_node == curr.from_node, else 0",
    "connects_to_next": "1 if same arc or curr.to_node == next_top5.from_node, else 0",
    "connects_2hop_prev": "1 if reachable within 1-2 directed graph hops from prev_top5 candidates",
    "connects_2hop_next": "1 if reaches next_top5 candidates within 1-2 directed graph hops",
    "has_reverse_arc": "1 if opposite-direction twin (to_node, from_node) exists in .arcs",
}


def run_eda_extras() -> None:
    eda_dir = OUTPUTS_DIR / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    integrity_rows = []
    conn_rows = []
    align_rows = []
    all_dts = []

    for split_name, split_dir in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        for tid in list_trip_ids(split_dir):
            trip = load_trip(split_dir, tid, include_route=True)
            n_nodes = len(trip.nodes_lon)
            n_arcs = len(trip.arcs_from)

            # Integrity checks
            track_nan = int(
                np.isnan(trip.track_lon).sum()
                + np.isnan(trip.track_lat).sum()
                + np.isnan(trip.track_ts).sum()
            )
            node_nan = int(np.isnan(trip.nodes_lon).sum() + np.isnan(trip.nodes_lat).sum())
            oob_nodes = int(
                ((trip.arcs_from < 0) | (trip.arcs_from >= n_nodes) | (trip.arcs_to < 0) | (trip.arcs_to >= n_nodes)).sum()
            )
            oob_route = 0
            if trip.route_arcs is not None:
                oob_route = int(((trip.route_arcs < 0) | (trip.route_arcs >= n_arcs)).sum())

            integrity_rows.append(
                {
                    "split": split_name,
                    "trip_id": tid,
                    "track_rows": len(trip.track_lon),
                    "node_rows": n_nodes,
                    "arc_rows": n_arcs,
                    "route_rows": len(trip.route_arcs) if trip.route_arcs is not None else 0,
                    "track_invalid": track_nan,
                    "node_invalid": node_nan,
                    "arc_invalid_node_refs": oob_nodes,
                    "route_invalid_arc_ids": oob_route,
                }
            )

            # Weak connected components
            rows_adj = np.concatenate([trip.arcs_from, trip.arcs_to])
            cols_adj = np.concatenate([trip.arcs_to, trip.arcs_from])
            adj = csr_matrix((np.ones(len(rows_adj)), (rows_adj, cols_adj)), shape=(n_nodes, n_nodes))
            n_comp, labels = connected_components(adj, directed=False, return_labels=True)
            comp_sizes = np.bincount(labels)
            largest_comp = int(comp_sizes.max()) if len(comp_sizes) else 0
            in_deg = np.bincount(trip.arcs_to, minlength=n_nodes)
            out_deg = np.bincount(trip.arcs_from, minlength=n_nodes)
            isolated = int(np.sum((in_deg + out_deg) == 0))

            conn_rows.append(
                {
                    "split": split_name,
                    "trip_id": tid,
                    "nodes": n_nodes,
                    "arcs": n_arcs,
                    "isolated_nodes": isolated,
                    "weak_components": int(n_comp),
                    "largest_component_nodes": largest_comp,
                    "largest_component_pct": round(100.0 * largest_comp / max(1, n_nodes), 2),
                }
            )

            # Sampling intervals
            dts = np.diff(trip.track_ts)
            all_dts.extend(dts[dts > 0].tolist())

            # GPS-to-nearest-reference-route diagnostic (per-trip local metric projection)
            if trip.route_arcs is not None and len(trip.route_arcs) > 0:
                proj = LocalProjection.from_lon_lat(trip.track_lon, trip.track_lat)
                track_xy = proj.to_xy(trip.track_lon, trip.track_lat)
                nodes_xy = proj.to_xy(trip.nodes_lon, trip.nodes_lat)
                r_a = nodes_xy[trip.arcs_from[trip.route_arcs]]
                r_b = nodes_xy[trip.arcs_to[trip.route_arcs]]
                min_dists = np.empty(len(track_xy), dtype=np.float64)
                for idx_pt in range(len(track_xy)):
                    min_dists[idx_pt] = float(
                        np.min(point_to_segments_geometry(track_xy[idx_pt], r_a, r_b)["dist_m"])
                    )
                align_rows.append(
                    {
                        "split": split_name,
                        "trip_id": tid,
                        "gps_points": len(min_dists),
                        "median_ref_dist_m": round(float(np.median(min_dists)), 2),
                        "p90_ref_dist_m": round(float(np.quantile(min_dists, 0.90)), 2),
                        "p95_ref_dist_m": round(float(np.quantile(min_dists, 0.95)), 2),
                        "max_ref_dist_m": round(float(np.max(min_dists)), 2),
                        "within_50m_pct": round(float(np.mean(min_dists <= 50.0) * 100.0), 2),
                    }
                )

    pd.DataFrame(integrity_rows).to_csv(eda_dir / "integrity_report.csv", index=False)
    pd.DataFrame(conn_rows).to_csv(eda_dir / "network_connectivity_summary.csv", index=False)
    pd.DataFrame(align_rows).sort_values("p90_ref_dist_m", ascending=False).to_csv(
        eda_dir / "alignment_summary.csv", index=False
    )

    # 0-30s sampling interval histogram
    dt_arr = np.asarray(all_dts)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(dt_arr[dt_arr <= 30.0], bins=60, color="#2563eb", edgecolor="white")
    ax.set_title("GPS Sampling Interval Distribution (0–30 s across all 90 trips)")
    ax.set_xlabel("Sampling interval dt (seconds)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(eda_dir / "gps_sampling_intervals_0_30s.png", dpi=160)
    plt.close(fig)

    # Feature dictionary CSV
    comp_dir = OUTPUTS_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"feature": f, "description": FEATURE_DESCRIPTIONS[f]} for f in FEATURE_NAMES]
    ).to_csv(comp_dir / "feature_dictionary.csv", index=False)

    # Extra report chart: Student-Test Top-1 Accuracy & Macro F1 across all 6 models
    df_cmp = pd.read_csv(comp_dir / "all_models_comparison.csv")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_cmp))
    w = 0.27
    ax2.bar(x - w, df_cmp["Cls_Top1_Accuracy"], width=w, label="Cls-Only Top-1 Acc", color="#f59e0b")
    ax2.bar(x, df_cmp["HMM_Top1_Accuracy"], width=w, label="Classifier + HMM Top-1 Acc", color="#16a34a")
    ax2.bar(x + w, df_cmp["Student_Test_Macro_F1"], width=w, label="Student-Test Macro F1", color="#2563eb")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_cmp["Model"], rotation=18, ha="right")
    ax2.set_ylim(0.6, 1.0)
    ax2.set_title("Student-Testing Top-1 Accuracy and Macro F1-score Across All 6 Models")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(comp_dir / "student_test_top1_and_macro_f1.png", dpi=160)
    plt.close(fig2)

    # Extra report chart: Untuned vs Bayesian-Tuned XGBoost comparison
    tune_dir = OUTPUTS_DIR / "tuning"
    tc = json.loads((tune_dir / "tuning_comparison.json").read_text())
    metrics_keys = [
        ("CV Macro F1", tc["untuned"]["cv_macro_f1"], tc["tuned"]["cv_macro_f1"]),
        ("Test Macro F1", tc["untuned"]["student_test_macro_f1"], tc["tuned"]["student_test_macro_f1"]),
        ("Cls Top-1 Acc", tc["untuned"]["cls_top1_acc"], tc["tuned"]["cls_top1_acc"]),
        ("HMM Top-1 Acc", tc["untuned"]["hmm_top1_acc"], tc["tuned"]["hmm_top1_acc"]),
        ("HMM Seq LCS F1", tc["untuned"]["hmm_gapfilled_seq_f1"], tc["tuned"]["hmm_gapfilled_seq_f1"]),
    ]
    labels_t = [m[0] for m in metrics_keys]
    u_vals = [m[1] for m in metrics_keys]
    t_vals = [m[2] for m in metrics_keys]
    fig3, ax3 = plt.subplots(figsize=(9, 4.8))
    xt = np.arange(len(labels_t))
    ax3.bar(xt - 0.18, u_vals, width=0.36, label="Untuned XGBoost + Base HMM", color="#64748b")
    ax3.bar(xt + 0.18, t_vals, width=0.36, label="Optuna-Tuned XGBoost + Tuned HMM", color="#16a34a")
    ax3.set_xticks(xt)
    ax3.set_xticklabels(labels_t)
    ax3.set_ylim(0.75, 0.96)
    ax3.set_title("Untuned vs. Bayesian-Tuned XGBoost & HMM Performance")
    ax3.legend()
    plt.tight_layout()
    fig3.savefig(tune_dir / "untuned_vs_tuned_comparison.png", dpi=160)
    plt.close(fig3)

    # Trip 00000002 Folium EDA map (gap-broken trajectory + start/end markers + OpenStreetMap)
    t2 = load_trip(TRAIN_DIR, "00000002", include_route=True)
    m = folium.Map(
        location=[float(np.mean(t2.track_lat)), float(np.mean(t2.track_lon))],
        zoom_start=13,
        tiles="OpenStreetMap",
    )
    gps_fg = folium.FeatureGroup(name="GPS Trajectory (gap-broken > 5s)", show=True)
    start_i = 0
    for i in range(1, len(t2.track_ts)):
        if t2.track_ts[i] - t2.track_ts[i - 1] > 5.0:
            if i - start_i >= 2:
                folium.PolyLine(
                    list(zip(t2.track_lat[start_i:i], t2.track_lon[start_i:i])),
                    color="#2563eb",
                    weight=3,
                    opacity=0.8,
                ).add_to(gps_fg)
            start_i = i
    if len(t2.track_ts) - start_i >= 2:
        folium.PolyLine(
            list(zip(t2.track_lat[start_i:], t2.track_lon[start_i:])),
            color="#2563eb",
            weight=3,
            opacity=0.8,
        ).add_to(gps_fg)
    folium.Marker([float(t2.track_lat[0]), float(t2.track_lon[0])], tooltip="Trip Start", icon=folium.Icon(color="green")).add_to(gps_fg)
    folium.Marker([float(t2.track_lat[-1]), float(t2.track_lon[-1])], tooltip="Trip End", icon=folium.Icon(color="red")).add_to(gps_fg)
    gps_fg.add_to(m)

    ref_fg = folium.FeatureGroup(name="Reference Route (.route)", show=True)
    for aid in t2.route_arcs:
        u, v = t2.arcs_from[aid], t2.arcs_to[aid]
        folium.PolyLine(
            [[float(t2.nodes_lat[u]), float(t2.nodes_lon[u])], [float(t2.nodes_lat[v]), float(t2.nodes_lon[v])]],
            color="#16a34a",
            weight=4,
            opacity=0.85,
        ).add_to(ref_fg)
    ref_fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(eda_dir / "folium_train_00000002.html"))

    print("Supplemental EDA and report charts generated successfully.")


if __name__ == "__main__":
    run_eda_extras()
