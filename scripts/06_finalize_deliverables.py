"""Finalize all PDF-mandated deliverables without re-training any models:
1. Enrich dataset_inventory.csv with file availability, file byte sizes, and trajectory bounding boxes.
2. Add road-network connectivity & trajectory pattern plots + data_quality_and_assumptions.md.
3. Enrich all_models_comparison.csv with binary Accuracy and Candidate Miss Rate from saved JSONs.
4. Add Classifier-Only vs. HMM/Viterbi per-trip comparison plot.
5. Upgrade all Folium HTML maps with floating color legends, Start/End markers, fit_bounds, and layer controls.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.candidates import generate_candidates
from pathsnap.config import ARTIFACTS_DIR, OUTPUTS_DIR, STUDENT_TEST_DIR, TOP_K_CANDIDATES, TRAIN_DIR
from pathsnap.data_loader import list_trip_ids, load_trip
from pathsnap.features import build_trip_features
from pathsnap.geo import LocalProjection, haversine_m
from pathsnap.hmm import viterbi_reconstruct

LEGEND_HTML = """
<div style="
    position: fixed;
    bottom: 28px;
    left: 28px;
    z-index: 9999;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 10px 14px;
    font-family: system-ui, sans-serif;
    font-size: 12px;
    color: #0f172a;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
">
  <div style="font-weight: 700; margin-bottom: 6px;">PathSnap Map Legend</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#2563eb;margin-right:6px;"></span>Raw GPS Observations (.track)</div>
  <div><span style="display:inline-block;width:14px;height:3px;background:#94a3b8;margin-right:6px;vertical-align:middle;"></span>Local Road Network (.nodes/.arcs)</div>
  <div><span style="display:inline-block;width:14px;height:4px;background:#16a34a;margin-right:6px;vertical-align:middle;"></span>Reference Route (.route)</div>
  <div><span style="display:inline-block;width:14px;height:3px;background:#f59e0b;margin-right:6px;vertical-align:middle;"></span>Classifier-Only Prediction</div>
  <div><span style="display:inline-block;width:14px;height:3px;background:#dc2626;margin-right:6px;vertical-align:middle;"></span>Classifier + HMM/Viterbi Route</div>
</div>
"""


def update_inventory_and_eda_plots() -> None:
    eda_dir = OUTPUTS_DIR / "eda"
    rows = []
    for split_name, split_dir in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        for tid in list_trip_ids(split_dir):
            tdir = split_dir / tid
            f_stats = {}
            for ext in ["track", "nodes", "arcs", "route"]:
                p = tdir / f"{tid}.{ext}"
                f_stats[f"has_{ext}"] = bool(p.exists())
                f_stats[f"{ext}_bytes"] = int(p.stat().st_size) if p.exists() else 0

            trip = load_trip(split_dir, tid, include_route=True)
            proj = LocalProjection.from_lon_lat(trip.track_lon, trip.track_lat)
            nodes_xy = proj.to_xy(trip.nodes_lon, trip.nodes_lat)

            dts = np.diff(trip.track_ts)
            step_m = haversine_m(
                trip.track_lon[:-1], trip.track_lat[:-1],
                trip.track_lon[1:], trip.track_lat[1:],
            )
            speeds = step_m / np.maximum(dts, 1e-3)
            ab = nodes_xy[trip.arcs_to] - nodes_xy[trip.arcs_from]
            arc_lens = np.sqrt(np.sum(ab * ab, axis=1))
            used_nodes = np.unique(np.concatenate([trip.arcs_from, trip.arcs_to]))
            isolated_nodes = len(trip.nodes_lon) - len(used_nodes)

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
                    **f_stats,
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
                    "min_lon": float(trip.track_lon.min()),
                    "max_lon": float(trip.track_lon.max()),
                    "min_lat": float(trip.track_lat.min()),
                    "max_lat": float(trip.track_lat.max()),
                    "center_lon": float(np.median(trip.track_lon)),
                    "center_lat": float(np.median(trip.track_lat)),
                }
            )

    df_inv = pd.DataFrame(rows)
    df_inv.to_csv(eda_dir / "dataset_inventory.csv", index=False)

    # Connectivity visualisations
    conn_df = pd.read_csv(eda_dir / "network_connectivity_summary.csv")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(data=conn_df, x="weak_components", hue="split", bins=25, ax=axes[0])
    axes[0].set_title("Weakly Connected Road-Graph Components per Trip")
    axes[0].set_xlabel("Number of Weak Components")

    sns.histplot(data=conn_df, x="largest_component_pct", hue="split", bins=25, ax=axes[1])
    axes[1].set_title("Nodes in Largest Connected Component (%)")
    axes[1].set_xlabel("Largest Component Coverage (%)")
    plt.tight_layout()
    fig.savefig(eda_dir / "eda_network_connectivity.png", dpi=160)
    plt.close(fig)

    # Trajectory pattern & reference-route distance diagnostic plot
    align_df = pd.read_csv(eda_dir / "alignment_summary.csv")
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.8))
    t7 = load_trip(STUDENT_TEST_DIR, "00000007", include_route=True)
    proj7 = LocalProjection.from_lon_lat(t7.track_lon, t7.track_lat)
    xy7 = proj7.to_xy(t7.track_lon, t7.track_lat)
    nxy7 = proj7.to_xy(t7.nodes_lon, t7.nodes_lat)
    for idx_r, aid in enumerate(t7.route_arcs):
        u, v = t7.arcs_from[aid], t7.arcs_to[aid]
        axes2[0].plot(
            [nxy7[u, 0], nxy7[v, 0]],
            [nxy7[u, 1], nxy7[v, 1]],
            color="#16a34a",
            linewidth=2.5,
            label="Reference Route (.route)" if idx_r == 0 else None,
        )
    axes2[0].plot(xy7[:, 0], xy7[:, 1], color="#2563eb", linewidth=1.0, alpha=0.85, label="Raw GPS (.track)")
    axes2[0].set_title("Trip 00000007 Trajectory Pattern (Local Metric Meters)")
    axes2[0].set_xlabel("Local Easting x (m)")
    axes2[0].set_ylabel("Local Northing y (m)")
    axes2[0].legend()

    sns.histplot(data=align_df, x="median_ref_dist_m", hue="split", bins=25, ax=axes2[1])
    axes2[1].set_title("Median GPS-to-Reference-Route Distance per Trip (m)")
    axes2[1].set_xlabel("Median Distance to .route (m)")
    plt.tight_layout()
    fig2.savefig(eda_dir / "eda_trajectory_patterns.png", dpi=160)
    plt.close(fig2)

    (eda_dir / "data_quality_and_assumptions.md").write_text(
        """# Data-Quality Issues, Assumptions, and Pipeline Decisions

1. **Global Multi-City Coordinate Distribution:**
   - Trips span longitudes `[-100.42°, 137.24°]` and latitudes `[-34.64°, 61.70°]` across 65+ regional clusters.
   - *Decision:* Never use raw `(lon, lat)` coordinates as ML features or a fixed regional UTM zone. Every trip is projected independently into local metric meters `(x, y)` centered on its median coordinate using WGS84 radii of curvature.

2. **Two-Way Directed Road Segments:**
   - ~97.7% of road segments in urban extracts have an opposite-direction twin `(v, u)` sharing the exact same physical geometry as `(u, v)`.
   - *Decision:* Distance features alone cannot distinguish opposite lanes; windowed GPS movement heading (`cos_heading_diff`, `cos_heading_weighted`) and graph transition continuity are included.

3. **Long Highway Arcs (up to 9.3 km) vs. Midpoint Indexing:**
   - While median arc length is ~25.5 m, highway arcs reach `9,326.9 m` (`train`) and `7,007.2 m` (`student_test`).
   - *Decision:* Arcs are densified every `25.0 m` prior to `cKDTree` spatial indexing (`TOP_K=10`, `radius=150 m`), achieving a `0.3810%` Candidate Miss Rate on `student_test`.

4. **Irregular Timestamp Gaps & Multipath Speed Spikes:**
   - Although ~96.5% (`train`) and ~98.6% (`student_test`) of steps are 1 s apart, 40 `train` trips and 2 `student_test` trips (`00000065`: 210 s, `00000076`: 32 s) contain gaps >10 s (up to `153,064 s` in `train`), and raw GPS jitter creates implied speeds up to `180.14 m/s`.
   - *Decision:* Speeds are smoothed and clipped to `[0, 60] m/s`, gaps use `log1p(dt)`, and HMM transitions bridge multi-hop road paths via bounded Dijkstra search.

5. **Reference Route Limitations & Disconnected Graph Islands:**
   - Road networks contain multiple weak components per trip (map-border clipping), and `.route` itself has 13 disjoint transitions in `train` and 1 in `student_test` (`00000076`). Furthermore, outlier training trips `00000054` and `00000033` have median GPS-to-route distances >120 m.
   - *Decision:* Offline `.track`→`.route` alignment uses order-preserving monotonic dynamic programming, and the Viterbi decoder includes a finite `teleport_penalty` fallback so disconnected components never cause decoding failure.
"""
    )


def enrich_comparison_and_hmm_charts() -> None:
    comp_dir = OUTPUTS_DIR / "comparison"
    tune_dir = OUTPUTS_DIR / "tuning"

    m1 = json.loads((OUTPUTS_DIR / "model1_lightgbm" / "model1_results.json").read_text())
    m1["key_config"] = "n_estimators=250, lr=0.06, num_leaves=63, min_child=40"
    m2 = json.loads((OUTPUTS_DIR / "model2_xgboost" / "model2_results.json").read_text())
    m2["key_config"] = "n_estimators=250, lr=0.06, max_depth=7, tree_method=hist"

    all_res = [m1, m2]
    for fn in [
        "model3_logreg_results.json",
        "model4_extratrees_results.json",
        "model5_pytorch_mlp_results.json",
        "model6_nystroem_svc_results.json",
        "model7_random_forest_results.json",
    ]:
        fp = comp_dir / fn
        if fp.exists():
            all_res.append(json.loads(fp.read_text()))

    miss_rate_test = round(float(m1["student_test_candidate_miss_rate"]), 5)
    rows = []
    for r in all_res:
        cv = r["cv_metrics"]
        te = r["student_test_candidate_metrics"]
        re_ = r["student_test_route_eval"]
        rows.append(
            {
                "Model": r["model_name"],
                "Algorithm_Family": r["algorithm_family"],
                "Key_Configuration": r.get("key_config", ""),
                "CV_Binary_Accuracy": round(cv["accuracy"], 4),
                "CV_Macro_F1": round(cv["macro_f1"], 4),
                "CV_Macro_F1_Std": round(cv.get("fold_macro_f1_std", 0.0), 4),
                "Student_Test_Binary_Accuracy": round(te["accuracy"], 4),
                "Student_Test_Precision": round(te["precision"], 4),
                "Student_Test_Recall": round(te["recall"], 4),
                "Student_Test_Macro_F1": round(te["macro_f1"], 4),
                "Log_Loss": round(te["log_loss"], 4),
                "Candidate_Miss_Rate": miss_rate_test,
                "Cls_Top1_Accuracy": round(re_["obs_top1_acc_cls"], 4),
                "HMM_Top1_Accuracy": round(re_["obs_top1_acc_hmm"], 4),
                "Cls_Route_Continuity": round(re_["mean_obs_continuity_cls"], 4),
                "HMM_Route_Continuity": round(re_["mean_obs_continuity_hmm"], 4),
                "Cls_Seq_LCS_F1": round(re_["mean_seq_lcs_f1_cls"], 4),
                "HMM_Raw_Seq_LCS_F1": round(re_["mean_seq_lcs_f1_hmm"], 4),
                "HMM_GapFilled_Seq_F1": round(re_["mean_seq_lcs_f1_hmm_gapfilled"], 4),
                "Train_Time_s": round(r["train_time_s"], 2),
                "Inference_Time_s": round(r["student_test_inference_time_s"], 3),
            }
        )
    pd.DataFrame(rows).to_csv(comp_dir / "all_models_comparison.csv", index=False)

    # Per-trip Classifier-Only vs HMM/Viterbi chart on student_test (Tuned XGBoost)
    pt_df = pd.read_csv(tune_dir / "tuned_student_test_per_trip.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(pt_df))
    w = 0.36
    axes[0].bar(x - w / 2, pt_df["cls_top1_acc"], width=w, label="Classifier Only", color="#f59e0b")
    axes[0].bar(x + w / 2, pt_df["hmm_top1_acc"], width=w, label="Classifier + HMM/Viterbi", color="#16a34a")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pt_df["trip_id"], rotation=30, ha="right")
    axes[0].set_ylim(0.5, 1.02)
    axes[0].set_title("Per-Trip Observation Top-1 Accuracy (Student Test)")
    axes[0].legend()

    axes[1].bar(x - w / 2, pt_df["cls_seq_lcs_f1"], width=w, label="Classifier Only", color="#f59e0b")
    axes[1].bar(x + w / 2, pt_df["hmm_gapfilled_seq_f1"], width=w, label="Classifier + HMM/Viterbi", color="#dc2626")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pt_df["trip_id"], rotation=30, ha="right")
    axes[1].set_ylim(0.2, 1.05)
    axes[1].set_title("Per-Trip Sequence LCS F1 vs. Reference .route (Student Test)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(tune_dir / "cls_vs_hmm_comparison.png", dpi=160)
    plt.close(fig)


def rebuild_folium_maps_with_legend() -> None:
    bundle = joblib.load(ARTIFACTS_DIR / "model_bundle.joblib")
    model = bundle["model"]
    scaler = bundle["scaler"]
    hmm_cfg = bundle["hmm_config"]
    maps_dir = OUTPUTS_DIR / "final_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    for tid in ["00000000", "00000007", "00000056"]:
        trip = load_trip(STUDENT_TEST_DIR, tid, include_route=True)
        cand = generate_candidates(trip, top_k=TOP_K_CANDIDATES)
        x_mat, obs_idx, _ = build_trip_features(trip, cand)
        probs = model.predict_proba(scaler.transform(x_mat))[:, 1]

        n_obs = len(trip.track_lon)
        probs_per_obs = [np.empty(0) for _ in range(n_obs)]
        cls_pred = np.full(n_obs, -1, dtype=np.int64)
        order_breaks = np.flatnonzero(np.diff(obs_idx)) + 1
        starts = np.concatenate(([0], order_breaks))
        ends = np.concatenate((order_breaks, [len(obs_idx)]))
        for s_i, e_i in zip(starts, ends):
            t_i = int(obs_idx[s_i])
            p_slice = probs[s_i:e_i]
            probs_per_obs[t_i] = p_slice
            cls_pred[t_i] = int(cand.cand_arc_ids[t_i][int(np.argmax(p_slice))])

        _, hmm_filled = viterbi_reconstruct(trip, cand, probs_per_obs, cfg=hmm_cfg)

        center = [float(np.median(trip.track_lat)), float(np.median(trip.track_lon))]
        m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

        lon_pad, lat_pad = 0.0045, 0.0032
        min_lon, max_lon = float(trip.track_lon.min()) - lon_pad, float(trip.track_lon.max()) + lon_pad
        min_lat, max_lat = float(trip.track_lat.min()) - lat_pad, float(trip.track_lat.max()) + lat_pad

        u_lat = trip.nodes_lat[trip.arcs_from]
        u_lon = trip.nodes_lon[trip.arcs_from]
        v_lat = trip.nodes_lat[trip.arcs_to]
        v_lon = trip.nodes_lon[trip.arcs_to]
        in_box = (
            ((u_lat >= min_lat) & (u_lat <= max_lat) & (u_lon >= min_lon) & (u_lon <= max_lon))
            | ((v_lat >= min_lat) & (v_lat <= max_lat) & (v_lon >= min_lon) & (v_lon <= max_lon))
        )
        box_arcs = np.flatnonzero(in_box)
        if len(box_arcs) > 4000:
            box_arcs = box_arcs[:: int(np.ceil(len(box_arcs) / 4000))]

        net_fg = folium.FeatureGroup(name="Road Network (Local)", show=True)
        for aid in box_arcs:
            u, v = int(trip.arcs_from[aid]), int(trip.arcs_to[aid])
            folium.PolyLine(
                [[float(trip.nodes_lat[u]), float(trip.nodes_lon[u])], [float(trip.nodes_lat[v]), float(trip.nodes_lon[v])]],
                color="#94a3b8",
                weight=1.4,
                opacity=0.55,
            ).add_to(net_fg)
        net_fg.add_to(m)

        if trip.route_arcs is not None:
            ref_fg = folium.FeatureGroup(name="Reference Route (.route)", show=True)
            for aid in trip.route_arcs:
                u, v = int(trip.arcs_from[aid]), int(trip.arcs_to[aid])
                folium.PolyLine(
                    [[float(trip.nodes_lat[u]), float(trip.nodes_lon[u])], [float(trip.nodes_lat[v]), float(trip.nodes_lon[v])]],
                    color="#16a34a",
                    weight=6,
                    opacity=0.65,
                    tooltip=f"Reference arc {aid}",
                ).add_to(ref_fg)
            ref_fg.add_to(m)

        cls_fg = folium.FeatureGroup(name="Classifier-Only Prediction", show=False)
        cls_dedup = [int(a) for i, a in enumerate(cls_pred) if a >= 0 and (i == 0 or a != cls_pred[i - 1])]
        for aid in cls_dedup:
            u, v = int(trip.arcs_from[aid]), int(trip.arcs_to[aid])
            folium.PolyLine(
                [[float(trip.nodes_lat[u]), float(trip.nodes_lon[u])], [float(trip.nodes_lat[v]), float(trip.nodes_lon[v])]],
                color="#f59e0b",
                weight=4,
                opacity=0.8,
                tooltip=f"Classifier-only arc {aid}",
            ).add_to(cls_fg)
        cls_fg.add_to(m)

        hmm_fg = folium.FeatureGroup(name="Predicted Route (XGBoost + HMM/Viterbi)", show=True)
        for aid in hmm_filled:
            u, v = int(trip.arcs_from[aid]), int(trip.arcs_to[aid])
            folium.PolyLine(
                [[float(trip.nodes_lat[u]), float(trip.nodes_lon[u])], [float(trip.nodes_lat[v]), float(trip.nodes_lon[v])]],
                color="#dc2626",
                weight=3.2,
                opacity=0.95,
                tooltip=f"HMM predicted arc {aid}",
            ).add_to(hmm_fg)
        hmm_fg.add_to(m)

        gps_fg = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
        step = max(1, n_obs // 600)
        for idx in range(0, n_obs, step):
            folium.CircleMarker(
                [float(trip.track_lat[idx]), float(trip.track_lon[idx])],
                radius=2.4,
                color="#2563eb",
                fill=True,
                fill_opacity=0.85,
                tooltip=f"GPS obs #{idx} (t={trip.track_ts[idx]:.0f}s)",
            ).add_to(gps_fg)
        folium.Marker(
            [float(trip.track_lat[0]), float(trip.track_lon[0])],
            tooltip=f"Trip {tid} Start",
            icon=folium.Icon(color="green"),
        ).add_to(gps_fg)
        folium.Marker(
            [float(trip.track_lat[-1]), float(trip.track_lon[-1])],
            tooltip=f"Trip {tid} End",
            icon=folium.Icon(color="red"),
        ).add_to(gps_fg)
        gps_fg.add_to(m)

        m.fit_bounds([[float(trip.track_lat.min()), float(trip.track_lon.min())], [float(trip.track_lat.max()), float(trip.track_lon.max())]])
        folium.LayerControl(collapsed=False).add_to(m)
        m.get_root().html.add_child(folium.Element(LEGEND_HTML))
        m.save(str(maps_dir / f"folium_final_{tid}.html"))


def main() -> None:
    update_inventory_and_eda_plots()
    enrich_comparison_and_hmm_charts()
    rebuild_folium_maps_with_legend()
    print("All PDF-mandated tables, plots, and Folium maps finalized.")


if __name__ == "__main__":
    main()
