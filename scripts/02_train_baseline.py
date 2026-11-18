"""Stage 2-3: Feature preparation, Model 1 (LightGBM) training with GroupKFold CV,
HMM/Viterbi evaluation, artifact export, and Folium comparison maps."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import folium
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.alignment import align_track_to_route
from pathsnap.candidates import CandidateTable, generate_candidates
from pathsnap.config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    STUDENT_TEST_DIR,
    TOP_K_CANDIDATES,
    TRAIN_DIR,
    TRAIN_MAX_ALIGN_DIST_M,
)
from pathsnap.data_loader import TripData, list_trip_ids, load_trip
from pathsnap.features import FEATURE_NAMES, build_trip_features
from pathsnap.hmm import HMMConfig, viterbi_reconstruct
from pathsnap.metrics import (
    compute_candidate_metrics,
    compute_route_continuity,
    compute_sequence_metrics,
)


def prepare_split_dataset(split_name: str, split_dir: Path, cache_path: Path) -> dict:
    """Build or load cached unscaled candidate features and aligned reference labels."""
    if cache_path.exists():
        print(f"Loading cached {split_name} features from {cache_path}")
        return joblib.load(cache_path)

    t0 = time.perf_counter()
    trip_ids = list_trip_ids(split_dir)

    x_chunks, y_chunks, trip_idx_chunks = [], [], []
    obs_chunks, arc_chunks = [], []
    trip_meta: list[dict] = []
    total_obs = 0
    missed_obs = 0

    for t_num, tid in enumerate(trip_ids):
        trip = load_trip(split_dir, tid, include_route=True)
        cand = generate_candidates(trip, top_k=TOP_K_CANDIDATES)
        alignment = align_track_to_route(trip, cand.track_xy, cand.nodes_xy)
        x_trip, obs_idx, cand_arc_id = build_trip_features(trip, cand)

        aligned_arcs = alignment.aligned_arc_ids
        align_dists = alignment.point_to_ref_dist_m
        y_trip = (cand_arc_id == aligned_arcs[obs_idx]).astype(np.int64)

        # For training only, exclude severe GPS outage observations far from .route so label noise
        # does not corrupt classifier probabilities. Never filter student_test during evaluation.
        if split_name == "train":
            valid_obs_mask = align_dists <= TRAIN_MAX_ALIGN_DIST_M
            keep_row_mask = valid_obs_mask[obs_idx]
            x_trip = x_trip[keep_row_mask]
            y_trip = y_trip[keep_row_mask]
            obs_idx = obs_idx[keep_row_mask]
            cand_arc_id = cand_arc_id[keep_row_mask]

        # Count observations whose aligned true arc is absent from top-K candidates.
        for o_i in range(len(aligned_arcs)):
            if split_name == "train" and align_dists[o_i] > TRAIN_MAX_ALIGN_DIST_M:
                continue
            total_obs += 1
            if int(aligned_arcs[o_i]) not in cand.cand_arc_ids[o_i]:
                missed_obs += 1

        x_chunks.append(x_trip)
        y_chunks.append(y_trip)
        trip_idx_chunks.append(np.full(len(y_trip), t_num, dtype=np.int64))
        obs_chunks.append(obs_idx)
        arc_chunks.append(cand_arc_id)

        trip_meta.append(
            {
                "trip_id": tid,
                "n_obs": len(trip.track_lon),
                "aligned_arcs": aligned_arcs,
                "align_median_dist_m": float(np.median(alignment.point_to_ref_dist_m)),
            }
        )
        if (t_num + 1) % 20 == 0 or (t_num + 1) == len(trip_ids):
            print(f"  [{split_name}] processed {t_num + 1}/{len(trip_ids)} trips")

    payload = {
        "trip_ids": trip_ids,
        "trip_meta": trip_meta,
        "X": np.vstack(x_chunks),
        "y": np.concatenate(y_chunks),
        "trip_group": np.concatenate(trip_idx_chunks),
        "obs_idx": np.concatenate(obs_chunks),
        "cand_arc_id": np.concatenate(arc_chunks),
        "candidate_miss_rate": float(missed_obs / max(1, total_obs)),
        "total_obs": int(total_obs),
        "missed_obs": int(missed_obs),
        "prep_time_s": float(time.perf_counter() - t0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=3)
    print(
        f"[{split_name}] rows={len(payload['y'])}, obs={total_obs}, "
        f"candidate_miss_rate={payload['candidate_miss_rate']:.4%}, "
        f"time={payload['prep_time_s']:.1f}s"
    )
    return payload


def evaluate_trip_predictions(
    split_dir: Path,
    trip_ids: list[str],
    trip_meta: list[dict],
    probs: np.ndarray,
    trip_group: np.ndarray,
    obs_idx: np.ndarray,
    cand_arc_id: np.ndarray,
    hmm_cfg: HMMConfig,
) -> dict:
    """Compare Classifier-only vs. Classifier+HMM/Viterbi across trips."""
    cls_correct, hmm_correct, total_obs = 0, 0, 0
    cls_cont_list, hmm_cont_list = [], []
    cls_seq_f1_list, hmm_seq_f1_list, hmm_filled_f1_list = [], [], []
    per_trip_results = []

    for t_num, tid in enumerate(trip_ids):
        trip = load_trip(split_dir, tid, include_route=True)
        cand = generate_candidates(trip, top_k=TOP_K_CANDIDATES)
        aligned_true = trip_meta[t_num]["aligned_arcs"]

        mask = trip_group == t_num
        p_trip = probs[mask]
        o_trip = obs_idx[mask]
        a_trip = cand_arc_id[mask]

        n_obs = len(trip.track_lon)
        probs_per_obs: list[np.ndarray] = [np.empty(0) for _ in range(n_obs)]
        cls_pred = np.full(n_obs, -1, dtype=np.int64)

        # Slice candidate probabilities per observation preserving candidate order.
        order_breaks = np.flatnonzero(np.diff(o_trip)) + 1
        starts = np.concatenate(([0], order_breaks))
        ends = np.concatenate((order_breaks, [len(o_trip)]))
        for s_i, e_i in zip(starts, ends):
            t_i = int(o_trip[s_i])
            p_slice = p_trip[s_i:e_i]
            probs_per_obs[t_i] = p_slice
            cls_pred[t_i] = int(a_trip[s_i + int(np.argmax(p_slice))])

        hmm_pred, hmm_filled = viterbi_reconstruct(trip, cand, probs_per_obs, cfg=hmm_cfg)

        cls_acc = float(np.mean(cls_pred == aligned_true))
        hmm_acc = float(np.mean(hmm_pred == aligned_true))
        cls_correct += int(np.sum(cls_pred == aligned_true))
        hmm_correct += int(np.sum(hmm_pred == aligned_true))
        total_obs += n_obs

        cls_cont = compute_route_continuity(cls_pred, trip.arcs_from, trip.arcs_to)
        hmm_cont = compute_route_continuity(hmm_pred, trip.arcs_from, trip.arcs_to)
        cls_cont_list.append(cls_cont["obs_continuity"])
        hmm_cont_list.append(hmm_cont["obs_continuity"])

        cls_seq = compute_sequence_metrics(cls_pred, trip.route_arcs)
        hmm_seq = compute_sequence_metrics(hmm_pred, trip.route_arcs)
        hmm_fill_seq = compute_sequence_metrics(np.array(hmm_filled, dtype=np.int64), trip.route_arcs)

        cls_seq_f1_list.append(cls_seq["seq_lcs_f1"])
        hmm_seq_f1_list.append(hmm_seq["seq_lcs_f1"])
        hmm_filled_f1_list.append(hmm_fill_seq["seq_lcs_f1"])

        per_trip_results.append(
            {
                "trip_id": tid,
                "n_obs": n_obs,
                "cls_top1_acc": cls_acc,
                "hmm_top1_acc": hmm_acc,
                "cls_obs_continuity": cls_cont["obs_continuity"],
                "hmm_obs_continuity": hmm_cont["obs_continuity"],
                "cls_disjoint_jumps": cls_cont["disjoint_jumps"],
                "hmm_disjoint_jumps": hmm_cont["disjoint_jumps"],
                "cls_seq_lcs_f1": cls_seq["seq_lcs_f1"],
                "hmm_seq_lcs_f1": hmm_seq["seq_lcs_f1"],
                "hmm_gapfilled_seq_f1": hmm_fill_seq["seq_lcs_f1"],
            }
        )

    return {
        "obs_top1_acc_cls": float(cls_correct / max(1, total_obs)),
        "obs_top1_acc_hmm": float(hmm_correct / max(1, total_obs)),
        "mean_obs_continuity_cls": float(np.mean(cls_cont_list)),
        "mean_obs_continuity_hmm": float(np.mean(hmm_cont_list)),
        "mean_seq_lcs_f1_cls": float(np.mean(cls_seq_f1_list)),
        "mean_seq_lcs_f1_hmm": float(np.mean(hmm_seq_f1_list)),
        "mean_seq_lcs_f1_hmm_gapfilled": float(np.mean(hmm_filled_f1_list)),
        "per_trip": per_trip_results,
    }


def save_comparison_folium_map(
    split_dir: Path,
    trip_id: str,
    model,
    scaler: StandardScaler,
    hmm_cfg: HMMConfig,
    out_html: Path,
) -> None:
    """Save 4-layer Folium map: GPS, road network, reference route, and HMM predicted route."""
    trip = load_trip(split_dir, trip_id, include_route=True)
    cand = generate_candidates(trip, top_k=TOP_K_CANDIDATES)
    x_mat, obs_idx, _ = build_trip_features(trip, cand)
    probs = model.predict_proba(scaler.transform(x_mat))[:, 1]

    n_obs = len(trip.track_lon)
    probs_per_obs: list[np.ndarray] = [np.empty(0) for _ in range(n_obs)]
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

    lon_pad, lat_pad = 0.005, 0.0035
    min_lon, max_lon = trip.track_lon.min() - lon_pad, trip.track_lon.max() + lon_pad
    min_lat, max_lat = trip.track_lat.min() - lat_pad, trip.track_lat.max() + lat_pad

    net_fg = folium.FeatureGroup(name="Road Network (Local)", show=True)
    for u, v in zip(trip.arcs_from, trip.arcs_to):
        la1, lo1 = trip.nodes_lat[u], trip.nodes_lon[u]
        la2, lo2 = trip.nodes_lat[v], trip.nodes_lon[v]
        if (min_lat <= la1 <= max_lat and min_lon <= lo1 <= max_lon) or (
            min_lat <= la2 <= max_lat and min_lon <= lo2 <= max_lon
        ):
            folium.PolyLine([[la1, lo1], [la2, lo2]], color="#cbd5e1", weight=1.5, opacity=0.65).add_to(net_fg)
    net_fg.add_to(m)

    if trip.route_arcs is not None:
        ref_fg = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        for aid in trip.route_arcs:
            u, v = trip.arcs_from[aid], trip.arcs_to[aid]
            folium.PolyLine(
                [[trip.nodes_lat[u], trip.nodes_lon[u]], [trip.nodes_lat[v], trip.nodes_lon[v]]],
                color="#16a34a",
                weight=6,
                opacity=0.65,
            ).add_to(ref_fg)
        ref_fg.add_to(m)

    cls_fg = folium.FeatureGroup(name="Classifier-Only Route", show=False)
    for aid in cls_pred:
        if aid < 0:
            continue
        u, v = trip.arcs_from[aid], trip.arcs_to[aid]
        folium.PolyLine(
            [[trip.nodes_lat[u], trip.nodes_lon[u]], [trip.nodes_lat[v], trip.nodes_lon[v]]],
            color="#f59e0b",
            weight=4,
            opacity=0.8,
        ).add_to(cls_fg)
    cls_fg.add_to(m)

    hmm_fg = folium.FeatureGroup(name="Predicted Route (LightGBM + HMM/Viterbi)", show=True)
    for aid in hmm_filled:
        u, v = trip.arcs_from[aid], trip.arcs_to[aid]
        folium.PolyLine(
            [[trip.nodes_lat[u], trip.nodes_lon[u]], [trip.nodes_lat[v], trip.nodes_lon[v]]],
            color="#dc2626",
            weight=3.2,
            opacity=0.95,
        ).add_to(hmm_fg)
    hmm_fg.add_to(m)

    gps_fg = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    step = max(1, n_obs // 600)
    for idx in range(0, n_obs, step):
        folium.CircleMarker(
            [float(trip.track_lat[idx]), float(trip.track_lon[idx])],
            radius=2.3,
            color="#2563eb",
            fill=True,
            fill_opacity=0.85,
        ).add_to(gps_fg)
    gps_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))


def main() -> None:
    cache_dir = OUTPUTS_DIR / "cache"
    model_out_dir = OUTPUTS_DIR / "model1_lightgbm"
    model_out_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_data = prepare_split_dataset("train", TRAIN_DIR, cache_dir / "train_features_v2.joblib")
    test_data = prepare_split_dataset(
        "student_test", STUDENT_TEST_DIR, cache_dir / "student_test_features_v2.joblib"
    )

    x_train, y_train, groups_train = train_data["X"], train_data["y"], train_data["trip_group"]
    x_test, y_test = test_data["X"], test_data["y"]

    base_params = {
        "n_estimators": 250,
        "learning_rate": 0.06,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # 5-Fold GroupKFold CV on the 80 training trips (scaler fit strictly inside each fold).
    gkf = GroupKFold(n_splits=CV_FOLDS)
    oof_probs = np.zeros(len(y_train), dtype=np.float64)
    fold_macro_f1, fold_logloss, fold_train_times = [], [], []
    last_eval_result = {}

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(x_train, y_train, groups=groups_train)):
        t_start = time.perf_counter()
        scaler_fold = StandardScaler()
        x_tr_s = scaler_fold.fit_transform(x_train[tr_idx])
        x_va_s = scaler_fold.transform(x_train[va_idx])

        clf = lgb.LGBMClassifier(**base_params)
        clf.fit(
            x_tr_s,
            y_train[tr_idx],
            eval_set=[(x_tr_s, y_train[tr_idx]), (x_va_s, y_train[va_idx])],
            eval_names=["train", "val"],
            eval_metric="binary_logloss",
        )
        fold_train_times.append(time.perf_counter() - t_start)
        p_va = clf.predict_proba(x_va_s)[:, 1]
        oof_probs[va_idx] = p_va

        f_metrics = compute_candidate_metrics(y_train[va_idx], p_va)
        fold_macro_f1.append(f_metrics["macro_f1"])
        fold_logloss.append(f_metrics["log_loss"])
        if fold_idx == 0:
            last_eval_result = clf.evals_result_
        print(
            f"  Fold {fold_idx + 1}/{CV_FOLDS}: macro_f1={f_metrics['macro_f1']:.4f}, "
            f"log_loss={f_metrics['log_loss']:.4f}, time={fold_train_times[-1]:.2f}s"
        )

    cv_metrics = compute_candidate_metrics(y_train, oof_probs)
    cv_metrics["fold_macro_f1_mean"] = float(np.mean(fold_macro_f1))
    cv_metrics["fold_macro_f1_std"] = float(np.std(fold_macro_f1))

    # Plot real LightGBM iteration training vs validation log-loss curve (Fold 1).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(last_eval_result["train"]["binary_logloss"], label="Train Log Loss (Fold 1)")
    ax.plot(last_eval_result["val"]["binary_logloss"], label="Val Log Loss (Fold 1)")
    ax.set_title("LightGBM (Model 1) Training vs. Validation Log Loss")
    ax.set_xlabel("Boosting Iteration")
    ax.set_ylabel("Binary Log Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(model_out_dir / "lightgbm_logloss_curve.png", dpi=160)
    plt.close(fig)

    # Fit final scaler + LightGBM on all 80 training trips.
    t_fit = time.perf_counter()
    final_scaler = StandardScaler()
    x_train_scaled = final_scaler.fit_transform(x_train)
    final_model = lgb.LGBMClassifier(**base_params)
    final_model.fit(x_train_scaled, y_train)
    fit_time_s = time.perf_counter() - t_fit

    # Evaluate on the 10 student_test trips (untouched during scaler/model fitting).
    t_inf = time.perf_counter()
    x_test_scaled = final_scaler.transform(x_test)
    test_probs = final_model.predict_proba(x_test_scaled)[:, 1]
    inf_time_s = time.perf_counter() - t_inf

    test_cand_metrics = compute_candidate_metrics(y_test, test_probs)
    hmm_cfg = HMMConfig()
    route_eval = evaluate_trip_predictions(
        STUDENT_TEST_DIR,
        test_data["trip_ids"],
        test_data["trip_meta"],
        test_probs,
        test_data["trip_group"],
        test_data["obs_idx"],
        test_data["cand_arc_id"],
        hmm_cfg=hmm_cfg,
    )

    # Feature importance plot.
    imp_df = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    imp_df.to_csv(model_out_dir / "lightgbm_feature_importance.csv", index=False)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    ax_imp.barh(imp_df["feature"].head(15)[::-1], imp_df["importance"].head(15)[::-1], color="#2563eb")
    ax_imp.set_title("Top 15 Feature Importances — Model 1 (LightGBM)")
    ax_imp.set_xlabel("Split Importance")
    plt.tight_layout()
    fig_imp.savefig(model_out_dir / "lightgbm_feature_importance.png", dpi=160)
    plt.close(fig_imp)

    # Save exported model bundle for FastAPI inference.
    bundle = {
        "model_name": "LightGBM Baseline (Model 1)",
        "algorithm_family": "Boosting",
        "model": final_model,
        "scaler": final_scaler,
        "feature_names": FEATURE_NAMES,
        "hmm_config": hmm_cfg,
        "top_k_candidates": TOP_K_CANDIDATES,
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(bundle, ARTIFACTS_DIR / "model_bundle.joblib", compress=3)

    report = {
        "model_name": "LightGBM (Baseline)",
        "algorithm_family": "Boosting",
        "hyperparameters": base_params,
        "train_candidate_miss_rate": train_data["candidate_miss_rate"],
        "student_test_candidate_miss_rate": test_data["candidate_miss_rate"],
        "cv_metrics": cv_metrics,
        "student_test_candidate_metrics": test_cand_metrics,
        "student_test_route_eval": route_eval,
        "train_time_s": float(fit_time_s),
        "student_test_inference_time_s": float(inf_time_s),
    }
    (model_out_dir / "model1_results.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(route_eval["per_trip"]).to_csv(
        model_out_dir / "student_test_per_trip.csv", index=False
    )

    save_comparison_folium_map(
        STUDENT_TEST_DIR,
        "00000007",
        final_model,
        final_scaler,
        hmm_cfg,
        model_out_dir / "folium_student_test_00000007_pred.html",
    )

    print("\n=== MODEL 1 (LightGBM) SUMMARY ===")
    print(f"Train Candidate Miss Rate:        {train_data['candidate_miss_rate']:.4%}")
    print(f"Student Test Candidate Miss Rate: {test_data['candidate_miss_rate']:.4%}")
    print(f"CV Macro F1:                      {cv_metrics['macro_f1']:.4f} (+/- {cv_metrics['fold_macro_f1_std']:.4f})")
    print(f"Student Test Macro F1:            {test_cand_metrics['macro_f1']:.4f}")
    print(f"Student Test Log Loss:            {test_cand_metrics['log_loss']:.4f}")
    print(f"Obs Top-1 Acc (Classifier Only):  {route_eval['obs_top1_acc_cls']:.4%}")
    print(f"Obs Top-1 Acc (Classifier + HMM): {route_eval['obs_top1_acc_hmm']:.4%}")
    print(f"Obs Continuity (Classifier Only): {route_eval['mean_obs_continuity_cls']:.4%}")
    print(f"Obs Continuity (HMM/Viterbi):     {route_eval['mean_obs_continuity_hmm']:.4%}")
    print(f"Seq LCS F1 (Classifier Only):     {route_eval['mean_seq_lcs_f1_cls']:.4%}")
    print(f"Seq LCS F1 (HMM/Viterbi):         {route_eval['mean_seq_lcs_f1_hmm']:.4%}")
    print(f"Seq LCS F1 (HMM Gap-Filled):      {route_eval['mean_seq_lcs_f1_hmm_gapfilled']:.4%}")


if __name__ == "__main__":
    main()
