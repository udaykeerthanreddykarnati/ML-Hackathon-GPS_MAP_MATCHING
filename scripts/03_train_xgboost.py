"""Model 2: XGBoost (histogram boosting) training, 5-fold GroupKFold CV,
HMM/Viterbi evaluation, and head-to-head comparison against Model 1 (LightGBM)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathsnap.config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    STUDENT_TEST_DIR,
    TOP_K_CANDIDATES,
)
from pathsnap.features import FEATURE_NAMES
from pathsnap.hmm import HMMConfig
from pathsnap.metrics import compute_candidate_metrics

# Reuse exact evaluation logic from 02_train_baseline for strict comparability.
import importlib
baseline_mod = importlib.import_module("02_train_baseline")
evaluate_trip_predictions = baseline_mod.evaluate_trip_predictions


def main() -> None:
    cache_dir = OUTPUTS_DIR / "cache"
    out_dir = OUTPUTS_DIR / "model2_xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_cache = cache_dir / "train_features_v2.joblib"
    test_cache = cache_dir / "student_test_features_v2.joblib"
    if not train_cache.exists() or not test_cache.exists():
        raise RuntimeError("Feature caches not found. Run scripts/02_train_baseline.py first.")

    train_data = joblib.load(train_cache)
    test_data = joblib.load(test_cache)

    x_train, y_train, groups_train = train_data["X"], train_data["y"], train_data["trip_group"]
    x_test, y_test = test_data["X"], test_data["y"]

    xgb_params = {
        "n_estimators": 250,
        "learning_rate": 0.06,
        "max_depth": 7,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "reg_alpha": 0.5,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    gkf = GroupKFold(n_splits=CV_FOLDS)
    oof_probs = np.zeros(len(y_train), dtype=np.float64)
    fold_macro_f1, fold_logloss, fold_train_times = [], [], []
    fold1_history: dict = {}

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(x_train, y_train, groups=groups_train)):
        t_start = time.perf_counter()
        scaler_fold = StandardScaler()
        x_tr_s = scaler_fold.fit_transform(x_train[tr_idx])
        x_va_s = scaler_fold.transform(x_train[va_idx])

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(
            x_tr_s,
            y_train[tr_idx],
            eval_set=[(x_tr_s, y_train[tr_idx]), (x_va_s, y_train[va_idx])],
            verbose=False,
        )
        fold_train_times.append(time.perf_counter() - t_start)

        p_va = clf.predict_proba(x_va_s)[:, 1]
        oof_probs[va_idx] = p_va

        f_metrics = compute_candidate_metrics(y_train[va_idx], p_va)
        fold_macro_f1.append(f_metrics["macro_f1"])
        fold_logloss.append(f_metrics["log_loss"])
        if fold_idx == 0:
            fold1_history = clf.evals_result()
        print(
            f"  Fold {fold_idx + 1}/{CV_FOLDS}: macro_f1={f_metrics['macro_f1']:.4f}, "
            f"log_loss={f_metrics['log_loss']:.4f}, time={fold_train_times[-1]:.2f}s"
        )

    cv_metrics = compute_candidate_metrics(y_train, oof_probs)
    cv_metrics["fold_macro_f1_mean"] = float(np.mean(fold_macro_f1))
    cv_metrics["fold_macro_f1_std"] = float(np.std(fold_macro_f1))

    # Plot real XGBoost iteration train/validation log-loss curve (Fold 1).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(fold1_history["validation_0"]["logloss"], label="Train Log Loss (Fold 1)")
    ax.plot(fold1_history["validation_1"]["logloss"], label="Val Log Loss (Fold 1)")
    ax.set_title("XGBoost (Model 2) Training vs. Validation Log Loss")
    ax.set_xlabel("Boosting Iteration")
    ax.set_ylabel("Binary Log Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "xgboost_logloss_curve.png", dpi=160)
    plt.close(fig)

    # Fit final scaler + XGBoost on all 80 training trips.
    t_fit = time.perf_counter()
    final_scaler = StandardScaler()
    x_train_scaled = final_scaler.fit_transform(x_train)
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(x_train_scaled, y_train, verbose=False)
    fit_time_s = time.perf_counter() - t_fit

    # Evaluate on the 10 student_test trips.
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
    imp_df.to_csv(out_dir / "xgboost_feature_importance.csv", index=False)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    ax_imp.barh(imp_df["feature"].head(15)[::-1], imp_df["importance"].head(15)[::-1], color="#059669")
    ax_imp.set_title("Top 15 Feature Importances — Model 2 (XGBoost)")
    ax_imp.set_xlabel("Gain Importance")
    plt.tight_layout()
    fig_imp.savefig(out_dir / "xgboost_feature_importance.png", dpi=160)
    plt.close(fig_imp)

    bundle = {
        "model_name": "XGBoost Hist (Model 2)",
        "algorithm_family": "Boosting",
        "model": final_model,
        "scaler": final_scaler,
        "feature_names": FEATURE_NAMES,
        "hmm_config": hmm_cfg,
        "top_k_candidates": TOP_K_CANDIDATES,
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(bundle, ARTIFACTS_DIR / "model2_xgboost_bundle.joblib", compress=3)

    report = {
        "model_name": "XGBoost (Hist)",
        "algorithm_family": "Boosting",
        "hyperparameters": xgb_params,
        "train_candidate_miss_rate": train_data["candidate_miss_rate"],
        "student_test_candidate_miss_rate": test_data["candidate_miss_rate"],
        "cv_metrics": cv_metrics,
        "student_test_candidate_metrics": test_cand_metrics,
        "student_test_route_eval": route_eval,
        "train_time_s": float(fit_time_s),
        "student_test_inference_time_s": float(inf_time_s),
    }
    (out_dir / "model2_results.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(route_eval["per_trip"]).to_csv(
        out_dir / "student_test_per_trip.csv", index=False
    )

    # Check against Model 1 and promote to active bundle if superior on HMM Seq F1 / Log Loss.
    m1_path = OUTPUTS_DIR / "model1_lightgbm" / "model1_results.json"
    promoted = False
    if m1_path.exists():
        m1 = json.loads(m1_path.read_text())
        m1_seq = m1["student_test_route_eval"]["mean_seq_lcs_f1_hmm_gapfilled"]
        m2_seq = route_eval["mean_seq_lcs_f1_hmm_gapfilled"]
        m1_top1 = m1["student_test_route_eval"]["obs_top1_acc_hmm"]
        m2_top1 = route_eval["obs_top1_acc_hmm"]
        print("\n=== MODEL 1 (LightGBM) vs. MODEL 2 (XGBoost) ===")
        print(f"CV Macro F1:            {m1['cv_metrics']['macro_f1']:.4f}  vs  {cv_metrics['macro_f1']:.4f}")
        print(f"Student Test Macro F1:  {m1['student_test_candidate_metrics']['macro_f1']:.4f}  vs  {test_cand_metrics['macro_f1']:.4f}")
        print(f"Student Test Log Loss:  {m1['student_test_candidate_metrics']['log_loss']:.4f}  vs  {test_cand_metrics['log_loss']:.4f}")
        print(f"HMM Obs Top-1 Acc:      {m1_top1:.4%}  vs  {m2_top1:.4%}")
        print(f"HMM Gap-Filled Seq F1:  {m1_seq:.4%}  vs  {m2_seq:.4%}")
        print(f"Inference Time (s):     {m1['student_test_inference_time_s']:.3f}s  vs  {inf_time_s:.3f}s")
        if (m2_top1, m2_seq) > (m1_top1, m1_seq):
            joblib.dump(bundle, ARTIFACTS_DIR / "model_bundle.joblib", compress=3)
            promoted = True
    print(f"\nPromoted XGBoost to active artifacts/model_bundle.joblib: {promoted}")


if __name__ == "__main__":
    main()
