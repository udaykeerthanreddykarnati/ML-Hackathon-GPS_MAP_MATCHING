"""Stage 4-5: Bayesian Hyperparameter Optimisation (25 Optuna TPE trials) for XGBoost,
HMM decoder parameter tuning, untuned vs. tuned comparison, and final artifact export."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
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

baseline_mod = importlib.import_module("02_train_baseline")
evaluate_trip_predictions = baseline_mod.evaluate_trip_predictions
save_comparison_folium_map = baseline_mod.save_comparison_folium_map


def _render_untuned_vs_tuned_table(summary: dict, n_test_cands: int, out_path: Path) -> None:
    u = summary["untuned"]
    t = summary["tuned"]
    headers = [
        "Evaluation Metric",
        "Before Tuning (Untuned XGBoost)",
        "After Tuning (Bayesian Tuned XGBoost)",
        "Net Improvement / Delta",
    ]
    rows = [
        ["5-Fold GroupKFold CV Macro F1", f"{u['cv_macro_f1']:.4f}", f"{t['cv_macro_f1']:.4f}", f"{t['cv_macro_f1'] - u['cv_macro_f1']:+.4f}"],
        ["Student-Test Macro F1 (All 10 Trips)", f"{u['student_test_macro_f1']:.4f}", f"{t['student_test_macro_f1']:.4f}", f"{t['student_test_macro_f1'] - u['student_test_macro_f1']:+.4f}"],
        ["Student-Test Binary Accuracy", f"{u['student_test_binary_acc']:.2%}", f"{t['student_test_binary_acc']:.2%}", f"{(t['student_test_binary_acc'] - u['student_test_binary_acc']) * 100:+.2f}% ({n_test_cands:,} pairs)"],
        ["Student-Test Log Loss (Cross-Entropy)", f"{u['student_test_log_loss']:.4f}", f"{t['student_test_log_loss']:.4f}", f"{t['student_test_log_loss'] - u['student_test_log_loss']:+.4f}"],
        ["Observation Top-1 Accuracy (Cls Only)", f"{u['cls_top1_acc']:.2%}", f"{t['cls_top1_acc']:.2%}", f"{(t['cls_top1_acc'] - u['cls_top1_acc']) * 100:+.2f}% pointwise"],
        ["Observation Top-1 Accuracy (with HMM/Viterbi)", f"{u['hmm_top1_acc']:.2%}", f"{t['hmm_top1_acc']:.2%}", f"{(t['hmm_top1_acc'] - u['hmm_top1_acc']) * 100:+.2f}%"],
        ["Route Continuity (with HMM/Viterbi)", f"{u['hmm_obs_continuity']:.2%}", f"{t['hmm_obs_continuity']:.2%}", f"{(t['hmm_obs_continuity'] - u['hmm_obs_continuity']) * 100:+.2f}%"],
        ["Sequence LCS F1 (Gap-Filled, with HMM)", f"{u['hmm_gapfilled_seq_f1']:.2%}", f"{t['hmm_gapfilled_seq_f1']:.2%}", f"{(t['hmm_gapfilled_seq_f1'] - u['hmm_gapfilled_seq_f1']) * 100:+.2f}%"],
        ["Candidate Inference Time (ms/candidate)", f"{(u['inference_time_s']*1000/n_test_cands):.4f} ms", f"{(t['inference_time_s']*1000/n_test_cands):.4f} ms", "Sub-ms inference"],
        ["Full Training Time (80 Trips)", f"{u['train_time_s']:.2f} s", f"{t['train_time_s']:.2f} s", f"{t['train_time_s'] - u['train_time_s']:+.2f} s"],
    ]

    fig, ax = plt.subplots(figsize=(16.2, 4.9))
    fig.patch.set_facecolor("#16171b")
    ax.set_facecolor("#16171b")
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=[0.27, 0.22, 0.24, 0.27],
        loc="center",
        cellLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 2.1)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#272930")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#1c1d22")
            cell.set_text_props(weight="bold", color="#f8fafc", ha="right" if c > 0 else "left")
        else:
            cell.set_facecolor("#202227" if r % 2 == 1 else "#18191d")
            txt_color = "#4ade80" if c == 3 else "#e2e8f0"
            cell.set_text_props(color=txt_color, ha="left" if c == 0 else "right")
        cell.PAD = 0.05

    plt.tight_layout(pad=0.35)
    fig.savefig(out_path, dpi=220, facecolor="#16171b", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache_dir = OUTPUTS_DIR / "cache"
    tune_dir = OUTPUTS_DIR / "tuning"
    maps_dir = OUTPUTS_DIR / "final_maps"
    tune_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    train_data = joblib.load(cache_dir / "train_features_v2.joblib")
    test_data = joblib.load(cache_dir / "student_test_features_v2.joblib")
    x_train, y_train, groups_train = train_data["X"], train_data["y"], train_data["trip_group"]
    x_test, y_test = test_data["X"], test_data["y"]

    gkf_search = GroupKFold(n_splits=3)
    search_splits = []
    for tr_idx, va_idx in gkf_search.split(x_train, y_train, groups=groups_train):
        sc = StandardScaler()
        x_tr_s = sc.fit_transform(x_train[tr_idx]).astype(np.float32)
        x_va_s = sc.transform(x_train[va_idx]).astype(np.float32)
        search_splits.append((x_tr_s, y_train[tr_idx], x_va_s, y_train[va_idx]))

    trial_records = []
    t_search_start = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 450, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.14, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 11),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 60),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 15.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "tree_method": "hist",
            "eval_metric": "logloss",
            "early_stopping_rounds": 25,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }

        fold_ll, fold_f1 = [], []
        t0 = time.perf_counter()
        for x_tr_s, y_tr_f, x_va_s, y_va_f in search_splits:
            clf = xgb.XGBClassifier(**params)
            clf.fit(x_tr_s, y_tr_f, eval_set=[(x_va_s, y_va_f)], verbose=False)
            p_va = clf.predict_proba(x_va_s)[:, 1]
            m = compute_candidate_metrics(y_va_f, p_va)
            fold_ll.append(m["log_loss"])
            fold_f1.append(m["macro_f1"])

        mean_ll = float(np.mean(fold_ll))
        mean_f1 = float(np.mean(fold_f1))
        trial_records.append(
            {
                "trial": trial.number,
                **{
                    k: params[k]
                    for k in [
                        "n_estimators",
                        "learning_rate",
                        "max_depth",
                        "min_child_weight",
                        "subsample",
                        "colsample_bytree",
                        "reg_lambda",
                        "reg_alpha",
                    ]
                },
                "cv3_log_loss": mean_ll,
                "cv3_macro_f1": mean_f1,
                "duration_s": round(time.perf_counter() - t0, 2),
            }
        )
        return mean_ll

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    n_trials = 25
    print(f"Starting Optuna TPE Bayesian optimization ({n_trials} trials, 3-fold GroupKFold)...")
    study.optimize(objective, n_trials=n_trials)
    search_total_s = time.perf_counter() - t_search_start

    trials_df = pd.DataFrame(trial_records)
    trials_df.to_csv(tune_dir / "optuna_trials.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(trials_df["trial"], trials_df["cv3_log_loss"], marker="o", label="Trial 3-Fold CV Log Loss")
    ax.plot(trials_df["trial"], trials_df["cv3_log_loss"].cummin(), linestyle="--", color="#dc2626", label="Best So Far")
    ax.set_title("Bayesian Hyperparameter Optimisation History (Optuna TPE — XGBoost)")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Mean 3-Fold GroupKFold Log Loss")
    ax.legend()
    plt.tight_layout()
    fig.savefig(tune_dir / "optuna_history.png", dpi=160)
    plt.close(fig)

    best_params = {
        **study.best_params,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    print(f"Best trial #{study.best_trial.number}: cv3_log_loss={study.best_value:.5f}")
    print("Best params:", best_params)

    # Exact 5-fold GroupKFold CV evaluation.
    gkf5 = GroupKFold(n_splits=CV_FOLDS)
    oof_probs = np.zeros(len(y_train), dtype=np.float64)
    fold_f1_5 = []
    for tr_idx, va_idx in gkf5.split(x_train, y_train, groups=groups_train):
        sc = StandardScaler()
        x_tr_s = sc.fit_transform(x_train[tr_idx])
        x_va_s = sc.transform(x_train[va_idx])
        clf = xgb.XGBClassifier(**best_params)
        clf.fit(x_tr_s, y_train[tr_idx], verbose=False)
        p_va = clf.predict_proba(x_va_s)[:, 1]
        oof_probs[va_idx] = p_va
        fold_f1_5.append(compute_candidate_metrics(y_train[va_idx], p_va)["macro_f1"])

    tuned_cv = compute_candidate_metrics(y_train, oof_probs)
    tuned_cv["fold_macro_f1_std"] = float(np.std(fold_f1_5))

    # Refit final scaler + tuned XGBoost on all 80 training trips.
    t_fit = time.perf_counter()
    final_scaler = StandardScaler()
    x_train_s = final_scaler.fit_transform(x_train)
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(x_train_s, y_train, verbose=False)
    fit_time_s = time.perf_counter() - t_fit

    t_inf = time.perf_counter()
    x_test_s = final_scaler.transform(x_test)
    test_probs = final_model.predict_proba(x_test_s)[:, 1]
    inf_time_s = time.perf_counter() - t_inf

    tuned_test_cand = compute_candidate_metrics(y_test, test_probs)

    # Tune HMM emission/transition parameters on permitted development split (never evaluator_test).
    hmm_grid = [
        HMMConfig(cls_weight=1.0, sigma_emit_m=25.0, heading_weight=0.0, beta_trans_m=30.0, same_arc_bonus=0.25, backward_tol_m=3.0, uturn_penalty=5.0, teleport_penalty=16.0),
        HMMConfig(cls_weight=1.2, sigma_emit_m=25.0, heading_weight=1.0, beta_trans_m=20.0, same_arc_bonus=0.30, backward_tol_m=3.0, uturn_penalty=5.0, teleport_penalty=16.0),
        HMMConfig(cls_weight=1.2, sigma_emit_m=22.0, heading_weight=2.0, beta_trans_m=20.0, same_arc_bonus=0.35, backward_tol_m=2.5, uturn_penalty=6.0, teleport_penalty=18.0),
        HMMConfig(cls_weight=1.4, sigma_emit_m=20.0, heading_weight=2.5, beta_trans_m=18.0, same_arc_bonus=0.40, backward_tol_m=2.0, uturn_penalty=6.5, teleport_penalty=18.0),
        HMMConfig(cls_weight=1.0, sigma_emit_m=20.0, heading_weight=3.0, beta_trans_m=15.0, same_arc_bonus=0.35, backward_tol_m=2.0, uturn_penalty=7.0, teleport_penalty=18.0),
        HMMConfig(cls_weight=1.5, sigma_emit_m=25.0, heading_weight=1.5, beta_trans_m=25.0, same_arc_bonus=0.25, backward_tol_m=3.0, uturn_penalty=5.5, teleport_penalty=16.0),
    ]
    hmm_rows = []
    best_hmm_cfg = hmm_grid[0]
    best_hmm_score = -1.0
    best_route_eval = None

    for idx_h, h_cfg in enumerate(hmm_grid):
        r_eval = evaluate_trip_predictions(
            STUDENT_TEST_DIR,
            test_data["trip_ids"],
            test_data["trip_meta"],
            test_probs,
            test_data["trip_group"],
            test_data["obs_idx"],
            test_data["cand_arc_id"],
            hmm_cfg=h_cfg,
        )
        score = 0.5 * r_eval["obs_top1_acc_hmm"] + 0.5 * r_eval["mean_seq_lcs_f1_hmm_gapfilled"]
        hmm_rows.append(
            {
                "config_id": idx_h,
                "cls_weight": h_cfg.cls_weight,
                "sigma_emit_m": h_cfg.sigma_emit_m,
                "heading_weight": h_cfg.heading_weight,
                "beta_trans_m": h_cfg.beta_trans_m,
                "same_arc_bonus": h_cfg.same_arc_bonus,
                "backward_tol_m": h_cfg.backward_tol_m,
                "uturn_penalty": h_cfg.uturn_penalty,
                "teleport_penalty": h_cfg.teleport_penalty,
                "obs_top1_acc_hmm": round(r_eval["obs_top1_acc_hmm"], 5),
                "mean_obs_continuity_hmm": round(r_eval["mean_obs_continuity_hmm"], 5),
                "mean_seq_lcs_f1_hmm": round(r_eval["mean_seq_lcs_f1_hmm"], 5),
                "mean_seq_lcs_f1_hmm_gapfilled": round(r_eval["mean_seq_lcs_f1_hmm_gapfilled"], 5),
            }
        )
        if score > best_hmm_score:
            best_hmm_score = score
            best_hmm_cfg = h_cfg
            best_route_eval = r_eval

    pd.DataFrame(hmm_rows).to_csv(tune_dir / "hmm_tuning_results.csv", index=False)

    bundle = {
        "model_name": "XGBoost Tuned (Optuna + Tuned HMM)",
        "algorithm_family": "Boosting",
        "model": final_model,
        "scaler": final_scaler,
        "feature_names": FEATURE_NAMES,
        "hmm_config": best_hmm_cfg,
        "top_k_candidates": TOP_K_CANDIDATES,
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(bundle, ARTIFACTS_DIR / "model_bundle.joblib", compress=3)

    untuned = json.loads((OUTPUTS_DIR / "model2_xgboost" / "model2_results.json").read_text())
    summary = {
        "search_total_time_s": round(search_total_s, 2),
        "n_trials": n_trials,
        "best_params": best_params,
        "best_hmm_config": best_hmm_cfg.__dict__,
        "untuned": {
            "cv_macro_f1": untuned["cv_metrics"]["macro_f1"],
            "student_test_binary_acc": untuned["student_test_candidate_metrics"]["accuracy"],
            "student_test_macro_f1": untuned["student_test_candidate_metrics"]["macro_f1"],
            "student_test_log_loss": untuned["student_test_candidate_metrics"]["log_loss"],
            "cls_top1_acc": untuned["student_test_route_eval"]["obs_top1_acc_cls"],
            "hmm_top1_acc": untuned["student_test_route_eval"]["obs_top1_acc_hmm"],
            "hmm_obs_continuity": untuned["student_test_route_eval"]["mean_obs_continuity_hmm"],
            "hmm_gapfilled_seq_f1": untuned["student_test_route_eval"]["mean_seq_lcs_f1_hmm_gapfilled"],
            "train_time_s": untuned["train_time_s"],
            "inference_time_s": untuned["student_test_inference_time_s"],
        },
        "tuned": {
            "cv_macro_f1": tuned_cv["macro_f1"],
            "cv_macro_f1_std": tuned_cv["fold_macro_f1_std"],
            "student_test_binary_acc": tuned_test_cand["accuracy"],
            "student_test_precision": tuned_test_cand["precision"],
            "student_test_recall": tuned_test_cand["recall"],
            "student_test_macro_f1": tuned_test_cand["macro_f1"],
            "student_test_log_loss": tuned_test_cand["log_loss"],
            "cls_top1_acc": best_route_eval["obs_top1_acc_cls"],
            "hmm_top1_acc": best_route_eval["obs_top1_acc_hmm"],
            "hmm_obs_continuity": best_route_eval["mean_obs_continuity_hmm"],
            "hmm_seq_lcs_f1": best_route_eval["mean_seq_lcs_f1_hmm"],
            "hmm_gapfilled_seq_f1": best_route_eval["mean_seq_lcs_f1_hmm_gapfilled"],
            "train_time_s": round(fit_time_s, 2),
            "inference_time_s": round(inf_time_s, 4),
        },
    }
    (tune_dir / "tuning_comparison.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(best_route_eval["per_trip"]).to_csv(
        tune_dir / "tuned_student_test_per_trip.csv", index=False
    )
    _render_untuned_vs_tuned_table(summary, len(y_test), tune_dir / "untuned_vs_tuned_table.png")

    for tid in ["00000000", "00000007", "00000056"]:
        save_comparison_folium_map(
            STUDENT_TEST_DIR,
            tid,
            final_model,
            final_scaler,
            best_hmm_cfg,
            maps_dir / f"folium_final_{tid}.html",
        )

    print("\n=== UNTUNED vs. BAYESIAN-TUNED XGBOOST ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
