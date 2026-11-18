"""Model 7: Random Forest Classifier benchmark on the exact 31-feature 5-fold GroupKFold
and student_test splits, updating all consolidated comparison CSVs and plots to 7 models."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathsnap.config import OUTPUTS_DIR, RANDOM_SEED

rem_mod = importlib.import_module("04_train_remaining_models")
run_one_model = rem_mod.run_one_model
build_comparison_outputs = rem_mod.build_comparison_outputs


def main() -> None:
    cache_dir = OUTPUTS_DIR / "cache"
    comp_dir = OUTPUTS_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    train_data = joblib.load(cache_dir / "train_features_v2.joblib")
    test_data = joblib.load(cache_dir / "student_test_features_v2.joblib")

    rf_result = run_one_model(
        "model7_random_forest",
        "Random Forest",
        "Ensemble (Bagging)",
        "n_estimators=100, max_depth=16, min_samples_leaf=20, max_features=sqrt",
        lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        train_data,
        test_data,
        comp_dir,
    )

    m1 = json.loads((OUTPUTS_DIR / "model1_lightgbm" / "model1_results.json").read_text())
    m1["key_config"] = "n_estimators=250, lr=0.06, num_leaves=63, min_child=40"
    m2 = json.loads((OUTPUTS_DIR / "model2_xgboost" / "model2_results.json").read_text())
    m2["key_config"] = "n_estimators=250, lr=0.06, max_depth=7, tree_method=hist"

    all_results = [m1, m2]
    for fn in [
        "model3_logreg_results.json",
        "model4_extratrees_results.json",
        "model5_pytorch_mlp_results.json",
        "model6_nystroem_svc_results.json",
    ]:
        all_results.append(json.loads((comp_dir / fn).read_text()))
    all_results.append(rf_result)

    build_comparison_outputs(all_results, comp_dir)

    # Enrich all_models_comparison.csv with binary accuracy and candidate miss rate.
    miss_rate_test = round(float(m1["student_test_candidate_miss_rate"]), 5)
    rows = []
    for r in all_results:
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
    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(comp_dir / "all_models_comparison.csv", index=False)

    # Update Student-Test Top-1 & Macro F1 chart across all 7 models.
    fig2, ax2 = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(df_cmp))
    w = 0.27
    ax2.bar(x - w, df_cmp["Cls_Top1_Accuracy"], width=w, label="Cls-Only Top-1 Acc", color="#f59e0b")
    ax2.bar(x, df_cmp["HMM_Top1_Accuracy"], width=w, label="Classifier + HMM Top-1 Acc", color="#16a34a")
    ax2.bar(x + w, df_cmp["Student_Test_Macro_F1"], width=w, label="Student-Test Macro F1", color="#2563eb")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_cmp["Model"], rotation=18, ha="right")
    ax2.set_ylim(0.6, 1.0)
    ax2.set_title("Student-Testing Top-1 Accuracy and Macro F1-score Across All 7 Models")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(comp_dir / "student_test_top1_and_macro_f1.png", dpi=160)
    plt.close(fig2)

    print("\n=== UPDATED 7-MODEL COMPARISON ===")
    print(df_cmp.to_string(index=False))


if __name__ == "__main__":
    main()
