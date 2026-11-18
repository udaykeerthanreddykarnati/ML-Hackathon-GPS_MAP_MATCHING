"""Models 3-6: LogisticRegression, ExtraTrees, PyTorch MLP, and Nystroem+Calibrated LinearSVC.
Evaluates all models with identical 5-fold GroupKFold CV and student_test HMM/Viterbi,
and generates consolidated comparison tables, confusion matrices, and plots."""

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
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathsnap.config import ARTIFACTS_DIR, CV_FOLDS, OUTPUTS_DIR, RANDOM_SEED, STUDENT_TEST_DIR
from pathsnap.hmm import HMMConfig
from pathsnap.metrics import compute_candidate_metrics

baseline_mod = importlib.import_module("02_train_baseline")
evaluate_trip_predictions = baseline_mod.evaluate_trip_predictions


class _MLPNet(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PyTorchMLPWrapper:
    """Mini-batch PyTorch MLP with epoch-wise train/validation binary log-loss tracking."""

    def __init__(
        self,
        in_dim: int,
        epochs: int = 12,
        batch_size: int = 4096,
        lr: float = 2e-3,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.in_dim = in_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model: _MLPNet | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def fit(
        self,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray | None = None,
        y_va: np.ndarray | None = None,
    ) -> "PyTorchMLPWrapper":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = _MLPNet(self.in_dim).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        ds = TensorDataset(
            torch.from_numpy(np.asarray(x_tr, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_tr, dtype=np.float32)),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.train_losses = []
        self.val_losses = []

        for _ in range(self.epochs):
            self.model.train()
            running_loss, seen = 0.0, 0
            for bx, by in loader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(by)
                seen += len(by)
            self.train_losses.append(running_loss / max(1, seen))

            if x_va is not None and y_va is not None:
                self.model.eval()
                val_ds = TensorDataset(
                    torch.from_numpy(np.asarray(x_va, dtype=np.float32)),
                    torch.from_numpy(np.asarray(y_va, dtype=np.float32)),
                )
                val_loader = DataLoader(val_ds, batch_size=16384, shuffle=False)
                v_loss, v_seen = 0.0, 0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx = bx.to(self.device)
                        by = by.to(self.device)
                        v_loss += float(criterion(self.model(bx), by).item()) * len(by)
                        v_seen += len(by)
                self.val_losses.append(v_loss / max(1, v_seen))

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        x_arr = np.asarray(x, dtype=np.float32)
        probs = np.empty(len(x_arr), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(x_arr), 32768):
                end = min(len(x_arr), start + 32768)
                bx = torch.from_numpy(x_arr[start:end]).to(self.device)
                p = torch.sigmoid(self.model(bx)).cpu().numpy()
                probs[start:end] = p
        return np.column_stack((1.0 - probs, probs))


def run_one_model(
    key: str,
    display_name: str,
    family: str,
    key_config: str,
    make_estimator,
    train_data: dict,
    test_data: dict,
    out_dir: Path,
) -> dict:
    """Run 5-fold GroupKFold CV + final fit + student_test & HMM/Viterbi evaluation."""
    print(f"\n=== Training {display_name} ({family}) ===")
    x_train, y_train, groups_train = train_data["X"], train_data["y"], train_data["trip_group"]
    x_test, y_test = test_data["X"], test_data["y"]

    gkf = GroupKFold(n_splits=CV_FOLDS)
    oof_probs = np.zeros(len(y_train), dtype=np.float64)
    fold_macro_f1, fold_times = [], []
    mlp_history = None

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(x_train, y_train, groups=groups_train)):
        t0 = time.perf_counter()
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_train[tr_idx])
        x_va_s = scaler.transform(x_train[va_idx])

        est = make_estimator()
        if isinstance(est, PyTorchMLPWrapper):
            est.fit(x_tr_s, y_train[tr_idx], x_va_s, y_train[va_idx])
            if fold_idx == 0:
                mlp_history = (list(est.train_losses), list(est.val_losses))
        else:
            est.fit(x_tr_s, y_train[tr_idx])

        dt = time.perf_counter() - t0
        fold_times.append(dt)
        p_va = est.predict_proba(x_va_s)[:, 1]
        oof_probs[va_idx] = p_va
        fm = compute_candidate_metrics(y_train[va_idx], p_va)
        fold_macro_f1.append(fm["macro_f1"])
        print(
            f"  Fold {fold_idx + 1}/{CV_FOLDS}: macro_f1={fm['macro_f1']:.4f}, "
            f"log_loss={fm['log_loss']:.4f}, time={dt:.2f}s"
        )

    cv_metrics = compute_candidate_metrics(y_train, oof_probs)
    cv_metrics["fold_macro_f1_mean"] = float(np.mean(fold_macro_f1))
    cv_metrics["fold_macro_f1_std"] = float(np.std(fold_macro_f1))

    if mlp_history is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(range(1, len(mlp_history[0]) + 1), mlp_history[0], marker="o", label="Train BCE Loss (Fold 1)")
        ax.plot(range(1, len(mlp_history[1]) + 1), mlp_history[1], marker="s", label="Val BCE Loss (Fold 1)")
        ax.set_title("PyTorch MLP (Model 5) Epoch Training vs. Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Cross-Entropy Loss")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "mlp_epoch_loss_curve.png", dpi=160)
        plt.close(fig)

    # Fit on all 80 training trips.
    t_fit = time.perf_counter()
    final_scaler = StandardScaler()
    x_tr_all = final_scaler.fit_transform(x_train)
    final_est = make_estimator()
    final_est.fit(x_tr_all, y_train)
    fit_time_s = time.perf_counter() - t_fit

    # Predict on 10 student_test trips.
    t_inf = time.perf_counter()
    x_te_all = final_scaler.transform(x_test)
    test_probs = final_est.predict_proba(x_te_all)[:, 1]
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

    result = {
        "key": key,
        "model_name": display_name,
        "algorithm_family": family,
        "key_config": key_config,
        "cv_metrics": cv_metrics,
        "student_test_candidate_metrics": test_cand_metrics,
        "student_test_route_eval": route_eval,
        "train_time_s": float(fit_time_s),
        "student_test_inference_time_s": float(inf_time_s),
    }
    (out_dir / f"{key}_results.json").write_text(json.dumps(result, indent=2))
    print(
        f"-> {display_name}: CV_F1={cv_metrics['macro_f1']:.4f}, "
        f"Test_F1={test_cand_metrics['macro_f1']:.4f}, LogLoss={test_cand_metrics['log_loss']:.4f}, "
        f"ClsTop1={route_eval['obs_top1_acc_cls']:.4%}, HMMTop1={route_eval['obs_top1_acc_hmm']:.4%}, "
        f"HMMSeqF1={route_eval['mean_seq_lcs_f1_hmm_gapfilled']:.4%}, InfTime={inf_time_s:.3f}s"
    )
    return result


def build_comparison_outputs(all_results: list[dict], comp_dir: Path) -> None:
    """Create consolidated comparison CSV, performance bar charts, and confusion matrices."""
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
                "CV_Macro_F1": round(cv["macro_f1"], 4),
                "CV_Macro_F1_Std": round(cv.get("fold_macro_f1_std", 0.0), 4),
                "Cls_Top1_Accuracy": round(re_["obs_top1_acc_cls"], 4),
                "HMM_Top1_Accuracy": round(re_["obs_top1_acc_hmm"], 4),
                "Cls_Route_Continuity": round(re_["mean_obs_continuity_cls"], 4),
                "HMM_Route_Continuity": round(re_["mean_obs_continuity_hmm"], 4),
                "HMM_GapFilled_Seq_F1": round(re_["mean_seq_lcs_f1_hmm_gapfilled"], 4),
                "Student_Test_Macro_F1": round(te["macro_f1"], 4),
                "Student_Test_Precision": round(te["precision"], 4),
                "Student_Test_Recall": round(te["recall"], 4),
                "Log_Loss": round(te["log_loss"], 4),
                "Train_Time_s": round(r["train_time_s"], 2),
                "Inference_Time_s": round(r["student_test_inference_time_s"], 3),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(comp_dir / "all_models_comparison.csv", index=False)

    # Bar charts across all 6 models.
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = df["Model"]

    axes[0, 0].barh(names, df["CV_Macro_F1"], xerr=df["CV_Macro_F1_Std"], color="#2563eb")
    axes[0, 0].set_title("5-Fold GroupKFold CV Macro F1-score")
    axes[0, 0].set_xlim(0.5, 1.0)

    x_pos = np.arange(len(df))
    w = 0.35
    axes[0, 1].bar(x_pos - w / 2, df["Cls_Top1_Accuracy"], width=w, label="Classifier Only", color="#f59e0b")
    axes[0, 1].bar(x_pos + w / 2, df["HMM_Top1_Accuracy"], width=w, label="Classifier + HMM", color="#16a34a")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(names, rotation=20, ha="right")
    axes[0, 1].set_title("Student-Test Observation Top-1 Accuracy (Before vs. After HMM)")
    axes[0, 1].set_ylim(0.5, 1.0)
    axes[0, 1].legend()

    axes[1, 0].barh(names, df["Log_Loss"], color="#dc2626")
    axes[1, 0].set_title("Student-Test Binary Log Loss (Lower is Better)")

    axes[1, 1].barh(names, df["Inference_Time_s"], color="#7c3aed")
    axes[1, 1].set_title("Student-Test Inference Time (s, 179k Candidate Pairs)")

    plt.tight_layout()
    fig.savefig(comp_dir / "model_comparison_charts.png", dpi=160)
    plt.close(fig)

    # Confusion matrices grid.
    n_models = len(all_results)
    cols = 3
    rows_g = int(np.ceil(n_models / cols))
    fig_cm, axes_cm = plt.subplots(rows_g, cols, figsize=(4.5 * cols, 4 * rows_g))
    axes_flat = np.atleast_1d(axes_cm).ravel()
    for i, r in enumerate(all_results):
        cm = np.array(r["student_test_candidate_metrics"]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes_flat[i],
            xticklabels=["Neg (0)", "Pos (1)"],
            yticklabels=["Neg (0)", "Pos (1)"],
        )
        axes_flat[i].set_title(r["model_name"])
        axes_flat[i].set_xlabel("Predicted")
        axes_flat[i].set_ylabel("True")
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    fig_cm.savefig(comp_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig_cm)


def main() -> None:
    cache_dir = OUTPUTS_DIR / "cache"
    comp_dir = OUTPUTS_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    train_data = joblib.load(cache_dir / "train_features_v2.joblib")
    test_data = joblib.load(cache_dir / "student_test_features_v2.joblib")
    in_dim = train_data["X"].shape[1]

    specs = [
        (
            "model3_logreg",
            "Logistic Regression",
            "Linear & Probabilistic",
            "C=1.0, solver=lbfgs, max_iter=500",
            lambda: LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED),
        ),
        (
            "model4_extratrees",
            "Extra Trees",
            "Ensemble (Bagging)",
            "n_estimators=120, max_depth=18, min_samples_leaf=20",
            lambda: ExtraTreesClassifier(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=20,
                max_features="sqrt",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
        ),
        (
            "model5_pytorch_mlp",
            "PyTorch MLP",
            "Neural Networks",
            "128-64-1 BN+ReLU+Dropout, AdamW lr=2e-3, 10 epochs",
            lambda: PyTorchMLPWrapper(in_dim=in_dim, epochs=10, batch_size=4096, lr=2e-3),
        ),
        (
            "model6_nystroem_svc",
            "Nystroem RBF + Calibrated LinearSVC",
            "Kernel-Based",
            "Nystroem(rbf, gamma=0.05, m=120) + Calibrated LinearSVC(C=1.0)",
            lambda: Pipeline(
                [
                    (
                        "nystroem",
                        Nystroem(
                            kernel="rbf",
                            gamma=0.05,
                            n_components=120,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                    (
                        "cal_svc",
                        CalibratedClassifierCV(
                            LinearSVC(C=1.0, max_iter=1500, random_state=RANDOM_SEED),
                            cv=3,
                            method="sigmoid",
                        ),
                    ),
                ]
            ),
        ),
    ]

    new_results = [
        run_one_model(k, name, fam, cfg_str, fn, train_data, test_data, comp_dir)
        for k, name, fam, cfg_str, fn in specs
    ]

    # Combine with Model 1 (LightGBM) and Model 2 (XGBoost).
    all_results = []
    m1_path = OUTPUTS_DIR / "model1_lightgbm" / "model1_results.json"
    if m1_path.exists():
        m1 = json.loads(m1_path.read_text())
        m1["key_config"] = "n_estimators=250, lr=0.06, num_leaves=63, min_child=40"
        all_results.append(m1)
    m2_path = OUTPUTS_DIR / "model2_xgboost" / "model2_results.json"
    if m2_path.exists():
        m2 = json.loads(m2_path.read_text())
        m2["key_config"] = "n_estimators=250, lr=0.06, max_depth=7, tree_method=hist"
        all_results.append(m2)
    all_results.extend(new_results)

    build_comparison_outputs(all_results, comp_dir)
    print("\n=== ALL MODELS CONSOLIDATED COMPARISON ===")
    print(pd.read_csv(comp_dir / "all_models_comparison.csv").to_string(index=False))


if __name__ == "__main__":
    main()
