"""Evaluation metrics for candidate classification and route reconstruction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def collapse_consecutive(seq: np.ndarray | list[int]) -> list[int]:
    """Deduplicate adjacent identical arc IDs while preserving visit order."""
    out: list[int] = []
    for item in seq:
        val = int(item)
        if not out or out[-1] != val:
            out.append(val)
    return out


def compute_route_continuity(
    pred_obs_arcs: np.ndarray,
    arcs_from: np.ndarray,
    arcs_to: np.ndarray,
) -> dict[str, float]:
    """Compute fraction of valid consecutive transitions (same arc or v_prev == u_curr)."""
    n = len(pred_obs_arcs)
    if n <= 1:
        return {"obs_continuity": 1.0, "collapsed_continuity": 1.0, "disjoint_jumps": 0}

    same_or_conn = 0
    for a1, a2 in zip(pred_obs_arcs[:-1], pred_obs_arcs[1:]):
        if a1 == a2 or arcs_to[a1] == arcs_from[a2]:
            same_or_conn += 1

    collapsed = collapse_consecutive(pred_obs_arcs)
    if len(collapsed) <= 1:
        col_cont = 1.0
        jumps = 0
    else:
        conn = sum(
            1
            for a1, a2 in zip(collapsed[:-1], collapsed[1:])
            if arcs_to[a1] == arcs_from[a2]
        )
        jumps = (len(collapsed) - 1) - conn
        col_cont = conn / float(len(collapsed) - 1)

    return {
        "obs_continuity": same_or_conn / float(n - 1),
        "collapsed_continuity": col_cont,
        "disjoint_jumps": int(jumps),
    }


def _lcs_length(a: list[int], b: list[int]) -> int:
    """Compute Longest Common Subsequence length in O(min(|a|, |b|)) memory."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            cur = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = dp[j] if dp[j] >= dp[j - 1] else dp[j - 1]
            prev = cur
    return dp[-1]


def compute_sequence_metrics(
    pred_obs_arcs: np.ndarray,
    true_route_arcs: np.ndarray,
) -> dict[str, float]:
    """Compare collapsed predicted arc sequence against ground-truth .route sequence."""
    pred_seq = collapse_consecutive(pred_obs_arcs)
    true_seq = [int(x) for x in true_route_arcs]

    lcs = _lcs_length(pred_seq, true_seq)
    seq_prec = lcs / float(len(pred_seq)) if pred_seq else 0.0
    seq_rec = lcs / float(len(true_seq)) if true_seq else 0.0
    seq_f1 = (
        2.0 * seq_prec * seq_rec / (seq_prec + seq_rec)
        if (seq_prec + seq_rec) > 0
        else 0.0
    )

    pred_set, true_set = set(pred_seq), set(true_seq)
    inter = len(pred_set & true_set)
    union = len(pred_set | true_set)
    arc_set_jaccard = inter / float(union) if union else 0.0

    return {
        "seq_lcs_precision": seq_prec,
        "seq_lcs_recall": seq_rec,
        "seq_lcs_f1": seq_f1,
        "arc_set_jaccard": arc_set_jaccard,
        "pred_collapsed_len": len(pred_seq),
        "true_route_len": len(true_seq),
    }


def compute_candidate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Compute binary candidate classification metrics and confusion matrix."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1.0 - 1e-6))),
        "confusion_matrix": cm.tolist(),
    }
