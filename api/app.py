"""FastAPI backend (Laptop 1) for PathSnap distributed map-matching and server-side evaluation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import sys
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.alignment import align_track_to_route
from pathsnap.candidates import generate_candidates
from pathsnap.config import ARTIFACTS_DIR, EVAL_DIR, EVAL_SPLIT_NAME, TOP_K_CANDIDATES
from pathsnap.data_loader import list_trip_ids, load_trip
from pathsnap.features import build_trip_features
from pathsnap.hmm import HMMConfig, viterbi_reconstruct
from pathsnap.metrics import (
    compute_candidate_metrics,
    compute_route_continuity,
    compute_sequence_metrics,
)

BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.joblib"

@asynccontextmanager
async def _lifespan(app_instance: FastAPI):
    trips = list_trip_ids(EVAL_DIR)
    _log_cmd(
        f"STARTUP | model='{MODEL_BUNDLE['model_name']}' | eval_dir='{EVAL_DIR}' "
        f"| split='{EVAL_SPLIT_NAME}' | trips_found={len(trips)}"
    )
    yield


app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed map-matching & Viterbi route reconstruction service (Laptop 1).",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """Payload containing only the selected evaluation trip identifier."""

    trip_id: str


def _load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise RuntimeError(
            f"Model bundle not found at {BUNDLE_PATH}. Run scripts/02_train_baseline.py first."
        )
    return joblib.load(BUNDLE_PATH)


MODEL_BUNDLE = _load_bundle()


def _arcs_to_polylines(trip, arc_ids: list[int] | np.ndarray) -> list[list[list[float]]]:
    """Convert arc IDs into Leaflet-friendly [[lat1, lon1], [lat2, lon2]] segments."""
    segments: list[list[list[float]]] = []
    for aid in arc_ids:
        a = int(aid)
        if a < 0:
            continue
        u = int(trip.arcs_from[a])
        v = int(trip.arcs_to[a])
        segments.append(
            [
                [round(float(trip.nodes_lat[u]), 6), round(float(trip.nodes_lon[u]), 6)],
                [round(float(trip.nodes_lat[v]), 6), round(float(trip.nodes_lon[v]), 6)],
            ]
        )
    return segments


def _local_network_polylines(trip, max_arcs: int = 5000) -> list[list[list[float]]]:
    """Extract local road-network segments around the GPS trajectory bounding box."""
    lon_pad = 0.0045
    lat_pad = 0.0032
    min_lon = float(trip.track_lon.min()) - lon_pad
    max_lon = float(trip.track_lon.max()) + lon_pad
    min_lat = float(trip.track_lat.min()) - lat_pad
    max_lat = float(trip.track_lat.max()) + lat_pad

    u_lat = trip.nodes_lat[trip.arcs_from]
    u_lon = trip.nodes_lon[trip.arcs_from]
    v_lat = trip.nodes_lat[trip.arcs_to]
    v_lon = trip.nodes_lon[trip.arcs_to]

    in_box = (
        ((u_lat >= min_lat) & (u_lat <= max_lat) & (u_lon >= min_lon) & (u_lon <= max_lon))
        | ((v_lat >= min_lat) & (v_lat <= max_lat) & (v_lon >= min_lon) & (v_lon <= max_lon))
    )
    arc_indices = np.flatnonzero(in_box)
    if len(arc_indices) > max_arcs:
        step = int(np.ceil(len(arc_indices) / max_arcs))
        arc_indices = arc_indices[::step]

    return _arcs_to_polylines(trip, arc_indices)


def _log_cmd(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@app.get("/health")
def health() -> dict:
    """Report server health, loaded model metadata, and available evaluation trips."""
    trips = list_trip_ids(EVAL_DIR)
    return {
        "status": "ok",
        "model_name": MODEL_BUNDLE["model_name"],
        "algorithm_family": MODEL_BUNDLE["algorithm_family"],
        "eval_split": EVAL_SPLIT_NAME,
        "trip_count": len(trips),
    }


@app.get("/api/trips")
def get_trips(request: Request) -> dict:
    """List available evaluation trips (metadata from .track only; never reads .route)."""
    client_host = request.client.host if request.client else "unknown"
    trip_ids = list_trip_ids(EVAL_DIR)
    items = []
    for tid in trip_ids:
        trip = load_trip(EVAL_DIR, tid, include_route=False)
        items.append(
            {
                "trip_id": tid,
                "n_gps": int(len(trip.track_lon)),
                "duration_s": round(float(trip.track_ts[-1] - trip.track_ts[0]), 1),
                "n_nodes": int(len(trip.nodes_lon)),
                "n_arcs": int(len(trip.arcs_from)),
                "center_lat": round(float(np.median(trip.track_lat)), 6),
                "center_lon": round(float(np.median(trip.track_lon)), 6),
            }
        )
    _log_cmd(
        f"GET /api/trips | client={client_host} | split={EVAL_SPLIT_NAME} | returned={len(items)} trips"
    )
    return {"eval_split": EVAL_SPLIT_NAME, "trips": items}


@app.post("/api/predict")
def predict_trip(payload: PredictRequest, request: Request) -> dict:
    """Run live map-matching inference and server-side evaluation for one trip."""
    client_host = request.client.host if request.client else "unknown"
    valid_ids = set(list_trip_ids(EVAL_DIR))
    if payload.trip_id not in valid_ids:
        _log_cmd(f"POST /api/predict | client={client_host} | trip={payload.trip_id} | status=404")
        raise HTTPException(status_code=404, detail=f"Unknown trip_id: {payload.trip_id}")

    t_total_start = time.perf_counter()

    # 1. Prediction pipeline strictly WITHOUT .route access.
    trip = load_trip(EVAL_DIR, payload.trip_id, include_route=False)

    t0 = time.perf_counter()
    cand = generate_candidates(
        trip, top_k=int(MODEL_BUNDLE.get("top_k_candidates", TOP_K_CANDIDATES))
    )
    cand_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    x_mat, obs_idx, cand_arc_id = build_trip_features(trip, cand)
    feat_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    scaler = MODEL_BUNDLE["scaler"]
    model = MODEL_BUNDLE["model"]
    probs = model.predict_proba(scaler.transform(x_mat))[:, 1]
    model_ms = (time.perf_counter() - t2) * 1000.0

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

    t3 = time.perf_counter()
    hmm_cfg: HMMConfig = MODEL_BUNDLE["hmm_config"]
    hmm_pred, hmm_filled = viterbi_reconstruct(trip, cand, probs_per_obs, cfg=hmm_cfg)
    viterbi_ms = (time.perf_counter() - t3) * 1000.0

    cls_cont = compute_route_continuity(cls_pred, trip.arcs_from, trip.arcs_to)
    hmm_cont = compute_route_continuity(hmm_pred, trip.arcs_from, trip.arcs_to)

    # 2. Server-side evaluation using internal .route (labels NEVER leave this function).
    eval_summary = None
    trip_with_ref = load_trip(EVAL_DIR, payload.trip_id, include_route=True)
    if trip_with_ref.route_arcs is not None and len(trip_with_ref.route_arcs) > 0:
        alignment = align_track_to_route(trip_with_ref, cand.track_xy, cand.nodes_xy)
        aligned_true = alignment.aligned_arc_ids
        y_true = (cand_arc_id == aligned_true[obs_idx]).astype(np.int64)
        cand_metrics = compute_candidate_metrics(y_true, probs)

        missed = sum(
            1
            for o_i in range(n_obs)
            if int(aligned_true[o_i]) not in cand.cand_arc_ids[o_i]
        )
        cls_seq = compute_sequence_metrics(cls_pred, trip_with_ref.route_arcs)
        hmm_seq = compute_sequence_metrics(hmm_pred, trip_with_ref.route_arcs)
        hmm_fill_seq = compute_sequence_metrics(
            np.array(hmm_filled, dtype=np.int64), trip_with_ref.route_arcs
        )

        eval_summary = {
            "reference_available": True,
            "candidate_miss_rate": round(float(missed / max(1, n_obs)), 5),
            "cls_candidate_macro_f1": round(float(cand_metrics["macro_f1"]), 5),
            "cls_candidate_log_loss": round(float(cand_metrics["log_loss"]), 5),
            "cls_obs_top1_acc": round(float(np.mean(cls_pred == aligned_true)), 5),
            "hmm_obs_top1_acc": round(float(np.mean(hmm_pred == aligned_true)), 5),
            "cls_seq_lcs_f1": round(float(cls_seq["seq_lcs_f1"]), 5),
            "hmm_seq_lcs_f1": round(float(hmm_seq["seq_lcs_f1"]), 5),
            "hmm_gapfilled_seq_lcs_f1": round(float(hmm_fill_seq["seq_lcs_f1"]), 5),
            "hmm_arc_set_jaccard": round(float(hmm_fill_seq["arc_set_jaccard"]), 5),
        }

    total_ms = (time.perf_counter() - t_total_start) * 1000.0

    # Downsample classifier-only segments deduplicated for clean map rendering.
    cls_collapsed = []
    for aid in cls_pred:
        a = int(aid)
        if a >= 0 and (not cls_collapsed or cls_collapsed[-1] != a):
            cls_collapsed.append(a)

    eval_log = (
        f"cls_top1={eval_summary['cls_obs_top1_acc']:.3f} "
        f"hmm_top1={eval_summary['hmm_obs_top1_acc']:.3f} "
        f"hmm_seq_f1={eval_summary['hmm_gapfilled_seq_lcs_f1']:.3f}"
        if eval_summary
        else "no_server_ref"
    )
    _log_cmd(
        f"POST /api/predict | client={client_host} | trip={payload.trip_id} | "
        f"gps={n_obs} | cands={len(obs_idx)} | time={total_ms:.1f}ms "
        f"(cand={cand_ms:.0f}, feat={feat_ms:.0f}, ml={model_ms:.0f}, hmm={viterbi_ms:.0f}) | "
        f"hmm_cont={hmm_cont['obs_continuity']:.4f} | {eval_log} | status=200"
    )

    return {
        "trip_id": payload.trip_id,
        "eval_split": EVAL_SPLIT_NAME,
        "model_name": MODEL_BUNDLE["model_name"],
        "n_gps": int(n_obs),
        "n_candidates_evaluated": int(len(obs_idx)),
        "gps_points": [
            [round(float(la), 6), round(float(lo), 6)]
            for la, lo in zip(trip.track_lat, trip.track_lon)
        ],
        "local_network_segments": _local_network_polylines(trip),
        "cls_route_segments": _arcs_to_polylines(trip, cls_collapsed),
        "hmm_route_segments": _arcs_to_polylines(trip, hmm_filled),
        "hmm_predicted_arc_ids": [int(a) for a in hmm_filled],
        "continuity": {
            "cls_obs_continuity": round(float(cls_cont["obs_continuity"]), 5),
            "hmm_obs_continuity": round(float(hmm_cont["obs_continuity"]), 5),
            "cls_disjoint_jumps": int(cls_cont["disjoint_jumps"]),
            "hmm_disjoint_jumps": int(hmm_cont["disjoint_jumps"]),
            "cls_collapsed_arcs": len(cls_collapsed),
            "hmm_gapfilled_arcs": len(hmm_filled),
        },
        "timings_ms": {
            "candidates_ms": round(cand_ms, 2),
            "features_ms": round(feat_ms, 2),
            "model_ms": round(model_ms, 2),
            "viterbi_ms": round(viterbi_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "evaluation": eval_summary,
    }
