"""Vectorized segment-densified cKDTree candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from .config import ARC_SAMPLE_STEP_M, MAX_CAND_RADIUS_M, TOP_K_CANDIDATES
from .data_loader import TripData
from .geo import LocalProjection, point_to_segments_geometry


@dataclass
class CandidateTable:
    """Candidate road arcs and raw point-to-segment geometry for one trip."""

    proj: LocalProjection
    track_xy: np.ndarray
    nodes_xy: np.ndarray
    arc_len_m: np.ndarray
    arc_unit_dir: np.ndarray
    cand_arc_ids: list[np.ndarray]
    cand_geom: list[dict[str, np.ndarray]]


def densify_arcs(
    nodes_xy: np.ndarray,
    arcs_from: np.ndarray,
    arcs_to: np.ndarray,
    step_m: float = ARC_SAMPLE_STEP_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample points along every directed arc segment for fast spatial indexing."""
    a_xy = nodes_xy[arcs_from]
    b_xy = nodes_xy[arcs_to]
    ab = b_xy - a_xy
    arc_len_m = np.sqrt(np.sum(ab * ab, axis=1))
    safe_len = np.where(arc_len_m > 1e-9, arc_len_m, 1.0)
    arc_unit_dir = ab / safe_len[:, None]

    counts = np.clip(np.ceil(arc_len_m / step_m).astype(np.int64) + 1, 2, 400)
    sample_arc_ids = np.repeat(np.arange(len(arcs_from), dtype=np.int64), counts)

    # Vectorized local parameter t in [0, 1] per segment without Python loops.
    cum_counts = np.cumsum(counts)
    starts = cum_counts - counts
    local_idx = np.arange(len(sample_arc_ids), dtype=np.float64) - np.repeat(starts, counts)
    denom = np.repeat(counts - 1, counts).astype(np.float64)
    t = local_idx / np.maximum(denom, 1.0)

    sample_xy = a_xy[sample_arc_ids] + t[:, None] * ab[sample_arc_ids]
    return sample_xy, sample_arc_ids, arc_len_m, arc_unit_dir


def generate_candidates(
    trip: TripData,
    top_k: int = TOP_K_CANDIDATES,
    radius_m: float = MAX_CAND_RADIUS_M,
    step_m: float = ARC_SAMPLE_STEP_M,
) -> CandidateTable:
    """Generate top-K nearest candidate road arcs for every GPS observation in a trip."""
    proj = LocalProjection.from_lon_lat(trip.track_lon, trip.track_lat)
    track_xy = proj.to_xy(trip.track_lon, trip.track_lat)
    nodes_xy = proj.to_xy(trip.nodes_lon, trip.nodes_lat)

    sample_xy, sample_arc_ids, arc_len_m, arc_unit_dir = densify_arcs(
        nodes_xy, trip.arcs_from, trip.arcs_to, step_m=step_m
    )
    tree = cKDTree(sample_xy)
    ball_hits = tree.query_ball_point(track_xy, r=radius_m)

    cand_arc_ids: list[np.ndarray] = []
    cand_geom: list[dict[str, np.ndarray]] = []
    k_fallback = min(len(sample_xy), max(64, top_k * 6))

    for t_idx in range(len(track_xy)):
        pt = track_xy[t_idx]
        hits = ball_hits[t_idx]
        unique_arcs = (
            np.unique(sample_arc_ids[np.asarray(hits, dtype=np.int64)])
            if hits
            else np.empty(0, dtype=np.int64)
        )

        # Fallback nearest-neighbor query if fewer than top_k candidates lie within radius_m.
        if len(unique_arcs) < top_k and k_fallback > 0:
            _, nn_idx = tree.query(pt, k=k_fallback)
            nn_arcs = np.unique(sample_arc_ids[np.atleast_1d(nn_idx)])
            unique_arcs = np.unique(np.concatenate([unique_arcs, nn_arcs]))

        seg_a = nodes_xy[trip.arcs_from[unique_arcs]]
        seg_b = nodes_xy[trip.arcs_to[unique_arcs]]
        geom = point_to_segments_geometry(pt, seg_a, seg_b)

        order = np.argsort(geom["dist_m"])[:top_k]
        cand_arc_ids.append(unique_arcs[order])
        cand_geom.append({k: v[order] for k, v in geom.items()})

    return CandidateTable(
        proj=proj,
        track_xy=track_xy,
        nodes_xy=nodes_xy,
        arc_len_m=arc_len_m,
        arc_unit_dir=arc_unit_dir,
        cand_arc_ids=cand_arc_ids,
        cand_geom=cand_geom,
    )
