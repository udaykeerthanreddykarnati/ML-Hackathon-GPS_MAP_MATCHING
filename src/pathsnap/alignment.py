"""Offline-only monotonic alignment of .track GPS points onto .route ground truth."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from .candidates import densify_arcs
from .data_loader import TripData
from .geo import estimate_track_kinematics, point_to_segments_geometry


@dataclass
class RouteAlignment:
    """Aligned per-observation reference arc IDs and route-index positions."""

    aligned_route_pos: np.ndarray
    aligned_arc_ids: np.ndarray
    point_to_ref_dist_m: np.ndarray


def align_track_to_route(
    trip: TripData,
    track_xy: np.ndarray,
    nodes_xy: np.ndarray,
    sigma_dist_m: float = 25.0,
    heading_weight: float = 1.5,
) -> RouteAlignment:
    """Align GPS points monotonically onto the ordered .route sequence via sparse Viterbi DP."""
    if trip.route_arcs is None or len(trip.route_arcs) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no .route loaded for alignment.")

    route = trip.route_arcs
    m_len = len(route)
    t_len = len(track_xy)

    r_from = trip.arcs_from[route]
    r_to = trip.arcs_to[route]
    sample_xy, sample_pos, route_arc_lens, _ = densify_arcs(
        nodes_xy, r_from, r_to, step_m=20.0
    )
    cum_len = np.concatenate(([0.0], np.cumsum(route_arc_lens)))

    kin = estimate_track_kinematics(track_xy, trip.track_ts)
    headings = kin["headings"]
    heading_conf = kin["heading_conf"]

    tree = cKDTree(sample_xy)
    k_q = min(len(sample_xy), 96)
    _, nn_indices = tree.query(track_xy, k=k_q)
    if nn_indices.ndim == 1:
        nn_indices = nn_indices[:, None]

    # Precompute route arc endpoints for exact distances.
    r_a = nodes_xy[r_from]
    r_b = nodes_xy[r_to]

    def eval_states(t_idx: int, pos_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        geom = point_to_segments_geometry(track_xy[t_idx], r_a[pos_arr], r_b[pos_arr])
        cos_sim = np.sum(geom["unit_dir"] * headings[t_idx][None, :], axis=1)
        emit_cost = (geom["dist_m"] / sigma_dist_m) ** 2 + (
            heading_weight * heading_conf[t_idx] * (1.0 - cos_sim)
        )
        return emit_cost, geom["dist_m"]

    dp_states: list[np.ndarray] = []
    dp_costs: list[np.ndarray] = []
    dp_back: list[np.ndarray] = []
    dp_dists: list[np.ndarray] = []

    # Step 0 candidate route positions.
    pos0 = np.unique(sample_pos[nn_indices[0]])
    if 0 not in pos0:
        pos0 = np.unique(np.append(pos0, 0))
    cost0, dist0 = eval_states(0, pos0)
    # Mild tie-breaker favoring earlier start position on self-intersecting routes.
    cost0 = cost0 + 0.01 * pos0
    dp_states.append(pos0)
    dp_costs.append(cost0)
    dp_back.append(np.full(len(pos0), -1, dtype=np.int64))
    dp_dists.append(dist0)

    for t in range(1, t_len):
        prev_pos = dp_states[t - 1]
        prev_cost = dp_costs[t - 1]

        near_pos = np.unique(sample_pos[nn_indices[t]])
        # Keep previous active positions and immediate successors to avoid dead ends.
        best_prev_pos = int(prev_pos[np.argmin(prev_cost)])
        extra = np.array(
            [p for p in (best_prev_pos, min(m_len - 1, best_prev_pos + 1)) if p >= 0],
            dtype=np.int64,
        )
        curr_pos = np.unique(np.concatenate([near_pos, extra]))
        # Trim to closest 28 positions by Euclidean midpoint/endpoint check if large.
        if len(curr_pos) > 28:
            g_tmp = point_to_segments_geometry(track_xy[t], r_a[curr_pos], r_b[curr_pos])
            keep = np.argsort(g_tmp["dist_m"])[:28]
            curr_pos = np.sort(curr_pos[keep])

        emit_cost, dist_t = eval_states(t, curr_pos)
        gps_step_m = float(np.linalg.norm(track_xy[t] - track_xy[t - 1]))

        curr_cost = np.full(len(curr_pos), np.inf, dtype=np.float64)
        back_idx = np.full(len(curr_pos), -1, dtype=np.int64)

        for j, m_curr in enumerate(curr_pos):
            valid_mask = prev_pos <= m_curr
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                m_prev_vals = prev_pos[valid_indices]
                route_step_m = cum_len[m_curr] - cum_len[m_prev_vals]
                # Penalize large mismatch between route advance and GPS step distance.
                step_diff = np.abs(route_step_m - gps_step_m) / max(sigma_dist_m, 10.0)
                hop_pen = 0.03 * (m_curr - m_prev_vals)
                cand_totals = prev_cost[valid_indices] + 0.35 * step_diff + hop_pen
                best_local = int(np.argmin(cand_totals))
                curr_cost[j] = cand_totals[best_local] + emit_cost[j]
                back_idx[j] = int(valid_indices[best_local])
            else:
                # Rare GPS backward jump fallback: heavy penalty instead of infeasible -inf.
                back_pen = 20.0 + 2.0 * (prev_pos - m_curr)
                cand_totals = prev_cost + back_pen
                best_local = int(np.argmin(cand_totals))
                curr_cost[j] = cand_totals[best_local] + emit_cost[j]
                back_idx[j] = best_local

        dp_states.append(curr_pos)
        dp_costs.append(curr_cost)
        dp_back.append(back_idx)
        dp_dists.append(dist_t)

    # Backtrack from minimum final state (preferring end of route on ties).
    end_costs = dp_costs[-1] + 0.01 * np.maximum(0, (m_len - 1) - dp_states[-1])
    cur_idx = int(np.argmin(end_costs))

    aligned_pos = np.zeros(t_len, dtype=np.int64)
    aligned_dist = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len - 1, -1, -1):
        aligned_pos[t] = dp_states[t][cur_idx]
        aligned_dist[t] = dp_dists[t][cur_idx]
        cur_idx = int(dp_back[t][cur_idx])

    return RouteAlignment(
        aligned_route_pos=aligned_pos,
        aligned_arc_ids=route[aligned_pos],
        point_to_ref_dist_m=aligned_dist,
    )
