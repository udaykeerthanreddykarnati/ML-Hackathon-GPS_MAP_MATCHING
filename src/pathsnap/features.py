"""Coordinate-invariant feature extraction for (GPS observation, candidate arc) pairs."""

from __future__ import annotations

import numpy as np

from .candidates import CandidateTable
from .data_loader import TripData
from .geo import estimate_track_kinematics, point_to_segments_geometry

FEATURE_NAMES = [
    "dist_to_arc_m",
    "signed_lateral_m",
    "abs_lateral_m",
    "proj_frac",
    "dist_to_from_m",
    "dist_to_to_m",
    "arc_len_m",
    "log_arc_len",
    "dist_rank",
    "dist_diff_best",
    "dist_ratio_best",
    "cos_heading_diff",
    "sin_heading_diff",
    "heading_conf",
    "cos_heading_weighted",
    "gps_speed_mps",
    "log_dt_prev",
    "log_dt_next",
    "prev_obs_dist_m",
    "next_obs_dist_m",
    "prev2_obs_dist_m",
    "next2_obs_dist_m",
    "mean_5pt_dist_m",
    "proj_delta_m",
    "from_in_deg",
    "to_out_deg",
    "connects_from_prev",
    "connects_to_next",
    "connects_2hop_prev",
    "connects_2hop_next",
    "has_reverse_arc",
]


def _compute_graph_stats(
    n_nodes: int,
    arcs_from: np.ndarray,
    arcs_to: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], list[list[int]]]:
    """Compute node degrees, reverse-twin flags, and 1-hop node incidence lists."""
    in_deg = np.bincount(arcs_to, minlength=n_nodes).astype(np.float64)
    out_deg = np.bincount(arcs_from, minlength=n_nodes).astype(np.float64)

    base = np.int64(n_nodes + 1)
    fwd_keys = arcs_from.astype(np.int64) * base + arcs_to.astype(np.int64)
    rev_keys = arcs_to.astype(np.int64) * base + arcs_from.astype(np.int64)
    sorted_fwd = np.sort(fwd_keys)
    idx = np.searchsorted(sorted_fwd, rev_keys)
    valid = idx < len(sorted_fwd)
    has_rev = np.zeros(len(arcs_from), dtype=np.float64)
    has_rev[valid] = (sorted_fwd[idx[valid]] == rev_keys[valid]).astype(np.float64)

    out_neighbors: list[list[int]] = [[] for _ in range(n_nodes)]
    in_neighbors: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in zip(arcs_from, arcs_to):
        ui, vi = int(u), int(v)
        out_neighbors[ui].append(vi)
        in_neighbors[vi].append(ui)

    return in_deg, out_deg, has_rev, out_neighbors, in_neighbors


def build_trip_features(
    trip: TripData,
    cand: CandidateTable,
    neighbor_top_n: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 31-feature matrix X and index arrays (obs_idx, cand_arc_id) for one trip.

    Never reads .route or any reference label.
    """
    n_obs = len(cand.track_xy)
    n_nodes = len(cand.nodes_xy)
    in_deg, out_deg, has_rev, out_neighbors, in_neighbors = _compute_graph_stats(
        n_nodes, trip.arcs_from, trip.arcs_to
    )
    kin = estimate_track_kinematics(cand.track_xy, trip.track_ts)

    # Precompute 1-hop and 2-hop reachable endpoint node sets from top candidates at each step.
    top_arcs_per_step: list[set[int]] = []
    top_to_1hop: list[set[int]] = []
    top_to_2hop: list[set[int]] = []
    top_from_1hop: list[set[int]] = []
    top_from_2hop: list[set[int]] = []

    for t in range(n_obs):
        c_top = cand.cand_arc_ids[t][:neighbor_top_n]
        top_arcs_per_step.append(set(int(a) for a in c_top))
        to1 = {int(trip.arcs_to[a]) for a in c_top}
        from1 = {int(trip.arcs_from[a]) for a in c_top}
        to2 = set(to1)
        for node in to1:
            to2.update(out_neighbors[node])
        from2 = set(from1)
        for node in from1:
            from2.update(in_neighbors[node])
        top_to_1hop.append(to1)
        top_to_2hop.append(to2)
        top_from_1hop.append(from1)
        top_from_2hop.append(from2)

    total_rows = sum(len(a) for a in cand.cand_arc_ids)
    x_mat = np.zeros((total_rows, len(FEATURE_NAMES)), dtype=np.float32)
    obs_idx_arr = np.zeros(total_rows, dtype=np.int64)
    arc_id_arr = np.zeros(total_rows, dtype=np.int64)

    row_cursor = 0
    for t in range(n_obs):
        arcs_t = cand.cand_arc_ids[t]
        k = len(arcs_t)
        if k == 0:
            continue
        geom = cand.cand_geom[t]
        dist_m = geom["dist_m"]
        min_d = float(dist_m[0])

        u_nodes = trip.arcs_from[arcs_t]
        v_nodes = trip.arcs_to[arcs_t]
        seg_a = cand.nodes_xy[u_nodes]
        seg_b = cand.nodes_xy[v_nodes]

        prev_xy = cand.track_xy[t - 1] if t > 0 else cand.track_xy[t]
        next_xy = cand.track_xy[t + 1] if t + 1 < n_obs else cand.track_xy[t]
        prev2_xy = cand.track_xy[t - 2] if t > 1 else prev_xy
        next2_xy = cand.track_xy[t + 2] if t + 2 < n_obs else next_xy

        prev_geom = point_to_segments_geometry(prev_xy, seg_a, seg_b)
        next_geom = point_to_segments_geometry(next_xy, seg_a, seg_b)
        prev2_dist = point_to_segments_geometry(prev2_xy, seg_a, seg_b)["dist_m"]
        next2_dist = point_to_segments_geometry(next2_xy, seg_a, seg_b)["dist_m"]
        prev_dist = prev_geom["dist_m"]
        next_dist = next_geom["dist_m"]

        mean_5pt = 0.2 * (prev2_dist + prev_dist + dist_m + next_dist + next2_dist)
        # Signed forward progress (meters) along this directed arc from t-1 to t (or t to t+1 at t=0).
        if t > 0:
            proj_delta_m = (geom["proj_frac"] - prev_geom["proj_frac"]) * geom["arc_len_m"]
        else:
            proj_delta_m = (next_geom["proj_frac"] - geom["proj_frac"]) * geom["arc_len_m"]

        h_vec = kin["headings"][t]
        h_conf = float(kin["heading_conf"][t])
        u_dir = geom["unit_dir"]
        cos_h = u_dir[:, 0] * h_vec[0] + u_dir[:, 1] * h_vec[1]
        sin_h = h_vec[0] * u_dir[:, 1] - h_vec[1] * u_dir[:, 0]

        conn_prev = np.zeros(k, dtype=np.float64)
        conn_2hop_prev = np.zeros(k, dtype=np.float64)
        if t > 0:
            p_arcs = top_arcs_per_step[t - 1]
            p_to1 = top_to_1hop[t - 1]
            p_to2 = top_to_2hop[t - 1]
            for j in range(k):
                a_id = int(arcs_t[j])
                u_id = int(u_nodes[j])
                if a_id in p_arcs or u_id in p_to1:
                    conn_prev[j] = 1.0
                    conn_2hop_prev[j] = 1.0
                elif u_id in p_to2:
                    conn_2hop_prev[j] = 1.0

        conn_next = np.zeros(k, dtype=np.float64)
        conn_2hop_next = np.zeros(k, dtype=np.float64)
        if t + 1 < n_obs:
            n_arcs_set = top_arcs_per_step[t + 1]
            n_from1 = top_from_1hop[t + 1]
            n_from2 = top_from_2hop[t + 1]
            for j in range(k):
                a_id = int(arcs_t[j])
                v_id = int(v_nodes[j])
                if a_id in n_arcs_set or v_id in n_from1:
                    conn_next[j] = 1.0
                    conn_2hop_next[j] = 1.0
                elif v_id in n_from2:
                    conn_2hop_next[j] = 1.0

        sl = slice(row_cursor, row_cursor + k)
        obs_idx_arr[sl] = t
        arc_id_arr[sl] = arcs_t

        x_mat[sl, 0] = dist_m
        x_mat[sl, 1] = geom["signed_lateral_m"]
        x_mat[sl, 2] = np.abs(geom["signed_lateral_m"])
        x_mat[sl, 3] = geom["proj_frac"]
        x_mat[sl, 4] = geom["dist_from_m"]
        x_mat[sl, 5] = geom["dist_to_m"]
        x_mat[sl, 6] = geom["arc_len_m"]
        x_mat[sl, 7] = np.log1p(geom["arc_len_m"])
        x_mat[sl, 8] = np.arange(k, dtype=np.float64)
        x_mat[sl, 9] = dist_m - min_d
        x_mat[sl, 10] = (dist_m + 1.0) / (min_d + 1.0)
        x_mat[sl, 11] = cos_h
        x_mat[sl, 12] = sin_h
        x_mat[sl, 13] = h_conf
        x_mat[sl, 14] = cos_h * h_conf
        x_mat[sl, 15] = kin["speed_mps"][t]
        x_mat[sl, 16] = np.log1p(kin["dt_prev"][t])
        x_mat[sl, 17] = np.log1p(kin["dt_next"][t])
        x_mat[sl, 18] = prev_dist
        x_mat[sl, 19] = next_dist
        x_mat[sl, 20] = prev2_dist
        x_mat[sl, 21] = next2_dist
        x_mat[sl, 22] = mean_5pt
        x_mat[sl, 23] = proj_delta_m
        x_mat[sl, 24] = in_deg[u_nodes]
        x_mat[sl, 25] = out_deg[v_nodes]
        x_mat[sl, 26] = conn_prev
        x_mat[sl, 27] = conn_next
        x_mat[sl, 28] = conn_2hop_prev
        x_mat[sl, 29] = conn_2hop_next
        x_mat[sl, 30] = has_rev[arcs_t]

        row_cursor += k

    return x_mat, obs_idx_arr, arc_id_arr
