"""HMM and Viterbi sequential route reconstruction over the directed road graph."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np

from .candidates import CandidateTable
from .data_loader import TripData
from .geo import estimate_track_kinematics


@dataclass
class HMMConfig:
    """Parameters governing HMM emission and road-network transition log-probabilities."""

    cls_weight: float = 1.2
    sigma_emit_m: float = 25.0
    heading_weight: float = 1.5
    beta_trans_m: float = 20.0
    same_arc_bonus: float = 0.3
    backward_tol_m: float = 3.0
    uturn_penalty: float = 5.0
    teleport_penalty: float = 16.0
    min_prob: float = 1e-6


def _build_adj_list(
    n_nodes: int,
    arcs_from: np.ndarray,
    arcs_to: np.ndarray,
    arc_len_m: np.ndarray,
) -> list[list[tuple[int, float, int]]]:
    """Build directed adjacency list: u -> [(v, length_m, arc_id), ...]."""
    adj: list[list[tuple[int, float, int]]] = [[] for _ in range(n_nodes)]
    for aid, (u, v, w) in enumerate(zip(arcs_from, arcs_to, arc_len_m)):
        adj[int(u)].append((int(v), float(w), int(aid)))
    return adj


def _bounded_dijkstra(
    adj: list[list[tuple[int, float, int]]],
    src: int,
    targets: set[int],
    max_dist: float,
) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
    """Shortest path distances & parent pointers from src up to max_dist, stopping once targets settle."""
    dist: dict[int, float] = {src: 0.0}
    parent: dict[int, tuple[int, int]] = {}
    remaining = set(targets) - {src}
    if not remaining:
        return dist, parent

    pq: list[tuple[float, int]] = [(0.0, src)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist.get(u, np.inf):
            continue
        if u in remaining:
            remaining.remove(u)
            if not remaining:
                break
        if d_u > max_dist:
            break
        for v, w, aid in adj[u]:
            nd = d_u + w
            if nd <= max_dist and nd < dist.get(v, np.inf):
                dist[v] = nd
                parent[v] = (u, aid)
                heapq.heappush(pq, (nd, v))
    return dist, parent


def viterbi_reconstruct(
    trip: TripData,
    cand: CandidateTable,
    cand_probs_per_obs: list[np.ndarray],
    cfg: HMMConfig | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Run Viterbi decoding over candidate arcs and return (per_obs_arcs, gap_filled_route).

    Handles GPS gaps (dt > 1s) via bounded shortest-path routing, penalizes backward/reverse-twin
    travel, and handles disconnected components or missing candidates via finite fallback penalties.
    """
    if cfg is None:
        cfg = HMMConfig()

    n_obs = len(cand.track_xy)
    if n_obs == 0:
        return np.empty(0, dtype=np.int64), []

    adj = _build_adj_list(
        len(cand.nodes_xy), trip.arcs_from, trip.arcs_to, cand.arc_len_m
    )
    kin = estimate_track_kinematics(cand.track_xy, trip.track_ts)
    headings = kin["headings"]
    heading_conf = kin["heading_conf"]

    valid_steps = [t for t in range(n_obs) if len(cand.cand_arc_ids[t]) > 0]
    if not valid_steps:
        return np.full(n_obs, -1, dtype=np.int64), []

    def emission_scores(t_idx: int) -> np.ndarray:
        p = np.clip(cand_probs_per_obs[t_idx], cfg.min_prob, 1.0 - cfg.min_prob)
        d = cand.cand_geom[t_idx]["dist_m"]
        u_dir = cand.cand_geom[t_idx]["unit_dir"]
        cos_h = u_dir[:, 0] * headings[t_idx][0] + u_dir[:, 1] * headings[t_idx][1]
        dir_pen = cfg.heading_weight * heading_conf[t_idx] * (1.0 - cos_h)
        return cfg.cls_weight * np.log(p) - 0.5 * (d / cfg.sigma_emit_m) ** 2 - dir_pen

    v_scores: list[np.ndarray] = [emission_scores(valid_steps[0])]
    back_ptrs: list[np.ndarray] = [
        np.full(len(cand.cand_arc_ids[valid_steps[0]]), -1, dtype=np.int64)
    ]

    sp_cache: dict[tuple[int, int], tuple[dict[int, float], dict[int, tuple[int, int]]]] = {}

    for step_idx in range(1, len(valid_steps)):
        t_prev = valid_steps[step_idx - 1]
        t_curr = valid_steps[step_idx]

        prev_arcs = cand.cand_arc_ids[t_prev]
        curr_arcs = cand.cand_arc_ids[t_curr]
        prev_proj = cand.cand_geom[t_prev]["proj_frac"]
        curr_proj = cand.cand_geom[t_curr]["proj_frac"]

        d_gps = float(np.linalg.norm(cand.track_xy[t_curr] - cand.track_xy[t_prev]))
        dt = max(1.0, float(trip.track_ts[t_curr] - trip.track_ts[t_prev]))
        max_road_dist = min(4500.0, max(350.0, d_gps * 2.2 + 200.0 + 15.0 * min(dt, 60.0)))
        dist_bucket = int(np.ceil(max_road_dist / 100.0) * 100)

        dst_nodes = {int(trip.arcs_from[int(a)]) for a in curr_arcs}
        prev_scores = v_scores[-1]
        emit_curr = emission_scores(t_curr)

        n_prev = len(prev_arcs)
        n_curr = len(curr_arcs)
        trans_mat = np.full((n_prev, n_curr), -np.inf, dtype=np.float64)

        for i in range(n_prev):
            a_i = int(prev_arcs[i])
            u_i = int(trip.arcs_from[a_i])
            v_i = int(trip.arcs_to[a_i])
            len_i = float(cand.arc_len_m[a_i])
            rem_i = max(0.0, (1.0 - float(prev_proj[i])) * len_i)

            sp_dists: dict[int, float] | None = None

            for j in range(n_curr):
                a_j = int(curr_arcs[j])
                u_j = int(trip.arcs_from[a_j])
                v_j = int(trip.arcs_to[a_j])
                len_j = float(cand.arc_len_m[a_j])
                entry_j = max(0.0, float(curr_proj[j]) * len_j)

                if a_i == a_j:
                    delta_s = (float(curr_proj[j]) - float(prev_proj[i])) * len_i
                    back_pen = max(0.0, -delta_s - cfg.backward_tol_m) / cfg.beta_trans_m
                    d_route = max(0.0, delta_s)
                    trans_mat[i, j] = (
                        cfg.same_arc_bonus
                        - abs(d_route - d_gps) / cfg.beta_trans_m
                        - back_pen
                    )
                elif v_i == u_j:
                    uturn_pen = cfg.uturn_penalty if (u_i == v_j) else 0.0
                    d_route = rem_i + entry_j
                    trans_mat[i, j] = -abs(d_route - d_gps) / cfg.beta_trans_m - uturn_pen
                else:
                    if sp_dists is None:
                        ckey = (v_i, dist_bucket)
                        if ckey not in sp_cache:
                            sp_cache[ckey] = _bounded_dijkstra(adj, v_i, dst_nodes, max_road_dist)
                        sp_dists, _ = sp_cache[ckey]
                    if u_j in sp_dists:
                        d_route = rem_i + sp_dists[u_j] + entry_j
                        uturn_pen = cfg.uturn_penalty if (u_i == v_j and v_i == u_j) else 0.0
                        trans_mat[i, j] = -abs(d_route - d_gps) / cfg.beta_trans_m - uturn_pen
                    else:
                        trans_mat[i, j] = -cfg.teleport_penalty - d_gps / (2.0 * cfg.beta_trans_m)

        totals = prev_scores[:, None] + trans_mat
        best_prev = np.argmax(totals, axis=0)
        best_vals = totals[best_prev, np.arange(n_curr)] + emit_curr
        v_scores.append(best_vals)
        back_ptrs.append(best_prev.astype(np.int64))

    chosen_idx = np.zeros(len(valid_steps), dtype=np.int64)
    cur = int(np.argmax(v_scores[-1]))
    for s in range(len(valid_steps) - 1, -1, -1):
        chosen_idx[s] = cur
        cur = int(back_ptrs[s][cur])

    pred_obs_arcs = np.full(n_obs, -1, dtype=np.int64)
    for s, t_idx in enumerate(valid_steps):
        pred_obs_arcs[t_idx] = cand.cand_arc_ids[t_idx][chosen_idx[s]]

    for t_idx in range(n_obs):
        if pred_obs_arcs[t_idx] == -1:
            pred_obs_arcs[t_idx] = pred_obs_arcs[t_idx - 1] if t_idx > 0 else pred_obs_arcs[valid_steps[0]]

    gap_filled_route: list[int] = []
    for a_val in pred_obs_arcs:
        a_int = int(a_val)
        if not gap_filled_route:
            gap_filled_route.append(a_int)
            continue
        prev_a = gap_filled_route[-1]
        if a_int == prev_a:
            continue
        v_prev = int(trip.arcs_to[prev_a])
        u_curr = int(trip.arcs_from[a_int])
        if v_prev != u_curr:
            sp_d, sp_p = _bounded_dijkstra(adj, v_prev, {u_curr}, 2500.0)
            if u_curr in sp_d:
                bridge: list[int] = []
                node = u_curr
                while node != v_prev and node in sp_p:
                    p_node, p_arc = sp_p[node]
                    bridge.append(p_arc)
                    node = p_node
                gap_filled_route.extend(reversed(bridge))
        gap_filled_route.append(a_int)

    return pred_obs_arcs, gap_filled_route
