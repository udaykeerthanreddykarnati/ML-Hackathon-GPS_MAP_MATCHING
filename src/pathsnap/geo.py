"""Geodesic and per-trip local metric projection utilities."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014


@dataclass(frozen=True)
class LocalProjection:
    """Local WGS84 tangent-plane projection centered at (lon0, lat0) in meters."""

    lon0: float
    lat0: float
    m_per_deg_lon: float
    m_per_deg_lat: float

    @classmethod
    def from_lon_lat(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        """Fit local tangent-plane scale factors using the median trajectory coordinate."""
        lon0 = float(np.median(lon))
        lat0 = float(np.median(lat))
        phi = np.deg2rad(lat0)
        sin_phi = np.sin(phi)
        denom = np.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
        r_n = WGS84_A / denom
        r_m = WGS84_A * (1.0 - WGS84_E2) / (denom**3)
        m_per_deg_lat = (np.pi / 180.0) * r_m
        m_per_deg_lon = (np.pi / 180.0) * r_n * np.cos(phi)
        return cls(
            lon0=lon0,
            lat0=lat0,
            m_per_deg_lon=float(m_per_deg_lon),
            m_per_deg_lat=float(m_per_deg_lat),
        )

    def to_xy(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Project (lon, lat) arrays into metric (x, y) coordinates shaped (N, 2)."""
        x = (np.asarray(lon, dtype=np.float64) - self.lon0) * self.m_per_deg_lon
        y = (np.asarray(lat, dtype=np.float64) - self.lat0) * self.m_per_deg_lat
        return np.column_stack((x, y))

    def to_lon_lat(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert metric (x, y) coordinates back to (lon, lat)."""
        lon = self.lon0 + np.asarray(x, dtype=np.float64) / self.m_per_deg_lon
        lat = self.lat0 + np.asarray(y, dtype=np.float64) / self.m_per_deg_lat
        return lon, lat


def haversine_m(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray | float:
    """Compute geodesic distance in meters using the Haversine formula."""
    r_earth = 6371000.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp * 0.5) ** 2 + np.cos(p1) * np.cos(p2) * (np.sin(dl * 0.5) ** 2)
    return 2.0 * r_earth * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def point_to_segments_geometry(
    pt_xy: np.ndarray,
    seg_a_xy: np.ndarray,
    seg_b_xy: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute exact point-to-segment distance, projection fraction, lateral sign, and arc vectors.

    Args:
        pt_xy: Single point shaped (2,).
        seg_a_xy: Segment start points shaped (M, 2).
        seg_b_xy: Segment end points shaped (M, 2).
    """
    ab = seg_b_xy - seg_a_xy
    ab_len2 = np.sum(ab * ab, axis=1)
    ab_len = np.sqrt(np.maximum(ab_len2, 1e-12))

    ap = pt_xy[None, :] - seg_a_xy
    safe_len2 = np.where(ab_len2 > 1e-12, ab_len2, 1.0)
    proj_raw = np.sum(ap * ab, axis=1) / safe_len2
    proj_frac = np.where(ab_len2 > 1e-12, np.clip(proj_raw, 0.0, 1.0), 0.0)

    closest = seg_a_xy + proj_frac[:, None] * ab
    diff = pt_xy[None, :] - closest
    dist_m = np.sqrt(np.sum(diff * diff, axis=1))

    # Signed lateral offset: positive = left of directed segment A -> B.
    cross = ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]
    signed_lateral_m = np.where(ab_len > 1e-6, cross / ab_len, 0.0)

    dist_from_m = np.sqrt(np.sum(ap * ap, axis=1))
    bp = pt_xy[None, :] - seg_b_xy
    dist_to_m = np.sqrt(np.sum(bp * bp, axis=1))

    unit_dir = ab / ab_len[:, None]

    return {
        "dist_m": dist_m,
        "signed_lateral_m": signed_lateral_m,
        "proj_frac": proj_frac,
        "dist_from_m": dist_from_m,
        "dist_to_m": dist_to_m,
        "arc_len_m": ab_len,
        "unit_dir": unit_dir,
    }


def estimate_track_kinematics(
    track_xy: np.ndarray,
    track_ts: np.ndarray,
    min_disp_m: float = 4.0,
    max_window: int = 5,
) -> dict[str, np.ndarray]:
    """Estimate smooth GPS headings, confidence flags, speeds, and time gaps."""
    n = len(track_xy)
    headings = np.zeros((n, 2), dtype=np.float64)
    heading_conf = np.zeros(n, dtype=np.float64)
    speed_mps = np.zeros(n, dtype=np.float64)
    dt_prev = np.ones(n, dtype=np.float64)
    dt_next = np.ones(n, dtype=np.float64)

    if n >= 2:
        diffs = np.diff(track_xy, axis=0)
        step_dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        dts = np.maximum(np.diff(track_ts), 1e-3)
        step_speeds = np.clip(step_dists / dts, 0.0, 60.0)

        dt_prev[1:] = dts
        dt_prev[0] = dts[0]
        dt_next[:-1] = dts
        dt_next[-1] = dts[-1]

        speed_mps[0] = step_speeds[0]
        speed_mps[-1] = step_speeds[-1]
        if n > 2:
            speed_mps[1:-1] = 0.5 * (step_speeds[:-1] + step_speeds[1:])

    for i in range(n):
        best_vec = None
        # Expand symmetric/forward-backward window until displacement exceeds GPS jitter.
        for w in range(1, max_window + 1):
            lo = max(0, i - w)
            hi = min(n - 1, i + w)
            if hi == lo:
                continue
            vec = track_xy[hi] - track_xy[lo]
            norm = float(np.hypot(vec[0], vec[1]))
            if norm >= min_disp_m:
                best_vec = vec / norm
                break
        if best_vec is not None:
            headings[i] = best_vec
            heading_conf[i] = 1.0
        elif n >= 2:
            lo = max(0, i - 1)
            hi = min(n - 1, i + 1)
            vec = track_xy[hi] - track_xy[lo]
            norm = float(np.hypot(vec[0], vec[1]))
            if norm > 1e-6:
                headings[i] = vec / norm
                heading_conf[i] = float(np.clip(norm / min_disp_m, 0.0, 1.0))

    return {
        "headings": headings,
        "heading_conf": heading_conf,
        "speed_mps": speed_mps,
        "dt_prev": dt_prev,
        "dt_next": dt_next,
    }
