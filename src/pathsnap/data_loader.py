"""Raw trip data loaders enforcing strict separation of reference routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class TripData:
    """In-memory representation of one PathSnap trip."""

    trip_id: str
    track_lon: np.ndarray
    track_lat: np.ndarray
    track_ts: np.ndarray
    nodes_lon: np.ndarray
    nodes_lat: np.ndarray
    arcs_from: np.ndarray
    arcs_to: np.ndarray
    route_arcs: np.ndarray | None = None


def list_trip_ids(split_dir: Path) -> list[str]:
    """Return sorted trip IDs found inside a partition directory."""
    if not split_dir.exists():
        return []
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def load_trip(
    split_dir: Path,
    trip_id: str,
    include_route: bool = False,
) -> TripData:
    """Load a trip's .track, .nodes, .arcs, and optionally .route (offline/eval scoring only)."""
    tdir = split_dir / trip_id
    if not tdir.exists():
        raise FileNotFoundError(f"Trip directory not found: {tdir}")

    track = np.loadtxt(tdir / f"{trip_id}.track", dtype=np.float64)
    if track.ndim == 1:
        track = track[None, :]
    nodes = np.loadtxt(tdir / f"{trip_id}.nodes", dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes[None, :]
    arcs = np.loadtxt(tdir / f"{trip_id}.arcs", dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs[None, :]

    route_arcs = None
    if include_route:
        route_path = tdir / f"{trip_id}.route"
        if route_path.exists():
            route_arcs = np.loadtxt(route_path, dtype=np.int64)
            if route_arcs.ndim == 0:
                route_arcs = route_arcs[None]

    return TripData(
        trip_id=trip_id,
        track_lon=track[:, 0],
        track_lat=track[:, 1],
        track_ts=track[:, 2],
        nodes_lon=nodes[:, 0],
        nodes_lat=nodes[:, 1],
        arcs_from=arcs[:, 0],
        arcs_to=arcs[:, 1],
        route_arcs=route_arcs,
    )
