"""Build and sample 2-D Cartesian slab profiles from WorldBuilder inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

_SECONDS_PER_YEAR = 365.0 * 24 * 3600


@dataclass(frozen=True)
class SlabGeometry:
    """Trench x and segment lengths in m, end dips in degrees, and speed in m/yr."""

    trench_x: float
    segment_lengths: tuple[float, ...]
    dip_angles: tuple[tuple[float, float], ...]
    subduction_speed: float


@dataclass(frozen=True)
class SlabProfile:
    """Slab (x, y) points and their reference distance from the trench, in m."""

    points: NDArray
    distance: NDArray
    surface_y: float

    @property
    def depth(self) -> NDArray:
        """Depth below the model surface, in m."""
        return self.surface_y - self.points[:, 1]


def read_worldbuilder_slab(
    path: str | PathLike[str], *, feature_name: str = "Slab"
) -> SlabGeometry:
    """Read the geometry and first specified subducting velocity of a named slab.

    This reader assumes a Cartesian cross-section that extends toward positive x
    from the trench. It uses the x coordinate of the first trench point.
    """
    with Path(path).open(encoding="utf-8") as stream:
        world = json.load(stream)
    if world.get("coordinate system", {}).get("model", "cartesian") != "cartesian":
        raise ValueError("Slab profiles require a Cartesian WorldBuilder case.")

    slab = next(
        (
            feature
            for feature in world.get("features", [])
            if feature.get("name") == feature_name
        ),
        None,
    )
    if slab is None:
        raise ValueError(f"WorldBuilder feature {feature_name!r} not found in {path}.")

    try:
        trench_x = float(slab["coordinates"][0][0])
        lengths = tuple(float(segment["length"]) for segment in slab["segments"])
        dips = tuple(
            (float(start), float(end))
            for start, end in (segment["angle"] for segment in slab["segments"])
        )
        speed = next(
            float(model["subducting velocity"])
            for model in slab.get("temperature models", [])
            if "subducting velocity" in model
        )
    except (KeyError, IndexError, StopIteration, TypeError, ValueError) as exc:
        raise ValueError(
            f"Feature {feature_name!r} needs trench coordinates, segments with "
            "lengths and dip pairs, and a subducting velocity."
        ) from exc

    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Subduction speed must be positive and finite.")
    return SlabGeometry(trench_x, lengths, dips, speed)


def build_slab_profile(
    geometry: SlabGeometry, *, surface_y: float, spacing: float = 2e3
) -> SlabProfile:
    """Trace a slab from the trench at fixed intervals of arc length in m.

    The spacing continues across segment junctions. The slab tip is included
    even when the last interval is shorter. Dips vary linearly with distance
    along each segment, so segment positions can be integrated analytically.
    """
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Profile spacing must be positive and finite.")

    lengths = np.asarray(geometry.segment_lengths)
    dips = np.asarray(geometry.dip_angles)
    if not len(lengths) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Slab segment lengths must be positive and finite.")
    if dips.shape != (len(lengths), 2) or not np.isfinite(dips).all():
        raise ValueError("Each slab segment needs two finite dip angles.")
    if not np.isfinite([geometry.trench_x, surface_y]).all():
        raise ValueError("Trench coordinates must be finite.")

    segment_ends = np.cumsum(lengths, dtype=float)
    segment_starts = np.concatenate(([0.0], segment_ends[:-1]))
    total_length = segment_ends[-1]
    distance = np.arange(0.0, total_length, spacing)
    # Drop a station indistinguishable from the tip at floating-point precision.
    if len(distance) > 1 and distance[-1] >= np.nextafter(total_length, 0):
        distance = distance[:-1]
    distance = np.append(distance, total_length)

    start_dips = np.deg2rad(dips[:, 0])
    turns = np.deg2rad(dips[:, 1] - dips[:, 0])
    displacements = _segment_displacement(lengths, start_dips, turns)
    starts = np.array([geometry.trench_x, surface_y]) + np.vstack(
        ([0.0, 0.0], np.cumsum(displacements[:-1], axis=0))
    )

    segments = np.searchsorted(segment_ends[:-1], distance, side="right")
    along_segment = distance - segment_starts[segments]
    partial_turns = turns[segments] * (along_segment / lengths[segments])
    points = starts[segments] + _segment_displacement(
        along_segment, start_dips[segments], partial_turns
    )
    return SlabProfile(points, distance, surface_y)


def _segment_displacement(distance: NDArray, dip: NDArray, turn: NDArray) -> NDArray:
    # The chord of a circular arc points along the tangent at its midpoint.
    # sinc also gives the straight-segment limit when the dip does not change.
    half_turn = turn / 2
    chord = distance * np.sinc(half_turn / np.pi)
    midpoint_dip = dip + half_turn
    return chord[:, None] * np.column_stack(
        (np.cos(midpoint_dip), -np.sin(midpoint_dip))
    )


def offset_slab_profile(profile: SlabProfile, offset: float) -> SlabProfile:
    """Shift a profile by a signed normal distance in m.

    Positive offsets go into the slab for a rightward, downward profile.
    Offset points inherit distances measured along the reference profile.
    """
    if not np.isfinite(offset):
        raise ValueError("Profile offset must be finite.")
    tangents = np.gradient(profile.points, profile.distance, axis=0)
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("A profile point has no direction for calculating its offset.")
    points = profile.points + offset * normals / lengths[:, None]
    return SlabProfile(points, profile.distance, profile.surface_y)


def slab_travel_time(
    distance: ArrayLike, speed: float, *, seconds_per_year: float = _SECONDS_PER_YEAR
) -> NDArray:
    """Convert distance in m and speed in m/yr to elapsed time in seconds.

    The default year has 365 days, matching the HaMaGeoLib MOW notebook.
    """
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Subduction speed must be positive and finite.")
    if not np.isfinite(seconds_per_year) or seconds_per_year <= 0:
        raise ValueError("Seconds per year must be positive and finite.")
    return np.asarray(distance) / (speed / seconds_per_year)


def slab_mask(
    profile: SlabProfile,
    query_points: ArrayLike,
    *,
    min_distance: float,
    max_distance: float,
) -> NDArray[np.bool_]:
    """Select a signed band around the reference profile, in m.

    Queries have shape (..., 2). Distance is measured to the nearest sampled
    point on the reference profile and is positive below it.
    """
    queries = np.asarray(query_points)
    if queries.ndim < 2 or queries.shape[-1] != 2:
        raise ValueError("query_points must end with (x, y) coordinate pairs.")
    if min_distance > max_distance:
        raise ValueError("min_distance must not exceed max_distance.")

    distances, indices = cKDTree(profile.points).query(queries)
    signed = np.where(
        queries[..., 1] < profile.points[indices, 1], distances, -distances
    )
    return (signed >= min_distance) & (signed <= max_distance)
