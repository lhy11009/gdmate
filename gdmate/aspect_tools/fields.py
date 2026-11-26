"""Load, sample, and grid ASPECT solution fields."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cached_property
from os import PathLike
from types import MappingProxyType

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


def read_point_fields(
    path: str | PathLike[str], names: Iterable[str]
) -> tuple[pv.DataSet, dict[str, NDArray]]:
    """Read a VTK solution file and return its mesh and named scalar point fields.

    The mesh contains the original coordinates in ``mesh.points``. Field
    values retain the units stored in the file.
    """
    mesh = pv.read(path)
    if not isinstance(mesh, pv.DataSet):
        raise ValueError(f"{path} must contain a single mesh dataset.")

    fields = {}
    for name in names:
        if name not in mesh.point_data:
            raise ValueError(f"{name!r} is not a point field in {path}.")
        fields[name] = _scalar_values(mesh.point_data[name], mesh.n_points, name)
    return mesh, fields


class PointFieldSampler2D:
    """Sample scalar point fields with cached nearest or linear interpolation.

    Coordinates and fields are copied so both methods use the same snapshot.
    The nearest-neighbor tree and the linear triangulation are built on the
    first use of their respective methods.
    """

    def __init__(self, points: ArrayLike, fields: Mapping[str, ArrayLike]) -> None:
        self._points = np.array(
            _points_2d(points, "points"), dtype=float, order="C", copy=True
        )
        if len(self._points) == 0:
            raise ValueError("points must contain at least one source point.")
        self._points.flags.writeable = False
        scalar_fields = {}
        for name, values in fields.items():
            field = _scalar_values(values, len(self._points), name).copy()
            field.flags.writeable = False
            scalar_fields[name] = field.view()
        self._fields = MappingProxyType(scalar_fields)

    @property
    def points(self) -> NDArray:
        """Return a read-only view of the source (x, y) coordinates."""
        return self._points.view()

    @property
    def fields(self) -> Mapping[str, NDArray]:
        """Return read-only source field values keyed by name."""
        return self._fields

    @cached_property
    def _tree(self) -> cKDTree:
        return cKDTree(self._points)

    @cached_property
    def _linear_interpolator(self) -> LinearNDInterpolator:
        values = np.column_stack(tuple(self._fields.values()))
        return LinearNDInterpolator(self._points, values)

    def sample(
        self, query_points: ArrayLike, *, method: str = "linear"
    ) -> dict[str, NDArray]:
        """Return field values for query (x, y) pairs using nearest or linear.

        Linear interpolation is the default and returns NaN outside the
        source convex hull. Use ``method="nearest"`` for nearest source values.
        """
        return self._interpolate(_points_2d(query_points, "query_points"), method)

    def grid(
        self, x: ArrayLike, y: ArrayLike, *, method: str = "linear"
    ) -> dict[str, NDArray]:
        """Return a grid of shape (len(y), len(x)) for each source field.

        Use the same nearest or linear methods as for ``sample``.
        """
        queries = np.stack(_grid_xy(x, y), axis=-1)
        return self._interpolate(queries, method)

    def _interpolate(self, queries: NDArray, method: str) -> dict[str, NDArray]:
        if method == "nearest":
            _, indices = self._tree.query(queries)
            return {name: values[indices] for name, values in self._fields.items()}
        if method == "linear":
            if not self._fields:
                return {}
            values = self._linear_interpolator(queries)
            return {name: values[..., i] for i, name in enumerate(self._fields)}
        raise ValueError("method must be 'nearest' or 'linear'.")


def sample_point_fields_2d(
    points: ArrayLike,
    fields: Mapping[str, ArrayLike],
    query_points: ArrayLike,
    *,
    method: str = "linear",
) -> dict[str, NDArray]:
    """Interpolate scalar point fields for a single set of (x, y) queries.

    Supply source and query coordinates in the same units. Linear
    interpolation returns NaN outside the source convex hull. Use
    ``method="nearest"`` to get the nearest source values everywhere.
    """
    return PointFieldSampler2D(points, fields).sample(query_points, method=method)


def _grid_xy(x: ArrayLike, y: ArrayLike) -> tuple[NDArray, NDArray]:
    x_axis, y_axis = np.asarray(x), np.asarray(y)
    if x_axis.ndim != 1 or y_axis.ndim != 1:
        raise ValueError("x and y must be one-dimensional grid axes.")
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    return grid_x, grid_y


def _points_2d(points: ArrayLike, name: str) -> NDArray:
    coordinates = np.asarray(points)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"{name} must be an array of (x, y) pairs.")
    return coordinates


def _scalar_values(values: ArrayLike, size: int, name: str) -> NDArray:
    field = np.asarray(values)
    if field.shape != (size,):
        raise ValueError(f"{name!r} must contain one scalar value per source point.")
    return field
