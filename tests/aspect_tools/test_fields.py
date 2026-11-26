"""Tests for reading, sampling, and gridding ASPECT point fields."""

import numpy as np
import pytest
from pyvista import PolyData
from scipy.interpolate import griddata

import gdmate.aspect_tools.fields as field_tools
from gdmate.aspect_tools import (
    PointFieldSampler2D,
    read_point_fields,
    sample_point_fields_2d,
)


@pytest.fixture
def mesh():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    result = PolyData(points)
    result.point_data["T"] = 273.0 + 10.0 * points[:, 0] + 20.0 * points[:, 1]
    result.point_data["p"] = 1e6 * (1.0 - points[:, 1])
    return result


def test_read_point_fields(tmp_path, mesh):
    path = tmp_path / "solution.vtp"
    mesh.save(path)

    loaded, fields = read_point_fields(path, ("p", "T"))

    np.testing.assert_array_equal(loaded.points, mesh.points)
    assert list(fields) == ["p", "T"]
    np.testing.assert_array_equal(fields["T"], mesh.point_data["T"])
    np.testing.assert_array_equal(fields["p"], mesh.point_data["p"])


def test_read_point_fields_reports_missing_point_field(tmp_path, mesh):
    path = tmp_path / "solution.vtp"
    mesh.cell_data["density"] = np.full(mesh.n_cells, 3300.0)
    mesh.save(path)

    with pytest.raises(ValueError, match="density.*point field"):
        read_point_fields(path, ("density",))


def test_sample_point_fields_2d(mesh):
    points = mesh.points[:, :2]
    fields = {name: mesh.point_data[name] for name in ("p", "T")}
    queries = [[0.1, 0.8], [0.9, 0.1]]
    result = sample_point_fields_2d(points, fields, queries, method="nearest")

    np.testing.assert_array_equal(result["p"], [0.0, 1e6])
    np.testing.assert_array_equal(result["T"], [293.0, 283.0])

    default = sample_point_fields_2d(points, fields, queries)
    np.testing.assert_allclose(default["p"], [2e5, 9e5])
    np.testing.assert_allclose(default["T"], [290.0, 284.0])


def test_sample_point_fields_2d_checks_field_length(mesh):
    with pytest.raises(ValueError, match="T.*one scalar"):
        sample_point_fields_2d(mesh.points[:, :2], {"T": [273.0]}, [[0.0, 0.0]])


def test_sampler_builds_tree_once_for_sampling_and_gridding(mesh, monkeypatch):
    coordinates = mesh.points[:, :2].copy()
    fields = {"T": mesh.point_data["T"], "p": mesh.point_data["p"]}
    build_calls = []
    original_tree = field_tools.cKDTree

    def track_tree(points):
        build_calls.append(points)
        return original_tree(points)

    monkeypatch.setattr(field_tools, "cKDTree", track_tree)
    sampler = PointFieldSampler2D(coordinates, fields)
    assert not build_calls

    coordinates[:] = 10
    np.testing.assert_array_equal(sampler.points, mesh.points[:, :2])
    np.testing.assert_array_equal(sampler.fields["T"], fields["T"])

    samples = sampler.sample([[0.1, 0.8], [0.9, 0.1]], method="nearest")
    grids = sampler.grid([0.1, 1.2], [0.1, 0.8], method="nearest")
    np.testing.assert_array_equal(samples["T"], [293.0, 283.0])
    np.testing.assert_array_equal(grids["T"], [[273.0, 283.0], [293.0, 303.0]])
    np.testing.assert_array_equal(grids["p"], [[1e6, 1e6], [0.0, 0.0]])
    assert len(build_calls) == 1


def test_sampler_reuses_linear_interpolator_and_matches_griddata(mesh, monkeypatch):
    points = mesh.points[:, :2]
    fields = {"T": mesh.point_data["T"].copy(), "p": mesh.point_data["p"]}
    build_calls = []
    original_interpolator = field_tools.LinearNDInterpolator

    def track_interpolator(coordinates, values):
        build_calls.append(coordinates)
        return original_interpolator(coordinates, values)

    monkeypatch.setattr(field_tools, "LinearNDInterpolator", track_interpolator)
    sampler = PointFieldSampler2D(points, fields)
    assert not build_calls

    queries = [[0.25, 0.75], [1.5, 0.5]]
    samples = sampler.sample(queries)
    x, y = [0.25, 1.5], [0.25, 0.75]
    grids = sampler.grid(x, y)
    for name, values in fields.items():
        np.testing.assert_allclose(
            samples[name], griddata(points, values, queries), equal_nan=True
        )
        np.testing.assert_allclose(
            grids[name],
            griddata(points, values, tuple(np.meshgrid(x, y))),
            equal_nan=True,
        )

    fields["T"][:] = 0
    np.testing.assert_array_equal(sampler.fields["T"], mesh.point_data["T"])
    np.testing.assert_allclose(
        sampler.sample(queries, method="linear")["T"], samples["T"], equal_nan=True
    )
    assert not sampler.fields["T"].flags.writeable
    assert len(build_calls) == 1


def test_sampler_rejects_unknown_method(mesh):
    sampler = PointFieldSampler2D(mesh.points[:, :2], {"T": mesh.point_data["T"]})

    with pytest.raises(ValueError, match="nearest.*linear"):
        sampler.grid([0.25], [0.75], method="cubic")
