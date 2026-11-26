"""Tests for Cartesian WorldBuilder slab profiles."""

import json

import numpy as np
import pytest

from gdmate.aspect_tools import (
    SlabGeometry,
    build_slab_profile,
    offset_slab_profile,
    read_worldbuilder_slab,
    slab_mask,
    slab_travel_time,
)


@pytest.fixture
def slab_file(tmp_path):
    path = tmp_path / "case.wb"
    path.write_text(
        json.dumps(
            {
                "coordinate system": {"model": "cartesian"},
                "features": [
                    {
                        "name": "Slab",
                        "coordinates": [[10, -1], [10, 1]],
                        "segments": [
                            {"length": 2, "angle": [0, 0]},
                            {"length": 2, "angle": [30, 30]},
                        ],
                        "temperature models": [{"subducting velocity": 0.1}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_read_worldbuilder_slab_and_build_profile(slab_file):
    geometry = read_worldbuilder_slab(slab_file)
    assert geometry.trench_x == 10
    assert geometry.subduction_speed == 0.1

    profile = build_slab_profile(geometry, surface_y=20, spacing=1)
    np.testing.assert_allclose(profile.distance, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(profile.depth, [0, 0, 0, 0.5, 1])
    np.testing.assert_allclose(
        profile.points[:, 0], [10, 11, 12, 12 + np.sqrt(3) / 2, 12 + np.sqrt(3)]
    )


def test_profile_samples_across_segment_boundaries(slab_file):
    profile = build_slab_profile(
        read_worldbuilder_slab(slab_file), surface_y=20, spacing=1.5
    )

    np.testing.assert_allclose(profile.distance, [0, 1.5, 3, 4])
    np.testing.assert_allclose(profile.depth, [0, 0, 0.5, 1])
    np.testing.assert_allclose(
        profile.points[:, 0], [10, 11.5, 12 + np.sqrt(3) / 2, 12 + np.sqrt(3)]
    )


def test_curved_segment_tip_is_independent_of_spacing():
    geometry = SlabGeometry(0, (np.pi / 2,), ((0, 90),), 0.1)

    profile = build_slab_profile(geometry, surface_y=2, spacing=10)

    np.testing.assert_allclose(profile.distance, [0, np.pi / 2])
    np.testing.assert_allclose(profile.points, [[0, 2], [1, 1]])


def test_missing_worldbuilder_slab_is_reported(slab_file):
    with pytest.raises(ValueError, match="Missing.*not found"):
        read_worldbuilder_slab(slab_file, feature_name="Missing")


def test_offset_and_time_preserve_reference_station(slab_file):
    profile = build_slab_profile(
        read_worldbuilder_slab(slab_file), surface_y=20, spacing=1
    )
    offset = offset_slab_profile(profile, 3)
    np.testing.assert_allclose(offset.points[0], [10, 17])
    np.testing.assert_allclose(
        offset.points[-1] - profile.points[-1], [-1.5, -3 * np.sqrt(3) / 2]
    )
    np.testing.assert_array_equal(offset.distance, profile.distance)
    np.testing.assert_allclose(
        slab_travel_time(offset.distance[[0, -1]], 0.1), [0, 1261440000]
    )

    with pytest.raises(ValueError, match="positive"):
        slab_travel_time(profile.distance, 0)


def test_slab_mask_respects_signed_band(slab_file):
    profile = build_slab_profile(
        read_worldbuilder_slab(slab_file), surface_y=20, spacing=1
    )
    queries = [[[10, 19], [11, 19]], [[10, 21], [10, 16]]]

    band = slab_mask(profile, queries, min_distance=0, max_distance=2)
    np.testing.assert_array_equal(band, [[True, True], [False, False]])
