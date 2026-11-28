"""Checks for the constant-condition grain-coarsening law."""

import numpy as np
import pytest

from gdmate.rheology_models.grain_growth import (
    GrainCoarseningModel,
    GrainCoarseningParameters,
)


def test_grain_coarsening_broadcasts_and_preserves_initial_size():
    model = GrainCoarseningModel(GrainCoarseningParameters(3, 2, 0, 0))

    result = model.grain_size_at_time([[1], [2]], [0, 1, 4], 1e9, 1200)

    np.testing.assert_allclose(result, np.sqrt([[1, 4, 13], [4, 7, 16]]))


def test_grain_coarsening_depends_on_pressure_and_temperature():
    model = GrainCoarseningModel(GrainCoarseningParameters(1e-3, 3, 4e4, 2e-6))

    assert model.grain_size_at_time(1e-6, 100, 2e9, 1200) < model.grain_size_at_time(
        1e-6, 100, 1e9, 1200
    )
    assert model.grain_size_at_time(1e-6, 100, 1e9, 1200) > model.grain_size_at_time(
        1e-6, 100, 1e9, 1000
    )


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ((0, 1, 1e9, 1200), "Initial grain size"),
        ((1e-6, -1, 1e9, 1200), "Elapsed time"),
        ((1e-6, 1, 1e9, 0), "Temperature"),
    ],
)
def test_grain_coarsening_rejects_invalid_conditions(inputs, message):
    model = GrainCoarseningModel(GrainCoarseningParameters(1e-3, 3, 4e4, 2e-6))

    with pytest.raises(ValueError, match=message):
        model.grain_size_at_time(*inputs)
