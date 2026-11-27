"""Evolution of a forward phase transition at fixed pressure and temperature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gdmate.rheology_models.transition_kinetics import TransitionKinetics


@dataclass(frozen=True)
class TransitionHistory:
    """Evolution at the requested elapsed times.

    Time and site saturation time are in seconds. The number density counts
    new grains per m³, and the equivalent mean grain diameter is in metres.
    Extended volume and transformed fraction are dimensionless.
    """

    time: NDArray
    grain_number_density: NDArray
    extended_volume: NDArray
    transformed_fraction: NDArray
    mean_grain_diameter: NDArray
    site_saturation_time: float
    is_saturated: NDArray[np.bool_]


def solve_fixed_conditions(
    kinetics: TransitionKinetics,
    pressure: float,
    temperature: float,
    times: ArrayLike,
    *,
    water_content_ppm: float | None = None,
    initial_grain_size: float | None = None,
    initial_transformed_fraction: float = 0.0,
) -> TransitionHistory:
    """Solve at constant P (Pa) and T (K) at the given elapsed times (s).

    The model starts with zero new nuclei and assumes zero critical nucleus
    radius, as in the fixed-condition MOW examples. The parent grain size
    in m determines the boundary area for surface nucleation. At site
    saturation, surface nucleation stops and existing grains keep growing.
    Volumetric nucleation has no site saturation in this solver.

    A pre-existing transformed fraction contributes to the initial extended
    volume. The grain diameter describes only the material transformed and
    grains formed during this run. It is NaN before any grains form.
    """
    time = np.asarray(times, dtype=float)
    if (
        time.ndim != 1
        or not time.size
        or not np.isfinite(time).all()
        or np.any(time < 0)
        or np.any(np.diff(time) < 0)
    ):
        raise ValueError(
            "Times must be a nonempty, nondecreasing array of elapsed seconds."
        )
    if not 0 <= initial_transformed_fraction <= 1:
        raise ValueError("Initial transformed fraction must be in [0, 1].")

    rates = kinetics.rates(
        pressure,
        temperature,
        water_content_ppm=water_content_ppm,
        initial_grain_size=initial_grain_size,
    )
    if rates.growth.ndim != 0:
        raise ValueError(
            "Fixed conditions require scalar pressure, temperature, and water content."
        )
    growth = float(rates.growth)
    nucleation = float(rates.volumetric_nucleation)
    surface_rate = float(rates.surface_nucleation)
    if (
        not np.isfinite([growth, nucleation, surface_rate]).all()
        or min(growth, nucleation, surface_rate) < 0
    ):
        raise ValueError("Growth and nucleation rates must be finite and nonnegative.")

    initial_fraction = initial_transformed_fraction
    if initial_fraction == 1:
        growth = nucleation = surface_rate = 0.0
        initial_volume = np.inf
    else:
        initial_volume = -np.log1p(-initial_fraction)

    saturation_time = float("inf")
    extra_volume = 0.0
    if surface_rate > 0 and growth > 0 and initial_grain_size is not None:
        saturation_time = float(1 / (np.cbrt(surface_rate) * np.cbrt(growth) ** 2))
        area_per_volume = (
            kinetics.parameters.grain_boundary_area_factor / initial_grain_size
        )
        extra_volume = area_per_volume * growth * np.maximum(time - saturation_time, 0)

    nucleation_time = np.minimum(time, saturation_time)
    grain_number = nucleation * nucleation_time
    # Integrate the constant-rate moment equations up to site saturation.
    growth_distance = growth * nucleation_time
    new_volume = (np.pi / 3) * grain_number * growth_distance**3 + extra_volume
    newly_transformed = (1 - initial_fraction) * -np.expm1(-new_volume)
    volume_per_grain = np.full_like(time, np.nan)
    np.divide(
        newly_transformed, grain_number, out=volume_per_grain, where=grain_number > 0
    )

    return TransitionHistory(
        time=time,
        grain_number_density=grain_number,
        extended_volume=initial_volume + new_volume,
        transformed_fraction=initial_fraction + newly_transformed,
        mean_grain_diameter=np.cbrt(6 * volume_per_grain / np.pi),
        site_saturation_time=saturation_time,
        is_saturated=time >= saturation_time,
    )
