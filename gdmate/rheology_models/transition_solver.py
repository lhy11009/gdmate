"""Evolve a forward phase transition at fixed conditions or along a history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

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
    time = _validate_history_inputs(times, initial_transformed_fraction)

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
    return _make_history(
        time, grain_number, new_volume, saturation_time, initial_fraction
    )


def solve_transition_history(
    kinetics: TransitionKinetics,
    pressure: ArrayLike,
    temperature: ArrayLike,
    times: ArrayLike,
    *,
    water_content_ppm: float | None = None,
    initial_grain_size: float | None = None,
    initial_transformed_fraction: float = 0.0,
    substeps_per_interval: int = 1,
) -> TransitionHistory:
    """Evolve forward kinetics along P (Pa), T (K), and time (s) stations.

    Linearly interpolate pressure and temperature to the midpoint of each
    substep, then hold the resulting rates fixed over that substep. The
    default of one substep uses the mean conditions of the two stations.
    Results are returned at the original stations.

    The initial state is set at the first time. A repeated time leaves it as
    it was. Intervals touching a nonfinite sample leave the state as it was
    and the missing stations have NaN diagnostics. Below-equilibrium rates
    are zero, so the existing state is preserved there as well.

    Surface saturation uses the accumulated area moment of the growing
    grains. The saturation time is in the same time coordinates as the
    input, or infinity if saturation is never reached.
    """
    if (
        isinstance(substeps_per_interval, bool)
        or not isinstance(substeps_per_interval, (int, np.integer))
        or substeps_per_interval < 1
    ):
        raise ValueError("substeps_per_interval must be a positive integer.")

    time = _validate_history_inputs(times, initial_transformed_fraction)
    pressure = np.asarray(pressure, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    if pressure.shape != time.shape or temperature.shape != time.shape:
        raise ValueError("Pressure, temperature, and times must have the same shape.")

    valid = np.isfinite(pressure) & np.isfinite(temperature)
    if np.any(temperature[valid] <= 0):
        raise ValueError("Temperature must be positive in kelvin at valid stations.")
    interval_durations = np.diff(time)
    advancing = valid[:-1] & valid[1:] & (interval_durations > 0)
    midpoint_fractions = (
        np.arange(substeps_per_interval) + 0.5
    ) / substeps_per_interval
    start_conditions = np.stack((pressure[:-1][advancing], temperature[:-1][advancing]))
    end_conditions = np.stack((pressure[1:][advancing], temperature[1:][advancing]))
    midpoint_pressure, midpoint_temperature = (
        start_conditions[..., None] * (1 - midpoint_fractions)
        + end_conditions[..., None] * midpoint_fractions
    )
    rates = kinetics.rates(
        midpoint_pressure,
        midpoint_temperature,
        water_content_ppm=water_content_ppm,
        initial_grain_size=initial_grain_size,
    )
    rate_shape = (interval_durations.size, substeps_per_interval)
    substep_growth_rates = np.zeros(rate_shape)
    substep_volumetric_nucleation_rates = np.zeros(rate_shape)
    if initial_transformed_fraction < 1:
        substep_growth_rates[advancing] = rates.growth
        substep_volumetric_nucleation_rates[advancing] = rates.volumetric_nucleation
    if (
        not np.isfinite(substep_growth_rates).all()
        or not np.isfinite(substep_volumetric_nucleation_rates).all()
        or np.any(substep_growth_rates < 0)
        or np.any(substep_volumetric_nucleation_rates < 0)
    ):
        raise ValueError("Growth and nucleation rates must be finite and nonnegative.")

    area_per_volume = 0.0
    if kinetics.nucleation_law.process_type == "surface":
        # The rate model validates the grain size for surface nucleation.
        assert initial_grain_size is not None
        area_per_volume = (
            kinetics.parameters.grain_boundary_area_factor / initial_grain_size
        )
    # For constant rates X2 = (pi/3) I_v G² t³. This threshold reduces to the
    # fixed-condition criterion I_surface G² t³ = 1.
    saturation_area = (np.pi / 3) * area_per_volume
    saturation_time = float("inf")
    moments = np.zeros(4)
    grain_number = np.full(time.size, np.nan)
    new_volume = np.full(time.size, np.nan)
    if valid[0]:
        grain_number[0] = new_volume[0] = 0.0

    for index, interval_duration in enumerate(interval_durations):
        duration = interval_duration / substeps_per_interval
        for substep, (growth_rate, volumetric_nucleation_rate) in enumerate(
            zip(
                substep_growth_rates[index],
                substep_volumetric_nucleation_rates[index],
                strict=True,
            )
        ):
            if not (growth_rate or volumetric_nucleation_rate):
                continue
            if np.isfinite(saturation_time):
                moments[3] += area_per_volume * growth_rate * duration
                continue

            advanced = _advance_moments(
                moments, growth_rate, volumetric_nucleation_rate, duration
            )
            if saturation_area > 0 and advanced[2] >= saturation_area:
                offset = _time_to_saturation(
                    moments,
                    growth_rate,
                    volumetric_nucleation_rate,
                    duration,
                    saturation_area,
                )
                saturation_time = time[index] + substep * duration + offset
                advanced = _advance_moments(
                    moments, growth_rate, volumetric_nucleation_rate, offset
                )
                advanced[3] += area_per_volume * growth_rate * (duration - offset)
            moments = advanced
        if valid[index + 1]:
            grain_number[index + 1] = moments[0]
            new_volume[index + 1] = moments[3]

    return _make_history(
        time, grain_number, new_volume, saturation_time, initial_transformed_fraction
    )


def _advance_moments(
    state: NDArray,
    growth_rate: float,
    volumetric_nucleation_rate: float,
    duration: float,
) -> NDArray:
    """Integrate the four zero-nucleus-radius moments over a constant-rate interval."""
    number, diameter, area, volume = state
    travel = growth_rate * duration
    new_grain_density = volumetric_nucleation_rate * duration
    return np.array(
        [
            number + new_grain_density,
            diameter + travel * (2 * number + new_grain_density),
            area
            + np.pi * travel * (diameter + travel * (number + new_grain_density / 3)),
            volume
            + 4
            * travel
            * (
                area
                + np.pi
                * travel
                * (diameter / 2 + travel * (number / 3 + new_grain_density / 12))
            ),
        ]
    )


def _time_to_saturation(
    state: NDArray,
    growth_rate: float,
    volumetric_nucleation_rate: float,
    duration: float,
    target: float,
) -> float:
    fraction = brentq(
        lambda fraction: (
            _advance_moments(
                state, growth_rate, volumetric_nucleation_rate, duration * fraction
            )[2]
            - target
        ),
        0.0,
        1.0,
        xtol=np.finfo(float).tiny,
    )
    return duration * fraction


def _validate_history_inputs(
    times: ArrayLike, initial_transformed_fraction: float
) -> NDArray:
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
    return time


def _make_history(
    time: NDArray,
    grain_number: NDArray,
    new_volume: NDArray,
    saturation_time: float,
    initial_fraction: float,
) -> TransitionHistory:
    initial_volume = np.inf if initial_fraction == 1 else -np.log1p(-initial_fraction)
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
        is_saturated=(time >= saturation_time) & ~np.isnan(new_volume),
    )
