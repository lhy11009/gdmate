"""Reference checks for the olivine-to-wadsleyite rate laws."""

from math import exp

import numpy as np
import pytest

from gdmate.rheology_models.transition_kinetics import (
    InterfaceControlledGrowth,
    InterfaceControlledGrowthParameters,
    SurfaceNucleation,
    SurfaceNucleationParameters,
    TransitionBoundary,
    TransitionBoundaryParameters,
    TransitionKinetics,
    TransitionKineticsParameters,
)
from gdmate.rheology_models.transition_solver import solve_fixed_conditions


@pytest.fixture
def mow_kinetics():
    """Use the parameters from the HaMaGeoLib MOW surface-nucleation model."""
    return TransitionKinetics(
        boundary=TransitionBoundary(TransitionBoundaryParameters(13.5e9, 1740.0, 2e6)),
        growth_law=InterfaceControlledGrowth(
            InterfaceControlledGrowthParameters(exp(-18), 3.2, 274e3, 3.3e-6)
        ),
        nucleation_law=SurfaceNucleation(
            SurfaceNucleationParameters(1e30, 274e3, 3.3e-6, 0.46, 6e-4, 4.05e-5)
        ),
        parameters=TransitionKineticsParameters(3.16e-6, 6.7),
    )


def test_equilibrium_boundary(mow_kinetics):
    boundary = mow_kinetics.boundary
    temperatures = [1000.0, 1740.0, 2000.0]
    pressures = boundary.pressure_at(temperatures)
    np.testing.assert_allclose(pressures, [12.02e9, 13.5e9, 14.02e9])
    np.testing.assert_allclose(boundary.temperature_at(pressures), temperatures)


def test_surface_nucleation_and_growth_match_hamageolib(mow_kinetics):
    # HaMaGeoLib: get_kinetic_constants(1), 150 wt ppm H2O, 0.01 m grains.
    pressure = [13.5e9, 13.5e9, 13.5e9, 17e9, 13.5e9, 12e9]
    temperature = [873.15, 973.15, 1573.15, 1200, 1740, 1500]
    rates = mow_kinetics.rates(
        pressure, temperature, water_content_ppm=150, initial_grain_size=0.01
    )

    np.testing.assert_allclose(
        rates.growth,
        [
            5.6896893614437715e-18,
            4.898706456068756e-16,
            4.5198916650981173e-10,
            5.50491045441455e-13,
            0,
            0,
        ],
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        rates.surface_nucleation,
        [
            906450188355.2295,
            47892360981190.25,
            5.549159880738927e-7,
            3.233119804401826e18,
            0,
            0,
        ],
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        rates.volumetric_nucleation,
        [
            607321626198003.8,
            3.2087881857397468e16,
            0.0003717937120095081,
            2.1661902689492232e21,
            0,
            0,
        ],
        rtol=1e-12,
        atol=0,
    )


def test_rates_broadcast_and_use_water_and_grain_size(mow_kinetics):
    model = mow_kinetics
    rates = model.rates(
        17e9, [[1200], [1500]], water_content_ppm=[0, 150], initial_grain_size=0.01
    )
    dry = model.rates(17e9, 1200, water_content_ppm=0, initial_grain_size=0.01)
    coarse = model.rates(17e9, 1200, water_content_ppm=150, initial_grain_size=0.02)

    assert rates.growth.shape == (2, 2)
    np.testing.assert_array_equal(rates.growth[:, 0], 0.0)
    assert dry.surface_nucleation == coarse.surface_nucleation
    assert coarse.volumetric_nucleation == pytest.approx(
        rates.volumetric_nucleation[0, 1] / 2
    )


def test_rates_reject_nonphysical_temperature_and_grain_size(mow_kinetics):
    model = mow_kinetics

    with pytest.raises(ValueError, match="Temperature"):
        model.rates(13.5e9, 0, water_content_ppm=150, initial_grain_size=0.01)
    with pytest.raises(ValueError, match="grain size"):
        model.rates(13.5e9, 1200, water_content_ppm=150, initial_grain_size=0)


def test_mow_laws_report_their_process_and_match_assembled_rates(mow_kinetics):
    model = mow_kinetics
    growth = model.growth_law
    nucleation = model.nucleation_law
    assert growth.process_type == "interface_controlled"
    assert nucleation.process_type == "surface"

    pressure = np.array([17e9])
    temperature = np.array([1200.0])
    water = np.array([150.0])
    driving_force = model.parameters.molar_volume_change * (
        pressure - model.boundary.pressure_at(temperature)
    )
    rates = model.rates(
        pressure, temperature, water_content_ppm=water, initial_grain_size=0.01
    )
    np.testing.assert_allclose(
        growth.rate(pressure, temperature, driving_force, water), rates.growth
    )
    np.testing.assert_allclose(
        nucleation.rate(pressure, temperature, driving_force), rates.surface_nucleation
    )


def test_volumetric_nucleation_does_not_use_grain_boundary_area(mow_kinetics):
    class VolumetricNucleation:
        process_type = "volumetric"

        def rate(self, pressure, temperature, driving_force):
            return np.full_like(pressure, 5.0)

    model = TransitionKinetics(
        boundary=mow_kinetics.boundary,
        growth_law=mow_kinetics.growth_law,
        nucleation_law=VolumetricNucleation(),
        parameters=mow_kinetics.parameters,
    )
    rates = model.rates([17e9, 13.5e9], [1200, 1740], water_content_ppm=150)

    np.testing.assert_array_equal(rates.surface_nucleation, [0, 0])
    np.testing.assert_array_equal(rates.volumetric_nucleation, [5, 0])


def test_missing_inputs_are_reported_when_required(mow_kinetics):
    with pytest.raises(ValueError, match="Water content is required"):
        mow_kinetics.rates(17e9, 1200, initial_grain_size=0.01)
    with pytest.raises(ValueError, match="Initial grain size is required"):
        mow_kinetics.rates(17e9, 1200, water_content_ppm=150)
    with pytest.raises(ValueError, match="Water content must be finite"):
        mow_kinetics.rates(
            17e9, 1200, water_content_ppm=np.nan, initial_grain_size=0.01
        )


def test_water_independent_growth_and_volumetric_nucleation_need_neither_input(
    mow_kinetics,
):
    class WaterIndependentGrowth:
        process_type = "constant"
        requires_water_content = False

        def rate(self, pressure, temperature, driving_force, water_content_ppm):
            assert water_content_ppm is None
            return np.full_like(pressure, 2.0)

    class VolumetricNucleation:
        process_type = "volumetric"

        def rate(self, pressure, temperature, driving_force):
            return np.full_like(pressure, 5.0)

    model = TransitionKinetics(
        boundary=mow_kinetics.boundary,
        growth_law=WaterIndependentGrowth(),
        nucleation_law=VolumetricNucleation(),
        parameters=mow_kinetics.parameters,
    )
    rates = model.rates([17e9, 13.5e9], [1200, 1740])

    np.testing.assert_array_equal(rates.growth, [2, 0])
    np.testing.assert_array_equal(rates.volumetric_nucleation, [5, 0])


YEAR = 365 * 86400


@pytest.mark.parametrize(
    ("temperature", "saturation_time", "final_number", "final_fraction"),
    [
        (873.15, 32420993.18029507, 1.9689970301211196e22, 0.6994620665826814),
        (973.15, 443122.3345272988, 1.4218857118686127e22, 1.0),
        (1573.15, 206619942.82863322, 76819.99551944989, 1.0),
    ],
)
def test_fixed_mow_reference_cases(
    mow_kinetics, temperature, saturation_time, final_number, final_fraction
):
    result = solve_fixed_conditions(
        mow_kinetics,
        13.5e9,
        temperature,
        [0, saturation_time / 2, 2 * saturation_time, 10e6 * YEAR],
        water_content_ppm=150,
        initial_grain_size=0.01,
    )
    assert result.site_saturation_time == pytest.approx(saturation_time)
    np.testing.assert_array_equal(result.is_saturated, [False, False, True, True])
    np.testing.assert_allclose(
        result.grain_number_density[1:], [final_number / 2, final_number, final_number]
    )
    assert result.transformed_fraction[-1] == pytest.approx(final_fraction)
    assert result.mean_grain_diameter[-1] == pytest.approx(
        np.cbrt(6 * final_fraction / (np.pi * final_number))
    )
    assert np.isnan(result.mean_grain_diameter[0])


def test_fixed_solution_is_continuous_at_saturation(mow_kinetics):
    saturation = 443122.3345272988
    times = saturation * np.array([1 - 1e-6, 1, 1 + 1e-6])
    result = solve_fixed_conditions(
        mow_kinetics,
        13.5e9,
        973.15,
        times,
        water_content_ppm=150,
        initial_grain_size=0.01,
    )
    assert np.all(np.diff(result.extended_volume) >= 0)
    np.testing.assert_allclose(
        result.extended_volume, result.extended_volume[1], rtol=1e-4
    )


def test_below_equilibrium_preserves_initial_fraction(mow_kinetics):
    result = solve_fixed_conditions(
        mow_kinetics,
        12e9,
        1500,
        [0, 1e6 * YEAR],
        water_content_ppm=150,
        initial_grain_size=0.01,
        initial_transformed_fraction=0.25,
    )
    np.testing.assert_array_equal(result.transformed_fraction, [0.25, 0.25])
    np.testing.assert_array_equal(result.grain_number_density, [0, 0])
    assert np.isnan(result.mean_grain_diameter).all()
    assert np.isinf(result.site_saturation_time)


@pytest.mark.parametrize("initial_fraction", [0.25, 1.0])
def test_fixed_solution_respects_preexisting_transformation(
    mow_kinetics, initial_fraction
):
    solve_options = {"water_content_ppm": 150, "initial_grain_size": 0.01}
    times = [0, 1e4 * YEAR]
    fresh = solve_fixed_conditions(mow_kinetics, 13.5e9, 973.15, times, **solve_options)
    seeded = solve_fixed_conditions(
        mow_kinetics,
        13.5e9,
        973.15,
        times,
        initial_transformed_fraction=initial_fraction,
        **solve_options,
    )
    np.testing.assert_allclose(
        seeded.transformed_fraction,
        initial_fraction + (1 - initial_fraction) * fresh.transformed_fraction,
    )
    if initial_fraction == 1:
        np.testing.assert_array_equal(seeded.grain_number_density, 0)
