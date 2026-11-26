"""Composable equilibrium, growth, and nucleation laws for phase transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, NamedTuple, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Values used by the HaMaGeoLib MOW surface-nucleation parameter set.
_GAS_CONSTANT = 8.31446  # J/(mol K)
_BOLTZMANN_CONSTANT = 1.38e-23  # J/K


class TransitionBoundaryParameters(NamedTuple):
    """Reference pressure (Pa), temperature (K), and Clapeyron slope (Pa/K)."""

    reference_pressure: float
    reference_temperature: float
    clapeyron_slope: float


class InterfaceControlledGrowthParameters(NamedTuple):
    """Growth coefficients for pressure in Pa, temperature in K, and water in wt ppm."""

    prefactor: float  # m/(s K (wt ppm)^water_exponent)
    water_exponent: float
    activation_enthalpy: float  # J/mol
    activation_volume: float  # m³/mol


class SurfaceNucleationParameters(NamedTuple):
    """Coefficients for surface nucleation in nuclei per m² per second."""

    prefactor: float  # 1/(m² s K)
    activation_enthalpy: float  # J/mol
    activation_volume: float  # m³/mol
    interfacial_energy: float  # J/m²
    shape_factor: float  # dimensionless
    molar_volume: float  # m³/mol


class TransitionKineticsParameters(NamedTuple):
    """Coupling coefficients for the driving force and surface nucleation."""

    molar_volume_change: float  # positive driving-force coefficient in m³/mol
    grain_boundary_area_factor: float  # dimensionless area per volume times grain size


@dataclass(frozen=True)
class TransitionBoundary:
    """A linear phase boundary with explicit reference parameters."""

    parameters: TransitionBoundaryParameters

    def pressure_at(self, temperature: ArrayLike) -> NDArray:
        """Return the equilibrium pressure in Pa at temperature in K."""
        reference_pressure, reference_temperature, slope = self.parameters
        return reference_pressure + slope * (
            np.asarray(temperature) - reference_temperature
        )

    def temperature_at(self, pressure: ArrayLike) -> NDArray:
        """Return the equilibrium temperature in K at pressure in Pa."""
        reference_pressure, reference_temperature, slope = self.parameters
        if slope == 0:
            raise ValueError(
                "A zero Clapeyron slope has no unique equilibrium temperature."
            )
        return (
            reference_temperature + (np.asarray(pressure) - reference_pressure) / slope
        )


class GrowthLaw(Protocol):
    """A crystal growth mechanism that returns an interface speed in m/s."""

    @property
    def process_type(self) -> str: ...

    @property
    def requires_water_content(self) -> bool: ...

    def rate(
        self,
        pressure: NDArray,
        temperature: NDArray,
        driving_force: NDArray,
        water_content_ppm: NDArray | None,
    ) -> NDArray: ...


class NucleationLaw(Protocol):
    """A mechanism that returns nuclei per area or per volume per second."""

    @property
    def process_type(self) -> Literal["surface", "volumetric"]: ...

    def rate(
        self, pressure: NDArray, temperature: NDArray, driving_force: NDArray
    ) -> NDArray: ...


@dataclass(frozen=True)
class InterfaceControlledGrowth:
    """Interface-controlled crystal growth in m/s at positive driving force."""

    process_type: ClassVar[Literal["interface_controlled"]] = "interface_controlled"
    requires_water_content: ClassVar[bool] = True
    parameters: InterfaceControlledGrowthParameters

    def rate(
        self,
        pressure: NDArray,
        temperature: NDArray,
        driving_force: NDArray,
        water_content_ppm: NDArray | None,
    ) -> NDArray:
        """Evaluate the Hosoya law with water content in weight ppm."""
        if water_content_ppm is None:
            raise ValueError(
                "Water content is required for interface-controlled growth."
            )
        p = self.parameters
        thermal_energy = _GAS_CONSTANT * temperature
        activation_energy = p.activation_enthalpy + pressure * p.activation_volume
        driving_factor = -np.expm1(-driving_force / thermal_energy)
        return (
            p.prefactor
            * water_content_ppm**p.water_exponent
            * temperature
            * np.exp(-activation_energy / thermal_energy)
            * driving_factor
        )


@dataclass(frozen=True)
class SurfaceNucleation:
    """Nucleation per unit area per second at positive driving force."""

    process_type: ClassVar[Literal["surface"]] = "surface"
    parameters: SurfaceNucleationParameters

    def rate(
        self, pressure: NDArray, temperature: NDArray, driving_force: NDArray
    ) -> NDArray:
        """Evaluate surface nucleation using P in Pa, T in K, and force in J/mol."""
        p = self.parameters
        activation_energy = p.activation_enthalpy + pressure * p.activation_volume
        barrier = (
            16 * np.pi * p.shape_factor * p.molar_volume**2 * p.interfacial_energy**3
        ) / (3 * driving_force**2)
        return (
            p.prefactor
            * temperature
            * np.exp(-barrier / (_BOLTZMANN_CONSTANT * temperature))
            * np.exp(-activation_energy / (_GAS_CONSTANT * temperature))
        )


@dataclass(frozen=True)
class TransitionRates:
    """Growth in m/s and nucleation in 1/(m² s) and 1/(m³ s), respectively."""

    growth: NDArray
    surface_nucleation: NDArray
    volumetric_nucleation: NDArray


@dataclass(frozen=True)
class TransitionKinetics:
    """Combine a phase boundary with forward growth and nucleation laws."""

    boundary: TransitionBoundary
    growth_law: GrowthLaw
    nucleation_law: NucleationLaw
    parameters: TransitionKineticsParameters

    def rates(
        self,
        pressure: ArrayLike,
        temperature: ArrayLike,
        *,
        water_content_ppm: ArrayLike | None = None,
        initial_grain_size: float | None = None,
    ) -> TransitionRates:
        """Return forward rates for broadcastable P (Pa), T (K), and water content.

        All rates are zero at or below the equilibrium pressure. Supply water
        content in weight ppm if the growth law requires it. Surface nucleation
        requires initial grain size in m for conversion to a volumetric rate.
        """
        pressure = np.asarray(pressure, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        water = None
        if self.growth_law.requires_water_content:
            if water_content_ppm is None:
                raise ValueError(
                    "Water content is required by the selected growth law."
                )
            pressure, temperature, water = np.broadcast_arrays(
                pressure, temperature, np.asarray(water_content_ppm, dtype=float)
            )
            if not np.isfinite(water).all() or np.any(water < 0):
                raise ValueError(
                    "Water content must be finite and nonnegative in weight ppm."
                )
        else:
            pressure, temperature = np.broadcast_arrays(pressure, temperature)

        if not np.isfinite(pressure).all():
            raise ValueError("Pressure must be finite.")
        if not np.isfinite(temperature).all() or np.any(temperature <= 0):
            raise ValueError("Temperature must be finite and positive in kelvin.")

        area_per_volume = None
        nucleation_type = self.nucleation_law.process_type
        if nucleation_type == "surface":
            if initial_grain_size is None:
                raise ValueError(
                    "Initial grain size is required for surface nucleation."
                )
            if not np.isfinite(initial_grain_size) or initial_grain_size <= 0:
                raise ValueError(
                    "Initial grain size must be finite and positive in metres."
                )
            area_per_volume = (
                self.parameters.grain_boundary_area_factor / initial_grain_size
            )
        elif nucleation_type != "volumetric":
            raise ValueError("Nucleation process type must be surface or volumetric.")

        overpressure = pressure - self.boundary.pressure_at(temperature)
        active = overpressure > 0
        active_pressure = pressure[active]
        active_temperature = temperature[active]
        driving_force = self.parameters.molar_volume_change * overpressure[active]

        growth = np.zeros(pressure.shape)
        growth[active] = self.growth_law.rate(
            active_pressure,
            active_temperature,
            driving_force,
            None if water is None else water[active],
        )
        nucleation = np.zeros(pressure.shape)
        nucleation[active] = self.nucleation_law.rate(
            active_pressure, active_temperature, driving_force
        )
        if area_per_volume is not None:
            return TransitionRates(growth, nucleation, nucleation * area_per_volume)
        return TransitionRates(growth, np.zeros_like(nucleation), nucleation)
