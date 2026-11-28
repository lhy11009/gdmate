"""Post-transformation grain coarsening at fixed pressure and temperature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

_GAS_CONSTANT = 8.314462618  # J/(mol K), as used in HaMaGeoLib


class GrainCoarseningParameters(NamedTuple):
    """Coefficients for grain coarsening under constant conditions."""

    rate_constant: float  # m^exponent/s
    exponent: float
    activation_energy: float  # J/mol
    activation_volume: float  # m³/mol


@dataclass(frozen=True)
class GrainCoarseningModel:
    """Arrhenius grain coarsening with explicitly supplied parameters."""

    parameters: GrainCoarseningParameters

    def __post_init__(self) -> None:
        if not np.isfinite(self.parameters).all():
            raise ValueError("Grain growth parameters must be finite.")
        if self.parameters.rate_constant < 0 or self.parameters.exponent <= 0:
            raise ValueError("Rate constant must be nonnegative and exponent positive.")

    def grain_size_at_time(
        self,
        initial_grain_size: ArrayLike,
        elapsed_time: ArrayLike,
        pressure: ArrayLike,
        temperature: ArrayLike,
    ) -> NDArray:
        """Return grain diameter (m) after growth at constant P (Pa) and T (K).

        The elapsed time is in seconds. Each input can be a scalar or a
        broadcastable array. The law evolves the m-th power of grain size
        linearly in time, with m given by the growth exponent.
        """
        size = np.asarray(initial_grain_size, dtype=float)
        time = np.asarray(elapsed_time, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        temperature = np.asarray(temperature, dtype=float)

        if not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError(
                "Initial grain size must be finite and positive in metres."
            )
        if not np.isfinite(time).all() or np.any(time < 0):
            raise ValueError("Elapsed time must be finite and nonnegative in seconds.")
        if not np.isfinite(pressure).all():
            raise ValueError("Pressure must be finite in pascals.")
        if not np.isfinite(temperature).all() or np.any(temperature <= 0):
            raise ValueError("Temperature must be finite and positive in kelvin.")

        rate_constant, exponent, energy, volume = self.parameters
        arrhenius_factor = np.exp(
            -(energy + pressure * volume) / (_GAS_CONSTANT * temperature)
        )
        return (size**exponent + rate_constant * arrhenius_factor * time) ** (
            1 / exponent
        )
