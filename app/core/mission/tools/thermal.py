import numpy as np
import pytz
from langchain.tools import tool
from pydantic import BaseModel
from scipy.constants import Stefan_Boltzmann

from app.core.mission.utils.eclipse import EclipseResults
from app.core.mission.utils.power import PowerConfig
from app.core.mission.utils.propagator import PropagationResults
from app.core.mission.utils.sun_geometry import SunGeometryResults
from app.core.mission.utils.thermal import (
    ThermalConfig,
    ThermalResults,
    check_thermal_limits,
    compute_internal_heat,
    compute_radiation_heat,
    compute_solar_heat_input,
    propagate_temperature,
)


class ComputeSolarHeatInputTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    beta_deg: float
    thermal_config: ThermalConfig


class ComputeInternalHeatTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    power_config: PowerConfig


class ComputeRadiationHeatTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    temp_K: float
    thermal_config: ThermalConfig


class PropagateTemperatureTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    times: np.ndarray
    eclipsed: np.ndarray
    beta_deg: np.ndarray
    power_config: PowerConfig
    thermal_config: ThermalConfig


class CheckThermalLimitsTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    temperature_K: np.ndarray
    thermal_config: ThermalConfig


class ComputeThermalTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    propagation_results: PropagationResults
    eclipse_results: EclipseResults
    sun_geometry: SunGeometryResults
    power_config: PowerConfig
    thermal_config: ThermalConfig


SOLAR_CONSTANT = 1361  # W / m^2


@tool(args_schema=ComputeSolarHeatInputTool)
def compute_solar_heat_input_tool(
    beta_deg: float,
    thermal_config: ThermalConfig,
) -> float:
    """
    Calculate solar radiation heat input to the spacecraft.

    Computes the heat absorbed from solar radiation based on the beta angle
    (angle between orbital plane and Sun vector), spacecraft surface area,
    and material absorptivity. The effective solar flux is reduced by the
    cosine of the beta angle.

    Parameters
    ----------
    beta_deg : float
        Beta angle in degrees, representing the angle between the orbital
        plane and the Sun vector. Range: -90 to 90 degrees.
    thermal_config : ThermalConfig
        Thermal configuration dictionary containing:
        - 'area_m2': spacecraft surface area exposed to Sun in square meters (float)
        - 'absorptivity': solar absorptivity coefficient, range 0-1 (float)

    Returns
    -------
    float
        Solar heat input in watts (W). Returns 0 if the effective flux is negative.

    Notes
    -----
    The effective solar flux is calculated as:
        Q_solar = A × α × S × cos(β)

    where:
    - A = surface area (m²)
    - α = absorptivity (dimensionless, 0-1)
    - S = solar constant (1361 W/m²)
    - β = beta angle (degrees)

    The cosine factor accounts for the angle of incidence of solar radiation
    on the orbital plane. Negative values (when cos(β) < 0) are clamped to zero.
    """
    effective_flux = SOLAR_CONSTANT * np.cos(np.deg2rad(beta_deg))
    effective_flux = max(effective_flux, 0.0)
    return thermal_config["area_m2"] * thermal_config["absorptivity"] * effective_flux


@tool(args_schema=ComputeInternalHeatTool)
def compute_internal_heat_tool(
    power_config: PowerConfig,
) -> float:
    """
    Calculate internal heat generation from spacecraft electronics.

    Sums the heat dissipated by the spacecraft bus and payload, assuming
    all electrical power is ultimately converted to heat.

    Parameters
    ----------
    power_config : PowerConfig
        Power configuration dictionary containing:
        - 'bus_power_W': spacecraft bus power consumption in watts (float)
        - 'payload_power_W': payload power consumption in watts (float)

    Returns
    -------
    float
        Total internal heat generation in watts (W).

    Notes
    -----
    This function assumes 100% conversion of electrical power to heat, which
    is a reasonable approximation for spacecraft thermal modeling since nearly
    all electrical energy eventually dissipates as heat through resistive losses,
    processing, and radiation from electronics.
    """
    return power_config["bus_power_W"] + power_config["payload_power_W"]


@tool(args_schema=ComputeRadiationHeatTool)
def compute_radiation_heat_tool(temp_K: float, thermal_config: ThermalConfig):
    """
    Calculate radiative heat loss from the spacecraft to space.

    Uses the Stefan-Boltzmann law to compute heat radiated to the cold space
    environment (assumed to be at 0 K) based on the spacecraft's surface
    temperature, emissivity, and radiating area.

    Parameters
    ----------
    temp_K : float
        Current spacecraft temperature in Kelvin (K).
    thermal_config : ThermalConfig
        Thermal configuration dictionary containing:
        - 'emissivity': thermal emissivity coefficient, range 0-1 (float)
        - 'area_m2': spacecraft radiating surface area in square meters (float)

    Returns
    -------
    float
        Radiative heat loss in watts (W).

    Notes
    -----
    The Stefan-Boltzmann law for radiation heat loss:
        Q_rad = ε × σ × A × T⁴

    where:
    - ε = emissivity (dimensionless, 0-1)
    - σ = Stefan-Boltzmann constant (5.670374419×10⁻⁸ W⋅m⁻²⋅K⁻⁴)
    - A = radiating area (m²)
    - T = temperature (K)

    This assumes the spacecraft radiates to deep space at effectively 0 K.
    """
    return (
        thermal_config["emissivity"]
        * Stefan_Boltzmann
        * thermal_config["area_m2"]
        * temp_K**4
    )


@tool(args_schema=PropagateTemperatureTool)
def propagate_temperature_tool(
    times: np.ndarray,
    eclipsed: np.ndarray,
    beta_deg: np.ndarray,
    power_config: PowerConfig,
    thermal_config: ThermalConfig,
) -> np.ndarray:
    """
    Propagate spacecraft temperature over time using energy balance.

    Simulates the thermal evolution of the spacecraft by solving the energy
    balance equation at each time step. Accounts for solar heat input (when
    not in eclipse), internal heat generation, and radiative heat loss.

    Parameters
    ----------
    times : np.ndarray
        Array of datetime objects for each time step.
    eclipsed : np.ndarray
        Boolean array indicating eclipse status at each time step.
        True = in eclipse (no solar input), False = in sunlight.
    beta_deg : np.ndarray
        Array of beta angles in degrees at each time step.
    power_config : PowerConfig
        Power configuration dictionary containing:
        - 'bus_power_W': bus power consumption in watts (float)
        - 'payload_power_W': payload power consumption in watts (float)
    thermal_config : ThermalConfig
        Thermal configuration dictionary containing:
        - 'initial_temp_K': initial spacecraft temperature in Kelvin (float)
        - 'mass_kg': spacecraft mass in kilograms (float)
        - 'heat_capacity_J_per_kgK': specific heat capacity in J/(kg·K) (float)
        - 'area_m2': surface area in square meters (float)
        - 'absorptivity': solar absorptivity, range 0-1 (float)
        - 'emissivity': thermal emissivity, range 0-1 (float)

    Returns
    -------
    np.ndarray
        Array of temperatures in Kelvin at each time step.

    Notes
    -----
    Energy Balance Equation:
        m × c × dT/dt = Q_in - Q_out

    where:
    - m = spacecraft mass (kg)
    - c = specific heat capacity (J/(kg·K))
    - T = temperature (K)
    - Q_in = internal heat + solar heat (if not eclipsed) (W)
    - Q_out = radiative heat loss (W)

    Heat Sources:
    1. Internal: constant from electronics (bus + payload power)
    2. Solar: SOLAR_CONSTANT × A × α × cos(β), zero during eclipse

    Heat Sink:
    - Radiation: ε × σ × A × T⁴ (always active)

    The temperature is updated using forward Euler integration:
        T(t+dt) = T(t) + (Q_in - Q_out) × dt / (m × c)

    This is a simplified lumped-mass thermal model assuming:
    - Uniform spacecraft temperature (no thermal gradients)
    - Instantaneous heat distribution
    - Constant material properties
    """
    internal_heat = compute_internal_heat(power_config)

    propagated_temp = [thermal_config["initial_temp_K"]]
    curr_temp = thermal_config["initial_temp_K"]
    prev_time = times[0].replace(tzinfo=pytz.utc)

    for time, beta, isEclipsed in zip(times[1:], beta_deg[1:], eclipsed[1:]):
        time = time.replace(tzinfo=pytz.utc)
        dt = (time - prev_time).total_seconds()
        Q_in = internal_heat + (
            compute_solar_heat_input(beta, thermal_config) if not isEclipsed else 0
        )
        Q_out = compute_radiation_heat(curr_temp, thermal_config)
        curr_temp += ((Q_in - Q_out) * dt) / (
            thermal_config["mass_kg"] * thermal_config["heat_capacity_J_per_kgK"]
        )
        propagated_temp.append(curr_temp)
        prev_time = time

    return np.array(propagated_temp)


@tool(args_schema=CheckThermalLimitsTool)
def check_thermal_limits_tool(
    temperature_K: np.ndarray,
    thermal_config: ThermalConfig,
) -> dict:
    """
    Check if spacecraft temperatures stay within operational limits.

    Verifies that all temperatures in the propagation remain within the
    specified minimum and maximum allowable limits. Identifies thermal
    violations and reports extreme temperatures reached.

    Parameters
    ----------
    temperature_K : np.ndarray
        Array of spacecraft temperatures in Kelvin at each time step.
    thermal_config : ThermalConfig
        Thermal configuration dictionary containing:
        - 'Tmin_K': minimum allowable temperature in Kelvin (float)
        - 'Tmax_K': maximum allowable temperature in Kelvin (float)

    Returns
    -------
    dict
        Dictionary containing:
        - 'min_temp_K': minimum temperature reached during mission (float)
        - 'max_temp_K': maximum temperature reached during mission (float)
        - 'thermal_violation': True if any temperature exceeded limits (bool)

    Notes
    -----
    A thermal violation occurs if:
    - Any temperature > Tmax_K (overheating)
    - Any temperature < Tmin_K (overcooling)

    Typical spacecraft thermal limits:
    - Electronics: 233-323 K (-40°C to 50°C)
    - Batteries: 263-313 K (-10°C to 40°C)
    - Optics: tighter ranges depending on requirements

    This check is critical for:
    - Mission feasibility assessment
    - Thermal control system design validation
    - Identifying need for heaters or radiators
    """
    min_bearable_temp = thermal_config["Tmin_K"]
    max_bearable_temp = thermal_config["Tmax_K"]

    thermal_violation = any(temperature_K > max_bearable_temp) or any(
        temperature_K < min_bearable_temp
    )

    min_temp = min(temperature_K)
    max_temp = max(temperature_K)

    return {
        "min_temp_K": min_temp,
        "max_temp_K": max_temp,
        "thermal_violation": thermal_violation,
    }


@tool(args_schema=ComputeThermalTool)
def compute_thermal_tool(
    propagation_results: PropagationResults,
    eclipse_results: EclipseResults,
    sun_geometry: SunGeometryResults,
    power_config: PowerConfig,
    thermal_config: ThermalConfig,
) -> ThermalResults:
    """
    Perform complete thermal analysis for a spacecraft mission.

    This is the main entry point for thermal analysis. It propagates spacecraft
    temperature throughout the orbit considering solar input, eclipse periods,
    internal heat generation, and radiative cooling, then checks if temperatures
    remain within operational limits.

    Parameters
    ----------
    propagation_results : PropagationResults
        Dictionary containing spacecraft trajectory data with key:
        - 'time': list of datetime objects for the mission timeline (list[datetime])
    eclipse_results : EclipseResults
        Dictionary containing eclipse analysis results with key:
        - 'eclipsed': boolean array indicating eclipse status at each time step (np.ndarray)
    sun_geometry : SunGeometryResults
        Dictionary containing Sun geometry data with key:
        - 'beta_deg': array of beta angles in degrees at each time step (np.ndarray)
    power_config : PowerConfig
        Power configuration dictionary containing:
        - 'bus_power_W': spacecraft bus power consumption in watts (float)
        - 'payload_power_W': payload power consumption in watts (float)
    thermal_config : ThermalConfig
        Thermal configuration dictionary containing:
        - 'mass_kg': spacecraft mass in kilograms (float)
        - 'heat_capacity_J_per_kgK': specific heat capacity in J/(kg·K) (float)
        - 'area_m2': surface area in square meters (float)
        - 'absorptivity': solar absorptivity, range 0-1 (float)
        - 'emissivity': thermal emissivity, range 0-1 (float)
        - 'Tmin_K': minimum allowable temperature in Kelvin (float)
        - 'Tmax_K': maximum allowable temperature in Kelvin (float)
        - 'initial_temp_K': initial spacecraft temperature in Kelvin (float)

    Returns
    -------
    ThermalResults
        TypedDict containing:
        - 'min_temp_K': minimum temperature reached during mission in Kelvin (float)
        - 'max_temp_K': maximum temperature reached during mission in Kelvin (float)
        - 'thermal_violation': True if temperature limits were exceeded (bool)

    Notes
    -----
    Thermal Analysis Workflow:
    1. Extract time-series data from propagation, eclipse, and sun geometry
    2. Propagate temperature using energy balance equation
    3. Check if all temperatures remain within specified limits
    4. Report extreme temperatures and violation status

    Physical Model:
    - Lumped-mass thermal model (uniform spacecraft temperature)
    - Heat sources: solar radiation (when not eclipsed) + internal electronics
    - Heat sink: thermal radiation to space (Stefan-Boltzmann law)
    - Time integration using forward Euler method

    Key Assumptions:
    - Spacecraft behaves as a single thermal mass
    - No thermal gradients across spacecraft structure
    - Constant material properties (α, ε, c)
    - Space environment at 0 K
    - All electrical power converts to heat

    Applications:
    - Thermal control system design
    - Mission feasibility assessment
    - Heater/radiator sizing
    - Component placement planning
    - Survival heater requirements

    Typical Material Properties:
    - Aluminum: α ≈ 0.2-0.4, ε ≈ 0.05-0.2 (bare), c ≈ 900 J/(kg·K)
    - Black paint: α ≈ 0.95, ε ≈ 0.88
    - White paint: α ≈ 0.25, ε ≈ 0.90
    - MLI (Multi-Layer Insulation): α ≈ 0.15, ε ≈ 0.03
    """
    times = propagation_results["time"]
    eclipsed = eclipse_results["eclipsed"]
    beta_arr = sun_geometry["beta_deg"]

    propagated_temp = propagate_temperature(
        times, eclipsed, beta_arr, power_config, thermal_config
    )

    return check_thermal_limits(propagated_temp, thermal_config)
