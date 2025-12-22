from typing import List

from langchain.tools import tool
from pydantic import BaseModel

from app.core.mission.utils.communication import CommunicationResults
from app.core.mission.utils.duty_cycle import DutyCycleResults, PayloadConfig
from app.core.mission.utils.mission_summery import (
    MissionSummary,
    determine_limiting_subsystem,
    generate_explanation,
    generate_recommendations,
)
from app.core.mission.utils.power import PowerResults
from app.core.mission.utils.thermal import ThermalResults


class DetermineLimitingSubsystemTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    power: PowerResults
    thermal: ThermalResults
    duty_cycle: DutyCycleResults


class GenerateRecommendationsTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    limiting_subsystem: str


class GenerateExplanationTool(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    feasible: bool
    limiting_subsystem: str
    power: PowerResults


class BuildMissionSummary(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    power: PowerResults
    thermal: dict
    duty_cycle: DutyCycleResults
    comms: CommunicationResults
    payload_config: PayloadConfig


@tool(args_schema=DetermineLimitingSubsystemTool)
def determine_limiting_subsystem_tool(
    power: PowerResults,
    thermal: ThermalResults,
    duty_cycle: DutyCycleResults,
) -> str:
    """
    Identify the subsystem that limits mission feasibility.

    Evaluates mission-critical subsystems in priority order (power, thermal,
    sunlight availability, downlink capacity) to determine which constraint,
    if any, prevents the mission from being feasible.

    Parameters
    ----------
    power : PowerResults
        Dictionary containing power budget analysis results with key:
        - 'power_violation': True if battery SOC reached zero (bool)
    thermal : ThermalResults
        Dictionary containing thermal analysis results with key:
        - 'thermal_violation': True if temperature limits were exceeded (bool)
    duty_cycle : DutyCycleResults
        Dictionary containing duty cycle evaluation results with keys:
        - 'sunlight_feasible': True if sunlight availability supports duty cycle (bool)
        - 'downlink_feasible': True if downlink capacity handles generated data (bool)

    Returns
    -------
    str
        Name of the limiting subsystem: 'power', 'thermal', 'sunlight',
        'downlink', or 'none' if all constraints are satisfied.

    Notes
    -----
    Priority Order:
    1. Power - most critical for spacecraft survival
    2. Thermal - critical for component survival and operation
    3. Sunlight - limits payload operational capability
    4. Downlink - limits data return but not spacecraft survival

    This prioritization reflects the typical criticality hierarchy in
    spacecraft mission design.
    """
    if power["power_violation"]:
        return "power"
    if thermal["thermal_violation"]:
        return "thermal"
    if not duty_cycle["sunlight_feasible"]:
        return "sunlight"
    if not duty_cycle["downlink_feasible"]:
        return "downlink"
    return "none"


@tool(args_schema=GenerateRecommendationsTool)
def generate_recommendations_tool(
    limiting_subsystem: str,
) -> List[str]:
    """
    Generate actionable recommendations to address mission constraints.

    Provides specific suggestions for resolving issues identified with the
    limiting subsystem. Recommendations focus on design modifications or
    operational changes that could make the mission feasible.

    Parameters
    ----------
    limiting_subsystem : str
        Name of the limiting subsystem: 'power', 'thermal', 'sunlight',
        or 'downlink'. If 'none', no recommendations are needed.

    Returns
    -------
    List[str]
        List of recommendation strings. Returns empty list if no limiting
        subsystem (limiting_subsystem == 'none').

    Notes
    -----
    Recommendation Categories:

    **Power:**
    - Increase energy generation (larger solar panels)
    - Reduce energy consumption (lower duty cycle, efficient components)

    **Thermal:**
    - Improve heat rejection (lower absorptivity, more radiator area)
    - Reduce heat generation (lower duty cycle during high-heat periods)

    **Sunlight:**
    - Reduce operational requirements (lower duty cycle)
    - Change mission design (different orbit with less eclipse time)

    **Downlink:**
    - Increase downlink capability (higher data rate, more ground stations)
    - Reduce data volume (lower data rate, data compression, selective downlink)

    These recommendations are starting points for trade studies and should
    be evaluated against mission requirements, cost, and schedule constraints.
    """
    recs = []

    if limiting_subsystem == "power":
        recs.append("Increase solar panel area or reduce payload duty cycle.")
        recs.append("Reduce payload power consumption.")
    elif limiting_subsystem == "thermal":
        recs.append("Reduce surface absorptivity or increase radiator area.")
        recs.append("Lower payload duty cycle during sunlit periods.")
    elif limiting_subsystem == "sunlight":
        recs.append("Reduce payload duty cycle or use payload only in sunlight.")
        recs.append("Consider orbit with lower eclipse fraction.")
    elif limiting_subsystem == "downlink":
        recs.append("Increase downlink data rate or add ground stations.")
        recs.append("Reduce payload data generation rate.")

    return recs


@tool(args_schema=GenerateExplanationTool)
def generate_explanation_tool(
    feasible: bool,
    limiting_subsystem: str,
    power: PowerResults,
) -> str:
    """
    Generate a human-readable explanation of mission feasibility status.

    Creates a descriptive text summary explaining whether the mission is
    feasible and, if not, which subsystem constraint is violated and why.

    Parameters
    ----------
    feasible : bool
        True if the mission is feasible under current configuration,
        False otherwise.
    limiting_subsystem : str
        Name of the limiting subsystem: 'power', 'thermal', 'sunlight',
        'downlink', or 'none'.
    power : PowerResults
        Dictionary containing power budget analysis results with key:
        - 'min_soc': minimum state of charge reached (float)

    Returns
    -------
    str
        Explanation text describing the mission feasibility status and,
        if infeasible, the specific constraint violation.

    Notes
    -----
    The explanation provides:
    - Feasible missions: confirmation that all subsystems operate within limits
    - Infeasible missions: specific details about which constraint is violated
      and relevant metrics (e.g., minimum SOC for power violations)

    This text is intended for mission designers and stakeholders to quickly
    understand the results of the mission analysis.
    """
    if feasible:
        return (
            "The mission is feasible under the current configuration. "
            "All subsystems operate within acceptable limits, including "
            "power, thermal environment, payload duty cycle, and communication capability."
        )

    explanations = {
        "power": (
            f"Mission is power-limited. Battery state of charge drops below safe limits "
            f"(minimum SOC = {power['min_soc']:.2f})."
        ),
        "thermal": (
            "Mission violates thermal constraints. Spacecraft temperature exceeds allowable bounds."
        ),
        "sunlight": (
            "Mission is constrained by sunlight availability. "
            "Required payload ON-time cannot be accommodated within sunlit periods."
        ),
        "downlink": (
            "Mission is downlink-limited. Payload generates more data than can be downlinked during available ground contact windows."
        ),
    }

    return explanations.get(
        limiting_subsystem,
        "Mission is infeasible due to multiple subsystem constraints.",
    )


@tool(args_schema=BuildMissionSummary)
def build_mission_summary_tool(
    power: PowerResults,
    thermal: dict,
    duty_cycle: DutyCycleResults,
    comms: CommunicationResults,
    payload_config: PayloadConfig,
) -> MissionSummary:
    """
    Build a comprehensive mission feasibility summary.

    Aggregates results from all subsystem analyses to produce a complete
    mission summary including feasibility determination, limiting factors,
    key performance metrics, explanation, and recommendations.

    Parameters
    ----------
    power : PowerResults
        Dictionary containing power budget analysis results:
        - 'power_violation': True if battery SOC reached zero (bool)
        - 'min_soc': minimum state of charge reached, range 0-1 (float)
    thermal : dict
        Dictionary containing thermal analysis results:
        - 'thermal_violation': True if temperature limits exceeded (bool)
        - 'max_temp_K': maximum temperature reached in Kelvin (float)
        - 'min_temp_K': minimum temperature reached in Kelvin (float)
    duty_cycle : DutyCycleResults
        Dictionary containing duty cycle evaluation results:
        - 'overall_feasible': True if all duty cycle constraints satisfied (bool)
        - 'data_generated_Mb': total data generated in megabits (float)
        - 'data_downlinked_Mb': total data downlinked in megabits (float)
    comms : CommunicationResults
        Dictionary containing communication analysis results:
        - 'passes_per_day': average number of ground station passes per day (float)
    payload_config : PayloadConfig
        Payload configuration dictionary (used for generating recommendations):
        - 'payload_power_W': payload power consumption in watts (float)
        - 'data_rate_Mbps': data generation rate in Mbps (float)
        - 'duty_cycle': fraction of time payload is ON, range 0-1 (float)
        - 'requires_sunlight': whether payload needs sunlight (bool)
        - 'requires_contact': whether data must be downlinked (bool)

    Returns
    -------
    MissionSummary
        TypedDict containing comprehensive mission summary:
        - 'feasible': True if mission is feasible under current configuration (bool)
        - 'limiting_subsystem': name of constraint limiting feasibility (str)
        - 'key_metrics': dictionary of critical mission metrics (KeyMissionMetrics)
        - 'explanation': human-readable text explaining feasibility status (str)
        - 'recommendations': list of actionable suggestions for improvement (List[str])

    Notes
    -----
    Mission Feasibility Determination:
    A mission is deemed feasible if and only if ALL of the following are true:
    1. No power violations (battery never fully depletes)
    2. No thermal violations (temperature stays within limits)
    3. Duty cycle is overall feasible (sunlight and downlink constraints met)

    Key Metrics Included:
    - min_soc: Minimum battery state of charge (0-1), indicates power margin
    - max_temp_K: Maximum temperature reached, indicates thermal stress
    - min_temp_K: Minimum temperature reached, indicates cold survival
    - passes_per_day: Ground contact frequency, indicates communication opportunity
    - data_generated_Mb: Total mission data volume generated
    - data_downlinked_Mb: Total data volume that can be downlinked

    The summary provides:
    1. **Feasibility verdict**: clear yes/no determination
    2. **Root cause**: identifies which subsystem limits feasibility
    3. **Quantitative metrics**: key performance indicators
    4. **Qualitative explanation**: context for stakeholders
    5. **Actionable recommendations**: design modifications to address issues

    Use Cases:
    - Mission design reviews
    - Trade study documentation
    - Stakeholder communication
    - Design iteration planning
    - Requirements validation
    """
    feasible = (
        not power["power_violation"]
        and not thermal["thermal_violation"]
        and duty_cycle["overall_feasible"]
    )

    limiting = determine_limiting_subsystem(power, thermal, duty_cycle)

    explanation = generate_explanation(
        feasible, limiting, power, thermal, duty_cycle, comms
    )

    recommendations = generate_recommendations(limiting, payload_config)

    key_metrics = {
        "min_soc": power["min_soc"],
        "max_temp_K": thermal["max_temp_K"],
        "min_temp_K": thermal["min_temp_K"],
        "passes_per_day": comms["passes_per_day"],
        "data_generated_Mb": duty_cycle["data_generated_Mb"],
        "data_downlinked_Mb": duty_cycle["data_downlinked_Mb"],
    }

    return {
        "feasible": feasible,
        "limiting_subsystem": limiting,
        "key_metrics": key_metrics,
        "explanation": explanation,
        "recommendations": recommendations,
    }
