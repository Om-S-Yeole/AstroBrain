from app.core import *
from app.core.mission.tools import *

BASE_TOOL_REGISTRY = [
    body_from_str_tool,
    cross_tool,
    datetime_from_times_tool,
    datetime_to_jd_tool,
    deg2rad_tool,
    dot_tool,
    jd_to_datetime_tool,
    non_quantity_to_Quantity_tool,
    norm_tool,
    rad2deg_tool,
    rotation_matrix_from_ECI_to_perifocal_tool,
    rotation_matrix_from_perifocal_to_ECI_tool,
    unit_vec_tool,
    uuid_generator_tool,
    keplerian_to_cartesian,
    cartesian_to_keplerian,
    hohmann_transfer,
    hohmann_time_of_flight,
    bielliptic_transfer,
    plane_change,
    lambert_solver,
    universal_kepler,
    sgp4_propagate_tool,
    orbit_period,
    mean_motion,
    specific_energy,
    specific_angular_momentum,
    eccentricity_vector,
    true_anomaly_from_vectors,
    raan_from_vectors,
    argument_of_periapsis,
]


PROPAGATOR_TOOL_REGISTRY = [
    propagate_at_tool,
    propagate_tool,
]
VISIBILITY_TOOL_REGISTRY = [
    compute_elevation_angle_tool,
    compute_visibility_tool,
    ecef_from_lat_lon_alt_tool,
    extract_visibility_windows_tool,
    visibility_mask_tool,
]
SUN_GEOMETRY_TOOL_REGISTRY = [
    beta_angle_tool,
    compute_sun_geometry_tool,
    orbit_unit_normal_vector_tool,
    sun_vector_gcrs_tool,
]
ECLIPSE_TOOL_REGISTRY = [
    compute_eclipse_tool,
    extract_eclipse_windows_tool,
    is_in_umbra_tool,
    umbra_mask_tool,
]
POWER_TOOL_REGISTRY = [
    compute_power_budget_tool,
    compute_power_consumption_tool,
    compute_power_generation_tool,
    propagate_battery_soc_tool,
]
COMMUNICATION_TOOL_REGISTRY = [
    aggregate_ground_stations_tool,
    compute_communication_tool,
    compute_contact_duration_from_windows_tool,
    compute_passes_tool,
    compute_visibility_for_station_tool,
]
THERMAL_TOOL_REGISTRY = [
    check_thermal_limits_tool,
    compute_internal_heat_tool,
    compute_radiation_heat_tool,
    compute_solar_heat_input_tool,
    compute_thermal_tool,
    propagate_temperature_tool,
]
DUTY_CYCLE_TOOL_REGISTRY = [
    check_sunlight_constraint_tool,
    compute_data_downlinked_tool,
    compute_data_generated_tool,
    compute_payload_ON_time_tool,
    evaluate_duty_cycle_tool,
]

MISSION_SUMMARY_TOOL_REGISTRY = [
    build_mission_summary_tool,
    determine_limiting_subsystem_tool,
    generate_explanation_tool,
    generate_recommendations_tool,
]

# ALL_TOOL_REGISTRY = (
#     BASE_TOOL_REGISTRY
#     + PROPAGATOR_TOOL_REGISTRY
#     + VISIBILITY_TOOL_REGISTRY
#     + SUN_GEOMETRY_TOOL_REGISTRY
#     + ECLIPSE_TOOL_REGISTRY
#     + POWER_TOOL_REGISTRY
#     + COMMUNICATION_TOOL_REGISTRY
#     + THERMAL_TOOL_REGISTRY
#     + DUTY_CYCLE_TOOL_REGISTRY
#     + MISSION_SUMMARY_TOOL_REGISTRY
# )

ALL_TOOL_REGISTRY = {
    tool.name: tool
    for tool in (
        BASE_TOOL_REGISTRY
        + PROPAGATOR_TOOL_REGISTRY
        + VISIBILITY_TOOL_REGISTRY
        + SUN_GEOMETRY_TOOL_REGISTRY
        + ECLIPSE_TOOL_REGISTRY
        + POWER_TOOL_REGISTRY
        + COMMUNICATION_TOOL_REGISTRY
        + THERMAL_TOOL_REGISTRY
        + DUTY_CYCLE_TOOL_REGISTRY
        + MISSION_SUMMARY_TOOL_REGISTRY
    )
}
