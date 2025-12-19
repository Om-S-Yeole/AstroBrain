# app/core/mission/
# │
# ├── visibility.py -> 2 -> Done
# ├── eclipse.py -> 4 -> Done
# ├── sun_geometry.py -> 3 -> Done
# ├── power.py -> 5 -> Done
# ├── thermal.py -> 8
# ├── communication.py -> 6
# ├── duty_cycle.py -> 7
# ├── propagator.py -> 1 ->  Done
# └── __init__.py

from app.core.mission.eclipse import (
    compute_eclipse,
    extract_eclipse_windows,
    is_in_umbra,
    umbra_mask,
)
from app.core.mission.power import (
    compute_power_budget,
    compute_power_consumption,
    compute_power_generation,
    propagate_battery_soc,
)
from app.core.mission.propagator import OrbitPropagator
from app.core.mission.sun_geometry import (
    beta_angle,
    compute_sun_geometry,
    orbit_unit_normal_vector,
    sun_vector_gcrs,
)
from app.core.mission.visibility import (
    compute_elevation_angle,
    compute_visibility,
    ecef_from_lat_lon_alt,
    extract_visibility_windows,
    visibility_mask,
)

__all__ = [
    "OrbitPropagator",
    "compute_elevation_angle",
    "ecef_from_lat_lon_alt",
    "visibility_mask",
    "extract_visibility_windows",
    "compute_visibility",
    "beta_angle",
    "compute_sun_geometry",
    "orbit_unit_normal_vector",
    "sun_vector_gcrs",
    "compute_eclipse",
    "extract_eclipse_windows",
    "is_in_umbra",
    "umbra_mask",
    "compute_power_budget",
    "compute_power_consumption",
    "compute_power_generation",
    "propagate_battery_soc",
]
