from datetime import datetime
from typing import Tuple, TypedDict

import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import get_sun
from astropy.time import Time

from app.core.mission.utils.propagator import PropagationResults


class SunGeometryResults(TypedDict):
    sun_vec_eci: list
    beta_deg: list


def sun_vector_gcrs(time: datetime | Time) -> Tuple:
    """
    Compute the Sun position vector in GCRS (Geocentric Celestial Reference System) coordinates.

    Parameters
    ----------
    time : datetime or Time
        The time at which to compute the Sun vector. If datetime is provided,
        it will be converted to UTC timezone.

    Returns
    -------
    tuple of float
        A 3-tuple (x, y, z) representing the Sun's position vector in GCRS
        coordinates in kilometers.

    Raises
    ------
    TypeError
        If time is not a datetime or astropy Time object.
    """
    if not isinstance(time, (datetime, Time)):
        raise TypeError(f"Expected type of time is datetime, Time. Got {type(time)}")

    time = time.replace(tzinfo=pytz.utc)
    sun_vector = get_sun(Time(time))  # We will get in GCRS
    cart = sun_vector.cartesian
    return (cart.x.to(u.km).value, cart.y.to(u.km).value, cart.z.to(u.km).value)


def orbit_unit_normal_vector(
    r_eci: list | np.ndarray, v_eci: list | np.ndarray
) -> np.ndarray:
    """
    Calculate the unit normal vector to the orbital plane.

    The orbital plane normal is computed using the cross product of the
    position and velocity vectors (angular momentum vector), then normalized.

    Parameters
    ----------
    r_eci : list or np.ndarray
        Position vector in ECI coordinates (km).
    v_eci : list or np.ndarray
        Velocity vector in ECI coordinates (km/s).

    Returns
    -------
    np.ndarray
        Unit normal vector to the orbital plane (dimensionless).

    Raises
    ------
    TypeError
        If r_eci or v_eci are not list or numpy arrays.
    """
    if not isinstance(r_eci, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of r_eci is list or np.ndarray. Got {type(r_eci)}"
        )
    if not isinstance(v_eci, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of v_eci is list or np.ndarray. Got {type(v_eci)}"
        )

    r_eci = np.array(r_eci)
    v_eci = np.array(v_eci)
    h = np.cross(r_eci, v_eci)

    return h / np.linalg.norm(h)


def beta_angle(
    r_eci: list | np.ndarray, v_eci: list | np.ndarray, sun_vec_eci: list | np.ndarray
):
    """
    Calculate the beta angle for a spacecraft orbit.

    The beta angle is the angle between the orbital plane and the Sun vector.
    It determines the illumination conditions of the spacecraft and ranges
    from -90 to +90 degrees.

    Parameters
    ----------
    r_eci : list or np.ndarray
        Position vector in ECI coordinates (km).
    v_eci : list or np.ndarray
        Velocity vector in ECI coordinates (km/s).
    sun_vec_eci : list or np.ndarray
        Sun position vector in ECI coordinates (km).

    Returns
    -------
    float
        Beta angle in degrees.
    """
    sun_vec_eci = np.array(sun_vec_eci)
    return np.rad2deg(
        np.arcsin(
            orbit_unit_normal_vector(r_eci, v_eci)
            @ sun_vec_eci
            / np.linalg.norm(sun_vec_eci)
        )
    )


def compute_sun_geometry(propagation_results: PropagationResults) -> SunGeometryResults:
    """
    Compute Sun geometry parameters for an entire propagated orbit.

    For each time step in the propagation results, this function calculates
    the Sun position vector and the beta angle.

    Parameters
    ----------
    propagation_results : PropagationResults
        Dictionary containing propagation data with keys:
        - 'time': list of datetime objects
        - 'r_eci': list of position vectors in ECI coordinates (km)
        - 'v_eci': list of velocity vectors in ECI coordinates (km/s)

    Returns
    -------
    SunGeometryResults
        TypedDict containing:
        - 'sun_vec_eci': list of Sun position vectors in ECI coordinates (km)
        - 'beta_deg': list of beta angles in degrees
    """
    times = propagation_results["time"]
    r_vecs = propagation_results["r_eci"]
    v_vecs = propagation_results["v_eci"]

    sun_vecs = []
    beta_deg = []

    for time, r_vec, v_vec in zip(times, r_vecs, v_vecs):
        time = time.replace(tzinfo=pytz.utc)
        sun_vec = sun_vector_gcrs(time)
        beta = beta_angle(r_vec, v_vec, sun_vec)

        sun_vecs.append(sun_vec)
        beta_deg.append(beta)

    return {"sun_vec_eci": sun_vecs, "beta_deg": beta_deg}
