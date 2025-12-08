from datetime import datetime, timedelta
from math import pi, sqrt

import numpy as np
from astropy import units as u
from astropy.time import TimeDelta
from astropy.units import Quantity
from poliastro.bodies import Body, Earth
from poliastro.iod.vallado import lambert
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import ValladoPropagator
from sgp4.api import SGP4_ERRORS, Satrec

from app.core import body_from_str, datetime_to_jd, non_quantity_to_Quantity


def keplerian_to_cartesian(
    a: float | Quantity,
    ecc: float | Quantity,
    inc: float | Quantity,
    raan: float | Quantity,
    argp: float | Quantity,
    nu: float | Quantity,
    attractor: str | Body = Earth,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian orbital elements to Cartesian position and velocity vectors.

    This function converts classical orbital elements (Keplerian elements) to Cartesian
    coordinates (position and velocity vectors) in three-dimensional space. The conversion
    is performed relative to a specified central body (attractor).

    Parameters
    ----------
    a : float or astropy.units.Quantity
        Semi-major axis of the orbit. If float, assumed to be in kilometers.
    ecc : float or astropy.units.Quantity
        Eccentricity of the orbit (dimensionless). If float, treated as dimensionless.
    inc : float or astropy.units.Quantity
        Inclination of the orbit. If float, assumed to be in degrees.
    raan : float or astropy.units.Quantity
        Right ascension of the ascending node (RAAN). If float, assumed to be in degrees.
    argp : float or astropy.units.Quantity
        Argument of periapsis. If float, assumed to be in degrees.
    nu : float or astropy.units.Quantity
        True anomaly. If float, assumed to be in degrees.
    attractor : str or poliastro.bodies.Body, optional
        The central body around which the orbit is defined. Can be a string name
        (e.g., 'earth', 'mars') or a poliastro Body object. Default is Earth.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two numpy arrays:
        - r : numpy.ndarray
            Position vector [x, y, z] in kilometers.
        - v : numpy.ndarray
            Velocity vector [vx, vy, vz] in kilometers per second.

    Raises
    ------
    TypeError
        If any input parameter is not of the expected type (float or Quantity for
        orbital elements, str or Body for attractor).

    Examples
    --------
    >>> from astropy import units as u
    >>> r, v = keplerian_to_cartesian(
    ...     a=7000.0,
    ...     ecc=0.01,
    ...     inc=45.0,
    ...     raan=0.0,
    ...     argp=0.0,
    ...     nu=0.0,
    ...     attractor='earth'
    ... )
    >>> print(r)  # Position vector in km
    >>> print(v)  # Velocity vector in km/s

    >>> r, v = keplerian_to_cartesian(
    ...     a=7000 * u.km,
    ...     ecc=0.01 * u.one,
    ...     inc=45 * u.deg,
    ...     raan=0 * u.deg,
    ...     argp=0 * u.deg,
    ...     nu=0 * u.deg,
    ...     attractor=Earth
    ... )
    """
    if not isinstance(attractor, (str, Body)):
        raise TypeError(
            f"Expected type of attractor is either str or poliastro.bodies.Body. Got {type(attractor)}."
        )
    if not isinstance(a, (float, Quantity)):
        raise TypeError(
            f"Expected type of a is either float or astropy.units.Quantity. Got {type(a)}."
        )
    if not isinstance(ecc, (float, Quantity)):
        raise TypeError(
            f"Expected type of ecc is either float or astropy.units.Quantity. Got {type(ecc)}."
        )
    if not isinstance(inc, (float, Quantity)):
        raise TypeError(
            f"Expected type of inc is either float or astropy.units.Quantity. Got {type(inc)}."
        )
    if not isinstance(raan, (float, Quantity)):
        raise TypeError(
            f"Expected type of raan is either float or astropy.units.Quantity. Got {type(raan)}."
        )
    if not isinstance(argp, (float, Quantity)):
        raise TypeError(
            f"Expected type of argp is either float or astropy.units.Quantity. Got {type(argp)}."
        )
    if not isinstance(nu, (float, Quantity)):
        raise TypeError(
            f"Expected type of nu is either float or astropy.units.Quantity. Got {type(nu)}."
        )

    if isinstance(attractor, str):
        attractor = body_from_str(attractor.lower())

    a = non_quantity_to_Quantity(a, u.km)
    ecc = non_quantity_to_Quantity(ecc, u.one)
    inc = non_quantity_to_Quantity(inc, u.deg)
    raan = non_quantity_to_Quantity(raan, u.deg)
    argp = non_quantity_to_Quantity(argp, u.deg)
    nu = non_quantity_to_Quantity(nu, u.deg)

    orbit = Orbit.from_classical(attractor, a, ecc, inc, raan, argp, nu)

    return orbit.r.value, orbit.v.value


def cartesian_to_keplerian(
    r_vec: list | np.ndarray | Quantity,
    v_vec: list | np.ndarray | Quantity,
    attractor: str | Body = Earth,
):
    """
    Convert Cartesian position and velocity vectors to Keplerian orbital elements.

    This function converts three-dimensional Cartesian coordinates (position and
    velocity vectors) to classical orbital elements (Keplerian elements) relative
    to a specified central body (attractor).

    Parameters
    ----------
    r_vec : list, numpy.ndarray, or astropy.units.Quantity
        Position vector [x, y, z]. If list or ndarray, assumed to be in kilometers.
        Must be a 3-element vector.
    v_vec : list, numpy.ndarray, or astropy.units.Quantity
        Velocity vector [vx, vy, vz]. If list or ndarray, assumed to be in
        kilometers per second. Must be a 3-element vector.
    attractor : str or poliastro.bodies.Body, optional
        The central body around which the orbit is defined. Can be a string name
        (e.g., 'earth', 'mars') or a poliastro Body object. Default is Earth.

    Returns
    -------
    tuple
        A tuple containing six orbital elements (all as floats):
        - a : float
            Semi-major axis in kilometers.
        - ecc : float
            Eccentricity (dimensionless).
        - inc : float
            Inclination in degrees.
        - raan : float
            Right ascension of the ascending node (RAAN) in degrees.
        - argp : float
            Argument of periapsis in degrees.
        - nu : float
            True anomaly in degrees.

    Raises
    ------
    TypeError
        If `r_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `v_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `attractor` is not a string or poliastro.bodies.Body object.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.array([7000.0, 0.0, 0.0])
    >>> v = np.array([0.0, 7.5, 0.0])
    >>> a, ecc, inc, raan, argp, nu = cartesian_to_keplerian(r, v, attractor='earth')
    >>> print(f"Semi-major axis: {a} km")
    >>> print(f"Eccentricity: {ecc}")

    >>> from astropy import units as u
    >>> r = [7000.0, 0.0, 0.0] * u.km
    >>> v = [0.0, 7.5, 0.0] * u.km / u.s
    >>> elements = cartesian_to_keplerian(r, v, attractor=Earth)
    """
    if not isinstance(r_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of r_vec is either list or ndarray or Quantity. Got {type(r_vec)}"
        )
    if not isinstance(v_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of v_vec is either list or ndarray or Quantity. Got {type(v_vec)}"
        )
    if not isinstance(attractor, (str, Body)):
        raise TypeError(
            f"Expected type of attractor is either str or poliastro.bodies.Body. Got {type(attractor)}."
        )

    attractor = body_from_str(attractor)
    r_vec = non_quantity_to_Quantity(r_vec, u.km)
    v_vec = non_quantity_to_Quantity(v_vec, u.km / u.s)

    orbit = Orbit.from_vectors(attractor, r_vec, v_vec)

    return (
        orbit.a.value,
        orbit.ecc.value,
        orbit.inc.value,
        orbit.raan.value,
        orbit.argp.value,
        orbit.nu.value,
    )


def hohmann_transfer(
    r_1_vec: list | np.ndarray | Quantity,
    v_1_vec: list | np.ndarray | Quantity,
    r_2: int | float | Quantity,
    attractor: str | Body = Earth,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute delta-v requirements for a Hohmann transfer between two circular orbits.

    This function calculates the two delta-v maneuvers (Δv₁ and Δv₂) required to
    perform a Hohmann transfer from an initial circular orbit to a final circular
    orbit at a different altitude. The Hohmann transfer is an efficient two-impulse
    maneuver that uses an elliptical transfer orbit.

    Parameters
    ----------
    r_1_vec : list, numpy.ndarray, or astropy.units.Quantity
        Position vector [x, y, z] of the satellite in the initial orbit. If list
        or ndarray, assumed to be in kilometers. Must be a 3-element vector.
    v_1_vec : list, numpy.ndarray, or astropy.units.Quantity
        Velocity vector [vx, vy, vz] of the satellite in the initial orbit. If
        list or ndarray, assumed to be in kilometers per second. Must be a 3-element vector.
    r_2 : int, float, or astropy.units.Quantity
        Orbital radius of the final circular orbit. If int or float, assumed to
        be in kilometers.
    attractor : str or poliastro.bodies.Body, optional
        The central body around which the orbits are defined. Can be a string name
        (e.g., 'earth', 'mars') or a poliastro Body object. Default is Earth.

    Returns
    -------
    tuple of ndarrays
        A tuple containing three ndarray objects:
        - dv1 : numpy.ndarray
            First delta-v maneuver (Δv₁) at the initial orbit to enter the transfer orbit.
        - dv2 : numpy.ndarray
            Second delta-v maneuver (Δv₂) at the final orbit to circularize.
        - total_dv : numpy.ndarray
            Total delta-v required for the complete transfer (Δv₁ + Δv₂).

    Raises
    ------
    TypeError
        If `r_1_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `v_1_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `r_2` is not an int, float, or astropy.units.Quantity.
    TypeError
        If `attractor` is not a string or poliastro.bodies.Body object.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> # Initial orbit position and velocity (LEO at 400 km altitude)
    >>> r1 = np.array([6778.0, 0.0, 0.0])  # km
    >>> v1 = np.array([0.0, 7.67, 0.0])  # km/s
    >>> # Final orbit radius (GEO altitude)
    >>> r2 = 42164.0  # km
    >>> dv1, dv2, total = hohmann_transfer(r1, v1, r2, attractor='earth')
    >>> print(f"First burn: {dv1}")
    >>> print(f"Second burn: {dv2}")
    >>> print(f"Total delta-v: {total}")

    >>> # Using Quantity objects
    >>> r1 = [6778.0, 0.0, 0.0] * u.km
    >>> v1 = [0.0, 7.67, 0.0] * u.km / u.s
    >>> r2 = 42164.0 * u.km
    >>> dv1, dv2, total = hohmann_transfer(r1, v1, r2, attractor=Earth)
    """
    if not isinstance(r_1_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of r_1_vec is either list or np.ndarray or astropy.units.Quantity. Got {type(r_1_vec)}."
        )
    if not isinstance(v_1_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of v_1_vec is either list or np.ndarray or astropy.units.Quantity. Got {type(v_1_vec)}."
        )
    if not isinstance(r_2, (int, float, Quantity)):
        raise TypeError(
            f"Expected type of r_2 is either int or float or astropy.units.Quantity. Got {type(r_2)}."
        )
    if not isinstance(attractor, (str, Body)):
        raise TypeError(
            f"Expected type of attractor is either str or poliastro.bodies.Body. Got {type(attractor)}."
        )

    attractor = body_from_str(attractor)
    r_1_vec = non_quantity_to_Quantity(r_1_vec, u.km)
    v_1_vec = non_quantity_to_Quantity(v_1_vec, u.km / u.s)
    r_2 = non_quantity_to_Quantity(r_2, u.km)

    orbit_1 = Orbit.from_vectors(attractor, r_1_vec, v_1_vec)
    man = Maneuver.hohmann(orbit_1, r_2)

    return (man[0][1].value, man[1][1].value, man[0][1].value + man[1][1].value)


def hohmann_time_of_flight(
    r_1_vec: list | np.ndarray | Quantity,
    v_1_vec: list | np.ndarray | Quantity,
    r_2: int | float | Quantity,
    attractor: str | Body = Earth,
) -> float:
    """
    Calculate the time of flight for a Hohmann transfer between two circular orbits.

    This function computes the total time required to complete a Hohmann transfer
    maneuver from an initial circular orbit to a final circular orbit. The time of
    flight corresponds to half the period of the elliptical transfer orbit.

    Parameters
    ----------
    r_1_vec : list, numpy.ndarray, or astropy.units.Quantity
        Position vector [x, y, z] of the satellite in the initial orbit. If list
        or ndarray, assumed to be in kilometers. Must be a 3-element vector.
    v_1_vec : list, numpy.ndarray, or astropy.units.Quantity
        Velocity vector [vx, vy, vz] of the satellite in the initial orbit. If
        list or ndarray, assumed to be in kilometers per second. Must be a 3-element vector.
    r_2 : int, float, or astropy.units.Quantity
        Orbital radius of the final circular orbit. If int or float, assumed to
        be in kilometers.
    attractor : str or poliastro.bodies.Body, optional
        The central body around which the orbits are defined. Can be a string name
        (e.g., 'earth', 'mars') or a poliastro Body object. Default is Earth.

    Returns
    -------
    float
        The time of flight for the Hohmann transfer in seconds.

    Raises
    ------
    TypeError
        If `r_1_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `v_1_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `r_2` is not an int, float, or astropy.units.Quantity.
    TypeError
        If `attractor` is not a string or poliastro.bodies.Body object.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> # Initial orbit position and velocity (LEO at 400 km altitude)
    >>> r1 = np.array([6778.0, 0.0, 0.0])  # km
    >>> v1 = np.array([0.0, 7.67, 0.0])  # km/s
    >>> # Final orbit radius (GEO altitude)
    >>> r2 = 42164.0  # km
    >>> tof = hohmann_time_of_flight(r1, v1, r2, attractor='earth')
    >>> print(f"Time of flight: {tof} seconds")
    >>> print(f"Time of flight: {tof/3600:.2f} hours")

    >>> # Using Quantity objects
    >>> r1 = [6778.0, 0.0, 0.0] * u.km
    >>> v1 = [0.0, 7.67, 0.0] * u.km / u.s
    >>> r2 = 42164.0 * u.km
    >>> tof = hohmann_time_of_flight(r1, v1, r2, attractor=Earth)
    """
    if not isinstance(r_1_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of r_1_vec is either list or np.ndarray or astropy.units.Quantity. Got {type(r_1_vec)}."
        )
    if not isinstance(v_1_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of v_1_vec is either list or np.ndarray or astropy.units.Quantity. Got {type(v_1_vec)}."
        )
    if not isinstance(r_2, (int, float, Quantity)):
        raise TypeError(
            f"Expected type of r_2 is either int or float or astropy.units.Quantity. Got {type(r_2)}."
        )
    if not isinstance(attractor, (str, Body)):
        raise TypeError(
            f"Expected type of attractor is either str or poliastro.bodies.Body. Got {type(attractor)}."
        )

    attractor = body_from_str(attractor)
    r_1_vec = non_quantity_to_Quantity(r_1_vec, u.km)
    v_1_vec = non_quantity_to_Quantity(v_1_vec, u.km / u.s)
    r_2 = non_quantity_to_Quantity(r_2, u.km)

    orbit_1 = Orbit.from_vectors(attractor, r_1_vec, v_1_vec)
    man = Maneuver.hohmann(orbit_1, r_2)

    return man.get_total_time().value


def bielliptic_transfer(
    r_1_vec: list | np.ndarray | Quantity,
    v_1_vec: list | np.ndarray | Quantity,
    r_b: int | float | Quantity,
    r_2: int | float | Quantity,
    attractor: str | Body = Earth,
):
    """
    Compute delta-v requirements for a bi-elliptic transfer between two circular orbits.

    This function calculates the three delta-v maneuvers (Δv₁, Δv₂, and Δv₃) required
    to perform a bi-elliptic transfer from an initial circular orbit to a final circular
    orbit via an intermediate apoapsis. The bi-elliptic transfer uses two elliptical
    transfer orbits and can be more efficient than a Hohmann transfer for large orbital
    radius ratios (typically when r₂/r₁ > 11.94).

    Parameters
    ----------
    r_1_vec : list, numpy.ndarray, or astropy.units.Quantity
        Position vector [x, y, z] of the satellite in the initial orbit. If list
        or ndarray, assumed to be in kilometers. Must be a 3-element vector.
    v_1_vec : list, numpy.ndarray, or astropy.units.Quantity
        Velocity vector [vx, vy, vz] of the satellite in the initial orbit. If
        list or ndarray, assumed to be in kilometers per second. Must be a 3-element vector.
    r_b : int, float, or astropy.units.Quantity
        Orbital radius of the intermediate apoapsis (the highest point in the transfer).
        If int or float, assumed to be in kilometers. Must be greater than both r₁ and r₂.
    r_2 : int, float, or astropy.units.Quantity
        Orbital radius of the final circular orbit. If int or float, assumed to
        be in kilometers.
    attractor : str or poliastro.bodies.Body, optional
        The central body around which the orbits are defined. Can be a string name
        (e.g., 'earth', 'mars') or a poliastro Body object. Default is Earth.

    Returns
    -------
    tuple
        A tuple containing five elements:
        - dv1 : numpy.ndarray
            First delta-v maneuver (Δv₁) at the initial orbit to enter the first transfer orbit.
        - dv2 : numpy.ndarray
            Second delta-v maneuver (Δv₂) at the intermediate apoapsis.
        - dv3 : numpy.ndarray
            Third delta-v maneuver (Δv₃) at the final orbit to circularize.
        - total_dv : numpy.ndarray
            Total delta-v required for the complete transfer (Δv₁ + Δv₂ + Δv₃).
        - tof : float
            Total time of flight for the bi-elliptic transfer in seconds.

    Raises
    ------
    TypeError
        If `r_1_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `v_1_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `r_b` is not an int, float, or astropy.units.Quantity.
    TypeError
        If `r_2` is not an int, float, or astropy.units.Quantity.
    TypeError
        If `attractor` is not a string or poliastro.bodies.Body object.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> # Initial orbit position and velocity (LEO at 400 km altitude)
    >>> r1 = np.array([6778.0, 0.0, 0.0])  # km
    >>> v1 = np.array([0.0, 7.67, 0.0])  # km/s
    >>> # Intermediate apoapsis radius
    >>> rb = 100000.0  # km
    >>> # Final orbit radius (GEO altitude)
    >>> r2 = 42164.0  # km
    >>> dv1, dv2, dv3, total, tof = bielliptic_transfer(r1, v1, rb, r2, attractor='earth')
    >>> print(f"First burn: {dv1}")
    >>> print(f"Second burn: {dv2}")
    >>> print(f"Third burn: {dv3}")
    >>> print(f"Total delta-v: {total}")
    >>> print(f"Time of flight: {tof/3600:.2f} hours")

    >>> # Using Quantity objects
    >>> r1 = [6778.0, 0.0, 0.0] * u.km
    >>> v1 = [0.0, 7.67, 0.0] * u.km / u.s
    >>> rb = 100000.0 * u.km
    >>> r2 = 42164.0 * u.km
    >>> dv1, dv2, dv3, total, tof = bielliptic_transfer(r1, v1, rb, r2, attractor=Earth)
    """
    if not isinstance(r_1_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of r_1_vec is either list or np.ndarray or astropy.units.Quantity. Got {type(r_1_vec)}."
        )
    if not isinstance(v_1_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of v_1_vec is either list or np.ndarray or astropy.units.Quantity. Got {type(v_1_vec)}."
        )
    if not isinstance(r_b, (int, float, Quantity)):
        raise TypeError(
            f"Expected type of r_b is either int or float or astropy.units.Quantity. Got {type(r_b)}."
        )
    if not isinstance(r_2, (int, float, Quantity)):
        raise TypeError(
            f"Expected type of r_2 is either int or float or astropy.units.Quantity. Got {type(r_2)}."
        )
    if not isinstance(attractor, (str, Body)):
        raise TypeError(
            f"Expected type of attractor is either str or poliastro.bodies.Body. Got {type(attractor)}."
        )

    attractor = body_from_str(attractor)
    r_1_vec = non_quantity_to_Quantity(r_1_vec, u.km)
    v_1_vec = non_quantity_to_Quantity(v_1_vec, u.km / u.s)
    r_b = non_quantity_to_Quantity(r_b, u.km)
    r_2 = non_quantity_to_Quantity(r_2, u.km)

    orbit_1 = Orbit.from_vectors(attractor, r_1_vec, v_1_vec)
    man = Maneuver.bielliptic(orbit_1, r_b, r_2)

    return (
        man[0][1].value,
        man[1][1].value,
        man[2][1].value,
        man[0][1].value + man[1][1].value + man[2][1].value,
        man.get_total_time().value,
    )


def plane_change(v: list | np.ndarray | Quantity, delta_i: int | float | Quantity):
    """
    Calculate the delta-v required for a simple plane change maneuver.

    This function computes the delta-v necessary to change the orbital plane by a
    specified angle. A plane change is performed by firing the spacecraft's engines
    perpendicular to the original velocity vector. This is one of the most fuel-intensive
    orbital maneuvers.

    Parameters
    ----------
    v : list, numpy.ndarray, or astropy.units.Quantity
        Velocity vector [vx, vy, vz] or velocity magnitude at which the plane change
        is performed. If list or ndarray, assumed to be in kilometers per second.
    delta_i : int, float, or astropy.units.Quantity
        The change in inclination angle. If int or float, assumed to be in degrees.

    Returns
    -------
    astropy.units.Quantity
        The magnitude of delta-v required for the plane change maneuver, with units
        of velocity (km/s).

    Raises
    ------
    TypeError
        If `v` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `delta_i` is not an int, float, or astropy.units.Quantity.

    Notes
    -----
    The delta-v for a plane change is calculated using the formula:

    .. math::
        \\Delta v = 2v \\sin(\\Delta i / 2)

    where v is the orbital velocity and Δi is the change in inclination.

    Plane changes are most efficient when performed at the ascending or descending
    node of the orbit, and are more fuel-efficient at higher altitudes where orbital
    velocity is lower.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> # Calculate delta-v for a 30-degree plane change at 7.5 km/s
    >>> v = 7.5  # km/s
    >>> delta_i = 30.0  # degrees
    >>> dv = plane_change(v, delta_i)
    >>> print(f"Delta-v required: {dv}")

    >>> # Using Quantity objects
    >>> v = 7.5 * u.km / u.s
    >>> delta_i = 30.0 * u.deg
    >>> dv = plane_change(v, delta_i)

    >>> # Using velocity vector
    >>> v_vec = np.array([0.0, 7.5, 0.0])
    >>> delta_i = 45.0
    >>> dv = plane_change(v_vec, delta_i)
    """
    if not isinstance(v, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of v is either list or np.ndarray or astropy.units.Quantity. Got {type(v)}."
        )
    if not isinstance(delta_i, (int, float, Quantity)):
        raise TypeError(
            f"Expected type of delta_i is either int or float or astropy.units.Quantity. Got {type(delta_i)}."
        )

    v = non_quantity_to_Quantity(v, u.km / u.s)
    delta_i = non_quantity_to_Quantity(delta_i, u.deg)

    return 2 * v * np.sin(delta_i / 2)


def lambert_solver(
    r_1_vec: list | np.ndarray | Quantity,
    r_2_vec: list | np.ndarray | Quantity,
    tof: int | float | Quantity,
    prograde: bool = True,
    attractor: str | Body = Earth,
):
    """
    Solve Lambert's problem to find velocity vectors for orbital transfer.

    Lambert's problem involves determining the orbit that connects two position vectors
    in a specified time of flight. This function solves for the initial and final
    velocity vectors required for the transfer trajectory. This is fundamental for
    interplanetary mission design and rendezvous maneuvers.

    Parameters
    ----------
    r_1_vec : list, numpy.ndarray, or astropy.units.Quantity
        Initial position vector [x, y, z]. If list or ndarray, assumed to be in
        kilometers. Must be a 3-element vector.
    r_2_vec : list, numpy.ndarray, or astropy.units.Quantity
        Final position vector [x, y, z]. If list or ndarray, assumed to be in
        kilometers. Must be a 3-element vector.
    tof : int, float, or astropy.units.Quantity
        Time of flight for the transfer. If int or float, assumed to be in seconds.
    prograde : bool, optional
        If True, solve for prograde (short way) trajectory. If False, solve for
        retrograde (long way) trajectory. Default is True.
    attractor : str or poliastro.bodies.Body, optional
        The central body around which the transfer occurs. Can be a string name
        (e.g., 'earth', 'mars') or a poliastro Body object. Default is Earth.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two numpy arrays:
        - v1 : numpy.ndarray
            Initial velocity vector [vx, vy, vz] in kilometers per second required
            at the first position.
        - v2 : numpy.ndarray
            Final velocity vector [vx, vy, vz] in kilometers per second at the
            second position.

    Raises
    ------
    TypeError
        If `r_1_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `r_2_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `tof` is not an int, float, or astropy.units.Quantity.
    TypeError
        If `prograde` is not a boolean.
    TypeError
        If `attractor` is not a string or poliastro.bodies.Body object.

    Notes
    -----
    Lambert's problem is a classical problem in orbital mechanics. Given two position
    vectors and the time of flight between them, there are generally two possible
    solutions: a prograde (shorter) trajectory and a retrograde (longer) trajectory.
    The choice between them depends on mission requirements.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> # Define initial and final positions for a transfer
    >>> r1 = np.array([7000.0, 0.0, 0.0])  # km
    >>> r2 = np.array([0.0, 10000.0, 0.0])  # km
    >>> tof = 3600.0  # seconds (1 hour)
    >>> v1, v2 = lambert_solver(r1, r2, tof, prograde=True, attractor='earth')
    >>> print(f"Initial velocity: {v1} km/s")
    >>> print(f"Final velocity: {v2} km/s")

    >>> # Using Quantity objects
    >>> r1 = [7000.0, 0.0, 0.0] * u.km
    >>> r2 = [0.0, 10000.0, 0.0] * u.km
    >>> tof = 3600.0 * u.s
    >>> v1, v2 = lambert_solver(r1, r2, tof, prograde=True, attractor=Earth)

    >>> # Retrograde trajectory
    >>> v1, v2 = lambert_solver(r1, r2, tof, prograde=False, attractor='earth')
    """
    if not isinstance(r_1_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of r_1_vec is either list or np.ndarray or astropy.units.Quantity. Got {type(r_1_vec)}."
        )
    if not isinstance(r_2_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of r_2_vec is either list or np.ndarray or astropy.units.Quantity. Got {type(r_2_vec)}."
        )
    if not isinstance(tof, (int, float, Quantity)):
        raise TypeError(
            f"Expected type of tof is either int or float or astropy.units.Quantity. Got {type(tof)}."
        )
    if not isinstance(prograde, bool):
        raise TypeError(f"Expected type of prograde is bool. Got {type(prograde)}.")
    if not isinstance(attractor, (str, Body)):
        raise TypeError(
            f"Expected type of attractor is either str or poliastro.bodies.Body. Got {type(attractor)}."
        )

    attractor = body_from_str(attractor)
    k = attractor.k
    r_1_vec = non_quantity_to_Quantity(r_1_vec, u.km)
    r_2_vec = non_quantity_to_Quantity(r_2_vec, u.km)
    tof = non_quantity_to_Quantity(tof, u.s)

    v_1, v_2 = lambert(k, r_1_vec, r_2_vec, tof, prograde=prograde)

    return v_1.value, v_2.value


def universal_kepler(
    r_vec: list | np.ndarray | Quantity,
    v_vec: list | np.ndarray | Quantity,
    dt: int | float | timedelta | TimeDelta,
    attractor: str | Body = Earth,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate an orbit using the universal Kepler propagator.

    This function propagates an orbit forward or backward in time using the universal
    variable formulation of Kepler's equations. It solves the two-body problem to
    determine the future (or past) position and velocity vectors given initial conditions
    and a time increment. The Vallado propagator is used with high precision (500 iterations).

    Parameters
    ----------
    r_vec : list, numpy.ndarray, or astropy.units.Quantity
        Initial position vector [x, y, z]. If list or ndarray, assumed to be in
        kilometers. Must be a 3-element vector.
    v_vec : list, numpy.ndarray, or astropy.units.Quantity
        Initial velocity vector [vx, vy, vz]. If list or ndarray, assumed to be in
        kilometers per second. Must be a 3-element vector.
    dt : int, float, datetime.timedelta, or astropy.time.TimeDelta
        Time increment for propagation. If int or float, assumed to be in seconds.
        Positive values propagate forward in time, negative values propagate backward.
    attractor : str or poliastro.bodies.Body, optional
        The central body around which the orbit is propagated. Can be a string name
        (e.g., 'earth', 'mars') or a poliastro Body object. Default is Earth.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two numpy arrays:
        - r : numpy.ndarray
            Position vector [x, y, z] in kilometers at time t + dt.
        - v : numpy.ndarray
            Velocity vector [vx, vy, vz] in kilometers per second at time t + dt.

    Raises
    ------
    TypeError
        If `r_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `v_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `dt` is not an int, float, datetime.timedelta, or astropy.time.TimeDelta.
    TypeError
        If `attractor` is not a string or poliastro.bodies.Body object.

    Notes
    -----
    The universal Kepler propagator solves the two-body problem assuming only
    gravitational forces from the central body. It does not account for perturbations
    such as atmospheric drag, solar radiation pressure, or gravitational effects from
    other bodies. The Vallado propagator uses an iterative method with 500 iterations
    for high precision.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> from datetime import timedelta
    >>> # Initial position and velocity (LEO)
    >>> r0 = np.array([7000.0, 0.0, 0.0])  # km
    >>> v0 = np.array([0.0, 7.5, 0.0])  # km/s
    >>> # Propagate forward by 1 hour
    >>> dt = 3600.0  # seconds
    >>> r_new, v_new = universal_kepler(r0, v0, dt, attractor='earth')
    >>> print(f"Position after 1 hour: {r_new} km")
    >>> print(f"Velocity after 1 hour: {v_new} km/s")

    >>> # Using Quantity objects
    >>> r0 = [7000.0, 0.0, 0.0] * u.km
    >>> v0 = [0.0, 7.5, 0.0] * u.km / u.s
    >>> dt = 3600.0  # seconds
    >>> r_new, v_new = universal_kepler(r0, v0, dt, attractor=Earth)

    >>> # Using timedelta object
    >>> dt = timedelta(hours=1)
    >>> r_new, v_new = universal_kepler(r0, v0, dt, attractor='earth')

    >>> # Propagate backward in time
    >>> dt = -3600.0  # -1 hour
    >>> r_past, v_past = universal_kepler(r0, v0, dt, attractor='earth')
    """
    if not isinstance(r_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of r_vec is either list or np.ndarray or astropy.units.Quantity. Got {type(r_vec)}."
        )
    if not isinstance(v_vec, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of v_vec is either list or np.ndarray or astropy.units.Quantity. Got {type(v_vec)}."
        )
    if not isinstance(dt, (int, float, timedelta, TimeDelta)):
        raise TypeError(
            f"Expected type of dt is either int or float or timedelta or astropy.time.TimeDelta. Got {type(dt)}."
        )
    if not isinstance(attractor, (str, Body)):
        raise TypeError(
            f"Expected type of attractor is either str or poliastro.bodies.Body. Got {type(attractor)}."
        )

    attractor = body_from_str(attractor)
    r_vec = non_quantity_to_Quantity(r_vec, u.km)
    v_vec = non_quantity_to_Quantity(v_vec, u.km / u.s)

    if isinstance(dt, (int, float)):
        dt = TimeDelta(timedelta(seconds=dt))
    elif isinstance(dt, timedelta):
        dt = TimeDelta(dt)

    orbit = Orbit.from_vectors(attractor, r_vec, v_vec)
    new_orbit = orbit.propagate(dt, ValladoPropagator(numiter=500))

    return new_orbit.r.value, new_orbit.v.value


def sgp4_propagate(line_1: str, line_2: str, at_datetime: str | datetime):
    """
    Propagate a satellite orbit using the SGP4 model from TLE data.

    This function uses the Simplified General Perturbations 4 (SGP4) model to propagate
    a satellite's orbit from Two-Line Element (TLE) set data to a specific date and time.
    SGP4 is a widely-used analytical propagation model that accounts for perturbations
    such as Earth's oblateness, atmospheric drag, and solar/lunar gravitational effects.

    Parameters
    ----------
    line_1 : str
        First line of the Two-Line Element (TLE) set. Must be a valid TLE format
        string containing orbital elements and satellite information.
    line_2 : str
        Second line of the Two-Line Element (TLE) set. Must be a valid TLE format
        string containing additional orbital parameters.
    at_datetime : str or datetime.datetime
        The date and time at which to compute the satellite's position and velocity.
        If string, must be in the format 'YYYY-MM-DD HH:MM:SS'.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two numpy arrays:
        - r : numpy.ndarray
            Position vector [x, y, z] in kilometers in the TEME (True Equator Mean
            Equinox) reference frame.
        - v : numpy.ndarray
            Velocity vector [vx, vy, vz] in kilometers per second in the TEME
            reference frame.

    Raises
    ------
    TypeError
        If `line_1` is not a string.
    TypeError
        If `line_2` is not a string.
    TypeError
        If `at_datetime` is not a string or datetime.datetime object.
    RuntimeError
        If SGP4 propagation encounters an error (e.g., decayed satellite, invalid
        TLE data, or numerical issues).

    Notes
    -----
    The SGP4 model is designed for near-Earth satellites and uses mean orbital elements.
    The output is in the TEME coordinate system, which may need to be converted to
    other reference frames (e.g., GCRS, ITRS) for some applications.

    TLE data can be obtained from sources such as Space-Track.org, CelesTrak, or
    N2YO. TLE elements have limited validity periods and should be updated regularly
    for accurate predictions.

    Examples
    --------
    >>> from datetime import datetime
    >>> # ISS TLE example (fictional values for demonstration)
    >>> line1 = "1 25544U 98067A   21275.51261574  .00002182  00000-0  41420-4 0  9990"
    >>> line2 = "2 25544  51.6461 339.8014 0003045  24.8134  62.5806 15.48919393304228"
    >>> target_time = "2021-10-02 12:00:00"
    >>> r, v = sgp4_propagate(line1, line2, target_time)
    >>> print(f"Position: {r} km")
    >>> print(f"Velocity: {v} km/s")

    >>> # Using datetime object
    >>> target_time = datetime(2021, 10, 2, 12, 0, 0)
    >>> r, v = sgp4_propagate(line1, line2, target_time)

    >>> # Get current position of a satellite
    >>> from datetime import datetime
    >>> now = datetime.utcnow()
    >>> r, v = sgp4_propagate(line1, line2, now)
    """
    if not isinstance(line_1, str):
        raise TypeError(f"Expected type of line_1 is str. Got {type(line_1)}.")
    if not isinstance(line_2, str):
        raise TypeError(f"Expected type of line_2 is str. Got {type(line_2)}.")
    if not isinstance(at_datetime, (str, datetime)):
        raise TypeError(
            f"Expected type of at_datetime is str or datetime. Got {type(at_datetime)}."
        )

    if isinstance(at_datetime, str):
        at_datetime = datetime.strptime(at_datetime, format="%Y-%m-%d %H:%M:%S")

    _, whole_part, frac_part = datetime_to_jd(at_datetime)

    satellite = Satrec.twoline2rv(line_1, line_2)
    e, r, v = satellite.sgp4(whole_part, frac_part)

    if e != 0:
        raise RuntimeError(f"Error in SGP4 propagation. {SGP4_ERRORS[f'{e}']}")
    else:
        return np.array(r), np.array(v)


def orbit_period(a: int | float, mu: int | float) -> float:
    """
    Calculate the orbital period of a two-body orbit.

    This function computes the orbital period using Kepler's Third Law, which relates
    the period of an orbit to its semi-major axis and the gravitational parameter of
    the central body. This applies to elliptical, circular, and parabolic orbits.

    Parameters
    ----------
    a : int or float
        Semi-major axis of the orbit in kilometers.
    mu : int or float
        Standard gravitational parameter (GM) of the central body in km³/s².
        For Earth, μ ≈ 398600.4418 km³/s².

    Returns
    -------
    float
        The orbital period in seconds.

    Raises
    ------
    TypeError
        If `a` is not an int or float.
    TypeError
        If `mu` is not an int or float.

    Notes
    -----
    The orbital period is calculated using Kepler's Third Law:

    .. math::
        T = 2\\pi \\sqrt{\\frac{a^3}{\\mu}}

    where T is the period, a is the semi-major axis, and μ is the gravitational
    parameter of the central body.

    This formula is valid for all conic section orbits (elliptical and circular).
    For parabolic and hyperbolic orbits (e ≥ 1), the concept of period is not
    physically meaningful.

    Examples
    --------
    >>> # Calculate period of a circular orbit at 400 km altitude above Earth
    >>> a = 6378 + 400  # Earth radius + altitude in km
    >>> mu_earth = 398600.4418  # km³/s²
    >>> T = orbit_period(a, mu_earth)
    >>> print(f"Orbital period: {T:.2f} seconds")
    >>> print(f"Orbital period: {T/60:.2f} minutes")

    >>> # ISS orbit (approximately)
    >>> a_iss = 6778.0  # km
    >>> T_iss = orbit_period(a_iss, 398600.4418)
    >>> print(f"ISS period: {T_iss/60:.2f} minutes")

    >>> # Geostationary orbit
    >>> a_geo = 42164.0  # km
    >>> T_geo = orbit_period(a_geo, 398600.4418)
    >>> print(f"GEO period: {T_geo/3600:.2f} hours")
    """
    if not isinstance(a, (int, float)):
        raise TypeError(f"Expected type of a is int or float. Got {type(a)}.")
    if not isinstance(mu, (int, float)):
        raise TypeError(f"Expected type of mu is int or float. Got {type(mu)}.")

    return 2 * pi * sqrt(a**3 / mu)
