from datetime import datetime
from typing import TypedDict

import numpy as np
from astropy.coordinates import EarthLocation

from app.core.mission.utils.propagator import PropagationResults
from app.core.utils import rad2deg


class GroundStationConfigVisibility(TypedDict):
    """Configuration for a ground station location.

    Attributes
    ----------
    gs_lat : float
        Ground station latitude in degrees.
    gs_lon : float
        Ground station longitude in degrees.
    gs_alt : float
        Ground station altitude in meters.
    """

    gs_lat: float
    gs_lon: float
    gs_alt: float


class VisibilityResults(TypedDict):
    """Results from visibility computation.

    Attributes
    ----------
    elevation_deg : np.ndarray
        Elevation angles in degrees for each time step.
    visible : np.ndarray
        Boolean mask indicating visibility at each time step.
    windows : List[Dict]
        List of visibility windows with start, end, and maximum elevation.
    """

    elevation_deg: list
    visible: list
    windows: list[dict]


def compute_elevation_angle(sat_ecef: list | np.ndarray, gs_ecef: list | np.ndarray) -> float:
    """Compute elevation angle of satellite from ground station.

    Parameters
    ----------
    sat_ecef : list or np.ndarray
        Satellite position in ECEF coordinates [x, y, z] in kilometers.
    gs_ecef : list or np.ndarray
        Ground station position in ECEF coordinates [x, y, z] in kilometers.

    Returns
    -------
    float
        Elevation angle in degrees.

    Raises
    ------
    TypeError
        If sat_ecef or gs_ecef are not list or np.ndarray.
    """
    if not isinstance(sat_ecef, (list, np.ndarray)):
        raise TypeError(f"Expected type of sat_ecef is list or np.ndarray. Got {type(sat_ecef)}")
    if not isinstance(gs_ecef, (list, np.ndarray)):
        raise TypeError(f"Expected type of gs_ecef is list or np.ndarray. Got {type(gs_ecef)}")
    sat_ecef = np.array(sat_ecef)
    gs_ecef = np.array(gs_ecef)

    line_of_sight_vector = sat_ecef - gs_ecef
    gs_norm = np.linalg.norm(gs_ecef)
    los_norm = np.linalg.norm(line_of_sight_vector)
    gs_unit_vector = gs_ecef / gs_norm

    return rad2deg(np.arcsin((line_of_sight_vector @ gs_unit_vector) / los_norm))


def ecef_from_lat_lon_alt(lat: float, lon: float, alt: float) -> np.ndarray[float]:
    """Convert geodetic coordinates to ECEF coordinates.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    alt : float
        Altitude in meters.

    Returns
    -------
    np.ndarray[float]
        ECEF coordinates [x, y, z] in meters.

    Raises
    ------
    TypeError
        If lat, lon, or alt are not float.
    """
    if not isinstance(lat, float):
        raise TypeError(f"Expected type of lat is float. Got {type(lat)}")
    if not isinstance(lon, float):
        raise TypeError(f"Expected type of lon is float. Got {type(lon)}")
    if not isinstance(alt, float):
        raise TypeError(f"Expected type of alt is float. Got {type(alt)}")

    earthloc = EarthLocation.from_geodetic(lon, lat, alt)
    return np.array([earthloc.x.value, earthloc.y.value, earthloc.z.value])


def visibility_mask(
    sat_lat: list[float] | np.ndarray[float],
    sat_lon: list[float] | np.ndarray[float],
    sat_alt: list[float] | np.ndarray[float],
    gs_lat: float,
    gs_lon: float,
    gs_alt: float,
    min_elevation_deg: int | float = 10.0,
) -> tuple[np.ndarray[float], np.ndarray[bool]]:
    """Compute visibility mask for satellite passes over a ground station.

    Parameters
    ----------
    sat_lat : List[float] or np.ndarray[float]
        Satellite latitude values in degrees.
    sat_lon : List[float] or np.ndarray[float]
        Satellite longitude values in degrees.
    sat_alt : List[float] or np.ndarray[float]
        Satellite altitude values in meters.
    gs_lat : float
        Ground station latitude in degrees.
    gs_lon : float
        Ground station longitude in degrees.
    gs_alt : float
        Ground station altitude in meters.
    min_elevation_deg : int or float, optional
        Minimum elevation angle for visibility in degrees (default is 10.0).

    Returns
    -------
    tuple[np.ndarray[float], np.ndarray[bool]]
        Tuple containing elevation angles array and boolean visibility mask.

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If min_elevation_deg is not in the range [0, 90).
    """
    if not isinstance(sat_lat, (list, np.ndarray)):
        raise TypeError(f"Expected type of sat_lat is list or np.ndarray. Got {type(sat_lat)}")
    if not isinstance(sat_lon, (list, np.ndarray)):
        raise TypeError(f"Expected type of sat_lon is list or np.ndarray. Got {type(sat_lon)}")
    if not isinstance(sat_alt, (list, np.ndarray)):
        raise TypeError(f"Expected type of sat_alt is list or np.ndarray. Got {type(sat_alt)}")
    if not isinstance(gs_lat, float):
        raise TypeError(f"Expected type of gs_lat is float. Got {type(gs_lat)}")
    if not isinstance(gs_lon, float):
        raise TypeError(f"Expected type of gs_lon is float. Got {type(gs_lon)}")
    if not isinstance(gs_alt, float):
        raise TypeError(f"Expected type of gs_alt is float. Got {type(gs_alt)}")

    if not isinstance(min_elevation_deg, (int, float)):
        raise TypeError(
            f"Expected type of min_elevation_deg is int or float. Got {type(min_elevation_deg)}"
        )

    if not (min_elevation_deg >= 0 and min_elevation_deg < 90):
        raise ValueError(f"Expected min_elevation_angle in [0, 90). Got {min_elevation_deg}")

    sat_lat = np.array(sat_lat)
    sat_lon = np.array(sat_lon)
    sat_alt = np.array(sat_alt)

    gs_ecef = ecef_from_lat_lon_alt(gs_lat, gs_lon, gs_alt)

    elevation_angles = []

    for lat, lon, alt in zip(sat_lat, sat_lon, sat_alt):
        sat_ecef = ecef_from_lat_lon_alt(lat, lon, alt)
        elevation_angles.append(compute_elevation_angle(sat_ecef, gs_ecef))

    elevation_angles = np.array(elevation_angles)
    mask = elevation_angles >= min_elevation_deg

    return elevation_angles, mask


def extract_visibility_windows(
    times: list[datetime] | np.ndarray,
    elevation_angles: list[float] | np.ndarray,
    visible_mask: list[bool] | np.ndarray,
) -> list[dict]:
    """Extract continuous visibility windows from visibility mask.

    Parameters
    ----------
    times : List[datetime] or np.ndarray
        Array of datetime objects corresponding to each time step.
    elevation_angles : List[float] or np.ndarray
        Elevation angles in degrees for each time step.
    visible_mask : List[bool] or np.ndarray
        Boolean mask indicating visibility at each time step.

    Returns
    -------
    list[dict]
        List of dictionaries, each containing 'start', 'end', and 'max_ele' keys
        representing visibility window start time, end time, and maximum elevation angle.

    Raises
    ------
    TypeError
        If input types are incorrect.
    """
    if not isinstance(times, (list, np.ndarray)):
        raise TypeError(f"Expected type of times is list or np.ndarray. Got {type(times)}")
    if not isinstance(elevation_angles, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of elevation_angles is list or np.ndarray. Got {type(elevation_angles)}"
        )
    if not isinstance(visible_mask, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of visible_mask is list or np.ndarray. Got {type(visible_mask)}"
        )

    visibility_windows = []
    add_new = True

    for time, ele_angle, mask in zip(times, elevation_angles, visible_mask):
        if mask and add_new:
            visibility_windows.append({"start": time, "end": time, "max_ele": ele_angle})
            add_new = False
        elif mask and not add_new:
            visibility_windows[-1]["end"] = time
            visibility_windows[-1]["max_ele"] = (
                visibility_windows[-1]["max_ele"]
                if abs(visibility_windows[-1]["max_ele"]) > abs(ele_angle)
                else ele_angle
            )
        elif not mask:
            add_new = True

    return visibility_windows


def compute_visibility(
    propagation_results: PropagationResults,
    ground_station: GroundStationConfigVisibility,
    min_elevation_deg: int | float = 10.0,
) -> VisibilityResults:
    """Compute satellite visibility from a ground station.

    Parameters
    ----------
    propagation_results : PropagationResults
        Dictionary containing propagation results with keys: 'time', 'lat', 'lon', 'alt'.
    ground_station : GroundStationConfigVisibility
        Dictionary containing ground station configuration with keys: 'gs_lat', 'gs_lon', 'gs_alt'.
    min_elevation_deg : int or float, optional
        Minimum elevation angle for visibility in degrees (default is 10.0).

    Returns
    -------
    VisibilityResults
        Dictionary containing 'elevation_deg', 'visible', and 'windows' keys with
        elevation angles, visibility mask, and visibility windows respectively.
    """
    times = propagation_results["time"]
    sat_lat = propagation_results["lat"]
    sat_lon = propagation_results["lon"]
    sat_alt = propagation_results["alt"]

    gs_lat = ground_station["gs_lat"]
    gs_lon = ground_station["gs_lon"]
    gs_alt = ground_station["gs_alt"]

    ele_angles, mask = visibility_mask(
        sat_lat, sat_lon, sat_alt, gs_lat, gs_lon, gs_alt, min_elevation_deg
    )

    windows = extract_visibility_windows(times, ele_angles, mask)

    return {"elevation_deg": ele_angles, "visible": mask, "windows": windows}
