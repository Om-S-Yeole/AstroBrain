from datetime import datetime

import numpy as np
import pytz
from langchain.tools import tool
from poliastro.constants import R_earth
from pydantic import BaseModel

from app.core.mission.utils.eclipse import (
    EclipseResults,
    extract_eclipse_windows,
    is_in_umbra,
    umbra_mask,
)
from app.core.mission.utils.propagator import PropagationResults
from app.core.mission.utils.sun_geometry import SunGeometryResults


class IsInUmbraToolSchema(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    r_eci: list | np.ndarray
    sun_vec_eci: list | np.ndarray


class UmbraMaskToolSchema(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    r_eci: list | np.ndarray
    sun_vec_eci: list | np.ndarray


class ExtractEclipseWindowsToolSchema(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    times: list[datetime] | np.ndarray[datetime]
    eclipse_mask: list[bool] | np.ndarray[bool]


class ComputeEclipseSchema(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    propagation_results: PropagationResults
    sun_geometry: SunGeometryResults


@tool(args_schema=IsInUmbraToolSchema)
def is_in_umbra_tool(
    r_eci: list | np.ndarray, sun_vec_eci: list | np.ndarray
) -> bool:  # We do not assuem that unit vectors are passed
    """
    Determine if a spacecraft position is in Earth's umbra (full shadow).

    Uses a cylindrical shadow model to check if the spacecraft is in Earth's
    shadow cone. The function checks if the spacecraft is behind Earth relative
    to the Sun and within the shadow cylinder.

    Parameters
    ----------
    r_eci : list or np.ndarray
        Spacecraft position vector in ECI coordinates (km).
    sun_vec_eci : list or np.ndarray
        Sun position vector in ECI coordinates (km). Does not need to be normalized.

    Returns
    -------
    bool
        True if the spacecraft is in umbra, False otherwise.

    Raises
    ------
    TypeError
        If r_eci or sun_vec_eci are not list or numpy arrays.
    """
    if not isinstance(r_eci, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of r_eci is list or np.ndarray. Got {type(r_eci)}"
        )
    if not isinstance(sun_vec_eci, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of sun_vec_eci is list or np.ndarray. Got {type(sun_vec_eci)}"
        )

    r_eci = np.array(r_eci)
    sun_vec_eci = np.array(sun_vec_eci)

    s_hat = sun_vec_eci / np.linalg.norm(sun_vec_eci)
    r_parallel = np.dot(r_eci, s_hat)
    r_perp = np.linalg.norm(r_eci - r_parallel * s_hat)

    return r_parallel < 0 and r_perp < R_earth.value / 1000


@tool(args_schema=UmbraMaskToolSchema)
def umbra_mask_tool(r_eci: list | np.ndarray, sun_vec_eci: list | np.ndarray):
    """
    Generate a boolean mask indicating eclipse status for multiple positions.

    For each spacecraft position and corresponding Sun vector, determines
    whether the spacecraft is in Earth's umbra.

    Parameters
    ----------
    r_eci : list or np.ndarray
        List of spacecraft position vectors in ECI coordinates (km).
    sun_vec_eci : list or np.ndarray
        List of Sun position vectors in ECI coordinates (km).

    Returns
    -------
    np.ndarray
        Boolean array where True indicates the spacecraft is in umbra
        at that time step.

    Raises
    ------
    TypeError
        If r_eci or sun_vec_eci are not list or numpy arrays.
    """
    if not isinstance(r_eci, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of r_eci is list or np.ndarray. Got {type(r_eci)}"
        )
    if not isinstance(sun_vec_eci, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of sun_vec_eci is list or np.ndarray. Got {type(sun_vec_eci)}"
        )

    mask = []

    for r, sun in zip(r_eci, sun_vec_eci):
        r = np.array(r)
        sun = np.array(sun)
        mask.append(is_in_umbra(r, sun))

    return np.array(mask)


@tool(args_schema=ExtractEclipseWindowsToolSchema)
def extract_eclipse_windows_tool(
    times: list[datetime] | np.ndarray[datetime],
    eclipse_mask: list[bool] | np.ndarray[bool],
) -> list[dict]:
    """
    Extract continuous eclipse windows from an eclipse mask.

    Identifies periods of continuous eclipse and returns them as time windows
    with start time, end time, and duration.

    Parameters
    ----------
    times : list of datetime or np.ndarray of datetime
        Array of datetime objects corresponding to each propagation step.
    eclipse_mask : list of bool or np.ndarray of bool
        Boolean array indicating eclipse status at each time step.

    Returns
    -------
    list of dict
        List of eclipse windows, where each window is a dictionary with:
        - 'start': datetime object for eclipse start (UTC)
        - 'end': datetime object for eclipse end (UTC)
        - 'duration': duration in seconds (float)

    Raises
    ------
    TypeError
        If times or eclipse_mask are not list or numpy arrays.
    """
    if not isinstance(times, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of times is list or np.ndarray. Got {type(times)}"
        )
    if not isinstance(eclipse_mask, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of eclipse_mask is list or np.ndarray. Got {type(eclipse_mask)}"
        )

    eclipse_windows = []
    add_new = True

    for time, is_eclipsed in zip(times, eclipse_mask):
        time = time.replace(tzinfo=pytz.utc)
        if is_eclipsed and add_new:
            eclipse_windows.append({"start": time, "end": time, "duration": 0})
            add_new = False
        elif is_eclipsed and not add_new:
            eclipse_windows[-1]["end"] = time
            eclipse_windows[-1]["duration"] = (
                time - eclipse_windows[-1]["start"]
            ).total_seconds()
        elif not is_eclipsed:
            add_new = True

    return eclipse_windows


@tool(args_schema=ComputeEclipseSchema)
def compute_eclipse_tool(
    propagation_results: PropagationResults, sun_geometry: SunGeometryResults
) -> EclipseResults:
    """
    Compute eclipse information for an entire propagated orbit.

    Analyzes the spacecraft trajectory to determine eclipse periods, extract
    continuous eclipse windows, and calculate the fraction of time spent in eclipse.

    Parameters
    ----------
    propagation_results : PropagationResults
        Dictionary containing propagation data with keys:
        - 'time': list of datetime objects
        - 'r_eci': list of position vectors in ECI coordinates (km)
    sun_geometry : SunGeometryResults
        Dictionary containing Sun geometry data with key:
        - 'sun_vec_eci': list of Sun position vectors in ECI coordinates (km)

    Returns
    -------
    EclipseResults
        TypedDict containing:
        - 'eclipsed': boolean array indicating eclipse status at each time step
        - 'windows': list of eclipse window dictionaries with start, end, and duration
        - 'fraction_in_eclipse': fraction of orbit time spent in eclipse (float)
    """
    times = propagation_results["time"]
    r_vecs = propagation_results["r_eci"]
    sun_vecs = sun_geometry["sun_vec_eci"]

    mask = umbra_mask(r_vecs, sun_vecs)
    windows = extract_eclipse_windows(times, mask)
    frac_in_eclipse = np.sum(mask) / len(mask)

    return {
        "eclipsed": mask,
        "windows": windows,
        "fraction_in_eclipse": frac_in_eclipse,
    }
