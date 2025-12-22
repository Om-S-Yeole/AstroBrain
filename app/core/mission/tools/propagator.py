from datetime import datetime, timedelta
from typing import List, Tuple, Union

import numpy as np
from astropy.units import Quantity
from langchain.tools import tool
from poliastro.twobody import Orbit
from pydantic import BaseModel

from app.core.mission.utils.propagator import OrbitPropagator, PropagationResults


class Propagate(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    orbit_source: Union[
        Orbit,
        Tuple[
            Union[List, np.ndarray, Quantity],
            Union[List, np.ndarray, Quantity],
            Union[str, datetime],
        ],
        Tuple[str, str],
    ]
    start_time: str | datetime
    end_time: str | datetime
    step: int | float | timedelta


class PropagateAt(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    orbit_source: Union[
        Orbit,  # Poliastro orbit
        Tuple[
            Union[List, np.ndarray, Quantity],
            Union[List, np.ndarray, Quantity],
            Union[str, datetime],
        ],
        Tuple[str, str],
    ]
    times: List[datetime] | np.ndarray[datetime]


@tool(args_schema=Propagate)
def propagate_tool(
    orbit_source: Union[
        Orbit,  # Poliastro orbit
        Tuple[
            Union[List, np.ndarray, Quantity],
            Union[List, np.ndarray, Quantity],
            Union[str, datetime],
        ],  # (r_vec, v_vec, epoch) attractor = Earth always because of propagate function
        Tuple[str, str],  # TLE (line 1, line 2) always assume earth as the attractor
    ],
    start_time: str | datetime,
    end_time: str | datetime,
    step: int | float | timedelta,
) -> PropagationResults:
    """
    Propagate the orbit over a time range with uniform time steps.

    Parameters
    ----------
    start_time : str or datetime
        Start time for propagation. If string, format should be
        "YYYY-MM-DD HH:MM:SS".
    end_time : str or datetime
        End time for propagation. If string, format should be
        "YYYY-MM-DD HH:MM:SS".
    step : int, float, or timedelta
        Time step between propagation points. If int or float,
        interpreted as seconds.

    Returns
    -------
    PropagationResults
        Dictionary containing arrays of time, position vectors (r_eci),
        velocity vectors (v_eci), latitude, longitude, and altitude.

    Examples
    --------
    >>> results = propagator.propagate(
    ...     "2024-01-01 00:00:00",
    ...     "2024-01-01 01:00:00",
    ...     60
    ... )
    >>> print(results['time'])
    >>> print(results['lat'])
    """
    orbit_propagator = OrbitPropagator(orbit_source)
    propagation_results = orbit_propagator.propagate(start_time, end_time, step)

    return propagation_results


@tool(args_schema=PropagateAt)
def propagate_at_tool(
    orbit_source: Union[
        Orbit,  # Poliastro orbit
        Tuple[
            Union[List, np.ndarray, Quantity],
            Union[List, np.ndarray, Quantity],
            Union[str, datetime],
        ],  # (r_vec, v_vec, epoch) attractor = Earth always because of propagate function
        Tuple[str, str],  # TLE (line 1, line 2) always assume earth as the attractor
    ],
    times: List[datetime] | np.ndarray[datetime],
) -> PropagationResults:
    """
    Propagate the orbit at specific time points.

    Parameters
    ----------
    times : List[datetime] or np.ndarray[datetime]
        List or array of datetime objects at which to compute
        the orbit state.

    Returns
    -------
    PropagationResults
        Dictionary containing arrays of time, position vectors (r_eci),
        velocity vectors (v_eci), latitude, longitude, and altitude
        at the specified times.

    Examples
    --------
    >>> from datetime import datetime
    >>> times = [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 1, 0, 0)]
    >>> results = propagator.propagate_at(times)
    >>> print(results['r_eci'])
    """

    orbit_propagator = OrbitPropagator(orbit_source)
    propagation_results = orbit_propagator.propagate_at(times)

    return propagation_results
