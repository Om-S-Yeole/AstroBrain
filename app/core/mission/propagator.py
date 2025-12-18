from datetime import datetime, timedelta
from typing import List, Tuple, TypedDict, Union

import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy.time import Time
from astropy.units import Quantity
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import ValladoPropagator
from sgp4.api import Satrec

from app.core.orbital_tools import sgp4_propagate
from app.core.utils import jd_to_datetime, non_quantity_to_Quantity


class PropagationResults(TypedDict):
    """
    Type definition for orbit propagation results.

    Attributes
    ----------
    time : np.ndarray[datetime]
        Array of datetime objects representing the time points.
    r_eci : np.ndarray[Tuple[float, float, float]]
        Position vectors in ECI frame (x, y, z) in kilometers.
    v_eci : np.ndarray[Tuple[float, float, float]]
        Velocity vectors in ECI frame (vx, vy, vz) in km/s.
    lat : np.ndarray[float]
        Geodetic latitude in degrees.
    lon : np.ndarray[float]
        Geodetic longitude in degrees.
    alt : np.ndarray[float]
        Altitude above Earth's surface in kilometers.
    """

    time: np.ndarray[datetime]
    r_eci: np.ndarray[Tuple[float, float, float]]
    v_eci: np.ndarray[Tuple[float, float, float]]
    lat: np.ndarray[float]
    lon: np.ndarray[float]
    alt: np.ndarray[float]


class OrbitPropagator:
    """
    Orbital propagator for satellites and spacecraft.

    This class provides orbit propagation capabilities using Vallado's method.
    It supports multiple input formats including Poliastro Orbit objects,
    state vectors (position and velocity), and Two-Line Element (TLE) sets.

    Parameters
    ----------
    orbit_source : Union[Orbit, Tuple, Tuple[str, str]]
        The orbital information in one of the following formats:
        - Poliastro Orbit object
        - Tuple of (r_vec, v_vec, epoch) where r_vec and v_vec are position
          and velocity vectors, and epoch is a datetime or string
        - Tuple of (line1, line2) representing TLE format

    Attributes
    ----------
    orbit : Orbit
        The normalized Poliastro Orbit object.
    _propagator : ValladoPropagator
        The propagation method instance.

    Examples
    --------
    >>> from poliastro.twobody import Orbit
    >>> propagator = OrbitPropagator(orbit_obj)
    >>> results = propagator.propagate("2024-01-01 00:00:00", "2024-01-02 00:00:00", 60)
    """

    def __init__(
        self,
        orbit_source: Union[
            Orbit,  # Poliastro orbit
            Tuple[
                Union[List, np.ndarray, Quantity],
                Union[List, np.ndarray, Quantity],
                Union[str, datetime],
            ],  # (r_vec, v_vec, epoch) attractor = Earth always because of propagate function
            Tuple[
                str, str
            ],  # TLE (line 1, line 2) always assume earth as the attractor
        ],
    ):
        self.orbit: Orbit = self._normalize_orbit_source(orbit_source)
        self._propagator = ValladoPropagator(numiter=500)

    def propagate(
        self,
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
        if not isinstance(start_time, (str, datetime)):
            raise TypeError(
                f"Expected type of start_time is str or datetime. Got {type(start_time)}"
            )
        if not isinstance(end_time, (str, datetime)):
            raise TypeError(
                f"Expected type of end_time is str or datetime. Got {type(end_time)}"
            )
        if not isinstance(step, (int, float, timedelta)):
            raise TypeError(
                f"Expected type of step is int or float or timedelta. Got {type(step)}"
            )

        propagation_results = {
            "time": self._build_time_grid(start_time, end_time, step),
            "r_eci": [],
            "v_eci": [],
            "lat": [],
            "lon": [],
            "alt": [],
        }

        for grid_time in propagation_results["time"]:
            dt = Time(grid_time) - self.orbit.epoch
            propagated_orbit = self.orbit.propagate(dt, method=self._propagator)
            r = propagated_orbit.r
            v = propagated_orbit.v.value
            epoch = propagated_orbit.epoch
            lat, lon, alt = self._get_latitude_longitude_altitude(r, epoch)
            propagation_results["r_eci"].append(tuple(r.value))
            propagation_results["v_eci"].append(tuple(v))
            propagation_results["lat"].append(lat)
            propagation_results["lon"].append(lon)
            propagation_results["alt"].append(alt)

        propagation_results = {
            key: np.array(value) for key, value in propagation_results.items()
        }

        return propagation_results

    def propagate_at(
        self, times: List[datetime] | np.ndarray[datetime]
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
        if not isinstance(times, (list, np.ndarray)):
            raise TypeError(
                f"Expected type of times is list or np.ndarray. Got {type(times)}"
            )
        for id, time in enumerate(times):
            times[id] = time.replace(tzinfo=pytz.utc)

        propagation_results = {
            "time": times,
            "r_eci": [],
            "v_eci": [],
            "lat": [],
            "lon": [],
            "alt": [],
        }

        for grid_time in propagation_results["time"]:
            dt = Time(grid_time) - self.orbit.epoch
            propagated_orbit = self.orbit.propagate(dt, method=self._propagator)
            r = propagated_orbit.r
            v = propagated_orbit.v.value
            epoch = propagated_orbit.epoch
            lat, lon, alt = self._get_latitude_longitude_altitude(r, epoch)
            propagation_results["r_eci"].append(tuple(r.value))
            propagation_results["v_eci"].append(tuple(v))
            propagation_results["lat"].append(lat)
            propagation_results["lon"].append(lon)
            propagation_results["alt"].append(alt)

        propagation_results = {
            key: np.array(value) for key, value in propagation_results.items()
        }

        return propagation_results

    def _normalize_orbit_source(self, orbit_source) -> Orbit:
        """
        Convert various orbit source formats to a Poliastro Orbit object.

        Parameters
        ----------
        orbit_source : Union[Orbit, Tuple]
            The orbit source in various formats:
            - Poliastro Orbit object (returned as-is)
            - Tuple of (r_vec, v_vec, epoch) for state vectors
            - Tuple of (line1, line2) for TLE format

        Returns
        -------
        Orbit
            A Poliastro Orbit object.

        Raises
        ------
        ValueError
            If the orbit_source format is invalid or unrecognized.

        Notes
        -----
        - For state vectors, attractor is always Earth.
        - For TLE inputs, SGP4 propagation is used to compute initial state.
        - Epoch strings should be in "YYYY-MM-DD HH:MM:SS" format.
        """
        if isinstance(orbit_source, Orbit):
            return orbit_source

        elif isinstance(orbit_source, tuple):
            if len(orbit_source) == 3:
                attractor = Earth
                r_vec = non_quantity_to_Quantity(orbit_source[0], "km")
                v_vec = Quantity(orbit_source[1], u.km / u.s)
                epoch = orbit_source[2]
                if isinstance(epoch, str):
                    epoch = datetime.strptime(epoch, "%Y-%m-%d %H:%M:%S")
                epoch = epoch.replace(tzinfo=pytz.utc)
                epoch = Time(epoch)

                return Orbit.from_vectors(attractor, r_vec, v_vec, epoch)

            elif len(orbit_source) == 2:
                satrec_obj = Satrec.twoline2rv(orbit_source[0], orbit_source[1])
                attractor = Earth
                satellite_jd = satrec_obj.jdsatepoch + satrec_obj.jdsatepochF
                epoch_datetime = jd_to_datetime(satellite_jd)
                r, v = sgp4_propagate(orbit_source[0], orbit_source[1], epoch_datetime)
                epoch_datetime = Time(epoch_datetime)

                return Orbit.from_vectors(attractor, r, v, epoch_datetime)
        else:
            raise ValueError("Invalid orbit source")

    def _build_time_grid(
        self,
        start_time: str | datetime,
        end_time: str | datetime,
        step: int | float | timedelta,
    ) -> np.ndarray:
        """
        Build a uniform time grid from start to end with given step size.

        Parameters
        ----------
        start_time : str or datetime
            Start time of the grid. If string, format should be
            "YYYY-MM-DD HH:MM:SS".
        end_time : str or datetime
            End time of the grid. If string, format should be
            "YYYY-MM-DD HH:MM:SS".
        step : int, float, or timedelta
            Time step size. If int or float, interpreted as seconds.

        Returns
        -------
        np.ndarray
            Array of datetime objects from start_time to end_time with
            the specified step size.

        Notes
        -----
        All times are converted to UTC timezone.
        """
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

        start_time = start_time.replace(tzinfo=pytz.utc)
        end_time = end_time.replace(tzinfo=pytz.utc)

        if isinstance(step, (int, float)):
            step = timedelta(seconds=step)

        grid = [start_time]
        curr_time = start_time

        while curr_time <= end_time:
            curr_time += step
            grid.append(curr_time)

        return np.array(grid)

    def _get_latitude_longitude_altitude(self, r_eci: Quantity, t_utc: datetime | Time):
        """
        Convert ECI position to geodetic coordinates (latitude, longitude, altitude).

        Parameters
        ----------
        r_eci : Quantity
            Position vector in ECI (Earth-Centered Inertial) frame in kilometers.
        t_utc : datetime or Time
            UTC time corresponding to the position vector.

        Returns
        -------
        lat : float
            Geodetic latitude in degrees.
        lon : float
            Geodetic longitude in degrees.
        alt : float
            Altitude above Earth's surface in kilometers.

        Notes
        -----
        This method performs coordinate transformation from GCRS (Geocentric
        Celestial Reference System) to ITRS (International Terrestrial
        Reference System) to compute geodetic coordinates.
        """
        r = CartesianRepresentation(r_eci[0], r_eci[1], r_eci[2], u.km)
        gcrs = GCRS(r, obstime=Time(t_utc, scale="utc"))
        itrs = gcrs.transform_to(ITRS(obstime=Time(t_utc, scale="utc")))

        # Access the Cartesian representation of the ITRS coordinate
        itrs_cart = itrs.cartesian
        loc = EarthLocation.from_geocentric(itrs_cart.x, itrs_cart.y, itrs_cart.z, u.km)

        lat = loc.lat.to(u.deg).value
        lon = loc.lon.to(u.deg).value
        alt = loc.height.value

        return lat, lon, alt
