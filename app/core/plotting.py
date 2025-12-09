from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.units import Quantity
from matplotlib.figure import Figure
from poliastro.bodies import Body, Earth
from poliastro.maneuver import Maneuver
from poliastro.plotting import OrbitPlotter
from poliastro.plotting.orbit.backends import Plotly3D
from poliastro.twobody import Orbit

from app.core import body_from_str, non_quantity_to_Quantity


def plot_orbit_3d(
    r_vec: list | np.ndarray | Quantity,
    v_vec: list | np.ndarray | Quantity,
    attractor: str | Body = Earth,
    color: Literal["red", "green", "blue"] = "red",
) -> OrbitPlotter:
    """
    Create a 3D visualization of an orbit using Plotly.

    This function generates an interactive 3D plot of an orbital trajectory from
    initial position and velocity vectors. The orbit is visualized around the
    specified central body using the Plotly backend.

    Parameters
    ----------
    r_vec : list, numpy.ndarray, or astropy.units.Quantity
        Position vector [x, y, z]. If list or ndarray, assumed to be in kilometers.
        Must be a 3-element vector.
    v_vec : list, numpy.ndarray, or astropy.units.Quantity
        Velocity vector [vx, vy, vz]. If list or ndarray, assumed to be in
        kilometers per second. Must be a 3-element vector.
    attractor : str or poliastro.bodies.Body, optional
        The central body around which the orbit is plotted. Can be a string name
        (e.g., 'earth', 'mars') or a poliastro Body object. Default is Earth.
    color : {'red', 'green', 'blue'}, optional
        Color of the orbit trajectory in the plot. Default is 'red'.

    Returns
    -------
    poliastro.plotting.OrbitPlotter
        An OrbitPlotter object with the orbit plotted. This can be further
        customized or displayed using the plotter's methods.

    Raises
    ------
    TypeError
        If `r_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `v_vec` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `attractor` is not a string or poliastro.bodies.Body object.
    TypeError
        If `color` is not a string.
    ValueError
        If `color` is not one of 'red', 'green', or 'blue'.

    Examples
    --------
    >>> import numpy as np
    >>> # Plot a circular LEO orbit
    >>> r = np.array([7000.0, 0.0, 0.0])  # km
    >>> v = np.array([0.0, 7.546, 0.0])  # km/s
    >>> plotter = plot_orbit_3d(r, v, attractor='earth', color='blue')
    >>> plotter.show()  # Display the interactive plot

    >>> # Plot an elliptical orbit around Mars
    >>> from astropy import units as u
    >>> r = [5000.0, 0.0, 0.0] * u.km
    >>> v = [0.0, 4.0, 0.0] * u.km / u.s
    >>> plotter = plot_orbit_3d(r, v, attractor='mars', color='red')
    >>> plotter.show()
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
    if not isinstance(color, str):
        raise TypeError(f"Expected type of color is str. Got {type(color)}.")

    if color not in ["red", "green", "blue"]:
        raise ValueError(
            f"Only valid arguments of color are 'red', 'green' and 'blue'. Got '{color}'."
        )

    attractor = body_from_str(attractor)
    r_vec = non_quantity_to_Quantity(r_vec, u.km)
    v_vec = non_quantity_to_Quantity(v_vec, u.km / u.s)

    orbit = Orbit.from_vectors(attractor, r_vec, v_vec)
    plotter = OrbitPlotter(backend=Plotly3D())
    plotter.plot(orbit, color=color)

    return plotter


def plot_transfer(
    r_vec_1: list | np.ndarray | Quantity,
    v_vec_1: list | np.ndarray | Quantity,
    r_vec_2: list | np.ndarray | Quantity,
    v_vec_2: list | np.ndarray | Quantity,
    attractor: str | Body = Earth,
    color: Literal["red", "green", "blue"] = "red",
) -> OrbitPlotter:
    """
    Create a 3D visualization of an orbital transfer maneuver using Lambert's problem.

    This function generates an interactive 3D plot showing a transfer trajectory
    between two orbits. It uses Lambert's problem solver to compute the optimal
    transfer maneuver connecting the initial and final states.

    Parameters
    ----------
    r_vec_1 : list, numpy.ndarray, or astropy.units.Quantity
        Initial position vector [x, y, z]. If list or ndarray, assumed to be in
        kilometers. Must be a 3-element vector.
    v_vec_1 : list, numpy.ndarray, or astropy.units.Quantity
        Initial velocity vector [vx, vy, vz]. If list or ndarray, assumed to be
        in kilometers per second. Must be a 3-element vector.
    r_vec_2 : list, numpy.ndarray, or astropy.units.Quantity
        Final position vector [x, y, z]. If list or ndarray, assumed to be in
        kilometers. Must be a 3-element vector.
    v_vec_2 : list, numpy.ndarray, or astropy.units.Quantity
        Final velocity vector [vx, vy, vz]. If list or ndarray, assumed to be
        in kilometers per second. Must be a 3-element vector.
    attractor : str or poliastro.bodies.Body, optional
        The central body around which the transfer occurs. Can be a string name
        (e.g., 'earth', 'mars') or a poliastro Body object. Default is Earth.
    color : {'red', 'green', 'blue'}, optional
        Color of the transfer trajectory in the plot. Default is 'red'.

    Returns
    -------
    poliastro.plotting.OrbitPlotter
        An OrbitPlotter object with the initial orbit and transfer maneuver plotted.
        This can be further customized or displayed using the plotter's methods.

    Raises
    ------
    TypeError
        If `r_vec_1` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `v_vec_1` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `r_vec_2` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `v_vec_2` is not a list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `attractor` is not a string or poliastro.bodies.Body object.
    TypeError
        If `color` is not a string.
    ValueError
        If `color` is not one of 'red', 'green', or 'blue'.

    Notes
    -----
    The transfer trajectory is computed using Lambert's problem, which determines
    the orbit connecting two position vectors given the transfer time. The maneuver
    shows both the initial orbit and the transfer arc.

    Examples
    --------
    >>> import numpy as np
    >>> # Plot a transfer from LEO to higher orbit
    >>> r1 = np.array([7000.0, 0.0, 0.0])  # km
    >>> v1 = np.array([0.0, 7.546, 0.0])  # km/s
    >>> r2 = np.array([10000.0, 0.0, 0.0])  # km
    >>> v2 = np.array([0.0, 6.0, 0.0])  # km/s
    >>> plotter = plot_transfer(r1, v1, r2, v2, attractor='earth', color='green')
    >>> plotter.show()  # Display the interactive plot

    >>> # Plot Earth-Mars transfer
    >>> from astropy import units as u
    >>> r_earth = [149.6e6, 0.0, 0.0] * u.km
    >>> v_earth = [0.0, 29.78, 0.0] * u.km / u.s
    >>> r_mars = [227.9e6, 0.0, 0.0] * u.km
    >>> v_mars = [0.0, 24.07, 0.0] * u.km / u.s
    >>> plotter = plot_transfer(r_earth, v_earth, r_mars, v_mars,
    ...                         attractor='sun', color='red')
    >>> plotter.show()
    """
    if not isinstance(r_vec_1, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of r_vec_1 is either list or ndarray or Quantity. Got {type(r_vec_1)}"
        )
    if not isinstance(v_vec_1, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of v_vec_1 is either list or ndarray or Quantity. Got {type(v_vec_1)}"
        )
    if not isinstance(r_vec_2, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of r_vec_2 is either list or ndarray or Quantity. Got {type(r_vec_2)}"
        )
    if not isinstance(v_vec_2, (list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of v_vec_2 is either list or ndarray or Quantity. Got {type(v_vec_2)}"
        )
    if not isinstance(attractor, (str, Body)):
        raise TypeError(
            f"Expected type of attractor is either str or poliastro.bodies.Body. Got {type(attractor)}."
        )
    if not isinstance(color, str):
        raise TypeError(f"Expected type of color is str. Got {type(color)}.")

    if color not in ["red", "green", "blue"]:
        raise ValueError(
            f"Only valid arguments of color are 'red', 'green' and 'blue'. Got '{color}'."
        )

    r_vec_1 = non_quantity_to_Quantity(r_vec_1, u.km)
    v_vec_1 = non_quantity_to_Quantity(v_vec_1, u.km / u.s)
    r_vec_2 = non_quantity_to_Quantity(r_vec_2, u.km)
    v_vec_2 = non_quantity_to_Quantity(v_vec_2, u.km / u.s)

    orbit_1 = Orbit.from_vectors(attractor, r_vec_1, v_vec_1)
    orbit_2 = Orbit.from_vectors(attractor, r_vec_2, v_vec_2)

    man = Maneuver.lambert(orbit_1, orbit_2)

    plotter = OrbitPlotter(backend=Plotly3D())
    plotter.plot_maneuver(initial_orbit=orbit_1, maneuver=man, color=color)

    return plotter


def plot_altitude_vs_time(times: list | np.ndarray, alts: list | np.ndarray) -> Figure:
    """
    Create a 2D plot of altitude versus time.

    This function generates a line plot showing how altitude changes over time,
    useful for visualizing orbital decay, altitude variations, or mission profiles.

    Parameters
    ----------
    times : list or numpy.ndarray
        Array of time values. Can be in any time unit (seconds, minutes, hours,
        datetime objects, etc.) as long as they are consistent and plottable.
    alts : list or numpy.ndarray
        Array of altitude values corresponding to each time point. Should be in
        consistent units (typically kilometers or meters).

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib Figure object containing the altitude vs. time plot.
        This can be displayed using `plt.show()` or saved using `fig.savefig()`.

    Raises
    ------
    TypeError
        If `times` is not a list or numpy.ndarray.
    TypeError
        If `alts` is not a list or numpy.ndarray.

    Notes
    -----
    The plot is automatically formatted with:
    - X-axis label: "Time"
    - Y-axis label: "Altitude"
    - Tight layout to prevent label clipping

    The function uses matplotlib's default styling. For custom styling, you can
    modify the returned Figure object.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Plot altitude decay over time
    >>> times = np.linspace(0, 100, 50)  # seconds
    >>> alts = 400 - 0.5 * times  # km, linear decay
    >>> fig = plot_altitude_vs_time(times, alts)
    >>> plt.show()

    >>> # Plot ISS altitude profile
    >>> from datetime import datetime, timedelta
    >>> start = datetime(2025, 1, 1)
    >>> times = [start + timedelta(hours=i) for i in range(24)]
    >>> alts = [408 + 2 * np.sin(i * np.pi / 12) for i in range(24)]  # km
    >>> fig = plot_altitude_vs_time(times, alts)
    >>> fig.savefig('iss_altitude.png')

    >>> # Using lists
    >>> times = [0, 1, 2, 3, 4, 5]
    >>> alts = [300, 305, 310, 308, 312, 315]
    >>> fig = plot_altitude_vs_time(times, alts)
    >>> plt.show()
    """
    if not isinstance(times, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of times is either list or ndarray. Got {type(times)}."
        )
    if not isinstance(alts, (list, np.ndarray)):
        raise TypeError(
            f"Expected type of alts is either list or ndarray. Got {type(alts)}."
        )

    times = np.asarray(times)
    alts = np.asarray(alts)

    fig, ax = plt.subplots()
    ax.plot(times, alts)

    ax.set_xlabel("Time")
    ax.set_ylabel("Altitude")

    fig.tight_layout()
    return fig
