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
