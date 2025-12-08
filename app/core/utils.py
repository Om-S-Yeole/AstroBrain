from datetime import datetime

import numpy as np
from astropy.units import Quantity, Unit
from poliastro.bodies import (
    Body,
    Earth,
    Jupiter,
    Mars,
    Mercury,
    Moon,
    Neptune,
    Pluto,
    Saturn,
    Sun,
    Uranus,
    Venus,
)
from sgp4.api import jday


def body_from_str(planet: str | Body) -> Body:
    """
    Convert a string or Body object to a poliastro Body object.

    This function takes either a string representing a celestial body name or an
    existing Body object and returns the corresponding poliastro Body object. If
    a Body object is provided, it is returned as-is. If a string is provided, it
    is matched (case-insensitive) to one of the supported celestial bodies.

    Parameters
    ----------
    planet : str or poliastro.bodies.Body
        The name of the celestial body as a string (case-insensitive) or an
        existing Body object. Supported body names are: 'sun', 'mercury', 'venus',
        'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto', 'moon'.

    Returns
    -------
    poliastro.bodies.Body
        The poliastro Body object corresponding to the input.

    Raises
    ------
    TypeError
        If the input is neither a string nor a poliastro.bodies.Body object.
    ValueError
        If the input string does not match any of the supported celestial bodies.

    Examples
    --------
    >>> body_from_str('earth')
    <Body: Earth>

    >>> body_from_str('Mars')
    <Body: Mars>

    >>> from poliastro.bodies import Jupiter
    >>> body_from_str(Jupiter)
    <Body: Jupiter>
    """
    if not isinstance(planet, (str, Body)):
        raise TypeError(
            f"Expected type of planet is str or poliastro.bodies.Body. Got {type(planet)}."
        )

    if isinstance(planet, Body):
        return planet
    else:
        match planet.lower():
            case "sun":
                planet = Sun
            case "mercury":
                planet = Mercury
            case "venus":
                planet = Venus
            case "earth":
                planet = Earth
            case "mars":
                planet = Mars
            case "jupiter":
                planet = Jupiter
            case "saturn":
                planet = Saturn
            case "uranus":
                planet = Uranus
            case "neptune":
                planet = Neptune
            case "pluto":
                planet = Pluto
            case "moon":
                planet = Moon
            case _:
                raise ValueError(
                    f"Given body '{planet}' is not a valid argument. Supported bodies are 'sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto', 'moon'."
                )

        return planet


def non_quantity_to_Quantity(
    value: int | float | list | np.ndarray | Quantity, unit: Unit
):
    """
    Convert a numeric value to an astropy Quantity with specified units.

    This function takes a numeric value (int, float, list, numpy array, or existing Quantity)
    and attaches or replaces its units with the specified unit. If the input is already
    a Quantity, its units will be replaced with the new unit.

    Parameters
    ----------
    value : int, float, list, numpy.ndarray, or astropy.units.Quantity
        The numeric value to convert. Can be a scalar (int or float), a numpy array,
        or an existing Quantity object.
    unit : astropy.units.Unit
        The unit to attach to the value or to replace existing units with.

    Returns
    -------
    astropy.units.Quantity
        A Quantity object with the specified value and unit.

    Raises
    ------
    TypeError
        If `value` is not an int, float, list, numpy.ndarray, or astropy.units.Quantity.
    TypeError
        If `unit` is not an astropy.units.Unit object.

    Examples
    --------
    >>> from astropy import units as u
    >>> non_quantity_to_Quantity(5, u.km)
    <Quantity 5. km>

    >>> import numpy as np
    >>> non_quantity_to_Quantity(np.array([1, 2, 3]), u.m)
    <Quantity [1., 2., 3.] m>

    >>> non_quantity_to_Quantity(10 * u.m, u.km)
    <Quantity 10. km>
    """
    if not isinstance(value, (int, float, list, np.ndarray, Quantity)):
        raise TypeError(
            f"Expected type of value are either int or float or list or ndarray or astropy.units.Quantity. Got {type(value)}."
        )
    if not isinstance(unit, Unit):
        raise TypeError(
            f"Expected type of unit is astropy.units.Unit. Got {type(unit)}."
        )

    # Replace the units of given Quantity with given units if value is instance of Quantity. Else attach the given unit to the given value.
    if isinstance(value, list):
        value = np.array(value)
    return Quantity(value, unit)


def datetime_to_jd(dt: datetime) -> tuple[float, float, float]:
    """
    Convert a datetime object to Julian Date.

    Parameters
    ----------
    dt : datetime
        Datetime object to convert.

    Returns
    -------
    tuple[float, float, float]
        - Julian Date (float).
        - Whole part of the Julian Date (float).
        - Fractional part of the Julian Date (float).

    Notes
    -----
    - The function uses the SGP4 library's `jday` function for conversion.
    - The input datetime object must be timezone-aware.
    """
    whole_part, frac_part = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )
    julian_day = whole_part + frac_part
    return julian_day, whole_part, frac_part
