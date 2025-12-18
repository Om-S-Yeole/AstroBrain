import uuid
from datetime import datetime, timedelta, timezone
from math import cos, sin

import numpy as np
import pytz
from astropy import units as u
from astropy.units import Quantity
from langchain.tools import tool
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
from pydantic import BaseModel, Field
from sgp4.api import jday


class BodyFromStrSchema(BaseModel):
    """Input schema for the body_from_str tool.

    Defines the required parameters for converting a string representation
    of a celestial body name to a poliastro Body object.
    """

    model_config = {"arbitrary_types_allowed": True}

    planet: str | Body = Field(
        description="Name of celestial body (case-insensitive) or Body object. Supported: 'sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto', 'moon'"
    )


class NonQuantityToQuantitySchema(BaseModel):
    """Input schema for the non_quantity_to_Quantity tool.

    Defines the required parameters for converting numeric values to
    astropy Quantity objects with specified units.
    """

    model_config = {"arbitrary_types_allowed": True}

    value: int | float | list | np.ndarray | Quantity = Field(
        description="Numeric value to convert (scalar, list, array, or existing Quantity)"
    )
    unit: str = Field(description="Astropy unit to attach or replace")


class DatetimeToJDSchema(BaseModel):
    """Input schema for the datetime_to_jd tool.

    Defines the required parameters for converting a Python datetime
    object to Julian Date format.
    """

    model_config = {"arbitrary_types_allowed": True}
    dt: datetime = Field(
        description="Datetime object to convert (expected in UTC timezone)"
    )


class DatetimeFromTimesSchema(BaseModel):
    """Input schema for the datetime_from_times tool.

    Defines the required parameters for creating a UTC-aware datetime
    object from individual time components (year, month, day, etc.).
    """

    model_config = {"arbitrary_types_allowed": True}
    year: int = Field(description="Year (e.g., 2025)")
    month: int = Field(description="Month (1-12)")
    day: int = Field(description="Day of month (1-31)")
    hour: int = Field(description="Hour in 24-hour format (0-23)")
    minute: int = Field(description="Minute (0-59)")
    second: int = Field(description="Second (0-59)")


class Deg2RadSchema(BaseModel):
    """Input schema for the deg2rad tool.

    Defines the required parameters for converting an angle from
    degrees to radians.
    """

    model_config = {"arbitrary_types_allowed": True}
    angle_deg: int | float = Field(description="Angle in degrees to convert to radians")


class Rad2DegSchema(BaseModel):
    """Input schema for the rad2deg tool.

    Defines the required parameters for converting an angle from
    radians to degrees.
    """

    model_config = {"arbitrary_types_allowed": True}
    angle_rad: int | float = Field(description="Angle in radians to convert to degrees")


class NormSchema(BaseModel):
    """Input schema for the norm tool.

    Defines the required parameters for calculating the Euclidean norm
    (magnitude) of a vector.
    """

    model_config = {"arbitrary_types_allowed": True}
    vec: list | np.ndarray = Field(
        description="Vector to calculate magnitude of (any dimension)"
    )


class UnitVecSchema(BaseModel):
    """Input schema for the unit_vec tool.

    Defines the required parameters for calculating the unit vector
    (normalized vector) of a given vector.
    """

    model_config = {"arbitrary_types_allowed": True}
    vec: list | np.ndarray = Field(description="Vector to normalize (any dimension)")


class RotationMatrixPerifocalToECISchema(BaseModel):
    """Input schema for the rotation_matrix_from_perifocal_to_ECI tool.

    Defines the required parameters for computing the rotation matrix that
    transforms vectors from the perifocal (PQW) coordinate frame to the
    Earth-Centered Inertial (ECI) coordinate frame.
    """

    model_config = {"arbitrary_types_allowed": True}
    raan: int | float = Field(
        description="Right Ascension of Ascending Node (RAAN/Ω) in degrees"
    )
    i: int | float = Field(description="Inclination angle in degrees")
    argp: int | float = Field(description="Argument of periapsis (ω) in degrees")


class RotationMatrixECIToPerifocalSchema(BaseModel):
    """Input schema for the rotation_matrix_from_ECI_to_perifocal tool.

    Defines the required parameters for computing the rotation matrix that
    transforms vectors from the Earth-Centered Inertial (ECI) coordinate
    frame to the perifocal (PQW) coordinate frame.
    """

    model_config = {"arbitrary_types_allowed": True}
    raan: int | float = Field(
        description="Right Ascension of Ascending Node (RAAN/Ω) in degrees"
    )
    i: int | float = Field(description="Inclination angle in degrees")
    argp: int | float = Field(description="Argument of periapsis (ω) in degrees")


class CrossSchema(BaseModel):
    """Input schema for the cross tool.

    Defines the required parameters for calculating the cross product
    (vector product) of two 3D vectors.
    """

    model_config = {"arbitrary_types_allowed": True}
    a: list | np.ndarray = Field(description="First 3D vector")
    b: list | np.ndarray = Field(description="Second 3D vector")


class DotSchema(BaseModel):
    """Input schema for the dot tool.

    Defines the required parameters for calculating the dot product
    (scalar product) of two vectors.
    """

    model_config = {"arbitrary_types_allowed": True}
    a: list | np.ndarray = Field(description="First vector")
    b: list | np.ndarray = Field(
        description="Second vector (must have same length as first)"
    )


class UUIDGeneratorSchema(BaseModel):
    """Input schema for the uuid_generator tool.

    This tool requires no input parameters. It generates a unique UUID
    (Universally Unique Identifier) as a string.
    """

    pass  # No inputs required


class JDToDatetimeSchema(BaseModel):
    """Input schema for the jd_to_datetime tool."""

    model_config = {"arbitrary_types_allowed": True}
    jd: float = Field(description="Julian day to convert to datetime object")


@tool(args_schema=BodyFromStrSchema)
def body_from_str_tool(planet: str | Body) -> Body:
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


@tool(args_schema=NonQuantityToQuantitySchema)
def non_quantity_to_Quantity_tool(
    value: int | float | list | np.ndarray | Quantity, unit: str
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
    if not isinstance(unit, str):
        raise TypeError(f"Expected type of unit is str. Got {type(unit)}.")

    if isinstance(unit, str):
        unit = getattr(u, unit)

    # Replace the units of given Quantity with given units if value is instance of Quantity. Else attach the given unit to the given value.
    if isinstance(value, list):
        value = np.array(value)
    return Quantity(value, unit)


@tool(args_schema=DatetimeToJDSchema)
def datetime_to_jd_tool(dt: datetime) -> tuple[float, float, float]:
    """
    Convert a datetime object to Julian Date components.

    This function converts a Python datetime object to Julian Date format, which
    is commonly used in astronomy and astrodynamics. The Julian Date is returned
    as both a complete value and separated into whole and fractional parts for
    precision in calculations.

    Parameters
    ----------
    dt : datetime.datetime
        The datetime object to convert. Should include year, month, day, hour,
        minute, and second components. Expected to be in UTC timezone.

    Returns
    -------
    tuple of float
        A tuple containing three float values:
        - julian_day : float
            Complete Julian Date (JD) value.
        - whole_part : float
            Whole (integer) part of the Julian Date.
        - frac_part : float
            Fractional part of the Julian Date.

    Notes
    -----
    The function uses the SGP4 library's `jday` function for conversion. The
    separation into whole and fractional parts is useful for maintaining numerical
    precision in orbital calculations.

    Julian Date is a continuous count of days since the beginning of the Julian
    Period (January 1, 4713 BC, noon UTC).

    All datetime objects in this library are expected to be in UTC timezone.

    Examples
    --------
    >>> from datetime import datetime
    >>> import pytz
    >>> dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=pytz.utc)
    >>> jd, whole, frac = datetime_to_jd(dt)
    >>> print(f"Julian Date: {jd}")
    >>> print(f"Whole part: {whole}, Fractional part: {frac}")

    >>> # Convert current time
    >>> from datetime import datetime
    >>> now = datetime.now(pytz.utc)
    >>> jd, _, _ = datetime_to_jd(now)
    >>> print(f"Current JD: {jd}")
    """
    whole_part, frac_part = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )
    julian_day = whole_part + frac_part
    return julian_day, whole_part, frac_part


@tool(args_schema=DatetimeFromTimesSchema)
def datetime_from_times_tool(
    year: int, month: int, day: int, hour: int, minute: int, second: int
) -> datetime:
    """
    Create a UTC-aware datetime object from individual time components.

    This function constructs a Python datetime object from separate integer values
    for year, month, day, hour, minute, and second. The returned datetime is
    timezone-aware and set to UTC. It includes type validation for all input
    parameters to ensure data integrity.

    Parameters
    ----------
    year : int
        The year (e.g., 2025).
    month : int
        The month (1-12).
    day : int
        The day of the month (1-31, depending on month).
    hour : int
        The hour in 24-hour format (0-23).
    minute : int
        The minute (0-59).
    second : int
        The second (0-59).

    Returns
    -------
    datetime.datetime
        A timezone-aware datetime object (UTC) representing the specified date and time.

    Raises
    ------
    TypeError
        If any of the parameters is not an integer.
    ValueError
        If any parameter is out of the valid range (raised by datetime constructor).

    Notes
    -----
    All datetime objects in this library are configured to use UTC timezone
    for consistency in orbital calculations.

    Examples
    --------
    >>> datetime_from_times(2025, 1, 1, 12, 0, 0)
    datetime.datetime(2025, 1, 1, 12, 0, tzinfo=<UTC>)

    >>> # Create datetime for a specific orbital epoch
    >>> epoch = datetime_from_times(2025, 12, 14, 10, 30, 45)
    >>> print(epoch)
    2025-12-14 10:30:45+00:00

    >>> # New Year's Day 2024 at midnight UTC
    >>> ny_2024 = datetime_from_times(2024, 1, 1, 0, 0, 0)
    >>> print(ny_2024)
    2024-01-01 00:00:00+00:00
    >>> print(ny_2024.tzinfo)
    UTC
    """
    if not isinstance(year, int):
        raise TypeError(f"Expected type of year is int. Got {type(year)}")
    if not isinstance(month, int):
        raise TypeError(f"Expected type of month is int. Got {type(month)}")
    if not isinstance(day, int):
        raise TypeError(f"Expected type of day is int. Got {type(day)}")
    if not isinstance(hour, int):
        raise TypeError(f"Expected type of hour is int. Got {type(hour)}")
    if not isinstance(minute, int):
        raise TypeError(f"Expected type of minute is int. Got {type(minute)}")
    if not isinstance(second, int):
        raise TypeError(f"Expected type of second is int. Got {type(second)}")

    return datetime(year, month, day, hour, minute, second).replace(tzinfo=pytz.utc)


@tool(args_schema=Deg2RadSchema)
def deg2rad_tool(angle_deg: int | float) -> float:
    """
    Convert an angle from degrees to radians.

    This is a convenience wrapper around numpy's deg2rad function with added
    type checking to ensure the input is numeric.

    Parameters
    ----------
    angle_deg : int or float
        The angle in degrees to convert to radians.

    Returns
    -------
    float
        The angle converted to radians.

    Raises
    ------
    TypeError
        If `angle_deg` is not an int or float.

    Notes
    -----
    The conversion formula is:

    .. math::
        \\text{radians} = \\text{degrees} \\times \\frac{\\pi}{180}

    Examples
    --------
    >>> deg2rad(180)
    3.141592653589793

    >>> deg2rad(90)
    1.5707963267948966

    >>> deg2rad(45.0)
    0.7853981633974483
    """
    if not isinstance(angle_deg, (int, float)):
        raise TypeError(
            f"Expected type of angle_deg is int or float. Got {type(angle_deg)}."
        )
    return np.deg2rad(angle_deg)


@tool(args_schema=Rad2DegSchema)
def rad2deg_tool(angle_rad: int | float) -> float:
    """
    Convert an angle from radians to degrees.

    This is a convenience wrapper around numpy's rad2deg function with added
    type checking to ensure the input is numeric.

    Parameters
    ----------
    angle_rad : int or float
        The angle in radians to convert to degrees.

    Returns
    -------
    float
        The angle converted to degrees.

    Raises
    ------
    TypeError
        If `angle_rad` is not an int or float.

    Notes
    -----
    The conversion formula is:

    .. math::
        \\text{degrees} = \\text{radians} \\times \\frac{180}{\\pi}

    Examples
    --------
    >>> import numpy as np
    >>> rad2deg(np.pi)
    180.0

    >>> rad2deg(np.pi / 2)
    90.0

    >>> rad2deg(1.5707963267948966)
    90.0
    """
    if not isinstance(angle_rad, (int, float)):
        raise TypeError(
            f"Expected type of angle_rad is int or float. Got {type(angle_rad)}."
        )
    return np.rad2deg(angle_rad)


@tool(args_schema=NormSchema)
def norm_tool(vec: list | np.ndarray) -> float:
    """
    Calculate the Euclidean norm (magnitude) of a vector.

    This function computes the L2 norm (Euclidean length) of a vector, which is
    the square root of the sum of squared components. It's commonly used to find
    the magnitude of position or velocity vectors in orbital mechanics.

    Parameters
    ----------
    vec : list or numpy.ndarray
        The input vector. Can be of any dimension.

    Returns
    -------
    float
        The Euclidean norm (magnitude) of the vector.

    Raises
    ------
    TypeError
        If `vec` is not a list or numpy.ndarray.

    Notes
    -----
    The Euclidean norm is calculated as:

    .. math::
        ||\\vec{v}|| = \\sqrt{v_1^2 + v_2^2 + ... + v_n^2}

    For a 3D vector:

    .. math::
        ||\\vec{v}|| = \\sqrt{x^2 + y^2 + z^2}

    Examples
    --------
    >>> norm([3, 4])
    5.0

    >>> import numpy as np
    >>> norm(np.array([1, 0, 0]))
    1.0

    >>> # Calculate magnitude of a position vector
    >>> r = [7000.0, 0.0, 0.0]  # km
    >>> norm(r)
    7000.0
    """
    if not isinstance(vec, (list, np.ndarray)):
        raise TypeError(f"Expected type of vec is list or np.ndarray. Got {type(vec)}.")
    return np.linalg.norm(np.asarray(vec))


@tool(args_schema=UnitVecSchema)
def unit_vec_tool(vec: list | np.ndarray) -> np.ndarray:
    """
    Calculate the unit vector (normalized vector) of a given vector.

    This function computes the unit vector by dividing the input vector by its
    magnitude. The resulting vector has the same direction but a magnitude of 1.
    Unit vectors are useful for representing directions in space.

    Parameters
    ----------
    vec : list or numpy.ndarray
        The input vector to normalize. Can be of any dimension.

    Returns
    -------
    numpy.ndarray
        The unit vector in the same direction as the input, with magnitude 1.

    Raises
    ------
    TypeError
        If `vec` is not a list or numpy.ndarray.

    Notes
    -----
    The unit vector is calculated as:

    .. math::
        \\hat{v} = \\frac{\\vec{v}}{||\\vec{v}||}

    where ||v|| is the Euclidean norm of the vector.

    If the input vector has zero magnitude, this will result in a division by zero
    and return NaN or infinity values.

    Examples
    --------
    >>> unit_vec([3, 4])
    array([0.6, 0.8])

    >>> import numpy as np
    >>> unit_vec(np.array([1, 0, 0]))
    array([1., 0., 0.])

    >>> # Get direction from Earth to a satellite
    >>> r = [7000.0, 0.0, 0.0]  # km
    >>> r_hat = unit_vec(r)
    >>> print(r_hat)
    [1. 0. 0.]

    >>> # Verify magnitude is 1
    >>> np.linalg.norm(unit_vec([3, 4, 5]))
    1.0
    """
    if not isinstance(vec, (list, np.ndarray)):
        raise TypeError(f"Expected type of vec is list or np.ndarray. Got {type(vec)}.")
    vec = np.asarray(vec)
    return vec / np.linalg.norm(vec)


@tool(args_schema=RotationMatrixPerifocalToECISchema)
def rotation_matrix_from_perifocal_to_ECI_tool(
    raan: int | float, i: int | float, argp: int | float
) -> np.ndarray:
    """
    Compute the rotation matrix from perifocal to ECI coordinate frame.

    This function calculates the 3x3 rotation matrix that transforms vectors from
    the perifocal (PQW) coordinate system to the Earth-Centered Inertial (ECI)
    coordinate system. The perifocal frame is orbit-specific, while ECI is an
    inertial reference frame.

    Parameters
    ----------
    raan : int or float
        Right Ascension of the Ascending Node (RAAN or Ω) in degrees.
    i : int or float
        Inclination angle in degrees.
    argp : int or float
        Argument of periapsis (ω) in degrees.

    Returns
    -------
    numpy.ndarray
        A 3x3 rotation matrix that transforms perifocal coordinates to ECI coordinates.

    Raises
    ------
    TypeError
        If `raan` is not an int or float.
    TypeError
        If `i` is not an int or float.
    TypeError
        If `argp` is not an int or float.

    Notes
    -----
    The rotation matrix is constructed from three successive rotations:
    1. Rotation by argument of periapsis (ω) about the z-axis
    2. Rotation by inclination (i) about the x-axis
    3. Rotation by RAAN (Ω) about the z-axis

    The transformation is:

    .. math::
        \\vec{r}_{ECI} = R_{PQW \\rightarrow ECI} \\cdot \\vec{r}_{PQW}

    The perifocal frame (PQW) is defined as:
    - P-axis: Points toward periapsis
    - Q-axis: 90° ahead in the orbital plane
    - W-axis: Perpendicular to the orbital plane (angular momentum direction)

    Examples
    --------
    >>> # Get rotation matrix for ISS-like orbit
    >>> R = rotation_matrix_from_perifocal_to_ECI(raan=45, i=51.6, argp=0)
    >>> print(R.shape)
    (3, 3)

    >>> # Transform a position vector from perifocal to ECI
    >>> r_pqw = np.array([7000, 0, 0])  # Periapsis position in PQW
    >>> R = rotation_matrix_from_perifocal_to_ECI(30, 45, 60)
    >>> r_eci = R @ r_pqw
    >>> print(f"Position in ECI: {r_eci}")
    """

    if not isinstance(raan, (int, float)):
        raise TypeError(f"Expected type of raan is int or float. Got {type(raan)}.")
    if not isinstance(i, (int, float)):
        raise TypeError(f"Expected type of i is int or float. Got {type(i)}.")
    if not isinstance(argp, (int, float)):
        raise TypeError(f"Expected type of argp is int or float. Got {type(argp)}.")

    return np.array(
        [
            [
                cos(raan) * cos(argp) - sin(raan) * sin(argp) * cos(i),
                -cos(raan) * sin(argp) - sin(raan) * cos(i) * cos(argp),
                sin(raan) * sin(i),
            ],
            [
                sin(raan) * cos(argp) + cos(raan) * cos(i) * sin(argp),
                -sin(raan) * sin(argp) + cos(raan) * cos(i) * cos(argp),
                -cos(raan) * sin(i),
            ],
            [sin(i) * sin(argp), sin(i) * cos(argp), cos(i)],
        ]
    )


@tool(args_schema=RotationMatrixECIToPerifocalSchema)
def rotation_matrix_from_ECI_to_perifocal_tool(
    raan: int | float, i: int | float, argp: int | float
) -> np.ndarray:
    """
    Compute the rotation matrix from ECI to perifocal coordinate frame.

    This function calculates the 3x3 rotation matrix that transforms vectors from
    the Earth-Centered Inertial (ECI) coordinate system to the perifocal (PQW)
    coordinate system. This is the inverse/transpose of the perifocal-to-ECI
    transformation.

    Parameters
    ----------
    raan : int or float
        Right Ascension of the Ascending Node (RAAN or Ω) in degrees.
    i : int or float
        Inclination angle in degrees.
    argp : int or float
        Argument of periapsis (ω) in degrees.

    Returns
    -------
    numpy.ndarray
        A 3x3 rotation matrix that transforms ECI coordinates to perifocal coordinates.

    Raises
    ------
    TypeError
        If `raan` is not an int or float.
    TypeError
        If `i` is not an int or float.
    TypeError
        If `argp` is not an int or float.

    Notes
    -----
    This function returns the transpose of the perifocal-to-ECI rotation matrix,
    which is valid because rotation matrices are orthogonal (R⁻¹ = Rᵀ).

    The transformation is:

    .. math::
        \\vec{r}_{PQW} = R_{ECI \\rightarrow PQW} \\cdot \\vec{r}_{ECI}

    This is useful for:
    - Converting state vectors from ECI to orbital elements
    - Analyzing position/velocity in the orbital plane
    - Computing anomalies and orbital parameters

    Examples
    --------
    >>> # Get rotation matrix for ISS-like orbit
    >>> R = rotation_matrix_from_ECI_to_perifocal(raan=45, i=51.6, argp=0)
    >>> print(R.shape)
    (3, 3)

    >>> # Transform a position vector from ECI to perifocal
    >>> r_eci = np.array([5000, 5000, 3000])  # km in ECI
    >>> R = rotation_matrix_from_ECI_to_perifocal(30, 45, 60)
    >>> r_pqw = R @ r_eci
    >>> print(f"Position in PQW: {r_pqw}")

    >>> # Verify it's the transpose of the inverse transformation
    >>> R_pqw_to_eci = rotation_matrix_from_perifocal_to_ECI(30, 45, 60)
    >>> R_eci_to_pqw = rotation_matrix_from_ECI_to_perifocal(30, 45, 60)
    >>> np.allclose(R_eci_to_pqw, R_pqw_to_eci.T)
    True
    """

    if not isinstance(raan, (int, float)):
        raise TypeError(f"Expected type of raan is int or float. Got {type(raan)}.")
    if not isinstance(i, (int, float)):
        raise TypeError(f"Expected type of i is int or float. Got {type(i)}.")
    if not isinstance(argp, (int, float)):
        raise TypeError(f"Expected type of argp is int or float. Got {type(argp)}.")

    return rotation_matrix_from_perifocal_to_ECI(raan, i, argp).T


@tool(args_schema=CrossSchema)
def cross_tool(a: list | np.ndarray, b: list | np.ndarray) -> np.array:
    """
    Calculate the cross product of two vectors.

    This function computes the vector cross product (a × b) of two 3-dimensional
    vectors. The cross product is perpendicular to both input vectors and its
    magnitude equals the area of the parallelogram formed by the vectors.

    Parameters
    ----------
    a : list or numpy.ndarray
        The first vector. Should be a 3-element vector.
    b : list or numpy.ndarray
        The second vector. Should be a 3-element vector.

    Returns
    -------
    numpy.ndarray
        The cross product vector a × b, perpendicular to both input vectors.

    Raises
    ------
    TypeError
        If `a` is not a list or numpy.ndarray.
    TypeError
        If `b` is not a list or numpy.ndarray.

    Notes
    -----
    The cross product is calculated as:

    .. math::
        \\vec{a} \\times \\vec{b} = \\begin{vmatrix}
        \\hat{i} & \\hat{j} & \\hat{k} \\\\
        a_x & a_y & a_z \\\\
        b_x & b_y & b_z
        \\end{vmatrix}

    Which expands to:

    .. math::
        \\vec{a} \\times \\vec{b} = (a_y b_z - a_z b_y)\\hat{i} + (a_z b_x - a_x b_z)\\hat{j} + (a_x b_y - a_y b_x)\\hat{k}

    The magnitude is: ||a × b|| = ||a|| ||b|| sin(θ)

    The cross product follows the right-hand rule for determining direction.

    Examples
    --------
    >>> cross([1, 0, 0], [0, 1, 0])
    array([0, 0, 1])

    >>> import numpy as np
    >>> # Calculate angular momentum h = r × v
    >>> r = np.array([7000, 0, 0])  # km
    >>> v = np.array([0, 7.5, 0])  # km/s
    >>> h = cross(r, v)
    >>> print(f"Angular momentum: {h}")

    >>> # Using lists
    >>> cross([1, 2, 3], [4, 5, 6])
    array([-3,  6, -3])
    """
    if not isinstance(a, (list, np.ndarray)):
        raise TypeError(f"Expected type of a is list or np.ndarray. Got {type(a)}.")
    if not isinstance(b, (list, np.ndarray)):
        raise TypeError(f"Expected type of b is list or np.ndarray. Got {type(b)}.")

    return np.cross(np.asarray(a), np.asarray(b))


@tool(args_schema=DotSchema)
def dot_tool(a: list | np.ndarray, b: list | np.ndarray) -> float:
    """
    Calculate the dot product (scalar product) of two vectors.

    This function computes the dot product (a · b) of two vectors. The dot product
    is a scalar value that relates to the angle between the vectors and their
    magnitudes. It's positive when vectors point in similar directions, zero when
    perpendicular, and negative when pointing in opposite directions.

    Parameters
    ----------
    a : list or numpy.ndarray
        The first vector. Can be of any dimension, but both vectors must have
        the same length.
    b : list or numpy.ndarray
        The second vector. Must have the same length as the first vector.

    Returns
    -------
    float
        The dot product (scalar) of the two vectors.

    Raises
    ------
    TypeError
        If `a` is not a list or numpy.ndarray.
    TypeError
        If `b` is not a list or numpy.ndarray.

    Notes
    -----
    The dot product is calculated as:

    .. math::
        \\vec{a} \\cdot \\vec{b} = a_x b_x + a_y b_y + a_z b_z + ...

    Or equivalently:

    .. math::
        \\vec{a} \\cdot \\vec{b} = ||\\vec{a}|| ||\\vec{b}|| \\cos(\\theta)

    where θ is the angle between the vectors.

    Properties:
    - If a · b > 0: vectors point in similar directions (angle < 90°)
    - If a · b = 0: vectors are perpendicular (angle = 90°)
    - If a · b < 0: vectors point in opposite directions (angle > 90°)

    Examples
    --------
    >>> dot([1, 0, 0], [1, 0, 0])
    1.0

    >>> dot([1, 0, 0], [0, 1, 0])
    0.0

    >>> import numpy as np
    >>> # Calculate work done: W = F · d
    >>> force = np.array([10, 0, 0])  # N
    >>> displacement = np.array([5, 3, 0])  # m
    >>> work = dot(force, displacement)
    >>> print(f"Work done: {work} J")

    >>> # Check if velocity is radial or tangential
    >>> r = [7000, 0, 0]
    >>> v = [0, 7.5, 0]
    >>> radial_component = dot(r, v)
    >>> print(f"Radial velocity component: {radial_component}")
    """
    if not isinstance(a, (list, np.ndarray)):
        raise TypeError(f"Expected type of a is list or np.ndarray. Got {type(a)}.")
    if not isinstance(b, (list, np.ndarray)):
        raise TypeError(f"Expected type of b is list or np.ndarray. Got {type(b)}.")

    return np.dot(np.asarray(a), np.asarray(b))


@tool(args_schema=UUIDGeneratorSchema)
def uuid_generator_tool() -> str:
    """
    Generate a unique UUID (Universally Unique Identifier) as a string.

    This function creates a random UUID version 4 and returns it as a string
    representation. UUIDs are 128-bit identifiers that are practically guaranteed
    to be unique across space and time, making them ideal for generating unique
    identifiers without requiring a central coordination authority.

    Returns
    -------
    str
        A string representation of a UUID4 in the format:
        'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx' where x is a hexadecimal digit
        and y is one of 8, 9, a, or b.

    Notes
    -----
    This function uses Python's uuid.uuid4() which generates random UUIDs based
    on random numbers. The probability of collision is extremely low (approximately
    1 in 2^122).

    UUID4 format:
    - 32 hexadecimal digits
    - Displayed in 5 groups separated by hyphens
    - Total length: 36 characters (32 hex digits + 4 hyphens)

    Common use cases in this library:
    - Generating unique identifiers for orbital elements
    - Creating unique keys for database records
    - Tagging simulation runs or analysis sessions
    - Generating unique IDs for temporary files or objects

    Examples
    --------
    >>> uuid_str = uuid_generator()
    >>> print(uuid_str)
    '550e8400-e29b-41d4-a716-446655440000'  # Example output (will vary)

    >>> # Check format
    >>> uuid_str = uuid_generator()
    >>> len(uuid_str)
    36
    >>> uuid_str.count('-')
    4

    >>> # Generate multiple unique IDs
    >>> ids = [uuid_generator() for _ in range(3)]
    >>> len(set(ids))  # All should be unique
    3

    >>> # Use for tracking orbital simulations
    >>> simulation_id = uuid_generator()
    >>> print(f"Starting simulation {simulation_id}")

    See Also
    --------
    uuid.uuid4 : Python's built-in UUID generation function.
    """
    return str(uuid.uuid4())


@tool(args_schema=JDToDatetimeSchema)
def jd_to_datetime_tool(jd: float) -> datetime:
    """
    Convert Julian Date to a datetime object.

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    datetime
        Corresponding datetime object in UTC.

    Notes
    -----
    - The function accounts for fractional days and converts them to hours, minutes, and seconds.
    - The returned datetime object is timezone-aware and set to UTC.
    """
    jd += 0.5
    F, I = np.modf(jd)
    I = int(I)
    A = int((I - 1867216.25) / 36524.25)
    B = I + 1 + A - int(A / 4)
    C = B + 1524
    D = int((C - 122.1) / 365.25)
    E = int(365.25 * D)
    G = int((C - E) / 30.6001)
    day = C - E - int(30.6001 * G) + F
    month = G - 1 if G < 13.5 else G - 13
    year = D - 4716 if month > 2.5 else D - 4715
    return datetime(year, month, int(day), tzinfo=timezone.utc) + timedelta(
        days=day % 1
    )
