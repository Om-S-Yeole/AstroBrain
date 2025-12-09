from datetime import datetime
from math import cos, sin

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
    Convert a datetime object to Julian Date components.

    This function converts a Python datetime object to Julian Date format, which
    is commonly used in astronomy and astrodynamics. The Julian Date is returned
    as both a complete value and separated into whole and fractional parts for
    precision in calculations.

    Parameters
    ----------
    dt : datetime.datetime
        The datetime object to convert. Should include year, month, day, hour,
        minute, and second components.

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

    Examples
    --------
    >>> from datetime import datetime
    >>> dt = datetime(2025, 1, 1, 12, 0, 0)
    >>> jd, whole, frac = datetime_to_jd(dt)
    >>> print(f"Julian Date: {jd}")
    >>> print(f"Whole part: {whole}, Fractional part: {frac}")

    >>> # Convert current time
    >>> now = datetime.now()
    >>> jd, _, _ = datetime_to_jd(now)
    >>> print(f"Current JD: {jd}")
    """
    whole_part, frac_part = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )
    julian_day = whole_part + frac_part
    return julian_day, whole_part, frac_part


def deg2rad(angle_deg: int | float) -> float:
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


def rad2deg(angle_rad: int | float) -> float:
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


def norm(vec: list | np.ndarray) -> float:
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


def unit_vec(vec: list | np.ndarray) -> np.ndarray:
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


def rotation_matrix_from_perifocal_to_ECI(
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


def rotation_matrix_from_ECI_to_perifocal(
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


def cross(a: list | np.ndarray, b: list | np.ndarray) -> np.array:
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


def dot(a: list | np.ndarray, b: list | np.ndarray) -> float:
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
