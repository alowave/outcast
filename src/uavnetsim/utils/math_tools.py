"""Mathematical utility functions for UAV simulation.

This module provides various mathematical tools including:
- Distance calculations
- Unit conversions (linear/dB, watts/dBm, etc.)
- Coordinate transformations
- Geometric calculations
- Root-finding algorithms
"""

import random
from typing import Any, Sequence, Union

import numpy as np
from numpy import arctan2, degrees
from sympy import Expr, Symbol


def decision(probability: float) -> bool:
    """Determine a boolean outcome based on a probability.

    Args:
        probability: A value between 0 and 1.

    Returns:
        True if the random roll is less than the probability, False otherwise.
    """
    return random.random() < probability


def distance_to_line(
    p: Union[np.ndarray, list], p1: Union[np.ndarray, list], p2: Union[np.ndarray, list]
) -> Union[float, np.ndarray]:
    """Calculate the shortest perpendicular distance from a point (or points) to a line.

    This function supports both single points and batches of points in 2D and 3D space.
    It avoids NumPy 2.0 deprecation warnings for 2D cross products by calculating
    the 2D magnitude manually.

    Args:
        p: Coordinates of the point or points to measure.
        p1: First point defining the line. Must match the inner dimension of `p`.
        p2: Second point defining the line. Must match the inner dimension of `p`.

    Returns:
        The perpendicular distance or distances to the line. Returns a single float
        for one point and a 1D NumPy array for a batch.
    """
    p = np.asarray(p)
    v = np.asarray(p2) - np.asarray(p1)
    w = p - np.asarray(p1)

    if p.shape[-1] == 2:
        # 2D Case: Manual cross product (scalar magnitude)
        dist_numerator = np.abs(v[0] * w[..., 1] - v[1] * w[..., 0])
    else:
        # 3D Case: Magnitude of the cross product vector
        cross_product = np.cross(v, w)
        dist_numerator = np.linalg.norm(cross_product, axis=-1)

    return dist_numerator / np.linalg.norm(v)


def meshgrid2(*arrs: Sequence[Any]) -> tuple:
    """Create a coordinate meshgrid from multiple 1D arrays with reversed order.

    Args:
        *arrs: A sequence of 1D arrays.

    Returns:
        A tuple of N-dimensional coordinate arrays.
    """
    arrs = tuple(reversed(arrs))  # edit
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return tuple(ans)


def lin2db(linear_input: float) -> np.float64:
    """Convert a linear power ratio to decibels (dB).

    Args:
        linear_input: The linear power value.

    Returns:
        The value in dB.
    """
    return 10 * np.log10(linear_input)


def db2lin(db_input: float) -> float:
    """Convert a decibel (dB) value to a linear power ratio.

    Args:
        db_input: The value in dB.

    Returns:
        The linear power ratio.
    """
    return 10 ** (db_input / 10)


def w_to_dbm(power_w: float) -> float:
    """Convert power in watts to dBm.

    Args:
        power_w: Power in watts.

    Returns:
        Power in dBm.
    """
    return float(lin2db(power_w * 1e3))


def numpy_object_array(_object: Sequence[Any]) -> np.ndarray:
    """Convert a sequence into a 1D numpy object array.

    Args:
        _object: The input object or sequence to convert.

    Returns:
        A 1D NumPy array containing the object or objects.
    """
    obj_len = len(_object)
    object_type = type(_object[0]) if obj_len > 1 else type(_object)
    arr = np.empty(obj_len, dtype=object_type)
    if obj_len < 2:
        arr[0] = _object
        return arr
    else:
        for i in range(obj_len):
            arr[i] = _object[i]
        return arr


def rotate(inlist: list, n: int) -> list:
    """Rotate a list by n positions.

    Args:
        inlist: The input list.
        n: The number of positions to shift.

    Returns:
        The rotated list.
    """
    return inlist[n:] + inlist[:n]


# Unknown types, no usages found. Add types later.
def line_point_angle(
    length: float,
    point: Sequence[Any],
    angle_x: float = None,
    angle_y: float = None,
    rounding: bool = True,
):
    """Get endpoint from starting point, distance, and azimuth.

    Args:
        length: The distance to the new point.
        point: The starting `[x, y]` coordinates.
        angle_x: Angle in degrees from the x-axis.
        angle_y: Angle in degrees from the y-axis.
        rounding: Whether to round the trigonometric results to 5 decimals.

    Returns:
        A list `[x, y]` representing the destination point.

    Raises:
        ValueError: If neither `angle_x` nor `angle_y` is provided.
    """
    dest = [0, 0]
    if angle_x is not None:
        if rounding:
            dest[0] = point[0] + length * np.round(
                np.cos(angle_x / 180 * np.pi), decimals=5
            )
            dest[1] = point[1] + length * np.round(
                np.sin(angle_x / 180 * np.pi), decimals=5
            )
        else:
            dest[0] = point[0] + length * np.cos(angle_x / 180 * np.pi)
            dest[1] = point[1] + length * np.sin(angle_x / 180 * np.pi)
        return dest

    # BUG: angle_x is none, we still perform an operation? Possibly angle_x -> angle_y?
    elif angle_y is not None:
        dest[0] = point[0] + length * np.round(
            np.sin(angle_y / 180 * np.pi), decimals=5
        )
        dest[1] = point[1] + length * np.round(
            np.cos(angle_y / 180 * np.pi), decimals=5
        )
        return dest
    else:
        raise ValueError("No angle was provided!")


def point_inside_prlgm(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    """Check if a point (x, y) is inside a parallelogram defined by its vertices.

    Args:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
        poly: A list of vertex coordinates `[[x, y], [x, y], [x, y]]`.

    Returns:
        True if the point is inside, False otherwise.
    """
    inside = False
    xb = poly[0][0] - poly[1][0]
    yb = poly[0][1] - poly[1][1]
    xc = poly[2][0] - poly[1][0]
    yc = poly[2][1] - poly[1][1]
    xp = x - poly[1][0]
    yp = y - poly[1][1]
    d = xb * yc - yb * xc
    if d != 0:
        oned = 1.0 / d
        bb = (xp * yc - xc * yp) * oned
        cc = (xb * yp - xp * yb) * oned
        inside = (bb >= 0) & (cc >= 0) & (bb <= 1) & (cc <= 1)
    return inside


def get_mid_azimuth(
    p_origin: Sequence[float], p1: Sequence[float], p2: Sequence[float]
) -> tuple:
    """Calculate the middle azimuth between two points relative to an origin.

    Args:
        p_origin: The origin point `[x, y]`.
        p1: First target point `[x, y]`.
        p2: Second target point `[x, y]`.

    Returns:
        A tuple containing `(mid_azimuth, inverse_azim2, inverse_azim1)`.

    Raises:
        ValueError: If the polygon assumption is violated.
    """
    azim1 = get_azimuth(p_origin[0], p_origin[1], p1[0], p1[1])
    azim2 = get_azimuth(p_origin[0], p_origin[1], p2[0], p2[1])
    azim11 = angle_in_range(azim1 + 180, 360)
    azim22 = angle_in_range(azim2 + 180, 360)
    if angle_in_range(azim2 - azim1, 360) >= 180:
        return (
            angle_in_range(angle_in_range(azim2 - azim1, 360) / 2 + azim1, 360),
            azim22,
            azim11,
        )
    elif angle_in_range(azim1 - azim2, 360) >= 180:
        return (
            angle_in_range(angle_in_range(azim1 - azim2, 360) / 2 + azim2, 360),
            azim11,
            azim22,
        )
    else:
        raise ValueError("Ensure that polygon is convex!")


def get_azimuth(center_x: float, center_y: float, x: float, y: float) -> float:
    """Calculate the azimuth from a center point to a target point.

    Args:
        center_x: X-coordinate of the origin.
        center_y: Y-coordinate of the origin.
        x: X-coordinate of the target.
        y: Y-coordinate of the target.

    Returns:
        Bearing in degrees in the range `[0, 360)`.
    """
    angle = degrees(arctan2(y - center_y, x - center_x))
    bearing = (angle + 360) % 360
    return bearing


def angle_in_range(angle: float, max_angle: float) -> float:
    """Limit angle to a given range using modulo.

    Args:
        angle: The input angle.
        max_angle: The upper bound of the range, for example `360`.

    Returns:
        The normalized angle within `[0, max_angle)`.
    """
    angle = angle % max_angle
    return (angle + max_angle) % max_angle


def wh_to_joules(wh: float) -> float:
    """Convert Watt-hours to Joules.

    Args:
        wh: Energy in Watt-hours.

    Returns:
        Energy in Joules.
    """
    return wh * 3600


def joules_to_wh(joules: float) -> float:
    """Convert Joules to Watt-hours.

    Args:
        joules: Energy in Joules.

    Returns:
        Energy in Watt-hours.
    """
    return joules / 3600


def newton_raphson(
    f: Expr,
    f_derivative: Expr,
    variable: Symbol,
    initial_guess: float = 1,
    max_error: float = 1e-15,
    max_iter: int = 1000,
) -> tuple:
    """Find the root of a function using the Newton-Raphson method.

    Args:
        f: The SymPy function object.
        f_derivative: The derivative of the function.
        variable: The SymPy variable to solve for.
        initial_guess: The starting value for the iteration.
        max_error: The convergence threshold.
        max_iter: Maximum number of iterations allowed.

    Returns:
        A tuple `(root_estimate, final_error)`.
    """
    xn = initial_guess
    error = 10
    i = 0
    while error > max_error and i < max_iter:
        xn = xn - float(f.evalf(subs={variable: xn})) / float(
            f_derivative.evalf(subs={variable: xn})
        )
        error = float(f.evalf(subs={variable: xn}))
        i += 1

    return xn, float(f.evalf(subs={variable: xn}))
