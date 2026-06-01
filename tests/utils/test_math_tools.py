import math

import numpy as np
import pytest
import sympy
from numpy.testing import assert_allclose

from src.outcast.utils import math_tools

# --- Tests for Basic Logic & Conversions ---


def test_decision():
    # Edge cases: 0 and 1 probability
    assert math_tools.decision(1.0) is True
    assert math_tools.decision(0.0) is False
    # Check that it returns a boolean
    assert isinstance(math_tools.decision(0.5), bool)


def test_lin_db_conversions():
    # Test lin2db
    assert math_tools.lin2db(1.0) == 0.0
    assert math_tools.lin2db(100.0) == 20.0
    # Test db2lin
    assert math_tools.db2lin(0.0) == 1.0
    assert math_tools.db2lin(20.0) == 100.0
    # Round trip
    val = 15.5
    assert math_tools.db2lin(math_tools.lin2db(val)) == pytest.approx(val)


def test_energy_conversions():
    assert math_tools.wh_to_joules(1) == 3600
    assert math_tools.joules_to_wh(3600) == 1
    assert math_tools.wh_to_joules(0) == 0


# --- Tests for Geometry & Coordinates ---


def test_distance_to_line():
    # ==========================================
    # 1. Test 2D Single Points
    # ==========================================
    p_2d = np.array([0, 1])
    p1_2d = np.array([0, 0])
    p2_2d = np.array([1, 0])
    # Distance from (0,1) to X-axis should be 1.0
    assert math_tools.distance_to_line(p_2d, p1_2d, p2_2d) == pytest.approx(1.0)

    # Diagonal line
    p_diag = np.array([1, 0])
    p1_diag = np.array([0, 0])
    p2_diag = np.array([1, 1])
    # Distance from (1,0) to y=x is 1/sqrt(2)
    expected = 1 / math.sqrt(2)
    assert math_tools.distance_to_line(p_diag, p1_diag, p2_diag) == pytest.approx(
        expected
    )

    # ==========================================
    # 2. Test 2D Batch Processing
    # ==========================================
    p_batch_2d = np.array([[0, 1], [0, 2], [0, 5]])
    # Distances to the X-axis should be exactly the Y-coordinates
    expected_batch_2d = np.array([1.0, 2.0, 5.0])
    result_batch_2d = math_tools.distance_to_line(p_batch_2d, p1_2d, p2_2d)
    assert_allclose(result_batch_2d, expected_batch_2d)

    # ==========================================
    # 3. Test 3D Single Point (Bug fix check)
    # ==========================================
    p_3d = np.array([0, 1, 0])
    p1_3d = np.array([0, 0, 0])
    p2_3d = np.array([1, 0, 0])

    result_3d = math_tools.distance_to_line(p_3d, p1_3d, p2_3d)

    # Guard against the old bug: ensure it returns a float, not an array of shape 3
    assert isinstance(result_3d, (float, np.floating)), (
        f"Expected float, got {type(result_3d)}"
    )
    assert result_3d == pytest.approx(1.0)

    # ==========================================
    # 4. Test 3D Batch Processing
    # ==========================================
    # Points along Y axis and Z axis
    p_batch_3d = np.array([[0, 1, 0], [0, 2, 0], [0, 0, 3]])

    # Distances to the X-axis should be 1.0, 2.0, and 3.0
    expected_batch_3d = np.array([1.0, 2.0, 3.0])
    result_batch_3d = math_tools.distance_to_line(p_batch_3d, p1_3d, p2_3d)
    assert_allclose(result_batch_3d, expected_batch_3d)


def test_meshgrid2():
    x = [1, 2]
    y = [3, 4, 5]
    # Function reverses internal order: (y, x)
    yy_grid, xx_grid = math_tools.meshgrid2(x, y)

    assert xx_grid.shape == (3, 2)
    assert yy_grid.shape == (3, 2)

    # Because of the reversal:
    # xx_grid contains values from x, but is the SECOND output
    # yy_grid contains values from y, but is the FIRST output
    assert yy_grid[0, 0] == 3
    assert xx_grid[0, 0] == 1


def test_line_point_angle():
    start = [0, 0]
    # Test 90 degrees (North)
    res = math_tools.line_point_angle(10, start, angle_x=90)
    assert res[0] == pytest.approx(0, abs=1e-5)
    assert res[1] == pytest.approx(10)

    # Test angle_y branch
    res_y = math_tools.line_point_angle(10, start, angle_y=0)
    # angle_y 0 means sin(0) for x and cos(0) for y -> [0, 10]
    assert res_y == [0.0, 10.0]

    with pytest.raises(ValueError):
        math_tools.line_point_angle(10, start)


def test_get_azimuth():
    # East
    assert math_tools.get_azimuth(0, 0, 1, 0) == 0
    # North
    assert math_tools.get_azimuth(0, 0, 0, 1) == 90
    # West
    assert math_tools.get_azimuth(0, 0, -1, 0) == 180
    # South
    assert math_tools.get_azimuth(0, 0, 0, -1) == 270


def test_point_inside_prlgm():
    # Square from (0,0) to (1,1)
    poly = [[0, 1], [0, 0], [1, 0]]  # Points used to define vectors xb, yb and xc, yc
    # Center point
    assert math_tools.point_inside_prlgm(0.5, 0.5, poly) is True
    # Point outside
    assert math_tools.point_inside_prlgm(1.5, 0.5, poly) is False
    # On boundary
    assert math_tools.point_inside_prlgm(1.0, 1.0, poly) is True


def test_get_mid_azimuth():
    origin = [0, 0]
    p_east = [1, 0]  # azim1 = 0°
    p_north = [0, 1]  # azim2 = 90°

    # Branch check:
    # angle_in_range(0 - 90, 360) = 270.
    # 270 >= 180 is TRUE -> Executes the 'elif' block.
    # Returns: mid, azim11 (180), azim22 (270)
    mid, inv_val_a, inv_val_b = math_tools.get_mid_azimuth(origin, p_east, p_north)

    assert mid == pytest.approx(225)
    assert inv_val_a == 180  # azim1 + 180
    assert inv_val_b == 270  # azim2 + 180

    # Test the 'if' block:
    # azim1 = 90, azim2 = 0.
    # angle_in_range(0 - 90, 360) = 270 >= 180 is FALSE.
    # angle_in_range(90 - 0, 360) = 90 >= 180 is FALSE.
    # Wait—with (90, 0), it hits the 'else' and raises ValueError
    # Let's test a true 'if' block case (azim2 - azim1 >= 180)
    p_south = [0, -1]  # azim2 = 270
    # 270 - 0 = 270 (>= 180) -> hits 'if' block
    # Returns: mid, azim22 (90), azim11 (180)
    mid_if, inv_azim22, inv_azim11 = math_tools.get_mid_azimuth(origin, p_east, p_south)
    assert mid_if == pytest.approx(135)
    assert inv_azim22 == 90
    assert inv_azim11 == 180


# --- Tests for Utilities & Solvers ---


def test_numpy_object_array():
    # Single object
    res = math_tools.numpy_object_array([10])
    assert res.dtype == object or res.dtype == int
    assert res[0] == [10]

    # Sequence
    res_seq = math_tools.numpy_object_array([1, 2, 3])
    assert len(res_seq) == 3
    assert res_seq[1] == 2


def test_rotate():
    data = [1, 2, 3, 4]
    assert math_tools.rotate(data, 1) == [2, 3, 4, 1]
    assert math_tools.rotate(data, 0) == data
    assert math_tools.rotate(data, 4) == data


def test_newton_raphson():
    x = sympy.Symbol("x")
    f = x**2 - 4
    f_prime = 2 * x

    root, error = math_tools.newton_raphson(f, f_prime, x, initial_guess=5)
    assert root == pytest.approx(2.0)
    assert error == pytest.approx(0.0, abs=1e-10)
