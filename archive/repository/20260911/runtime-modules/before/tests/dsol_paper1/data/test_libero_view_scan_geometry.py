from __future__ import annotations

import math

import pytest

from scripts.dsol_paper1.scan_libero_hdf5_views import (
    _camera_displacement,
    _quaternion_geodesic_deg,
)


def test_quaternion_geodesic_treats_sign_as_equivalent() -> None:
    assert _quaternion_geodesic_deg([1, 0, 0, 0], [-1, 0, 0, 0]) == pytest.approx(
        0.0
    )


def test_quaternion_geodesic_reports_right_angle() -> None:
    half_angle = math.radians(45.0)
    rotated = [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]
    assert _quaternion_geodesic_deg([1, 0, 0, 0], rotated) == pytest.approx(90.0)


def test_camera_displacement_reports_physical_units() -> None:
    canonical = {
        "camera_position": [0.0, 0.0, 0.0],
        "camera_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    candidate = {
        "camera_position": [0.03, 0.04, 0.0],
        "camera_quaternion_wxyz": [0.0, 0.0, 0.0, 1.0],
    }
    displacement = _camera_displacement(canonical, candidate)
    assert displacement["translation_m"] == pytest.approx(0.05)
    assert displacement["rotation_geodesic_deg"] == pytest.approx(180.0)
