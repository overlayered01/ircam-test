import math

import cv2
import numpy as np
import pytest

from femto_fov.fov_math import (
    Intrinsics,
    border_pixels,
    focal_from_checkerboard,
    fov_from_focal,
    fov_from_physical,
    fov_pinhole,
    fov_undistorted,
    footprint_at_distance,
    overlap_ratio,
    project_between,
    undistort_to_normalized,
    valid_region_fov,
)


def test_pinhole_90deg():
    intr = Intrinsics(1000, 500, 500, 500, 500, 250)
    f = fov_pinhole(intr)
    assert f.h == pytest.approx(90.0)
    assert f.v == pytest.approx(2 * math.degrees(math.atan(0.5)))
    assert f.d == pytest.approx(2 * math.degrees(math.atan(math.hypot(1, 0.5))))


def test_pinhole_off_center_principal_point():
    centered = fov_pinhole(Intrinsics(1000, 500, 500, 500, 500, 250))
    shifted = fov_pinhole(Intrinsics(1000, 500, 500, 500, 600, 250))
    # 주점이 치우치면 좌/우 합이 줄어든다 (atan 비선형)
    assert shifted.h < centered.h


def test_undistorted_equals_pinhole_without_distortion():
    intr = Intrinsics(640, 576, 504, 504, 320, 288)
    assert fov_undistorted(intr).h == pytest.approx(fov_pinhole(intr).h)


def test_barrel_distortion_widens_fov():
    base = Intrinsics(640, 576, 504, 504, 320, 288)
    barrel = Intrinsics(640, 576, 504, 504, 320, 288, (-0.25, 0.05, 0, 0, 0, 0, 0, 0))
    assert fov_undistorted(barrel).h > fov_pinhole(base).h + 1.0


def test_undistort_roundtrip_rational():
    intr = Intrinsics(1024, 1024, 252, 252, 512, 512, (0.5, 0.1, 0.0, 0.0, 0.0, 0.8, 0.2, 0.02))
    n = np.array([[0.3, -0.2], [1.2, 0.4], [-1.0, -1.0]])
    pts3 = np.c_[n, np.ones(len(n))]
    px, _ = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), intr.K, intr.D)
    back = undistort_to_normalized(px.reshape(-1, 2), intr)
    np.testing.assert_allclose(back, n, atol=1e-6)


def test_undistort_roundtrip_fisheye():
    intr = Intrinsics(1024, 1024, 300, 300, 512, 512, (0.05, -0.01, 0.002, 0.0, 0, 0, 0, 0), "KANNALA_BRANDT4")
    n = np.array([[0.3, -0.2], [1.5, 0.4]])
    pts3 = np.c_[n, np.ones(len(n))].reshape(-1, 1, 3)
    px, _ = cv2.fisheye.projectPoints(pts3, np.zeros(3), np.zeros(3), intr.K, intr.D)
    back = undistort_to_normalized(px.reshape(-1, 2), intr)
    np.testing.assert_allclose(back, n, atol=1e-6)


def test_project_identity():
    intr = Intrinsics(640, 480, 500, 500, 320, 240)
    px = border_pixels(640, 480, 40)
    out = project_between(px, 1000, intr, intr, np.eye(3), np.zeros(3))
    np.testing.assert_allclose(out, px, atol=1e-6)


def test_project_baseline_shift():
    intr = Intrinsics(640, 480, 500, 500, 320, 240)
    out = project_between(np.array([[320.0, 240.0]]), 1000, intr, intr, np.eye(3), np.array([-32.0, 0, 0]))
    # 시차 = f · t / Z = 500 · 32 / 1000 = 16 px
    assert out[0, 0] == pytest.approx(320 - 16)


def test_overlap_full_and_half():
    assert overlap_ratio(np.array([[0, 0], [640, 0], [640, 480], [0, 480]]), 640, 480) == pytest.approx((1.0, 1.0))
    cov, inside = overlap_ratio(np.array([[320, 0], [960, 0], [960, 480], [320, 480]]), 640, 480)
    assert cov == pytest.approx(0.5)
    assert inside == pytest.approx(0.5)


def test_focal_from_synthetic_checkerboard():
    cols, rows, sq, z = 9, 6, 25.0, 800.0
    fx, fy = 610.0, 605.0
    K = np.array([[fx, 0, 640], [0, fy, 360], [0, 0, 1]])
    obj = np.zeros((rows * cols, 3))
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
    obj[:, 0] -= obj[:, 0].mean()
    obj[:, 1] -= obj[:, 1].mean()
    obj[:, 2] = z
    px, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, None)
    efx, efy, tilt = focal_from_checkerboard(px.reshape(-1, 2), cols, rows, sq, np.full(rows * cols, z))
    assert efx == pytest.approx(fx, rel=1e-6)
    assert efy == pytest.approx(fy, rel=1e-6)
    assert tilt == pytest.approx(0.0)


def test_physical_and_footprint_roundtrip():
    f = fov_from_physical(1000, 2000, 1000)
    assert f.h == pytest.approx(90.0)
    w, h = footprint_at_distance(f, 1000)
    assert w == pytest.approx(2000)
    assert h == pytest.approx(1000)


def test_fov_from_focal_matches_pinhole_centered():
    assert fov_from_focal(640, 576, 504, 504).h == pytest.approx(fov_pinhole(Intrinsics(640, 576, 504, 504, 320, 288)).h)


def test_valid_region_full_and_circular():
    intr = Intrinsics(512, 512, 128, 128, 256, 256)
    full = valid_region_fov(np.ones((512, 512), bool), intr)
    assert full.h == pytest.approx(fov_pinhole(intr).h, abs=0.01)
    yy, xx = np.mgrid[0:512, 0:512]
    circle = (xx - 255.5) ** 2 + (yy - 255.5) ** 2 <= 200**2
    c = valid_region_fov(circle, intr)
    assert c.h < full.h
    # 원형 마스크: 대각 범위 ≈ 수평 범위
    assert c.d == pytest.approx(c.h, abs=1.0)
