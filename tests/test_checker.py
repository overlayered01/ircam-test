"""체커보드 검출 → 초점거리 추정 전 과정을 합성 이미지로 검증 (장치 불필요)."""

import cv2
import numpy as np
import pytest

pytest.importorskip("pyorbbecsdk")

from fov_measure import detect_board  # noqa: E402
from femto_fov.fov_math import focal_from_checkerboard, fov_from_focal  # noqa: E402


def render_board(w, h, fx, fy, z, cols, rows, square, offset=(0.0, 0.0)):
    """정면을 향한 체커보드를 핀홀 카메라로 렌더링. cols/rows 는 내부 코너 수."""
    px_per_mm = 4
    nx, ny = cols + 1, rows + 1
    bw, bh = nx * square, ny * square
    margin = square
    tex = np.full((int((bh + 2 * margin) * px_per_mm), int((bw + 2 * margin) * px_per_mm)), 255, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                x0, y0 = int((margin + i * square) * px_per_mm), int((margin + j * square) * px_per_mm)
                tex[y0 : y0 + int(square * px_per_mm), x0 : x0 + int(square * px_per_mm)] = 0
    # 텍스처 픽셀 → 보드 평면 mm (보드 중심 원점) → 이미지 픽셀
    cx, cy = w / 2, h / 2
    sx, sy = fx / z / px_per_mm, fy / z / px_per_mm
    ox = cx + fx * offset[0] / z - (bw / 2 + margin) * fx / z
    oy = cy + fy * offset[1] / z - (bh / 2 + margin) * fy / z
    M = np.array([[sx, 0, ox], [0, sy, oy]], np.float64)
    img = cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_AREA, borderValue=200)
    return img


@pytest.mark.parametrize("w,h,fx", [(1280, 720, 749.2), (640, 576, 504.5), (3840, 2160, 2247.7)])
def test_detect_and_estimate_focal(w, h, fx):
    cols, rows, sq, z = 9, 6, 30.0, 900.0
    img = render_board(w, h, fx, fx, z, cols, rows, sq, offset=(40, -20))
    corners = detect_board(img, cols, rows)
    assert corners is not None
    efx, efy, tilt = focal_from_checkerboard(corners, cols, rows, sq, np.full(cols * rows, z))
    assert efx == pytest.approx(fx, rel=0.01)
    assert efy == pytest.approx(fx, rel=0.01)
    true = fov_from_focal(w, h, fx, fx)
    est = fov_from_focal(w, h, efx, efy)
    assert est.h == pytest.approx(true.h, abs=0.5)
