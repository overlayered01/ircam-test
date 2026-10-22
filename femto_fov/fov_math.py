"""화각(FOV) 계산용 순수 수학 함수 모음.

SDK에 의존하지 않으므로 장치 없이 단위 테스트할 수 있다.
모든 각도는 degree, 거리는 mm 단위를 사용한다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # OpenCV 순서: k1, k2, p1, p2, k3, k4, k5, k6 (Femto 는 Brown-Conrady K6 = OpenCV rational 모델)
    dist: tuple[float, ...] = (0.0,) * 8
    # "KANNALA_BRANDT4" 이면 fisheye 모델 (dist 앞 4개 = k1..k4)
    model: str = "BROWN_CONRADY_K6"

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def scaled(self, s: float) -> "Intrinsics":
        """이미지를 s 배 리사이즈했을 때의 내부 파라미터 (왜곡 계수는 정규화 좌표라 그대로)."""
        return Intrinsics(
            round(self.width * s), round(self.height * s),
            self.fx * s, self.fy * s, self.cx * s, self.cy * s, self.dist, self.model,
        )

    @property
    def is_fisheye(self) -> bool:
        return self.model == "KANNALA_BRANDT4"

    @property
    def D(self) -> np.ndarray:
        if self.is_fisheye:
            return np.array(self.dist[:4], dtype=np.float64)
        return np.array(self.dist, dtype=np.float64)

    def has_distortion(self) -> bool:
        return any(abs(v) > 1e-12 for v in self.dist)


@dataclass
class Fov:
    h: float
    v: float
    d: float

    def as_dict(self) -> dict:
        return {k: round(v, 2) for k, v in asdict(self).items()}

    def __str__(self) -> str:
        return f"H {self.h:6.2f}°  V {self.v:6.2f}°  D {self.d:6.2f}°"

    def short(self) -> str:
        """OpenCV 기본 폰트용 (° 를 그릴 수 없음)."""
        return f"H {self.h:.1f}  V {self.v:.1f}  D {self.d:.1f} deg"


def fov_pinhole(intr: Intrinsics) -> Fov:
    """왜곡을 무시한 핀홀 모델 FOV. 주점(cx, cy)이 중앙이 아니어도 좌/우를 따로 계산한다."""
    left = math.atan(intr.cx / intr.fx)
    right = math.atan((intr.width - intr.cx) / intr.fx)
    top = math.atan(intr.cy / intr.fy)
    bottom = math.atan((intr.height - intr.cy) / intr.fy)
    h = math.degrees(left + right)
    v = math.degrees(top + bottom)
    d = _diag_angle(
        np.array([-intr.cx / intr.fx, -intr.cy / intr.fy]),
        np.array([(intr.width - intr.cx) / intr.fx, (intr.height - intr.cy) / intr.fy]),
    )
    return Fov(h, v, d)


def fov_undistorted(intr: Intrinsics) -> Fov:
    """렌즈 왜곡을 반영한 실제 FOV.

    이미지 경계 픽셀(좌/우 중앙, 상/하 중앙, 대각 모서리)을 왜곡 보정해 정규화 좌표로 바꾼 뒤
    광선 사이 각도를 구한다. 광각(WFOV) 모드처럼 왜곡이 큰 경우 핀홀 값과 크게 다르다.
    """
    if not intr.has_distortion():
        return fov_pinhole(intr)
    w, h = intr.width, intr.height
    px = np.array(
        [
            [0.0, intr.cy],  # left
            [w, intr.cy],  # right
            [intr.cx, 0.0],  # top
            [intr.cx, h],  # bottom
            [0.0, 0.0],  # top-left
            [w, h],  # bottom-right
        ]
    )
    n = undistort_to_normalized(px, intr)
    hfov = _ray_angle(n[0], n[1])
    vfov = _ray_angle(n[2], n[3])
    dfov = _ray_angle(n[4], n[5])
    return Fov(hfov, vfov, dfov)


def undistort_to_normalized(px: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """픽셀 좌표 (N,2) → 왜곡 보정된 정규화 좌표 (N,2). 큰 왜곡에서도 수렴하도록 반복 횟수를 늘린다."""
    pts = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)
    if intr.is_fisheye:
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-10)
        return cv2.fisheye.undistortPoints(pts, intr.K, intr.D, criteria=criteria).reshape(-1, 2)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-10)
    if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4.x
        out = cv2.undistortPointsIter(pts, intr.K, intr.D, None, None, criteria)
    else:  # OpenCV 5.x: undistortPoints 가 criteria 를 받는다
        out = cv2.undistortPoints(pts, intr.K, intr.D, R=None, P=None, criteria=criteria)
    return out.reshape(-1, 2)


def pixels_to_angles(px: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """픽셀 좌표 (N,2) → 광축 기준 (수평각, 수직각) degree (N,2)."""
    n = undistort_to_normalized(px, intr)
    return np.degrees(np.arctan(n))


def _ray_angle(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.array([a[0], a[1], 1.0])
    rb = np.array([b[0], b[1], 1.0])
    cos = np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb))
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def _diag_angle(a: np.ndarray, b: np.ndarray) -> float:
    return _ray_angle(a, b)


def fov_from_focal(width: int, height: int, fx: float, fy: float) -> Fov:
    """주점이 중앙이라고 가정하고 초점거리만으로 FOV 계산 (실측 추정치용)."""
    return fov_pinhole(Intrinsics(width, height, fx, fy, width / 2, height / 2))


def fov_from_physical(distance_mm: float, width_mm: float, height_mm: float | None = None) -> Fov:
    """줄자 실측: 카메라에서 distance 만큼 떨어진 평면에서 화면 끝~끝 폭(높이)을 쟀을 때의 FOV."""
    h = math.degrees(2 * math.atan(width_mm / 2 / distance_mm))
    if height_mm is None:
        return Fov(h, float("nan"), float("nan"))
    v = math.degrees(2 * math.atan(height_mm / 2 / distance_mm))
    d = math.degrees(2 * math.atan(math.hypot(width_mm, height_mm) / 2 / distance_mm))
    return Fov(h, v, d)


def footprint_at_distance(fov: Fov, distance_mm: float) -> tuple[float, float]:
    """주어진 거리에서 화면이 덮는 폭/높이 (mm)."""
    w = 2 * distance_mm * math.tan(math.radians(fov.h) / 2)
    h = 2 * distance_mm * math.tan(math.radians(fov.v) / 2)
    return w, h


# ---------------------------------------------------------------------------
# 스트림 간 투영 (depth/IR 화각 경계를 color 이미지 위에 그리기 등)
# ---------------------------------------------------------------------------


def border_pixels(width: int, height: int, step: int = 8) -> np.ndarray:
    """이미지 테두리를 시계방향으로 따라가는 픽셀 좌표 (N,2)."""
    xs = np.arange(0, width + 1, step, dtype=np.float64)
    ys = np.arange(0, height + 1, step, dtype=np.float64)
    top = np.stack([xs, np.zeros_like(xs)], 1)
    right = np.stack([np.full_like(ys, width), ys], 1)
    bottom = np.stack([xs[::-1], np.full_like(xs, height)], 1)
    left = np.stack([np.zeros_like(ys), ys[::-1]], 1)
    return np.concatenate([top, right, bottom, left])


def project_between(
    px: np.ndarray,
    depth_mm: float | np.ndarray,
    src: Intrinsics,
    dst: Intrinsics,
    R: np.ndarray,
    t_mm: np.ndarray,
) -> np.ndarray:
    """src 카메라 픽셀을 거리 depth_mm(Z) 평면에 놓고 dst 카메라 픽셀로 투영한다.

    R, t 는 src → dst 변환 (P_dst = R · P_src + t).
    """
    n = undistort_to_normalized(px, src)
    z = np.broadcast_to(np.asarray(depth_mm, dtype=np.float64), (len(n),))
    pts3 = np.stack([n[:, 0] * z, n[:, 1] * z, z], 1)
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    tvec = np.asarray(t_mm, dtype=np.float64)
    if dst.is_fisheye:
        out, _ = cv2.fisheye.projectPoints(pts3.reshape(-1, 1, 3), rvec, tvec, dst.K, dst.D)
    else:
        out, _ = cv2.projectPoints(pts3, rvec, tvec, dst.K, dst.D)
    return out.reshape(-1, 2)


def project_border(
    src: Intrinsics, dst: Intrinsics, R: np.ndarray, t_mm: np.ndarray, depth_mm: float, step: int = 8
) -> np.ndarray:
    return project_between(border_pixels(src.width, src.height, step), depth_mm, src, dst, R, t_mm)


def overlap_ratio(polygon: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """투영된 다각형과 dst 이미지의 겹침 비율.

    반환: (dst 이미지 중 polygon 이 덮는 비율, polygon 중 dst 이미지 안에 들어오는 비율)
    """
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    img = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    poly_area = abs(cv2.contourArea(poly))
    if poly_area <= 0:
        return 0.0, 0.0
    hull = cv2.convexHull(poly)
    inter_area, _ = cv2.intersectConvexConvex(hull, img)
    return float(inter_area) / (width * height), float(inter_area) / float(cv2.contourArea(hull))


# ---------------------------------------------------------------------------
# 실측 (체커보드)
# ---------------------------------------------------------------------------


def focal_from_checkerboard(
    corners: np.ndarray, cols: int, rows: int, square_mm: float, depth_per_corner_mm: np.ndarray
) -> tuple[float, float, float]:
    """체커보드 코너와 코너별 거리(Z)로 fx, fy 를 추정한다 (보드가 정면을 향한다고 가정).

    인접 코너 간 픽셀 거리 d 와 실제 거리 s, 깊이 Z 로 f = d · Z / s.
    반환: (fx, fy, 보드 기울기 지표 = Z 최대-최소 / Z 평균)
    """
    c = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    z = np.asarray(depth_per_corner_mm, dtype=np.float64).reshape(rows, cols)
    zok = (z > 0) & np.isfinite(z)
    if zok.mean() < 0.5:
        raise ValueError("체커보드 위치의 깊이 값이 없습니다")
    zv = z[zok]
    # 인접 코너 대신 행/열 양 끝 코너 간 거리를 쓴다 (코너 위치 오차가 칸 수만큼 희석됨)
    fxs = [
        np.linalg.norm(c[r, -1] - c[r, 0]) * np.median(z[r][zok[r]]) / ((cols - 1) * square_mm)
        for r in range(rows)
        if zok[r].any()
    ]
    fys = [
        np.linalg.norm(c[-1, k] - c[0, k]) * np.median(z[:, k][zok[:, k]]) / ((rows - 1) * square_mm)
        for k in range(cols)
        if zok[:, k].any()
    ]
    fx = float(np.median(fxs))
    fy = float(np.median(fys))
    tilt = float((zv.max() - zv.min()) / zv.mean())
    return fx, fy, tilt


def valid_region_fov(mask: np.ndarray, intr: Intrinsics) -> Fov:
    """깊이 유효 픽셀 마스크의 실제 각도 범위.

    광축을 지나는 행/열에서 유효 픽셀의 양 끝을 찾아 각도로 변환한다.
    WFOV 처럼 유효 영역이 원형/육각형이면 센서 사각형 FOV 보다 작게 나온다.
    """
    m = np.asarray(mask, dtype=bool)
    row = int(round(intr.cy))
    col = int(round(intr.cx))
    xs = np.flatnonzero(m[row])
    ys = np.flatnonzero(m[:, col])
    if len(xs) == 0 or len(ys) == 0:
        return Fov(float("nan"), float("nan"), float("nan"))
    pts = np.array(
        [
            [xs[0], intr.cy],
            [xs[-1] + 1, intr.cy],
            [intr.cx, ys[0]],
            [intr.cx, ys[-1] + 1],
        ],
        dtype=np.float64,
    )
    n = undistort_to_normalized(pts, intr)
    # 대각: 주점을 지나는 대각선 방향으로 유효 영역 끝 탐색
    diag = _diag_extent(m, intr)
    return Fov(_ray_angle(n[0], n[1]), _ray_angle(n[2], n[3]), diag)


def _diag_extent(m: np.ndarray, intr: Intrinsics) -> float:
    h, w = m.shape
    ends = []
    for sx, sy in ((-1, -1), (1, 1)):
        last = None
        for k in range(0, max(w, h)):
            x = int(round(intr.cx + sx * k * w / math.hypot(w, h)))
            y = int(round(intr.cy + sy * k * h / math.hypot(w, h)))
            if not (0 <= x < w and 0 <= y < h):
                break
            if m[y, x]:
                last = (x, y)
        if last is None:
            return float("nan")
        ends.append(last)
    n = undistort_to_normalized(np.array(ends, dtype=np.float64), intr)
    return _ray_angle(n[0], n[1])
