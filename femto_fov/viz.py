"""시각화 도우미: depth/IR 컬러맵, FOV 경계·각도 격자 오버레이, 패널 배치."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .fov_math import Intrinsics

GREEN = (80, 220, 80)
CYAN = (230, 200, 40)
MAGENTA = (220, 80, 220)
YELLOW = (40, 220, 240)
WHITE = (255, 255, 255)


@lru_cache(maxsize=4)
def _depth_luts(near: float, far: float) -> tuple[np.ndarray, np.ndarray]:
    """mm(uint16) → 색 인덱스(0=무효, 1..255), 인덱스 → BGR (0 은 검정)."""
    d = np.arange(65536, dtype=np.float32)
    n = np.clip((d - near) / (far - near), 0, 1)
    idx = (1 + (1 - n) * 254).astype(np.uint8)
    idx[0] = 0
    cmap = cv2.applyColorMap(((np.arange(256) - 1).clip(0) * 255 // 254).astype(np.uint8).reshape(-1, 1),
                             cv2.COLORMAP_JET)
    cmap[0] = 0
    return idx, cmap.reshape(256, 1, 3)


def colorize_depth(depth_mm: np.ndarray, near: float = 250, far: float = 5000) -> np.ndarray:
    """룩업 테이블 방식 (부동소수 연산 없이 1024x1024 도 수 ms)."""
    d = depth_mm if depth_mm.dtype == np.uint16 else np.clip(depth_mm, 0, 65535).astype(np.uint16)
    idx, cmap = _depth_luts(float(near), float(far))
    return cv2.applyColorMap(idx[d], cmap)


def normalize_ir(ir: np.ndarray) -> np.ndarray:
    """ToF active IR 은 밝기 범위가 넓어(가까운 물체 포화) 제곱근으로 압축한 뒤 정규화한다."""
    v = np.sqrt(ir.astype(np.float32))
    sample = v[::4, ::4]  # 백분위는 샘플로 충분
    hi = np.percentile(sample, 99) if sample.any() else 1.0
    out = cv2.convertScaleAbs(v, alpha=255.0 / max(hi, 1.0))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def fit_scale(w: int, h: int, cell_w: int, cell_h: int) -> float:
    """셀에 맞추는 축소 비율 (1 이하). 확대는 grid() 의 fit 에서 한다."""
    return min(1.0, cell_w / w, cell_h / h)


def shrink(img: np.ndarray, s: float, nearest: bool = False) -> np.ndarray:
    """표시 크기로 축소 (확대는 하지 않음). depth 는 값이 섞이지 않도록 nearest."""
    if s >= 1.0:
        return img.copy()
    h, w = img.shape[:2]
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def draw_polygon(img: np.ndarray, pts: np.ndarray, color, thickness: int = 2, closed: bool = True) -> None:
    p = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], closed, color, thickness, cv2.LINE_AA)


def draw_angle_grid(img: np.ndarray, intr: Intrinsics, step_deg: int = 10, color=WHITE) -> None:
    """광축 기준 수평/수직 각도 격자. 렌즈 왜곡을 반영해 곡선으로 그린다."""
    k = max(1.0, img.shape[1] / 800)  # 화면에 축소 표시돼도 보이도록 해상도에 비례
    for pts, thick, label, org in _angle_grid_lines(_intr_key(intr), step_deg):
        draw_polygon(img, pts, color, int(thick * k), closed=False)
        if label:
            put_text(img, label, org, 0.45 * k, color)


def _intr_key(intr: Intrinsics) -> tuple:
    return (intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy, tuple(intr.dist), intr.model)


@lru_cache(maxsize=16)
def _angle_grid_lines(key: tuple, step_deg: int) -> list:
    w, h, fx, fy, cx, cy, dist, model = key
    intr = Intrinsics(w, h, fx, fy, cx, cy, dist, model)
    lim = 85
    samples = np.linspace(-lim, lim, 120)
    rvec = tvec = np.zeros(3)
    lines = []
    for a in range(-lim // step_deg * step_deg, lim + 1, step_deg):
        thick = 2 if a == 0 else 1
        for vertical in (True, False):
            t = np.tan(np.radians(a))
            s = np.tan(np.radians(samples))
            if vertical:  # 수평각 a 고정 → 세로선
                pts = np.stack([np.full_like(s, t), s * np.sqrt(1 + t * t), np.ones_like(s)], 1)
            else:  # 수직각 a 고정 → 가로선
                pts = np.stack([s * np.sqrt(1 + t * t), np.full_like(s, t), np.ones_like(s)], 1)
            px = _project(pts, intr, rvec, tvec)
            ok = (px[:, 0] > -50) & (px[:, 0] < w + 50) & (px[:, 1] > -50) & (px[:, 1] < h + 50)
            if ok.sum() < 2:
                continue
            mid = px[ok][len(px[ok]) // 2]
            label, org = "", (0, 0)
            if a != 0 and vertical:
                label, org = f"{a}", (int(mid[0]) + 3, int(cy) - 4)
            elif a != 0:
                label, org = f"{a}", (int(cx) + 3, int(mid[1]) - 3)
            lines.append((px[ok], thick, label, org))
    return lines


def _project(pts3: np.ndarray, intr: Intrinsics, rvec, tvec) -> np.ndarray:
    if intr.is_fisheye:
        out, _ = cv2.fisheye.projectPoints(pts3.reshape(-1, 1, 3), rvec, tvec, intr.K, intr.D)
    else:
        out, _ = cv2.projectPoints(pts3, rvec, tvec, intr.K, intr.D)
    return out.reshape(-1, 2)


def put_text(img: np.ndarray, text: str, org, scale: float = 0.55, color=WHITE) -> None:
    # OpenCV 5 는 두께에 따라 글자 간격이 달라지므로 외곽선은 같은 두께로 위치만 옮겨 그린다
    t = max(1, int(round(scale * 2)))
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2), (2, 0), (0, -2), (-2, 0)):
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, t, cv2.LINE_AA)


def put_lines(img: np.ndarray, lines: list[str], scale: float = 0.55, color=WHITE, origin=(10, 24)) -> None:
    step = int(40 * scale)
    origin = (origin[0], max(origin[1], int(30 * scale)))
    for i, line in enumerate(lines):
        put_text(img, line, (origin[0], origin[1] + i * step), scale, color)


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    canvas[y : y + nh, x : x + nw] = r
    return canvas


def grid(panels: list, cols: int, cell_w: int, cell_h: int, text_scale: float = 0.5) -> np.ndarray:
    """panels: (이미지, 텍스트 줄 목록) 리스트. 텍스트는 셀 크기로 축소한 뒤 그려 해상도와 무관하게 같은 크기."""
    rows = (len(panels) + cols - 1) // cols
    out = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, (img, lines) in enumerate(panels):
        r, c = divmod(i, cols)
        cell = fit(img, cell_w, cell_h)
        if lines:
            put_lines(cell, lines, text_scale)
        out[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = cell
    return out
