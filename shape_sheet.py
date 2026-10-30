"""인쇄용 도형 테스트 시트를 만든다 (A4, 300 dpi, 레이저 프린터 권장).

    python shape_sheet.py                 # shapes/shapes_filled.png, shapes/shapes_outline.png
    python shape_sheet.py --size 60 --stroke 3 --out shapes

한 장에 정삼각형 · 정사각형 · 정십자 · 원 · 정오각형 · 정육각형 여섯 개. filled 는 속이 찬 검정 도형(Shape mode filled),
outline 은 선으로만 그린 도형(Shape mode outline). 각 도형 아래에 이름과 크기(mm)를 적는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

A4_MM = (210, 297)
NAMES = ("triangle", "square", "cross", "circle", "pentagon", "hexagon")


def _regular(center, r, n, rot=-np.pi / 2):
    return np.array([(center[0] + r * np.cos(a + rot), center[1] + r * np.sin(a + rot))
                     for a in np.linspace(0, 2 * np.pi, n + 1)[:-1]], np.int32)


def _cross(center, size, arm):
    a, b = size / 2, arm / 2
    pts = [(-b, -a), (b, -a), (b, -b), (a, -b), (a, b), (b, b), (b, a), (-b, a), (-b, b), (-a, b), (-a, -b), (-b, -b)]
    return np.array([(center[0] + x, center[1] + y) for x, y in pts], np.int32)


def draw_shape(img: np.ndarray, name: str, center: tuple[int, int], size_px: int, stroke: int) -> None:
    """stroke 0 = 채움, 아니면 선 두께(픽셀). size_px = 외접 사각형 한 변."""
    r = size_px // 2
    th = -1 if stroke <= 0 else stroke
    if name == "square":
        cv2.rectangle(img, (center[0] - r, center[1] - r), (center[0] + r, center[1] + r), 0, th)
    elif name == "circle":
        cv2.circle(img, center, r, 0, th)
    elif name == "cross":
        poly = _cross(center, size_px, max(size_px // 5, stroke * 2 if stroke > 0 else 1))
        if stroke <= 0:
            cv2.fillPoly(img, [poly], 0)
        else:  # 선 도형 십자는 두 획
            cv2.line(img, (center[0], center[1] - r), (center[0], center[1] + r), 0, stroke)
            cv2.line(img, (center[0] - r, center[1]), (center[0] + r, center[1]), 0, stroke)
    else:
        n = {"triangle": 3, "pentagon": 5, "hexagon": 6}[name]
        poly = _regular(center, r, n)
        if stroke <= 0:
            cv2.fillPoly(img, [poly], 0)
        else:
            cv2.polylines(img, [poly], True, 0, stroke)


def make_sheet(size_mm: float, stroke_mm: float, dpi: int, title: str) -> np.ndarray:
    px = dpi / 25.4
    w, h = int(A4_MM[0] * px), int(A4_MM[1] * px)
    img = np.full((h, w), 255, np.uint8)
    size_px = int(size_mm * px)
    stroke = 0 if stroke_mm <= 0 else max(1, int(round(stroke_mm * px)))
    cols, rows = 2, 3
    cell_w, cell_h = w // cols, (h - int(20 * px)) // rows
    for i, name in enumerate(NAMES):
        cx = cell_w * (i % cols) + cell_w // 2
        cy = int(20 * px) + cell_h * (i // cols) + cell_h // 2 - int(8 * px)
        draw_shape(img, name, (cx, cy), size_px, stroke)
        label = f"{name}  {size_mm:.0f} mm"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
        cv2.putText(img, label, (cx - tw // 2, cy + size_px // 2 + int(12 * px)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3, cv2.LINE_AA)
    cv2.putText(img, title, (int(10 * px), int(14 * px)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=60.0, help="도형 외접 사각형 한 변 (mm)")
    ap.add_argument("--stroke", type=float, default=2.0, help="outline 시트의 선 두께 (mm)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("shapes"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for kind, stroke in (("filled", 0.0), ("outline", a.stroke)):
        img = make_sheet(a.size, stroke, a.dpi, f"IR shape test  {kind}  {a.size:.0f} mm  @ {a.dpi} dpi  (print at 100%)")
        path = a.out / f"shapes_{kind}.png"
        cv2.imwrite(str(path), img)
        print(f"{path}  {img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    main()
