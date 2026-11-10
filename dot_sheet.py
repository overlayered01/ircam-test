"""인쇄용 점 마커 시트를 만든다 (A4, 300 dpi, 레이저 프린터 권장). 점 지름마다 한 장.

    python dot_sheet.py                      # shapes/dots_4mm.png, dots_6mm.png, dots_8mm.png
    python dot_sheet.py --dots 3 5 10 --out shapes

한 장에 다섯 마커: 1점 / 같은 크기 2점 / 큰+작은 2점 / 1:2 간격 일직선 3점 / L 자 3점. 간격은 점 지름(d)의 배수로
뷰어의 묶기 기준(Dot group gap, 기본 3.5 d)에 맞춘다: 같은 마커 안 점 중심 간격 ≤ 3 d, 마커끼리는 ≥ 8 d.
    1        : 점 하나. 위치만.
    2        : 간격 2.5 d. 축 방향만 (앞뒤 모름).
    2s       : 큰 점(1.7 d) + 작은 점(d), 간격 2.5 d. 큰 점 → 작은 점이 방향.
    3I (1:2) : 일직선, 간격 1.5 d / 3 d. 외톨이 점 → 붙은 두 점 쪽이 방향.
    3L       : 직각 L, 팔 2.5 d. 꼭짓점 → 나머지 두 점의 중점 쪽이 방향.
라벨 글자는 점에서 5 d 이상 떨어뜨린다 (가까우면 글자가 점과 한 마커로 묶인다). 아래의 100 mm 선으로 인쇄 배율을 확인한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

A4_MM = (210, 297)

# (이름, 점들: (x, y, 반지름) 단위 d) — 좌표는 마커 중심 기준
MARKERS: list[tuple[str, list[tuple[float, float, float]]]] = [
    ("1", [(0.0, 0.0, 0.5)]),
    ("2", [(-1.25, 0.0, 0.5), (1.25, 0.0, 0.5)]),
    ("2s  big->small", [(-1.25, 0.0, 0.85), (1.25, 0.0, 0.5)]),
    ("3I  1:2", [(-2.25, 0.0, 0.5), (-0.75, 0.0, 0.5), (2.25, 0.0, 0.5)]),
    ("3L", [(-1.25, -1.25, 0.5), (1.25, -1.25, 0.5), (-1.25, 1.25, 0.5)]),
]


def marker_extent(dots: list[tuple[float, float, float]]) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) 단위 d."""
    return (min(x - r for x, _, r in dots), max(x + r for x, _, r in dots),
            min(y - r for _, y, r in dots), max(y + r for _, y, r in dots))


def draw_marker(img: np.ndarray, dots: list[tuple[float, float, float]], cx: int, cy: int, d_px: float) -> None:
    for x, y, r in dots:
        cv2.circle(img, (int(round(cx + x * d_px)), int(round(cy + y * d_px))), int(round(r * d_px)), 0, -1, cv2.LINE_AA)


def make_sheets(d_mm: float, dpi: int, title: str) -> list[np.ndarray]:
    """지름 d_mm 의 다섯 마커를 A4 에 흘려 배치한다 (가로 여백 6 d, 줄 사이 4 d). 안 들어가면 다음 장으로."""
    px = dpi / 25.4
    w, h = int(A4_MM[0] * px), int(A4_MM[1] * px)
    d = d_mm * px
    margin = int(15 * px)
    clear = 6 * d  # 마커 사이 가로 여백 (묶기 기준 3.5 d 보다 넉넉히)
    label_gap = 5 * d  # 점 ↔ 라벨 (가까우면 글자가 점과 한 마커로 묶인다)
    text_h = int(8 * px)
    top = margin + int(12 * px)
    bottom = h - margin - int(12 * px)  # 아래 100 mm 선 자리
    pages: list[np.ndarray] = []

    def new_page() -> np.ndarray:
        img = np.full((h, w), 255, np.uint8)
        cv2.putText(img, title + (f"   page {len(pages) + 1}" if pages else ""), (margin, margin),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 150, 1, cv2.LINE_AA)
        yb = h - margin  # 배율 확인용 100 mm 선
        cv2.line(img, (margin, yb), (margin + int(100 * px), yb), 150, 2)
        cv2.putText(img, "100 mm", (margin, yb - int(3 * px)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 150, 1, cv2.LINE_AA)
        pages.append(img)
        return img

    img = new_page()
    x, y, row_h = margin, top, 0.0
    for name, dots in MARKERS:
        x0, x1, y0, y1 = marker_extent(dots)
        mw, mh = (x1 - x0) * d, (y1 - y0) * d
        if x > margin and x + mw > w - margin:  # 줄 바꿈
            x, y, row_h = margin, y + row_h + label_gap + text_h + 4 * d, 0.0
        if y + mh + label_gap + text_h > bottom:  # 새 장
            img = new_page()
            x, y, row_h = margin, top, 0.0
        cx, cy = int(x - x0 * d), int(y - y0 * d)
        draw_marker(img, dots, cx, cy, d)
        # 라벨은 얇은 회색: 멀리서 흐려지면 종이에 묻혀 점으로 읽히지 않는다 (굵은 검정 글자 조각은 1점 마커로 보인다)
        cv2.putText(img, f"{name}   d={d_mm:g} mm", (int(x), int(y + mh + label_gap + text_h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 150, 1, cv2.LINE_AA)
        x += mw + clear
        row_h = max(row_h, mh)
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dots", type=float, nargs="+", default=[4.0, 6.0, 8.0], help="점 지름 mm (지름마다 한 장 이상)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("shapes"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for d in a.dots:
        pages = make_sheets(d, a.dpi, f"IR dot markers  d = {d:g} mm  @ {a.dpi} dpi  (print at 100%)")
        for i, img in enumerate(pages):
            path = a.out / (f"dots_{d:g}mm.png" if i == 0 else f"dots_{d:g}mm_p{i + 1}.png")
            cv2.imwrite(str(path), img)
            print(f"{path}  {img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    main()
