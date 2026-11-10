"""인쇄용 점 마커 시트 (A4, 300 dpi, 레이저 프린터, 100% 배율). 세 가지 시트를 만든다.

    python dot_sheet.py markers                        # 다섯 마커 × 지름 4/6/8 mm → print/dots/dots_markers.png
    python dot_sheet.py markers --dots 3 4 5 6 8 10    # 지름을 바꿔서 (안 들어가면 _p2, _p3 …)
    python dot_sheet.py sizes                          # 한 마커(3L)를 지름 2~12 mm 로 → print/dots/dots_sizes.png
    python dot_sheet.py sizes --kinds 1 3L --sizes 2 3 4 5 6 8 10 --grays 0 90 150   # 종류·크기·회색 농도 조합
    python dot_sheet.py cube                           # 8 cm 큐브 면 6개 (모서리 마커 + 가운데 그림 자리)
                                                       #   → print/cube/<stem>/<stem>.png (A4) + single/cube_face_*.png (낱장)
    python dot_sheet.py cube --face 80 --dot 4 --picture 38 --edge 2 --faces 1 2 2s 3I 3L 3L,1,3L,1
    python dot_sheet.py cube --faces combos12 --stem cube_faces_12   # 모서리마다 점 1개 또는 2개, 16가지 조합 (1,1,1,1 … 2,2,2,2)
    python dot_sheet.py cube --faces 1,1,1,1 1,1,1,2 1,1,2,1 1,2,1,1 2,1,1,1 --dot 3 4 5 6 8 --distance 2.7 2 --no-singles --stem cube_test
        # 2.7 m 에 둘 큐브를 2 m 에서 시험: 치수를 2/2.7 배로 줄인 시트를 점 지름(실제 3~8 mm)마다 한 장씩

마커 종류 (간격은 점 지름 d 의 배수. 뷰어 Dot group gap 기본 3.5 d 안에 들어가고, 마커끼리는 6 d 이상 떨어뜨린다):
    1   : 점 하나. 위치만.
    2   : 같은 크기 두 점, 간격 2.5 d. 축 방향만 (앞뒤 모름).
    2s  : 큰 점(1.7 d) + 작은 점(d), 간격 2.5 d. 큰 점 → 작은 점이 방향.
    3I  : 일직선, 간격 1.5 d / 3 d. 외톨이 점 → 붙은 두 점 쪽이 방향.
    3L  : 직각 L, 팔 2.5 d. 꼭짓점 → 나머지 두 점의 중점 쪽이 방향.
글자는 얇은 회색으로 점에서 5 d 이상 떨어뜨린다 (굵은 검정 글자 조각은 멀리서 점으로 읽힌다). 아래 100 mm 선으로 인쇄 배율을 확인한다.

큐브 면: 한 면 80 mm, 네 모서리에 마커 (L 의 꼭짓점이 면 모서리를 향하게), 가운데 그림 자리(점선). 그림의 어두운 부분이 점에서
묶기 거리(3.5 d) 안에 들어오면 점과 한 마커로 묶여 깨지므로, 그림 상자 ↔ 가장 가까운 점 거리를 계산해 모자라면 경고한다.
기본값(점 4 mm, 가장자리 2 mm, 그림 38 mm)이면 14 mm 이상이 된다. 그림을 키우려면 점을 줄이거나 그림 모서리를 둥글게/밝게 한다.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

A4_MM = (210, 297)

# 마커 점들 (x, y, 반지름) 단위 d — 왼쪽 위 모서리에 놓는 기준 자세 (꼭짓점/큰 점/외톨이 점이 왼쪽 위)
MARKERS: dict[str, list[tuple[float, float, float]]] = {
    "1": [(0.0, 0.0, 0.5)],
    "2": [(0.0, 0.0, 0.5), (2.5, 0.0, 0.5)],
    "2s": [(0.0, 0.0, 0.85), (2.5, 0.0, 0.5)],
    "3I": [(0.0, 0.0, 0.5), (3.0, 0.0, 0.5), (4.5, 0.0, 0.5)],  # 외톨이(0) · 붙은 둘(3, 4.5) → 방향은 +x
    "3L": [(0.0, 0.0, 0.5), (2.5, 0.0, 0.5), (0.0, 2.5, 0.5)],
}
LABELS = {"1": "1", "2": "2", "2s": "2s  big->small", "3I": "3I  1:2", "3L": "3L"}
GRAY = 150  # 글자·보조선 농도 (얇게)


def extent(dots) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) 단위 d (점 가장자리 포함)."""
    return (min(x - r for x, _, r in dots), max(x + r for x, _, r in dots),
            min(y - r for _, y, r in dots), max(y + r for _, y, r in dots))


def draw_dots(img, dots, ox: float, oy: float, d: float, ink: int = 0) -> None:
    """dots 의 (0,0) 이 (ox, oy) 픽셀에 오도록 그린다."""
    for x, y, r in dots:
        cv2.circle(img, (int(round(ox + x * d)), int(round(oy + y * d))), int(round(r * d)), ink, -1, cv2.LINE_AA)


def text(img, s: str, x: int, y: int, scale: float = 0.8) -> None:
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, GRAY, 1, cv2.LINE_AA)


class Pages:
    """A4 페이지 묶음: 제목, 100 mm 배율선, 흘려 넣기(flow) 배치."""

    def __init__(self, dpi: int, title: str):
        self.px = dpi / 25.4
        self.w, self.h = int(A4_MM[0] * self.px), int(A4_MM[1] * self.px)
        self.margin = int(10 * self.px)
        self.top = self.margin + int(8 * self.px)
        self.bottom = self.h - self.margin - int(5 * self.px)  # 아래 100 mm 선 자리
        self.title = title
        self.pages: list[np.ndarray] = []
        self.x = self.y = 0.0
        self.row_h = 0.0
        self.new_page()

    def new_page(self) -> np.ndarray:
        img = np.full((self.h, self.w), 255, np.uint8)
        text(img, self.title + (f"   page {len(self.pages) + 1}" if self.pages else ""), self.margin, self.margin + int(4 * self.px), 0.7)
        yb = self.h - self.margin
        cv2.line(img, (self.margin, yb), (self.margin + int(100 * self.px), yb), GRAY, 2)
        text(img, "100 mm", self.margin + int(102 * self.px), yb + int(2 * self.px), 0.6)
        self.pages.append(img)
        self.x, self.y, self.row_h = self.margin, self.top, 0.0
        return img

    @property
    def img(self) -> np.ndarray:
        return self.pages[-1]

    def newline(self) -> None:
        self.x = self.w  # 다음 place() 가 줄을 바꾼다

    def place(self, w: float, h: float, gap_x: float, gap_y: float) -> tuple[float, float]:
        """w×h 짜리 항목 자리를 잡아 왼쪽 위 좌표를 돌려준다. 줄이 차면 다음 줄, 장이 차면 다음 장."""
        if self.x > self.margin and self.x + w > self.w - self.margin:
            self.x, self.y, self.row_h = self.margin, self.y + self.row_h + gap_y, 0.0
        if self.y + h > self.bottom:
            self.new_page()
        x, y = self.x, self.y
        self.x += w + gap_x
        self.row_h = max(self.row_h, h)
        return x, y

    def save(self, out: Path, stem: str) -> list[Path]:
        paths = []
        for i, img in enumerate(self.pages):
            path = out / (f"{stem}.png" if i == 0 else f"{stem}_p{i + 1}.png")
            cv2.imwrite(str(path), img)
            print(f"{path}  {img.shape[1]}x{img.shape[0]}")
            paths.append(path)
        return paths


def put_marker(pg: Pages, kind: str, d_mm: float, ink: int = 0, label: str | None = None) -> None:
    """마커 하나 + 아래 작은 회색 라벨을 흘려 넣는다. 가로 여백 6 d (최소 6 mm), 라벨은 점에서 4 d 아래, 줄 사이 3 d."""
    d = d_mm * pg.px
    x0, x1, y0, y1 = extent(MARKERS[kind])
    mw, mh = (x1 - x0) * d, (y1 - y0) * d
    label = label if label is not None else f"{kind} {d_mm:g}mm"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    label_gap = 4 * d
    x, y = pg.place(max(mw, tw), mh + label_gap + th, max(6 * d, 6 * pg.px), 3 * d)
    draw_dots(pg.img, MARKERS[kind], x - x0 * d, y - y0 * d, d, ink)
    text(pg.img, label, x, y + mh + label_gap + th, 0.6)


# ------------------------------------------------------------------ 시트 1: 다섯 마커 × 지름
def sheet_markers(dots_mm: list[float], dpi: int, out: Path) -> list[Path]:
    pg = Pages(dpi, f"IR dot markers  d = {', '.join(f'{d:g}' for d in dots_mm)} mm  @ {dpi} dpi  (print at 100%)")
    for d in dots_mm:
        for kind in MARKERS:
            put_marker(pg, kind, d)
        pg.newline()  # 지름이 바뀌면 줄 바꿈
    return pg.save(out, "dots_markers")


# ------------------------------------------------------------------ 시트 2: 한 마커를 여러 크기·농도로
def sheet_sizes(kinds: list[str], sizes_mm: list[float], grays: list[int], dpi: int, out: Path) -> list[Path]:
    pg = Pages(dpi, f"IR dot sizes  {' '.join(kinds)}  d = {', '.join(f'{s:g}' for s in sizes_mm)} mm"
                    + (f"  gray {' '.join(map(str, grays))}" if grays != [0] else "") + f"  @ {dpi} dpi  (print at 100%)")
    for kind in kinds:
        for ink in grays:
            for s in sizes_mm:
                put_marker(pg, kind, s, ink, f"{kind} {s:g}mm" + (f" g{ink}" if ink else ""))
            pg.newline()
    return pg.save(out, "dots_sizes")


# ------------------------------------------------------------------ 시트 3: 큐브 면
def corner_dots(kind: str, corner: int, face: float, margin: float, d: float) -> list[tuple[float, float, float]]:
    """면(한 변 face, 단위 px) 의 corner(0 TL, 1 TR, 2 BR, 3 BL) 에 놓은 마커 점들 — 면 왼쪽 위 기준 픽셀 좌표.
    기준 자세(왼쪽 위)를 x/y 반전으로 돌리므로 L 의 꼭짓점·2s 의 큰 점·3I 의 외톨이 점이 늘 면 모서리 쪽이다."""
    base = MARKERS[kind]
    x0, _, y0, _ = extent(base)
    pts = []
    for x, y, r in base:
        px_ = margin + (x - x0) * d  # 왼쪽 위 기준
        py_ = margin + (y - y0) * d
        if corner in (1, 2):
            px_ = face - px_
        if corner in (2, 3):
            py_ = face - py_
        pts.append((px_, py_, r * d))
    return pts


def face_dots(spec: str, face: float, margin: float, d: float) -> list[tuple[float, float, float]]:
    """spec: 'kind' (네 모서리 모두) 또는 'k0,k1,k2,k3' (TL, TR, BR, BL). '-' 는 모서리를 비운다."""
    kinds = spec.split(",")
    if len(kinds) == 1:
        kinds = kinds * 4
    if len(kinds) != 4:
        raise ValueError(f"face spec must have 1 or 4 entries: {spec!r}")
    out = []
    for c, k in enumerate(kinds):
        if k and k != "-":
            out += corner_dots(k, c, face, margin, d)
    return out


def picture_clearance(dots, face: float, picture: float) -> float:
    """그림 상자(가운데 정사각형, 한 변 picture) 와 가장 가까운 점 가장자리 사이 거리 (px)."""
    lo, hi = (face - picture) / 2, (face + picture) / 2
    best = float("inf")
    for x, y, r in dots:
        dx = max(lo - x, 0.0, x - hi)
        dy = max(lo - y, 0.0, y - hi)
        best = min(best, float(np.hypot(dx, dy)) - r)
    return best


def draw_face(img, ox: int, oy: int, spec: str, face: float, margin: float, d: float, picture: float,
              label: str, guides: bool = True) -> list[tuple[float, float, float]]:
    dots = face_dots(spec, face, margin, d)
    for x, y, r in dots:
        cv2.circle(img, (int(round(ox + x)), int(round(oy + y))), int(round(r)), 0, -1, cv2.LINE_AA)
    if guides:
        f = int(round(face))
        # 재단 표시: 네 변 중앙의 짧은 눈금 (면 테두리 전체를 그리면 2 mm 옆의 점과 흐려져 붙는다). 자를 맞대고 자른다
        tick = max(8, int(face * 0.06))
        for (ax, ay, bx, by) in ((f // 2, -tick, f // 2, 0), (f // 2, f, f // 2, f + tick), (-tick, f // 2, 0, f // 2), (f, f // 2, f + tick, f // 2)):
            cv2.line(img, (ox + ax, oy + ay), (ox + bx, oy + by), GRAY, 1)
        lo, hi = int(round((face - picture) / 2)), int(round((face + picture) / 2))
        step = max(6, int(picture / 24))
        for t in range(lo, hi, step * 2):  # 그림 자리 점선
            t2 = min(t + step, hi)
            cv2.line(img, (ox + t, oy + lo), (ox + t2, oy + lo), GRAY, 1)
            cv2.line(img, (ox + t, oy + hi), (ox + t2, oy + hi), GRAY, 1)
            cv2.line(img, (ox + lo, oy + t), (ox + lo, oy + t2), GRAY, 1)
            cv2.line(img, (ox + hi, oy + t), (ox + hi, oy + t2), GRAY, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        text(img, label, ox + (f - tw) // 2, oy + (f + th) // 2, 0.7)
        text(img, "picture", ox + (f - 60) // 2, oy + (f + th) // 2 + int(th * 1.8), 0.6)
    return dots


def expand_faces(faces: list[str]) -> list[str]:
    """'combos12' → 모서리(TL,TR,BR,BL)마다 1점/2점인 16가지 조합 (2점 모서리 수가 적은 것부터)."""
    out = []
    for f in faces:
        if f == "combos12":
            combos = [",".join(c) for c in itertools.product("12", repeat=4)]
            out += sorted(combos, key=lambda c: (c.count("2"), c))
        else:
            out.append(f)
    return out


def sheet_cube(faces: list[str], face_mm: float, dot_mm: float, picture_mm: float, margin_mm: float | None,
               gap_d: float, dpi: int, out: Path, stem: str = "cube_faces", scale: float = 1.0,
               singles: bool = True) -> tuple[list[Path], float]:
    """A4 에 면들을 배치해 저장하고, 면마다 점만 그린 낱장 PNG(정확히 face_mm)도 저장한다. (경로들, 그림↔점 최소 거리 mm).
    scale < 1 이면 모든 치수를 그 배로 줄여 그린다 (실제 거리 대신 가까이서 시험할 때: 2.7 m 짜리를 2 m 에서 → 2/2.7).
    라벨과 출력은 실제 치수와 줄인 치수를 같이 적는다. 그림↔점 거리 검사는 실제 치수 기준(비율은 같다)."""
    px = dpi / 25.4
    k = scale * px
    face, d, picture = face_mm * k, dot_mm * k, picture_mm * k
    margin = (margin_mm if margin_mm is not None else 2.0) * k  # 면 가장자리 ↔ 점 가장자리
    sc = f"  x{scale:.3g} (face {face_mm * scale:.1f} mm, dot {dot_mm * scale:.2f} mm)" if scale != 1.0 else ""
    pg = Pages(dpi, f"Cube faces  {face_mm:g} mm  dot {dot_mm:g} mm  picture {picture_mm:g} mm{sc}  @ {dpi} dpi  (print at 100%)")
    worst = float("inf")
    paths = []
    for i, spec in enumerate(faces):
        # 면 사이는 6 d 이상 (이웃 면 모서리 마커끼리 묶이지 않게: 8 mm 로 붙이면 12 mm = 3 d 라 한 덩어리가 된다) → 보통 한 장에 4면
        gap = max(8 * px, 6 * d + 2 * margin)
        x, y = pg.place(face, face, gap, gap)
        label = f"face {i + 1}: {spec}" + (f"  d{dot_mm:g}" if scale != 1.0 else "")
        dots = draw_face(pg.img, int(x), int(y), spec, face, margin, d, picture, label)
        clear = picture_clearance(dots, face, picture) / k
        worst = min(worst, clear)
        if singles:
            single = np.full((int(round(face)), int(round(face))), 255, np.uint8)  # 인쇄소용 낱장: 점만
            draw_face(single, 0, 0, spec, face, margin, d, picture, label, guides=False)
            (out / "single").mkdir(parents=True, exist_ok=True)
            path = out / "single" / f"face_{i + 1}_{spec.replace(',', '-')}.png"
            cv2.imwrite(str(path), single)
            paths.append(path)
            print(f"{path}  {single.shape[1]}x{single.shape[0]} ({face_mm * scale:g} mm)  picture-dot {clear:.1f} mm")
    need = gap_d * dot_mm
    print(f"[{stem}] 그림 상자 ↔ 점 최소 거리 {worst:.1f} mm, 필요 ≥ {need:.1f} mm (Dot group gap {gap_d:g} × {dot_mm:g} mm)"
          + ("" if worst >= need else "  <- 모자람: 그림을 줄이거나(--picture) 점을 줄이거나 그림 모서리를 밝게"))
    return paths + pg.save(out, stem), worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet", nargs="?", default="markers", choices=["markers", "sizes", "cube"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", type=Path, default=None, help="출력 폴더 (기본: markers/sizes → print/dots, cube → print/cube/<stem>)")
    ap.add_argument("--dots", type=float, nargs="+", default=[4.0, 6.0, 8.0], help="markers: 점 지름 mm")
    ap.add_argument("--kinds", nargs="+", default=["3L"], choices=list(MARKERS), help="sizes: 마커 종류")
    ap.add_argument("--sizes", type=float, nargs="+", default=[2, 3, 4, 5, 6, 8, 10, 12], help="sizes: 점 지름 mm")
    ap.add_argument("--grays", type=int, nargs="+", default=[0], help="sizes: 잉크 농도 (0 검정 … 255 흰). 예 0 90 150")
    ap.add_argument("--faces", nargs="+", default=["1", "2", "2s", "3I", "3L", "3L,1,3L,1"],
                    help="cube: 면마다 마커. 'kind' = 네 모서리 같은 마커, 'TL,TR,BR,BL' = 모서리별 (- 는 비움)")
    ap.add_argument("--face", type=float, default=80.0, help="cube: 한 면 mm")
    ap.add_argument("--dot", type=float, nargs="+", default=[4.0], help="cube: 점 지름 mm (여러 개면 지름마다 시트, stem 에 _d{mm} 붙음)")
    ap.add_argument("--distance", type=float, nargs=2, metavar=("REAL_M", "TEST_M"), default=None,
                    help="cube: 실제 거리와 시험 거리 (m). 치수를 TEST/REAL 배로 줄여 가까이서 실제 거리처럼 보이게 (예 2.7 2)")
    ap.add_argument("--no-singles", action="store_true", help="cube: 낱장 PNG 는 만들지 않음 (시험 시트만)")
    ap.add_argument("--picture", type=float, default=38.0, help="cube: 가운데 그림 상자 한 변 mm")
    ap.add_argument("--edge", type=float, default=2.0, help="cube: 면 가장자리 ↔ 점 가장자리 mm")
    ap.add_argument("--gap", type=float, default=3.5, help="cube: 뷰어 Dot group gap (지름 배수) — 그림 간격 검사용")
    ap.add_argument("--stem", default="cube_faces", help="cube: 출력 파일 이름 앞부분 (낱장은 cube_face_N_spec.png)")
    a = ap.parse_args()
    if a.out is None:
        a.out = Path("print") / ("dots" if a.sheet != "cube" else f"cube/{a.stem}")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.sheet == "markers":
        sheet_markers(a.dots, a.dpi, a.out)
    elif a.sheet == "sizes":
        sheet_sizes(a.kinds, a.sizes, a.grays, a.dpi, a.out)
    else:
        scale = a.distance[1] / a.distance[0] if a.distance else 1.0
        for dot in a.dot:
            stem = a.stem + (f"_d{dot:g}" if len(a.dot) > 1 else "")
            sheet_cube(expand_faces(a.faces), a.face, dot, a.picture, a.edge, a.gap, a.dpi, a.out, stem, scale, not a.no_singles)


if __name__ == "__main__":
    main()
