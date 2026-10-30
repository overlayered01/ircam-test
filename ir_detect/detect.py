"""IR 8비트 영상에서 AprilTag 와 기본 도형(삼각형·사각형·십자·오각형·육각형·원)을 찾는다.

- AprilTag: pupil-apriltags(AprilTag 3) 가 있으면 그것을, 없으면 OpenCV ArUco 의 AprilTag 사전을 쓴다.
  두 검출기 모두 검은 테두리를 찾으므로 흰 테두리(inv) 태그나 밝은 배경의 어두운 태그는 invert 로 영상을 뒤집어 넣는다.
  pupil 결과는 hamming == 0, decision_margin >= min_margin 으로 거른다 (16h5 는 오검출이 많다).
- 도형: Otsu 이진화 → 외곽선 → 다각형 근사. 기본은 밝은 바탕(종이)의 어두운 도형, invert 면 어두운 바탕의 밝은 도형.
  도형마다 '정형 편차'(dev) 를 잰다: 변 길이·대각선·팔 길이가 이상적인 정삼각형/정사각형/정십자/원에서 얼마나 벗어났는지의 최대 비율.
  ShapeOptions.regular_only 면 dev <= tol 인 것만 남긴다 (예: tol 0.1 = 10% 이내).
- 큰 영상(DMK 2448x2048)은 max_width 로 줄여서 검출하고 좌표를 원본 스케일로 돌려준다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

FAMILIES = ("tag36h11", "tag25h9", "tag16h5", "tag36h10")
_CV_DICT = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}
SHAPE_TYPES = ("all", "triangle", "square", "cross", "circle", "rect", "pentagon", "hexagon")
SHAPE_TOLS = (0.05, 0.1, 0.15, 0.2, 0.3)
SHAPE_MIN_PX = (10, 20, 40, 80, 160)
SHAPE_MODES = ("both", "filled", "outline")  # 채워진 도형 / 선으로 그린 도형 / 둘 다


@dataclass
class TagDet:
    id: int
    corners: np.ndarray  # (4, 2) 원본 영상 픽셀
    hamming: int | None = None
    margin: float | None = None

    @property
    def center(self) -> tuple[float, float]:
        c = self.corners.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def size_px(self) -> float:
        """한 변 길이 평균 (픽셀)."""
        return float(np.mean([np.linalg.norm(self.corners[i] - self.corners[(i + 1) % 4]) for i in range(4)]))


@dataclass
class ShapeDet:
    name: str  # triangle / square / rect / cross / circle / pentagon / hexagon
    contour: np.ndarray  # (N, 2) 원본 영상 픽셀
    center: tuple[float, float]
    area: float
    size_px: float = 0.0  # 외접 사각형 짧은 변 (원본 픽셀)
    dev: float = 0.0  # 정형에서 벗어난 최대 비율 (0 = 완벽한 정형)

    @property
    def regular(self) -> bool:
        return self.dev <= 0.1


@dataclass(frozen=True)
class ShapeOptions:
    types: str = "all"  # SHAPE_TYPES 중 하나
    regular_only: bool = False  # 정형(정삼각형·정사각형·정십자·원…)만
    tol: float = 0.1  # 정형 허용 편차 (비율)
    min_px: int = 20  # 외접 사각형 짧은 변 최소 (원본 픽셀)
    mode: str = "both"  # SHAPE_MODES: filled = 채워진 도형(Otsu), outline = 선 도형(국소 이진화), both
    epsilon: float = 0.02  # approxPolyDP 둘레 비율
    max_area_frac: float = 0.5


@dataclass
class DetResult:
    tags: list[TagDet] = field(default_factory=list)
    shapes: list[ShapeDet] = field(default_factory=list)
    ms: float = 0.0
    backend: str = ""
    rejected: int = 0  # 태그 후보 중 디코딩 실패 (OpenCV) / 필터로 떨어진 것 (pupil)
    shapes_rejected: int = 0  # 도형 후보 중 종류·정형·크기 조건으로 떨어진 것

    def summary(self) -> str:
        parts = []
        if self.backend:
            ids = ",".join(str(t.id) for t in self.tags[:8]) + (",…" if len(self.tags) > 8 else "")
            parts.append(f"tags {len(self.tags)}" + (f" [{ids}]" if self.tags else "") + (f" (-{self.rejected})" if self.rejected else ""))
        if self.shapes or self.shapes_rejected:
            counts: dict[str, int] = {}
            for s in self.shapes:
                counts[s.name] = counts.get(s.name, 0) + 1
            parts.append("shapes " + (" ".join(f"{k}:{v}" for k, v in counts.items()) or "0") + (f" (-{self.shapes_rejected})" if self.shapes_rejected else ""))
        return "  ".join(parts) + f"  {self.ms:.0f}ms"


def prepare(gray: np.ndarray, max_width: int = 1280) -> tuple[np.ndarray, float]:
    """검출용으로 줄인 영상과 (원본 / 검출 영상) 배율."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] <= max_width:
        return gray, 1.0
    s = max_width / gray.shape[1]
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 1.0 / s


class AprilTagDetector:
    def __init__(self, family: str = "tag36h11", backend: str = "auto", min_margin: float = 10.0):
        if family not in _CV_DICT:
            raise ValueError(f"지원하지 않는 계열: {family}")
        self.family, self.min_margin = family, min_margin
        self._pupil = None
        if backend in ("auto", "pupil"):
            try:
                from pupil_apriltags import Detector

                # pupil-apriltags 는 tag36h10 을 모른다 → OpenCV 로
                self._pupil = Detector(families=family, nthreads=2, quad_decimate=1.0, refine_edges=True)
            except Exception:  # noqa: BLE001  ImportError 또는 미지원 계열
                if backend == "pupil":
                    raise
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._cv = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(_CV_DICT[family]), p)

    @property
    def backend(self) -> str:
        return "pupil-apriltags" if self._pupil is not None else "opencv-aruco"

    def detect(self, gray: np.ndarray, invert: bool = False, scale: float = 1.0) -> tuple[list[TagDet], int]:
        """(검출 목록, 걸러진 수). scale 은 좌표를 원본으로 돌릴 배율."""
        img = 255 - gray if invert else gray
        out, rejected = [], 0
        if self._pupil is not None:
            for d in self._pupil.detect(img):
                if d.hamming != 0 or d.decision_margin < self.min_margin:
                    rejected += 1
                    continue
                out.append(TagDet(int(d.tag_id), np.asarray(d.corners, np.float64) * scale, int(d.hamming), float(d.decision_margin)))
        else:
            corners, ids, rej = self._cv.detectMarkers(img)
            rejected = 0 if rej is None else len(rej)
            if ids is not None:
                for c, i in zip(corners, ids.flatten()):
                    out.append(TagDet(int(i), c.reshape(4, 2).astype(np.float64) * scale))
        out.sort(key=lambda t: t.id)
        return out, rejected


# --- 도형 분류 ---------------------------------------------------------------


def _edge_lengths(poly: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(poly[i] - poly[(i + 1) % len(poly)]) for i in range(len(poly))])


def _spread(v: np.ndarray) -> float:
    """값들이 평균에서 벗어난 최대 비율."""
    m = float(np.mean(v))
    return float(np.max(np.abs(v / m - 1.0))) if m > 0 else 1.0


def _convex_flags(poly: np.ndarray) -> np.ndarray:
    """꼭짓점마다 볼록(True)/오목(False). 외곽선 방향에 무관하게 다수 쪽을 볼록으로 본다."""
    n = len(poly)
    cross = []
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    cross = np.array(cross)
    sign = 1.0 if (cross > 0).sum() >= (cross < 0).sum() else -1.0
    return cross * sign >= 0


def classify(contour: np.ndarray, epsilon: float = 0.02) -> tuple[str, float, np.ndarray] | None:
    """외곽선 → (이름, 정형 편차, 근사 다각형). 해당 없으면 None."""
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if peri <= 0 or area <= 0:
        return None
    approx = cv2.approxPolyDP(contour, epsilon * peri, True).reshape(-1, 2).astype(np.float64)
    n = len(approx)
    # 꼭짓점 수는 epsilon 으로 정하되, 위치는 같은 수를 주는 가장 작은 epsilon 으로 다시 잡는다 (변 길이 오차 ↓)
    for eps in (0.004, 0.007, 0.01, 0.015):
        if eps >= epsilon:
            break
        fine = cv2.approxPolyDP(contour, eps * peri, True).reshape(-1, 2).astype(np.float64)
        if len(fine) == n:
            approx = fine
            break
    convex = cv2.isContourConvex(approx.astype(np.float32))
    circularity = 4 * np.pi * area / (peri * peri)

    if n == 3 and convex:
        return "triangle", _spread(_edge_lengths(approx)), approx
    if n == 4 and convex:
        sides = _edge_lengths(approx)
        diag = np.array([np.linalg.norm(approx[0] - approx[2]), np.linalg.norm(approx[1] - approx[3])])
        diag_dev = _spread(diag)  # 대각선이 같으면 직사각형 (각이 90°)
        side_dev = _spread(sides)
        opp_dev = max(abs(sides[0] / sides[2] - 1), abs(sides[1] / sides[3] - 1))  # 마주보는 변
        if side_dev <= 0.25:
            return "square", max(side_dev, diag_dev), approx
        return "rect", max(opp_dev, diag_dev), approx
    if n == 5 and convex:
        return "pentagon", _spread(_edge_lengths(approx)), approx
    if n == 6 and convex:
        return "hexagon", _spread(_edge_lengths(approx)), approx
    if n == 12 and not convex:
        r = _cross(approx)
        if r is not None:
            return r
    if n > 6 and convex and circularity > 0.75:
        (_, (w, h), _) = cv2.fitEllipse(contour) if len(contour) >= 5 else (None, (1.0, 1.0), 0)
        aspect = abs(min(w, h) / max(w, h) - 1) if max(w, h) > 0 else 1.0
        # 래스터화된 완전한 원도 circularity 가 0.9 안팎이라 0.92 기준으로 모자란 만큼만 더한다
        return "circle", max(aspect, max(0.0, 0.92 - circularity) * 2), approx
    # 가는 획 십자: 둘레 비율 epsilon 이 팔 폭보다 크면 끝변이 뭉개져 12각형이 안 나온다 → 더 작은 epsilon 으로 십자만 다시 본다
    for eps in (0.01, 0.005):
        if eps >= epsilon:
            continue
        fine = cv2.approxPolyDP(contour, eps * peri, True).reshape(-1, 2).astype(np.float64)
        if len(fine) == 12 and not cv2.isContourConvex(fine.astype(np.float32)):
            r = _cross(fine)
            if r is not None:
                return r
    return None


def _cross(approx: np.ndarray) -> tuple[str, float, np.ndarray] | None:
    """12각형 → 십자면 ('cross', dev, approx). 오목 꼭짓점 4개, 양 끝이 볼록인 변(팔 끝) 4개."""
    n = len(approx)
    flags = _convex_flags(approx)
    if (~flags).sum() != 4:
        return None
    lengths = _edge_lengths(approx)
    ends = np.array([flags[i] and flags[(i + 1) % n] for i in range(n)])  # 팔 끝 4개, 나머지 8개 = 팔 옆면
    if ends.sum() != 4:
        return None
    _, (w, h), _ = cv2.minAreaRect(approx.astype(np.float32))
    aspect = abs(min(w, h) / max(w, h) - 1) if max(w, h) > 0 else 1.0
    # 팔 폭(끝변)은 ±1px 양자화 몫을 빼고 본다: 선을 팽창한 십자는 폭이 5px 안팎이라 1px 이 20% 다
    width_dev = max(0.0, _spread(lengths[ends]) - 2.0 / float(np.mean(lengths[ends])))
    return "cross", max(width_dev, _spread(lengths[~ends]), aspect), approx


REGULAR_TYPES = ("triangle", "square", "cross", "circle", "pentagon", "hexagon")  # regular_only 에서 허용


class ShapeDetector:
    """채워진 도형(filled)과 선으로 그린 도형(outline)을 찾는다.

    - filled: 영상 전체 Otsu 이진화 → 바깥 외곽선. 인쇄한 검정 도형용.
    - outline: 국소(adaptive) 이진화로 가는 펜 선만 뽑아 닫힘·팽창으로 잇는다. 닫힌 윤곽(사각형·삼각형·원)은
      안쪽 구멍 외곽선을 분류하고(꼭짓점이 뭉개지지 않는다), 구멍이 없는 획(십자 두 줄)은 팽창된 바깥 외곽선을
      분류한다. 손으로 그린 도형이나, 종이 전체가 밝아 Otsu 임계값이 종이/배경 사이로 잡히는 장면용.
    - both: 둘 다 돌리고 같은 자리(중심 거리 < 짧은 변의 절반)는 filled 쪽만 남긴다.
    """

    def __init__(self, options: ShapeOptions = ShapeOptions()):
        self.options = options

    def detect(self, gray: np.ndarray, invert: bool = False, scale: float = 1.0,
               options: ShapeOptions | None = None) -> tuple[list[ShapeDet], int]:
        """(검출 목록, 조건으로 떨어진 후보 수)."""
        o = options or self.options
        out: list[ShapeDet] = []
        rejected: list[ShapeDet] = []
        if o.mode in ("filled", "both"):
            out, rejected = self._classify_all(self._filled_contours(gray, invert), gray, scale, o)
        if o.mode in ("outline", "both"):
            found, rej = self._classify_all(self._outline_contours(gray, invert), gray, scale, o)
            seen = out + rejected  # filled 가 이미 본 자리(채택이든 탈락이든)는 다시 세지 않는다
            out = out + [s for s in found if not any(_same_spot(s, f) for f in seen)]
            rejected = rejected + [s for s in rej if not any(_same_spot(s, f) for f in seen)]
        out.sort(key=lambda s: -s.area)
        return out, len(rejected)

    @staticmethod
    def _filled_contours(gray: np.ndarray, invert: bool) -> list[np.ndarray]:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mode = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV  # 기본: 어두운 도형 → 흰색
        _, bw = cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    @staticmethod
    def _outline_contours(gray: np.ndarray, invert: bool) -> list[np.ndarray]:
        w = gray.shape[1]
        block = max(11, (w // 20) | 1)  # 640px → 31, 1280px → 63 (홀수)
        grow = max(1, round(w / 320))  # 640px → 2: 1px 선이 5px 획이 되어 십자가 12각형으로 잡힌다
        mode = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
        bw0 = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, block, 8)
        bw = cv2.morphologyEx(bw0, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        bw = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=grow)
        contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return []
        hier = hier[0]
        out = []
        for i, (_, _, child, parent) in enumerate(hier):
            if parent != -1 or not _dark_stroke(gray, bw0, contours[i], invert):
                continue
            if child == -1:
                out.append(contours[i])  # 구멍 없는 획 (십자)
            while child != -1:
                out.append(contours[child])  # 닫힌 윤곽의 안쪽
                child = hier[child][0]
        return out

    @staticmethod
    def _classify_all(contours: list[np.ndarray], gray: np.ndarray, scale: float,
                      o: ShapeOptions) -> tuple[list[ShapeDet], list[ShapeDet]]:
        """(채택, 종류·정형·크기 조건으로 탈락) — 둘 다 원본 좌표의 ShapeDet."""
        total = gray.shape[0] * gray.shape[1]
        min_px = o.min_px / scale  # 검출 영상 픽셀로
        out, rejected = [], []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_px * min_px * 0.3 or area > total * o.max_area_frac:
                continue
            _, (w, h), _ = cv2.minAreaRect(c)
            if min(w, h) < min_px:
                continue
            r = classify(c, o.epsilon)
            if r is None:
                continue
            name, dev, _ = r
            m = cv2.moments(c)
            cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (0.0, 0.0)
            s = ShapeDet(name, c.reshape(-1, 2).astype(np.float64) * scale, (cx * scale, cy * scale),
                         area * scale * scale, min(w, h) * scale, dev)
            if (o.types != "all" and name != o.types) or (o.regular_only and (dev > o.tol or name not in REGULAR_TYPES)):
                rejected.append(s)
            else:
                out.append(s)
        return out, rejected


def _dark_stroke(gray: np.ndarray, bw: np.ndarray, contour: np.ndarray, invert: bool, margin: float = 8.0) -> bool:
    """획(이진화 흰 부분)이 바로 바깥 띠보다 어두우면 True (invert 면 밝으면 True).
    밝은 채움 도형의 가장자리(어두운 배경 쪽에 생기는 띠)를 선 도형으로 오인하지 않게 한다."""
    x, y, w, h = cv2.boundingRect(contour)
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.drawContours(roi, [contour - (x0, y0)], -1, 255, -1)
    stroke = (roi > 0) & (bw[y0:y1, x0:x1] > 0)
    band = (cv2.dilate(roi, np.ones((3, 3), np.uint8), iterations=3) > 0) & (roi == 0)
    if not stroke.any() or not band.any():
        return False
    g = gray[y0:y1, x0:x1]
    diff = float(g[band].mean()) - float(g[stroke].mean())
    return (-diff if invert else diff) > margin


def _same_spot(a: ShapeDet, b: ShapeDet) -> bool:
    d = np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
    return d < 0.5 * max(a.size_px, b.size_px)


class Pipeline:
    """메뉴 설정 → 검출기 묶음. 계열이 바뀌면 태그 검출기를 다시 만든다."""

    def __init__(self, backend: str = "auto", max_width: int = 1280):
        self.backend, self.max_width = backend, max_width
        self._tag: AprilTagDetector | None = None
        self._shape = ShapeDetector()

    def run(self, gray: np.ndarray, tags: bool, shapes: bool, family: str = "tag36h11", invert: bool = False,
            shape_options: ShapeOptions | None = None) -> DetResult:
        t0 = time.perf_counter()
        res = DetResult()
        if not (tags or shapes):
            return res
        small, scale = prepare(gray, self.max_width)
        if tags:
            if self._tag is None or self._tag.family != family:
                self._tag = AprilTagDetector(family, self.backend)
            res.tags, res.rejected = self._tag.detect(small, invert, scale)
            res.backend = self._tag.backend
        if shapes:
            res.shapes, res.shapes_rejected = self._shape.detect(small, invert, scale, shape_options)
        res.ms = (time.perf_counter() - t0) * 1000
        return res


def draw(img: np.ndarray, res: DetResult, s: float = 1.0, thickness: int = 2) -> None:
    """원본 좌표의 결과를 s 배로 줄인 표시 영상 위에 그린다."""
    for t in res.tags:
        pts = (t.corners * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (0, 255, 0), thickness)
        cv2.circle(img, tuple((t.corners[0] * s).astype(int)), 4, (0, 0, 255), -1)  # 첫 코너 = 방향
        cx, cy = t.center
        label = f"{t.id}" + (f" m{t.margin:.0f}" if t.margin is not None else "")
        cv2.putText(img, label, (int(cx * s) - 10, int(cy * s) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for sh in res.shapes:
        pts = (sh.contour * s).astype(np.int32).reshape(-1, 1, 2)
        color = (255, 160, 0) if sh.regular else (0, 140, 255)  # 정형 = 주황, 비정형 = 노랑빛
        cv2.polylines(img, [pts], True, color, thickness)
        cv2.putText(img, f"{sh.name} {sh.dev * 100:.0f}%", (int(sh.center[0] * s) - 24, int(sh.center[1] * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
