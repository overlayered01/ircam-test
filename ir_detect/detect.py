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
        flags = _convex_flags(approx)
        if (~flags).sum() == 4:  # 오목 꼭짓점 4개 = 십자
            lengths = _edge_lengths(approx)
            # 양 끝이 모두 볼록인 변 = 팔 끝 (4개), 나머지 = 팔 옆면 (8개)
            ends = np.array([flags[i] and flags[(i + 1) % n] for i in range(n)])
            if ends.sum() == 4:
                _, (w, h), _ = cv2.minAreaRect(approx.astype(np.float32))
                aspect = abs(min(w, h) / max(w, h) - 1) if max(w, h) > 0 else 1.0
                dev = max(_spread(lengths[ends]), _spread(lengths[~ends]), aspect)
                return "cross", dev, approx
    if n > 6 and convex and circularity > 0.75:
        (_, (w, h), _) = cv2.fitEllipse(contour) if len(contour) >= 5 else (None, (1.0, 1.0), 0)
        aspect = abs(min(w, h) / max(w, h) - 1) if max(w, h) > 0 else 1.0
        # 래스터화된 완전한 원도 circularity 가 0.9 안팎이라 0.92 기준으로 모자란 만큼만 더한다
        return "circle", max(aspect, max(0.0, 0.92 - circularity) * 2), approx
    return None


REGULAR_TYPES = ("triangle", "square", "cross", "circle", "pentagon", "hexagon")  # regular_only 에서 허용


class ShapeDetector:
    def __init__(self, options: ShapeOptions = ShapeOptions()):
        self.options = options

    def detect(self, gray: np.ndarray, invert: bool = False, scale: float = 1.0,
               options: ShapeOptions | None = None) -> tuple[list[ShapeDet], int]:
        """(검출 목록, 조건으로 떨어진 후보 수)."""
        o = options or self.options
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mode = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV  # 기본: 어두운 도형 → 흰색
        _, bw = cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = gray.shape[0] * gray.shape[1]
        min_px = o.min_px / scale  # 검출 영상 픽셀로
        out, rejected = [], 0
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
            if (o.types != "all" and name != o.types) or (o.regular_only and (dev > o.tol or name not in REGULAR_TYPES)):
                rejected += 1
                continue
            m = cv2.moments(c)
            cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (0.0, 0.0)
            out.append(ShapeDet(name, c.reshape(-1, 2).astype(np.float64) * scale, (cx * scale, cy * scale),
                                area * scale * scale, min(w, h) * scale, dev))
        out.sort(key=lambda s: -s.area)
        return out, rejected


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
