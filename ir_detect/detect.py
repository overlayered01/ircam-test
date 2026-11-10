"""IR 8비트 영상에서 AprilTag 와 기본 도형(삼각형·사각형·십자·오각형·육각형·원)을 찾는다.

- AprilTag: pupil-apriltags(AprilTag 3) 가 있으면 그것을, 없으면 OpenCV ArUco 의 AprilTag 사전을 쓴다.
  두 검출기 모두 검은 테두리를 찾으므로 흰 테두리(inv) 태그나 밝은 배경의 어두운 태그는 invert 로 영상을 뒤집어 넣는다.
  pupil 결과는 hamming == 0, decision_margin >= min_margin 으로 거른다 (16h5 는 오검출이 많다).
- 도형: Otsu 이진화 → 외곽선 → 다각형 근사. 기본은 밝은 바탕(종이)의 어두운 도형, invert 면 어두운 바탕의 밝은 도형.
  도형마다 '정형 편차'(dev) 를 잰다: 변 길이·대각선·팔 길이가 이상적인 정삼각형/정사각형/정십자/원에서 얼마나 벗어났는지의 최대 비율.
  ShapeOptions.regular_only 면 dev <= tol 인 것만 남긴다 (예: tol 0.1 = 10% 이내).
- 점 마커: 인쇄한 검정 점 1·2·3개를 지름 배수 거리로 묶어 개수(=ID)와 배치(방향)를 읽는다. DotDetector 참고.
- 큰 영상(DMK 2448x2048)은 max_width 로 줄여서 검출하고 좌표를 원본 스케일로 돌려준다.
"""

from __future__ import annotations

import threading
import time
from collections import deque
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
TAG_DECIMATES = (1.0, 2.0, 4.0)  # pupil quad_decimate: 사각형 후보 탐색 해상도 1/n (코너는 원본에서 보정)
TAG_SHARPENS = (0.25, 0.5, 1.0)  # pupil decode_sharpening: 비트 읽기 전 선명화. 1.0 이면 작은(18~22px)·흐린·회색 태그 검출률 ↑, 시간 동일
TAG_HAMMINGS = (0, 1, 2)  # 허용 비트 오류 수. 36h11(코드 간 거리 11)·25h9 는 1 까지 안전, 16h5(거리 5) 는 0 권장
TAG_MARGINS = (10, 20, 40, 80)  # pupil decision_margin 최소. 제대로 보이는 태그는 100 이상, 오검출은 대개 10~40
TAG_CONFIRMS = (1, 2, 3, 5)  # 같은 ID 가 같은 자리에 연속 n 프레임 나와야 인정 (1 = 바로)


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
    rate: float | None = None  # ShapeTracker: 최근 프레임 중 검출된 비율 (0~1)
    held: bool = False  # ShapeTracker: 이번 프레임엔 없지만 직전까지 보여 유지 중

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
    tags_pending: int = 0  # TagTracker 에서 아직 confirm 프레임을 못 채운 태그 수
    dots: list["DotMarker"] = field(default_factory=list)
    dots_rejected: int = 0  # 점 후보 중 모양·대비 조건으로 떨어진 것 + max_count 초과 그룹의 점
    dots_pending: int = 0  # 아직 confirm 프레임을 못 채운 마커 수

    def summary(self) -> str:
        parts = []
        if self.dots or self.dots_rejected or self.dots_pending:
            kinds = ",".join(m.kind for m in self.dots[:8]) + (",…" if len(self.dots) > 8 else "")
            parts.append(f"dots {len(self.dots)}" + (f" [{kinds}]" if self.dots else "") + (f" (-{self.dots_rejected})" if self.dots_rejected else "")
                         + (f" (+{self.dots_pending} new)" if self.dots_pending else ""))
        if self.backend:
            ids = ",".join(str(t.id) for t in self.tags[:8]) + (",…" if len(self.tags) > 8 else "")
            parts.append(f"tags {len(self.tags)}" + (f" [{ids}]" if self.tags else "") + (f" (-{self.rejected})" if self.rejected else "")
                         + (f" (+{self.tags_pending} new)" if self.tags_pending else ""))
        if self.shapes or self.shapes_rejected:
            counts: dict[str, int] = {}
            for s in self.shapes:
                counts[s.name] = counts.get(s.name, 0) + 1
            parts.append("shapes " + (" ".join(f"{k}:{v}" for k, v in counts.items()) or "0") + (f" (-{self.shapes_rejected})" if self.shapes_rejected else ""))
            rates = [s.rate for s in self.shapes if s.rate is not None]
            if rates:
                parts.append(f"hit {np.mean(rates) * 100:.0f}%")
        return "  ".join(parts) + f"  {self.ms:.0f}ms"


def prepare(gray: np.ndarray, max_width: int = 1280, upscale: float = 1.0) -> tuple[np.ndarray, float]:
    """검출용으로 줄인(또는 upscale 배 키운) 영상과 (원본 / 검출 영상) 배율.

    upscale 2 는 셀이 2~3px 밖에 안 되는 작은 태그용 (18px 태그 0/10 → 9/10). 픽셀이 4배라 시간도 4배 — 640x576 급에서 쓴다."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if upscale > 1.0:
        return cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC), 1.0 / upscale
    if gray.shape[1] <= max_width:
        return gray, 1.0
    s = max_width / gray.shape[1]
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 1.0 / s


class AprilTagDetector:
    def __init__(self, family: str = "tag36h11", backend: str = "auto", min_margin: float = 10.0, decimate: float = 1.0,
                 sharpen: float = 1.0):
        """decimate = pupil quad_decimate. 2 면 후보 탐색이 1/4 픽셀이라 4~6배 빠르지만 읽히는 최소 태그 크기가 2배로 커진다
        (36h11 기준 18px → 36px). 코너는 원본 해상도에서 다시 맞추므로 위치 정밀도는 거의 그대로.
        sharpen = pupil decode_sharpening (라이브러리 기본 0.25). 1.0 이면 18px 36h11 이 0/10 → 8/10, 회색(대비 60/150)
        20px 가 5/10 → 9/10, 큰 태그·시간은 그대로. 16h5 는 sharpen 에서 드물게 엉뚱한 ID 가 나오니 36h11 권장."""
        if family not in _CV_DICT:
            raise ValueError(f"지원하지 않는 계열: {family}")
        self.family, self.min_margin, self.decimate, self.sharpen = family, min_margin, decimate, sharpen
        self.max_hamming = 0  # 흐린 작은 태그는 한 비트가 틀려 hamming 1 로 나온다 (Pipeline.tag_hamming 으로 바꾼다)
        self._pupil = None
        if backend in ("auto", "pupil"):
            try:
                from pupil_apriltags import Detector

                # pupil-apriltags 는 tag36h10 을 모른다 → OpenCV 로
                self._pupil = Detector(families=family, nthreads=2, quad_decimate=decimate, refine_edges=True,
                                       decode_sharpening=sharpen)
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
                if d.hamming > self.max_hamming or d.decision_margin < self.min_margin:
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


def _spread_q(v: np.ndarray, px: float = 1.0) -> float:
    """픽셀 길이들의 편차에서 양자화 몫(px / 평균 길이)을 뺀 값. 20px 변은 1px 이 이미 5% 라 그만큼은 봐준다."""
    m = float(np.mean(v))
    return max(0.0, _spread(v) - px / m) if m > 0 else 1.0


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
    """외곽선 → (이름, 정형 편차, 근사 다각형). 해당 없으면 None.

    둘레 비율 epsilon 으로 꼭짓점을 잡는다. 작은 도형(20px 안팎)은 모서리가 뭉개져 꼭짓점이 남거나(사각형 → 육각형)
    잘린다(삼각형 꼭대기 → 사다리꼴). 그래서 첫 해석이 없거나 편차가 15% 를 넘으면 2배·3배 epsilon 으로도 보고
    편차가 가장 작은 해석을 고른다. 깨끗한 원이 거친 근사에서 육각형으로 둔갑하지 않게 좋은 해석은 그대로 둔다.
    """
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if peri <= 0 or area <= 0:
        return None
    best = _classify_at(contour, epsilon, peri, area)
    for k in (2, 3):
        if best is not None and (best[1] <= 0.15 or best[0] == "cross"):
            break  # 좋은 해석이거나 십자(비정형이어도 거친 근사로는 마름모가 될 뿐)면 그대로
        r = _classify_at(contour, k * epsilon, peri, area)
        # 거친 근사는 넓이가 원래 외곽선과 비슷할 때만 믿는다(0.7~1.3배): 십자를 팔 끝 4점으로 이으면 마름모(정사각형)가
        # 되는데 넓이가 두 배라 걸러진다. 꼭짓점이 외곽선 위에 찍히므로 모서리가 둥근 도형의 내접 다각형은 20% 쯤 작다.
        if r is not None and 0.7 <= cv2.contourArea(r[2].astype(np.float32)) / area <= 1.3 and (best is None or r[1] < best[1]):
            best = r
    return best


def _classify_at(contour: np.ndarray, epsilon: float, peri: float, area: float) -> tuple[str, float, np.ndarray] | None:
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
    if convex and 3 <= n <= 6:
        # 볼록 다각형 도형은 외곽선 자체도 볼록해야 한다. 울퉁불퉁한 덩어리(얼굴, 손)가 거친 근사에서 사각형으로
        # 둔갑하지 않게 solidity(넓이 / 볼록 껍질 넓이)로 거른다.
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or area / hull_area < 0.85:
            return None

    if n == 3 and convex:
        return "triangle", _spread_q(_edge_lengths(approx)), approx
    if n == 4 and convex:
        sides = _edge_lengths(approx)
        diag = np.array([np.linalg.norm(approx[0] - approx[2]), np.linalg.norm(approx[1] - approx[3])])
        diag_dev = _spread_q(diag)  # 대각선이 같으면 직사각형 (각이 90°)
        side_dev = _spread_q(sides)
        opp_dev = max(abs(sides[0] / sides[2] - 1), abs(sides[1] / sides[3] - 1))  # 마주보는 변
        if side_dev <= 0.25:
            return "square", max(side_dev, diag_dev), approx
        return "rect", max(opp_dev, diag_dev), approx
    if n == 5 and convex:
        return "pentagon", _spread_q(_edge_lengths(approx)), approx
    if n == 6 and convex:
        return "hexagon", _spread_q(_edge_lengths(approx)), approx
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
    # 가장 넓은 끝 하나는 뺀다: 팔 끝이 옆 글자나 손가락에 붙으면 그 끝만 넓어지는데 팔 길이·종횡비가 이미 십자를 보증한다
    width_dev = _spread_q(np.sort(lengths[ends])[:3], 2.0)
    # 팔 길이는 중심에서 팔 끝(끝변의 중점)까지로 잰다. 안쪽 오목 꼭짓점은 가는 획에서 위치가 몇 px 씩 흔들려
    # 옆변 길이로 재면 20% 넘게 튀지만 팔 끝은 안정적이다.
    centroid = approx.mean(axis=0)
    tips = np.array([(approx[i] + approx[(i + 1) % n]) / 2 for i in range(n) if ends[i]])
    arms = np.linalg.norm(tips - centroid, axis=1)
    return "cross", max(width_dev, _spread_q(arms), aspect), approx


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
        # RETR_EXTERNAL 이면 어두운 배경(= 흰 덩어리) 안의 밝은 종이(구멍) 위에 있는 도형은 돌아오지 않는다.
        # CCOMP 의 최상위(parent == -1)는 구멍 안의 덩어리도 포함한다.
        contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return []
        return [c for c, h in zip(contours, hier[0]) if h[3] == -1]

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
            if parent != -1:
                continue
            stroke = _dark_stroke(gray, bw0, contours[i], invert)
            if stroke is None:
                continue
            if child == -1:
                out.append(contours[i])  # 구멍 없는 획 (십자)
            while child != -1:
                # 닫힌 윤곽의 안쪽. 안쪽이 획보다 밝아야 선 도형이다 — 채운 도형은 가장자리 안쪽에 띠가 생기고
                # 그 구멍은 어두우므로 여기서 걸러 filled 결과와 겹치지 않는다.
                inner = _inside_mean(gray, contours[child]) - stroke
                if (-inner if invert else inner) > 8.0:
                    out.append(contours[child])
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


def _inside_mean(gray: np.ndarray, contour: np.ndarray) -> float:
    """외곽선 안쪽 평균 밝기."""
    x, y, w, h = cv2.boundingRect(contour)
    roi = np.zeros((h, w), np.uint8)
    cv2.drawContours(roi, [contour - (x, y)], -1, 255, -1)
    m = roi > 0
    return float(gray[y:y + h, x:x + w][m].mean()) if m.any() else 0.0


def _dark_stroke(gray: np.ndarray, bw: np.ndarray, contour: np.ndarray, invert: bool, margin: float = 8.0) -> float | None:
    """획(이진화 흰 부분)이 바로 바깥 띠보다 어두우면 획의 평균 밝기, 아니면 None (invert 면 밝아야 함).
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
        return None
    g = gray[y0:y1, x0:x1]
    stroke_mean = float(g[stroke].mean())
    diff = float(g[band].mean()) - stroke_mean
    return stroke_mean if (-diff if invert else diff) > margin else None


def _same_spot(a: ShapeDet, b: ShapeDet) -> bool:
    """중심이 작은 쪽 짧은 변의 절반 안에 있고 크기도 비슷(1.5배 이내)해야 같은 도형. 큰 후보(종이 전체 등)가
    그 안의 작은 도형을 삼키지 않게 한다."""
    small, big = sorted((a.size_px, b.size_px))
    if small <= 0 or big > 1.5 * small:
        return False
    d = np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
    return d < 0.5 * small


# ---------------------------------------------------------------- 점(dot) 마커 ----------------------------------------------------------------
# 인쇄한 검정 점 1·2·3개를 한 마커로 읽는다. 점 하나는 지름 6 px 이상이면 안정적으로 잡히므로(4~5 px 는 들쭉날쭉, 태그는 한 변 18~22 px)
# 같은 크기 마커로 태그보다 약 3배 거리까지 간다. 합성 검증: 6·9·12 px 점은 잡음 σ3·흐림 0.8 에서 5/5, 3~5 px 는 0~1/5.
# 묶기 기준은 절대 거리가 아니라 점 지름의 배수라 거리(스케일)에 따라 자동으로 맞는다. 방향은 점 배치에서 읽는다:
#   1점: 위치만.  2점: 크기가 다르면(큰 점 → 작은 점) 방향 확정, 같으면 축만(180° 모호).
#   3점 일직선: 간격이 다르면(1:2) 외톨이 점 → 붙은 두 점 쪽이 방향.  3점 L 자: 꼭짓점(내각이 가장 큰 점)에서 나머지 둘의 중점 쪽.
DOT_MIN_PX = (3, 4, 6, 8, 12)  # 점 지름 최소 (원본 픽셀). DMK 2448 에서 2 m·4 mm 점 ≈ 6 px → Detect width full 로
DOT_GAPS = (2.5, 3.5, 5.0)  # 같은 마커로 묶는 중심 간 거리 상한 (점 지름 배수). 마커끼리는 이보다 멀리 둔다
DOT_MAX_COUNT = 3
DOT_MIN_COUNTS = (1, 2)  # 1점 마커도 받을지. 멀리서 글자 조각·얼룩이 1점으로 읽히면 2 로


@dataclass(frozen=True)
class DotOptions:
    min_px: int = 4  # 점 지름 최소 (원본 픽셀)
    gap: float = 3.5  # 묶기 거리 (지름 배수)
    min_count: int = 1  # 이보다 적은 점 그룹(1점)은 버린다
    max_count: int = DOT_MAX_COUNT  # 이보다 많이 뭉친 그룹은 버린다 (글자 등)
    max_px_frac: float = 0.05  # 점 지름 최대 = 영상 폭 × 이 값 (2448 → 122 px, 640 → 32 px)
    min_circularity: float = 0.75  # 4πA/P². 원 1.0(흐리고 작으면 0.83~), 정사각형 0.785, 글자 획은 대개 0.3 이하
    min_solidity: float = 0.88  # 면적 / 볼록 껍질 면적. 찬 점 ≈ 1, 글자(B·A)·고리(O) 는 0.5~0.7
    min_contrast: float = 20.0  # 점 안쪽과 바로 바깥 띠의 밝기 차 최소
    size_ratio: float = 2.4  # 한 마커 안 점 지름 비 상한 (큰 점+작은 점 2점 마커는 1.5~2 배)


@dataclass
class Dot:
    center: tuple[float, float]  # 밝기 가중 중심 (원본 픽셀, 서브픽셀)
    diameter: float  # 면적 등가 지름 (원본 픽셀)
    area: float
    circularity: float
    contrast: float


@dataclass
class DotMarker:
    dots: list[Dot]
    kind: str = ""  # 1 / 2 / 2s(크기 다른 두 점) / 3I(일직선) / 3L
    angle: float | None = None  # 방향 (도, 영상 x 축 기준, y 아래쪽이 +). None = 1점
    sure: bool = True  # False = 축만 알고 앞뒤는 모름 (같은 크기 2점, 등간격 3점, 정삼각형)

    @property
    def count(self) -> int:
        return len(self.dots)

    @property
    def id(self) -> int:
        """TagTracker 와 호환: 점 개수가 ID."""
        return self.count

    @property
    def center(self) -> tuple[float, float]:
        xs = [d.center[0] for d in self.dots]
        ys = [d.center[1] for d in self.dots]
        return float(np.mean(xs)), float(np.mean(ys))

    @property
    def diameter(self) -> float:
        return float(np.mean([d.diameter for d in self.dots]))

    @property
    def size_px(self) -> float:
        """마커가 차지하는 폭(점 중심 간 최대 거리 + 지름). 추적 '같은 자리' 기준으로 쓰므로 최소 지름의 3배."""
        pts = np.array([d.center for d in self.dots])
        span = max((float(np.linalg.norm(a - b)) for a in pts for b in pts), default=0.0) + self.diameter
        return max(span, 3 * self.diameter)


def _angle_deg(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def classify_dots(dots: list[Dot]) -> DotMarker:
    """점 1~3개 → 종류·방향."""
    m = DotMarker(dots)
    pts = np.array([d.center for d in dots], float)
    if len(dots) == 1:
        m.kind = "1"
        return m
    if len(dots) == 2:
        big, small = sorted(dots, key=lambda d: -d.diameter)
        if big.diameter >= 1.35 * small.diameter:
            m.kind, m.sure = "2s", True
            m.angle = _angle_deg(np.array(small.center) - np.array(big.center))
        else:
            m.kind, m.sure = "2", False
            m.angle = _angle_deg(pts[1] - pts[0]) % 180.0
        return m
    # 3점: 가장 먼 두 점 P,Q 와 가운데 점 M
    d = {(i, j): float(np.linalg.norm(pts[i] - pts[j])) for i in range(3) for j in range(i + 1, 3)}
    (p, q), pq = max(d.items(), key=lambda kv: kv[1])
    mid = 3 - p - q
    pm, mq = d[tuple(sorted((p, mid)))], d[tuple(sorted((mid, q)))]
    u, v = pts[q] - pts[p], pts[mid] - pts[p]
    off = abs(float(u[0] * v[1] - u[1] * v[0])) / max(pq, 1e-6)  # M 의 직선 PQ 이탈 (2D 외적 / |PQ|)
    if off < 0.15 * pq:
        m.kind = "3I"
        near, far = (p, q) if pm < mq else (q, p)  # near = 가운데 점에 가까운 끝
        if min(pm, mq) / max(pm, mq) > 0.8:
            m.sure, m.angle = False, _angle_deg(pts[q] - pts[p]) % 180.0
        else:
            m.sure, m.angle = True, _angle_deg(pts[near] - pts[far])  # 외톨이 끝 → 붙은 두 점 쪽
        return m
    m.kind = "3L"
    angs = []
    for i in range(3):
        a, b = pts[(i + 1) % 3] - pts[i], pts[(i + 2) % 3] - pts[i]
        cosv = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6)
        angs.append(float(np.degrees(np.arccos(np.clip(cosv, -1, 1)))))
    order = np.argsort(angs)[::-1]
    corner = int(order[0])
    m.sure = angs[order[0]] - angs[order[1]] > 10.0
    others = (pts[(corner + 1) % 3] + pts[(corner + 2) % 3]) / 2
    m.angle = _angle_deg(others - pts[corner])
    return m


def group_dots(dots: list[Dot], gap: float, size_ratio: float) -> list[list[Dot]]:
    """중심 간 거리 ≤ gap × 평균 지름이고 지름 비가 size_ratio 이내인 점끼리 이어 붙인다 (단일 연결)."""
    n = len(dots)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = dots[i], dots[j]
            if max(a.diameter, b.diameter) > size_ratio * min(a.diameter, b.diameter):
                continue
            dist = float(np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]))
            if dist <= gap * 0.5 * (a.diameter + b.diameter):
                parent[find(i)] = find(j)
    groups: dict[int, list[Dot]] = {}
    for i, d in enumerate(dots):
        groups.setdefault(find(i), []).append(d)
    return list(groups.values())


class DotDetector:
    """인쇄한 검정 점(invert 면 밝은 점)을 찾아 1·2·3점 마커로 묶는다.

    국소(adaptive) 이진화 → 연결 성분 → 크기·종횡비·채움비·원형도·대비로 점만 남김 → 밝기 가중 서브픽셀 중심 →
    지름 배수 거리로 묶기 → 개수별 분류. 글자·선·큰 어두운 영역은 원형도/크기에서, 우연히 붙은 잡음은 지름 비·개수에서 떨어진다.
    """

    def __init__(self, options: DotOptions = DotOptions()):
        self.options = options

    def detect(self, gray: np.ndarray, invert: bool = False, scale: float = 1.0,
               options: DotOptions | None = None) -> tuple[list[DotMarker], int]:
        """(마커 목록, 떨어진 후보 수: 점 조건 탈락 + max_count 초과 그룹의 점)."""
        o = options or self.options
        dots, rejected = self._find_dots(gray, invert, scale, o)
        markers: list[DotMarker] = []
        for g in group_dots(dots, o.gap, o.size_ratio):
            if len(g) > o.max_count or len(g) < o.min_count:
                rejected += len(g)
                continue
            markers.append(classify_dots(g))
        markers.sort(key=lambda m: (-m.count, m.center))
        return markers, rejected

    @staticmethod
    def _find_dots(gray: np.ndarray, invert: bool, scale: float, o: DotOptions) -> tuple[list[Dot], int]:
        """2단계: ① 국소 평균보다 10 어두운 곳을 거칠게 떼어 후보 덩어리를 찾고, ② 덩어리마다 (핵심 밝기 + 바깥 밝기)/2 로
        다시 이진화한다. 흐려서 붙은 두 점은 사이 골이 중간 밝기보다 밝아 ②에서 갈라지고, 면적·지름도 ①보다 정확하다.
        DMK 2448x2048 전체에서 후보 2000개짜리 지저분한 장면도 40 ms 안쪽이 되도록 후보마다 하는 일을 줄였다."""
        h, w = gray.shape[:2]
        min_d = o.min_px / scale  # 검출 영상 픽셀
        max_d = max(min_d * 2, w * o.max_px_frac)
        mode = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
        src = cv2.GaussianBlur(gray, (3, 3), 0) if min_d >= 5 else gray
        # 국소 이진화 창 두 개의 합집합: 큰 창(4 × 최대 점)은 큰 점용, 작은 창(12 × 최소 점)은 종이 가장자리 근처 작은 점용 —
        # 큰 창 하나만 쓰면 종이 밖 어두운 배경이 국소 평균을 끌어내려 종이 가장자리 20 % 안쪽의 회색 점이 사라진다
        c = -10 if invert else 10
        big = int(max(11, (int(max_d * 4) // 2) * 2 + 1))
        small_ = int(max(15, (int(min_d * 12) // 2) * 2 + 1))
        bw = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, big, c)
        if small_ < big:
            bw |= cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, small_, c)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        min_area = np.pi * (min_d / 2) ** 2 * 0.5
        max_area = np.pi * (max_d / 2) ** 2 * 1.3
        g = gray.astype(np.float32)
        k3 = np.ones((3, 3), np.uint8)
        out: list[Dot] = []
        rejected = 0
        if n <= 1:
            return out, rejected
        # 1단계 거르기 (벡터): 면적, 변 길이, 종횡비 ≤ 1+gap (붙은 두세 점은 길쭉하다), 채움비 ≥ 0.3.
        # 점 하나보다 큰 덩어리는 길쭉할 때(붙은 점들)만 받는다 — 큰 어두운 영역에 2단계를 돌리면 느리다
        st = stats[1:]
        bw_, bh, area = st[:, 2].astype(np.float64), st[:, 3].astype(np.float64), st[:, 4].astype(np.float64)
        long_, short = np.maximum(bw_, bh), np.minimum(bw_, bh)
        aspect0 = long_ / short
        ok = ((area >= min_area) & (area <= max_area * 3) & (short >= min_d * 0.7) & (long_ <= max_d * (1 + o.gap))
              & (aspect0 <= 1 + o.gap) & (area / (bw_ * bh) >= 0.3) & ((long_ <= max_d * 1.2) | (aspect0 >= 1.3)))
        for i in np.nonzero(ok)[0] + 1:
            x, y, cw, ch = (int(v) for v in stats[i, :4])
            pad = 2
            x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + cw + pad), min(h, y + ch + pad)
            mask = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
            roi = g[y0:y1, x0:x1]
            grown = cv2.dilate(mask, k3)
            ring = (cv2.dilate(grown, k3) > 0) & (mask == 0)
            vals = roi[mask > 0]
            mean = float(vals.sum()) / vals.size
            half = vals[vals >= mean] if invert else vals[vals <= mean]  # 안쪽 진한 쪽 절반 (가장자리 반쯤 어두운 픽셀 제외)
            core = float(half.sum()) / half.size if half.size else mean
            rv = roi[ring]
            outside = float(rv.sum()) / rv.size if rv.size else core
            contrast = (core - outside) if invert else (outside - core)
            if contrast < o.min_contrast:
                rejected += 1
                continue
            mid = 0.5 * (core + outside)
            sub = (((roi >= mid) if invert else (roi <= mid)) & (grown > 0)).astype(np.uint8)
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
            wgt_all = np.clip((roi - outside) if invert else (outside - roi), 0, None)
            for j in range(1, n2):
                sx, sy, sw, sh, sa = (int(v) for v in st2[j])
                # 중간 밝기로 자르면 가장자리가 빠져 1단계보다 작다. 지름 4~5 px 점은 중심이 픽셀 경계에 걸리면 2x2 만 남으므로 면적 기준은 느슨하게
                if sa < min_area * 0.6 or sa > max_area or min(sw, sh) < min_d * 0.5 or max(sw, sh) > max_d:
                    continue
                aspect = sw / sh if sw > sh else sh / sw
                fill = sa / float(sw * sh)  # 원 0.785. 축에 나란한 사각형은 ~1 이라 큰 덩어리는 0.9 에서 자른다
                # 지름 6~9 px 원은 양자화 때문에 채움비가 0.9 를 넘기도 한다 → 80 px 이상 큰 덩어리에서만 0.9 로 사각형을 자른다
                if aspect > 1.6 or fill < 0.55 or fill > (1.0 if sa < 12 else 0.97 if sa < 80 else 0.9):
                    rejected += 1
                    continue
                m2 = (lab2 == j).astype(np.uint8)
                contours, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=len)
                per = cv2.arcLength(cnt, True)
                circ = 4 * np.pi * sa / (per * per) if per > 0 else 0.0
                if sa < 12:  # 아주 작은 점은 둘레가 거칠어 원형도가 낮게 나온다
                    circ = max(circ, fill / 0.785)
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                solidity = sa / hull_area if hull_area > 0 else 1.0
                if circ < o.min_circularity or solidity < o.min_solidity:
                    rejected += 1
                    continue
                # 밝기 가중 중심 (서브픽셀) + 유효 면적 (가장자리 반쯤 어두운 픽셀은 반만 센다)
                wgt = np.where(cv2.dilate(m2, k3) > 0, wgt_all, 0.0).astype(np.float32)
                mom = cv2.moments(wgt, binaryImage=False)
                if mom["m00"] <= 0:
                    continue
                cx, cy = x0 + mom["m10"] / mom["m00"], y0 + mom["m01"] / mom["m00"]
                eff_area = mom["m00"] / max(contrast, 1.0)
                diam = 2 * np.sqrt(eff_area / np.pi)
                if diam < min_d * 0.85:
                    continue
                out.append(Dot(((cx + 0.5) * scale, (cy + 0.5) * scale), diam * scale, eff_area * scale * scale, float(circ), float(contrast)))
        return out, rejected


class ShapeTracker:
    """연속 프레임의 도형을 자리(이름 + 중심 거리 < 짧은 변의 절반)로 이어 붙여 검출률을 센다.

    rate = 최근 window 프레임 중 검출된 비율. 이번 프레임에 안 보여도 hold 프레임 안에 보였으면 마지막 모양을
    held=True 로 돌려줘 화면이 깜빡이지 않는다. window 프레임 동안 한 번도 안 보이면 지운다.
    """

    @dataclass
    class Track:
        last: ShapeDet
        seen: deque  # 최근 프레임마다 True/False
        miss: int = 0  # 연속으로 안 보인 프레임 수

        @property
        def rate(self) -> float:
            return float(np.mean(self.seen)) if self.seen else 0.0

    def __init__(self, window: int = 30, hold: int = 10):
        self.window, self.hold = window, hold
        self.tracks: list[ShapeTracker.Track] = []

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, shapes: list[ShapeDet]) -> list[ShapeDet]:
        """이번 프레임 검출 목록 → rate 가 붙은 목록 (유지 중인 것 포함)."""
        matched: set[int] = set()
        out: list[ShapeDet] = []
        for s in shapes:
            best = None
            for i, t in enumerate(self.tracks):
                if i in matched or t.last.name != s.name or not _same_spot(s, t.last):
                    continue
                if best is None or s.area > 0 and abs(t.last.area - s.area) < abs(self.tracks[best].last.area - s.area):
                    best = i
            if best is None:
                self.tracks.append(self.Track(s, deque([True], maxlen=self.window)))
                best = len(self.tracks) - 1
            else:
                t = self.tracks[best]
                t.last, t.miss = s, 0
                t.seen.append(True)
            matched.add(best)
            s.rate, s.held = self.tracks[best].rate, False
            out.append(s)
        keep = []
        for i, t in enumerate(self.tracks):
            if i not in matched:
                t.seen.append(False)
                t.miss += 1
                if t.miss > self.window or not any(t.seen):
                    continue
                if t.miss <= self.hold:
                    held = ShapeDet(t.last.name, t.last.contour, t.last.center, t.last.area, t.last.size_px, t.last.dev, t.rate, True)
                    out.append(held)
            keep.append(t)
        self.tracks = keep
        return out


class TagTracker:
    """연속 프레임에서 같은 ID 가 같은 자리에 confirm 번 이어서 나와야 태그로 인정한다.

    16h5 처럼 코드가 적은 계열은 배경 무늬가 한 프레임짜리 가짜 ID 로 읽히는 일이 있다. 진짜 태그는 매 프레임 같은 자리에
    같은 ID 로 나오므로 두세 프레임만 이어서 보면 가짜가 걸러진다. 한 프레임 놓친 것은 봐주고(streak 유지), 두 프레임 연속
    안 보이면 트랙을 지운다. 같은 자리 = 중심 거리가 태그 한 변 이하. id·center·size_px 만 쓰므로 DotMarker(id = 점 개수)에도 쓴다.
    """

    @dataclass
    class Track:
        last: TagDet
        streak: int = 1  # 연속으로 보인 프레임 수
        miss: int = 0

    def __init__(self, max_miss: int = 1):
        self.max_miss = max_miss
        self.tracks: list[TagTracker.Track] = []

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, tags: list[TagDet], confirm: int = 1) -> tuple[list[TagDet], int]:
        """이번 프레임 검출 목록 → (confirm 을 채운 태그, 아직 못 채운 수)."""
        matched: set[int] = set()
        out, pending = [], 0
        for t in tags:
            best, best_d = None, 0.0
            for i, tr in enumerate(self.tracks):
                if i in matched or tr.last.id != t.id:
                    continue
                d = float(np.hypot(tr.last.center[0] - t.center[0], tr.last.center[1] - t.center[1]))
                if d <= max(tr.last.size_px, t.size_px) and (best is None or d < best_d):
                    best, best_d = i, d
            if best is None:
                self.tracks.append(self.Track(t))
                best = len(self.tracks) - 1
            else:
                tr = self.tracks[best]
                tr.last, tr.streak, tr.miss = t, tr.streak + 1, 0
            matched.add(best)
            if self.tracks[best].streak >= confirm:
                out.append(t)
            else:
                pending += 1
        keep = []
        for i, tr in enumerate(self.tracks):
            if i not in matched:
                tr.miss += 1
                if tr.miss > self.max_miss:
                    continue
            keep.append(tr)
        self.tracks = keep
        return out, pending


class Pipeline:
    """메뉴 설정 → 검출기 묶음. 계열이 바뀌면 태그 검출기를 다시 만든다. track=True 면 도형을 프레임 간 추적해 검출률을 붙인다."""

    def __init__(self, backend: str = "auto", max_width: int = 1280, track: bool = False, tag_decimate: float = 1.0,
                 tag_sharpen: float = 1.0, upscale: float = 1.0, tag_hamming: int = 0, tag_margin: float = 10.0,
                 tag_confirm: int = 1):
        self.backend, self.max_width, self.tag_decimate, self.tag_sharpen = backend, max_width, tag_decimate, tag_sharpen
        self.upscale = upscale  # > 1 이면 max_width 대신 그 배로 키워서 검출
        self.tag_hamming, self.tag_margin, self.tag_confirm = tag_hamming, tag_margin, tag_confirm
        self._tag: AprilTagDetector | None = None
        self._shape = ShapeDetector()
        self._dot = DotDetector()
        self.tracker = ShapeTracker() if track else None
        self.tag_tracker = TagTracker() if track else None
        self.dot_tracker = TagTracker() if track else None  # 점 마커도 같은 방식(개수 = ID)으로 confirm

    def run(self, gray: np.ndarray, tags: bool, shapes: bool, family: str = "tag36h11", invert: bool = False,
            shape_options: ShapeOptions | None = None, dots: bool = False, dot_options: DotOptions | None = None) -> DetResult:
        t0 = time.perf_counter()
        res = DetResult()
        if not (tags or shapes or dots):
            for tr in (self.tracker, self.tag_tracker, self.dot_tracker):
                if tr is not None:
                    tr.reset()
            return res
        small, scale = prepare(gray, self.max_width, self.upscale)
        if dots:
            res.dots, res.dots_rejected = self._dot.detect(small, invert, scale, dot_options)
            if self.dot_tracker is not None:
                res.dots, res.dots_pending = self.dot_tracker.update(res.dots, self.tag_confirm)  # type: ignore[arg-type]
        elif self.dot_tracker is not None:
            self.dot_tracker.reset()
        if tags:
            if (self._tag is None or self._tag.family != family or self._tag.decimate != self.tag_decimate
                    or self._tag.sharpen != self.tag_sharpen):
                self._tag = AprilTagDetector(family, self.backend, decimate=self.tag_decimate, sharpen=self.tag_sharpen)
            self._tag.max_hamming, self._tag.min_margin = self.tag_hamming, self.tag_margin
            res.tags, res.rejected = self._tag.detect(small, invert, scale)
            res.backend = self._tag.backend
            if self.tag_tracker is not None:
                res.tags, res.tags_pending = self.tag_tracker.update(res.tags, self.tag_confirm)
        elif self.tag_tracker is not None:
            self.tag_tracker.reset()
        if shapes:
            res.shapes, res.shapes_rejected = self._shape.detect(small, invert, scale, shape_options)
            if self.tracker is not None:
                res.shapes = self.tracker.update(res.shapes)
        elif self.tracker is not None:
            self.tracker.reset()
        res.ms = (time.perf_counter() - t0) * 1000
        return res


class DetectWorker:
    """Pipeline.run 을 별도 스레드에서 돌린다. submit() 은 바로 돌아오고 result 에 마지막 결과가 남는다.

    검출이 한 프레임 시간보다 길면(DMK 2448px 에서 AprilTag 46 ms vs 프레임 41 ms) 같은 스레드에서는 다음 프레임을 놓쳐
    표시가 절반 fps 로 떨어지고 간격이 튄다. 여기서는 검출 중에 들어온 프레임은 버리고(skipped) 항상 최신 프레임만 처리해
    영상은 카메라 fps 그대로 흐르고 검출 결과만 한두 프레임 늦게 따라온다. threaded=False 면 submit 안에서 바로 돌린다 (테스트용).
    """

    def __init__(self, pipeline: Pipeline, threaded: bool = True):
        self.pipeline = pipeline
        self.result = DetResult()
        self.skipped = 0  # 검출 중이라 버린 프레임 수 (누적)
        self._pending: tuple | None = None
        self._busy = False
        self._stop = False
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._loop, daemon=True) if threaded else None
        if self._thread is not None:
            self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, gray: np.ndarray, tags: bool, shapes: bool, family: str = "tag36h11", invert: bool = False,
               shape_options: ShapeOptions | None = None, max_width: int = 1280, dots: bool = False,
               dot_options: DotOptions | None = None, **params) -> bool:
        """검출을 요청한다. 스레드가 바쁘면 False 를 돌려주고 이 프레임은 건너뛴다.
        params 는 Pipeline 속성(tag_decimate, tag_sharpen, upscale, tag_hamming, tag_margin, tag_confirm)."""
        job = (gray, tags, shapes, family, invert, shape_options, max_width, dots, dot_options, params)
        if self._thread is None:
            self._run(*job)
            return True
        with self._cv:
            if self._busy or self._pending is not None:
                self.skipped += 1
                return False
            self._pending = job
            self._cv.notify()
        return True

    def _run(self, gray, tags, shapes, family, invert, shape_options, max_width, dots, dot_options, params) -> None:
        self.pipeline.max_width = max_width or 10**9
        for k, v in params.items():
            setattr(self.pipeline, k, v)
        self.result = self.pipeline.run(gray, tags, shapes, family, invert, shape_options, dots, dot_options)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job, self._pending, self._busy = self._pending, None, True
            try:
                self._run(*job)
            except Exception as e:  # noqa: BLE001  검출 오류로 뷰어가 죽지 않게
                print(f"detect error: {e}")
            finally:
                self._busy = False

    def wait(self, timeout: float = 5.0) -> None:
        """대기 중/진행 중인 검출이 끝날 때까지 (테스트·스냅샷용)."""
        t_end = time.perf_counter() + timeout
        while (self._busy or self._pending is not None) and time.perf_counter() < t_end:
            time.sleep(0.002)

    def close(self) -> None:
        if self._thread is None:
            return
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=2)


def draw(img: np.ndarray, res: DetResult, s: float = 1.0, thickness: int = 2) -> None:
    """원본 좌표의 결과를 s 배로 줄인 표시 영상 위에 그린다."""
    for t in res.tags:
        pts = (t.corners * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (0, 255, 0), thickness)
        cv2.circle(img, tuple((t.corners[0] * s).astype(int)), 4, (0, 0, 255), -1)  # 첫 코너 = 방향
        cx, cy = t.center
        label = f"{t.id}" + (f" m{t.margin:.0f}" if t.margin is not None else "")
        cv2.putText(img, label, (int(cx * s) - 10, int(cy * s) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for m in res.dots:
        color = (255, 0, 255) if m.sure else (200, 120, 255)  # 자홍 = 방향 확정, 연보라 = 축만 (앞뒤 모름)
        r = max(2, int(m.diameter * s / 2))
        for d in m.dots:
            cv2.circle(img, (int(round(d.center[0] * s)), int(round(d.center[1] * s))), r + 2, color, max(1, thickness - 1))
        cx, cy = m.center
        if m.count >= 3:
            pts = (np.array([d.center for d in m.dots]) * s).astype(np.float32)
            hull = cv2.convexHull(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [hull], True, color, 1)
        if m.angle is not None:
            L = max(12.0, m.size_px * s)
            a = np.radians(m.angle)
            tip = (int(cx * s + L * np.cos(a)), int(cy * s + L * np.sin(a)))
            if m.sure:
                cv2.arrowedLine(img, (int(cx * s), int(cy * s)), tip, color, thickness, tipLength=0.3)
            else:  # 축만: 양쪽으로 가는 선
                tail = (int(cx * s - L * np.cos(a)), int(cy * s - L * np.sin(a)))
                cv2.line(img, tail, tip, color, 1)
        label = f"D{m.kind}" + (f" {m.angle:.0f}" if m.angle is not None and m.sure else "")
        cv2.putText(img, label, (int(cx * s) + r + 6, int(cy * s) - r - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    for sh in res.shapes:
        pts = (sh.contour * s).astype(np.int32).reshape(-1, 1, 2)
        color = (255, 160, 0) if sh.regular else (0, 140, 255)  # 정형 = 주황, 비정형 = 노랑빛
        if sh.held:
            color = (160, 160, 160)  # 이번 프레임엔 못 봤지만 직전까지 보인 것
        cv2.polylines(img, [pts], True, color, max(1, thickness - 1) if sh.held else thickness)
        label = f"{sh.name} {sh.dev * 100:.0f}%" + (f" [{sh.rate * 100:.0f}%]" if sh.rate is not None else "")
        cv2.putText(img, label, (int(sh.center[0] * s) - 24, int(sh.center[1] * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
