"""AprilTag 이미지 파일 검증·검출 도우미.

태그 PNG (apriltag-imgs 형식: 셀당 1픽셀, 16h5 는 8x8 = 흰 여백 1 + 테두리 1 + 데이터 4x4) 를 읽어
- 데이터 비트가 OpenCV 사전의 어느 ID 와 일치하는지 (정상/반전 극성, 회전 포함)
- 확대·반전·회전·원근·노이즈·축소한 영상에서 실제로 검출되는지
확인한다. OpenCV ArUco 는 항상, pupil-apriltags 는 설치되어 있을 때만 쓴다.

두 검출기 모두 **검은 테두리 1셀** 을 찾는다. 흰 테두리(inv) 태그는 영상을 먼저 반전해야 하고,
바깥 흰 여백이 테두리와 붙어 2셀 두께가 되면 AprilTag 라이브러리는 디코딩에 실패한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FAMILIES = {
    "tag16h5": (cv2.aruco.DICT_APRILTAG_16h5, 4),
    "tag25h9": (cv2.aruco.DICT_APRILTAG_25h9, 5),
    "tag36h10": (cv2.aruco.DICT_APRILTAG_36h10, 6),
    "tag36h11": (cv2.aruco.DICT_APRILTAG_36h11, 6),
}


@dataclass
class TagFile:
    path: Path
    family: str
    expected_id: int | None  # 파일명 tag16_05_00001*.png 에서 읽은 ID, 못 읽으면 None
    cells: np.ndarray  # 셀 단위 uint8 (0/255), 여백 포함

    @property
    def n_data(self) -> int:
        return FAMILIES[self.family][1]

    def data_cells(self) -> np.ndarray:
        """가운데 데이터 비트 (1 = 흰)."""
        n = self.n_data
        m = (self.cells.shape[0] - n) // 2
        return (self.cells[m : m + n, m : m + n] > 127).astype(np.uint8)

    def tag_only(self, border: int = 1) -> np.ndarray:
        """여백을 떼고 테두리 `border` 셀 + 데이터만."""
        n = self.n_data + 2 * border
        m = (self.cells.shape[0] - n) // 2
        return self.cells[m : m + n, m : m + n]

    def is_inverted(self) -> bool:
        """테두리 셀이 흰색이면 반전(inv) 태그."""
        b = self.tag_only(1)
        return int(b[0, 0]) > 127


def family_from_name(name: str) -> str | None:
    """'tag16_05_00001_inv.png' → 'tag16h5'."""
    parts = name.lower().split("_")
    if len(parts) < 2 or not parts[0].startswith("tag") or not parts[1].isdigit():
        return None
    fam = f"{parts[0]}h{int(parts[1])}"
    return fam if fam in FAMILIES else None


def load_tag(path: str | Path, family: str | None = None) -> TagFile:
    path = Path(path)
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    fam = family or family_from_name(path.name)
    if fam is None:
        raise ValueError(f"{path.name}: 태그 계열을 파일명에서 알 수 없음. family= 로 지정")
    exp = None
    parts = path.stem.split("_")
    if len(parts) >= 3 and parts[2].isdigit():
        exp = int(parts[2])
    # 큰 이미지(인쇄용)면 셀 단위로 축소. apriltag-imgs 기준: 여백 1 + 테두리 1 + 데이터 + 테두리 1 + 여백 1
    cells = g
    if g.shape[0] > 32:
        n = FAMILIES[fam][1] + 4
        cells = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA)
    cells = np.where(cells > 127, 255, 0).astype(np.uint8)
    return TagFile(path, fam, exp, cells)


def dictionary(family: str) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(FAMILIES[family][0])


def dict_bits(family: str, tag_id: int) -> np.ndarray:
    d = dictionary(family)
    return cv2.aruco.Dictionary.getBitsFromByteList(d.bytesList[tag_id : tag_id + 1], d.markerSize)


def identify(tag: TagFile) -> list[tuple[int, str, int]]:
    """데이터 비트와 일치하는 (id, 'normal'|'inverted', 회전°) 목록."""
    d = dictionary(tag.family)
    data = tag.data_cells()
    out = []
    for tid in range(d.bytesList.shape[0]):
        ref = dict_bits(tag.family, tid)
        for k in range(4):
            r = np.rot90(data, k)
            if np.array_equal(r, ref):
                out.append((tid, "normal", k * 90))
            if np.array_equal(1 - r, ref):
                out.append((tid, "inverted", k * 90))
    return out


def render(cells: np.ndarray, px_per_cell: int = 40, margin_cells: int = 3, bg: int = 255) -> np.ndarray:
    """셀 이미지를 확대하고 배경색 여백을 두른다."""
    big = cv2.resize(cells, None, fx=px_per_cell, fy=px_per_cell, interpolation=cv2.INTER_NEAREST)
    m = px_per_cell * margin_cells
    return cv2.copyMakeBorder(big, m, m, m, m, cv2.BORDER_CONSTANT, value=bg)


def warp(img: np.ndarray) -> np.ndarray:
    """비스듬히 본 것처럼 원근 변형."""
    h, w = img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[40, 30], [w - 20, 60], [w - 50, h - 40], [30, h - 20]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=int(img[0, 0]))


def degrade(img: np.ndarray, contrast: float = 0.5, offset: float = 70, sigma: float = 12, seed: int = 0) -> np.ndarray:
    """대비 낮추고 가우시안 노이즈."""
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) * contrast + offset + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


class OpenCVDetector:
    name = "opencv-aruco"

    def __init__(self, family: str, border_bits: int = 1):
        p = cv2.aruco.DetectorParameters()
        p.markerBorderBits = border_bits
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._det = cv2.aruco.ArucoDetector(dictionary(family), p)

    def detect(self, img: np.ndarray) -> list[dict]:
        _, ids, _ = self._det.detectMarkers(img)
        return [] if ids is None else [{"id": int(i)} for i in ids.flatten()]


class AprilTagDetector:
    """pupil-apriltags (AprilTag 3). 미설치면 생성 시 ImportError."""

    name = "pupil-apriltags"

    def __init__(self, family: str):
        from pupil_apriltags import Detector

        self._det = Detector(families=family, nthreads=1, quad_decimate=1.0, refine_edges=True, decode_sharpening=0.25)

    def detect(self, img: np.ndarray) -> list[dict]:
        return [
            {"id": d.tag_id, "hamming": d.hamming, "margin": round(float(d.decision_margin), 1)}
            for d in self._det.detect(img)
        ]


def ids_of(dets: list[dict], max_hamming: int = 0, min_margin: float = 0.0) -> list[int]:
    """오검출 거르기: hamming 과 decision_margin 기준을 넘는 것만 (OpenCV 결과는 둘 다 없어 통과)."""
    return sorted(d["id"] for d in dets if d.get("hamming", 0) <= max_hamming and d.get("margin", float("inf")) >= min_margin)
