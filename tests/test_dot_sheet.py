"""dot_sheet 인쇄 시트: 만든 시트를 2 m 카메라 조건(1.5 px/mm, 흐림, 잡음)으로 줄여도 마커가 모두 읽혀야 한다."""

import cv2
import numpy as np
import pytest

import dot_sheet as ds
from ir_detect import detect as det


def as_camera(sheet: np.ndarray, ppm: float = 1.5, seed: int = 0) -> np.ndarray:
    """300 dpi 시트 → ppm px/mm 로 축소, 흐림 0.8, 종이 183/잉크 30, 어두운 방(60) 안에 놓고 잡음 σ3."""
    rng = np.random.default_rng(seed)
    s = ppm / (300 / 25.4)
    small = cv2.GaussianBlur(cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (0, 0), 0.8)
    h, w = small.shape
    frame = np.full((h + 100, w + 100), 60, np.float32)
    frame[50:50 + h, 50:50 + w] = small.astype(np.float32) * 0.6 + 30
    return np.clip(frame + rng.normal(0, 3, frame.shape), 0, 255).astype(np.uint8)


def kinds(frame: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in det.DotDetector(det.DotOptions(min_px=4)).detect(frame)[0]:
        out[m.kind] = out.get(m.kind, 0) + 1
    return out


def test_markers_sheet_all_detected(tmp_path):
    paths = ds.sheet_markers([4.0, 6.0, 8.0], 300, tmp_path)
    total: dict[str, int] = {}
    for p in paths:
        for k, v in kinds(as_camera(cv2.imread(str(p), 0))).items():
            total[k] = total.get(k, 0) + v
    assert total == {"1": 3, "2": 3, "2s": 3, "3I": 3, "3L": 3}


def test_sizes_sheet_gray_dots_near_paper_edge(tmp_path):
    """회색(150) 3 mm 점도 종이 위쪽 가장자리 근처(어두운 배경 옆)에서 읽힌다 — 작은 창 국소 이진화 덕."""
    paths = ds.sheet_sizes(["3L"], [3, 4, 6, 8], [0, 150], 300, tmp_path)
    assert len(paths) == 1
    assert kinds(as_camera(cv2.imread(str(paths[0]), 0))) == {"3L": 8}


def test_cube_faces_detected_and_clearance(tmp_path):
    faces = ["1", "2", "2s", "3I", "3L", "3L,1,3L,1"]
    paths, clearance = ds.sheet_cube(faces, 80.0, 4.0, 38.0, 2.0, 3.5, 300, tmp_path)
    assert clearance >= 3.5 * 4.0  # 그림 상자가 점과 묶이지 않는 거리
    singles = [p for p in paths if p.name.startswith("cube_face_")]
    assert len(singles) == 6 and cv2.imread(str(singles[0]), 0).shape == (945, 945)  # 80 mm @ 300 dpi
    total: dict[str, int] = {}
    for p in paths:
        if p.name.startswith("cube_faces"):
            for k, v in kinds(as_camera(cv2.imread(str(p), 0))).items():
                total[k] = total.get(k, 0) + v
    assert total == {"1": 6, "2": 4, "2s": 4, "3I": 4, "3L": 6}
    # 낱장(점만) 도 그대로 읽힌다: 3L 면은 꼭짓점이 네 모서리를 향한다
    face = cv2.imread(str([p for p in singles if "_5_" in p.name][0]), 0)
    markers = det.DotDetector(det.DotOptions(min_px=4)).detect(as_camera(face))[0]
    assert sorted(m.kind for m in markers) == ["3L"] * 4 and all(m.sure for m in markers)
    angles = sorted(m.angle % 360 for m in markers)  # TL 45°, TR 135°, BR 225°, BL 315° (꼭짓점 → 면 안쪽)
    assert all(abs(a - e) < 6 for a, e in zip(angles, [45, 135, 225, 315])), angles


def test_face_spec_validation():
    with pytest.raises(ValueError):
        ds.face_dots("3L,1", 100.0, 2.0, 4.0)
    assert len(ds.face_dots("3L,-,3L,-", 100.0, 2.0, 4.0)) == 6
