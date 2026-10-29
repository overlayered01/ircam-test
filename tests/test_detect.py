"""AprilTag / 도형 검출기 (합성 영상, 장치 불필요)."""

import cv2
import numpy as np
import pytest

from ir_detect import detect as det
from ir_detect.tags import dict_bits


def tag_image(family: str, tag_id: int, px: int = 20, margin: int = 4, inverted: bool = False) -> np.ndarray:
    """사전 비트로 태그 하나를 그린 흰 바탕 영상 (데이터 n×n + 검은 테두리 1 + 흰 여백)."""
    bits = dict_bits(family, tag_id)
    n = bits.shape[0]
    cells = np.zeros((n + 2, n + 2), np.uint8)
    cells[1:-1, 1:-1] = bits * 255
    img = det.cv2.resize(cells, None, fx=px, fy=px, interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(img, px * margin, px * margin, px * margin, px * margin, cv2.BORDER_CONSTANT, value=255)
    return 255 - img if inverted else img


def shapes_image() -> np.ndarray:
    img = np.full((600, 800), 230, np.uint8)  # 밝은 종이
    cv2.rectangle(img, (60, 60), (220, 220), 20, -1)  # square
    cv2.rectangle(img, (300, 80), (560, 200), 20, -1)  # rect
    cv2.circle(img, (680, 150), 80, 20, -1)  # circle
    cv2.fillPoly(img, [np.array([[120, 330], [40, 540], [200, 540]])], 20)  # triangle
    pts = np.array([(400 + 90 * np.cos(a), 430 + 90 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 7)[:-1]], np.int32)
    cv2.fillPoly(img, [pts], 20)  # hexagon
    return img


@pytest.mark.parametrize("backend", ["opencv", "auto"])
@pytest.mark.parametrize("family,tag_id", [("tag36h11", 7), ("tag16h5", 3), ("tag25h9", 11)])
def test_apriltag_detect(backend, family, tag_id):
    if backend == "auto":
        pytest.importorskip("pupil_apriltags")
    d = det.AprilTagDetector(family, backend)
    tags, _ = d.detect(tag_image(family, tag_id))
    assert [t.id for t in tags] == [tag_id]
    cx, cy = tags[0].center
    assert abs(tags[0].size_px - 20 * (dict_bits(family, tag_id).shape[0] + 2)) < 3
    assert abs(cx - cy) < 2  # 정사각 영상 가운데


def test_apriltag_inverted_needs_invert_flag():
    d = det.AprilTagDetector("tag36h11", "opencv")
    img = tag_image("tag36h11", 5, inverted=True)
    assert d.detect(img)[0] == []
    assert [t.id for t in d.detect(img, invert=True)[0]] == [5]


def test_shapes():
    found = det.ShapeDetector().detect(shapes_image())
    names = sorted(s.name for s in found)
    assert names == ["circle", "hexagon", "rect", "square", "triangle"]
    sq = next(s for s in found if s.name == "square")
    assert abs(sq.center[0] - 140) < 3 and abs(sq.center[1] - 140) < 3


def test_shapes_bright_on_dark_with_invert():
    img = 255 - shapes_image()
    assert det.ShapeDetector().detect(img) == [] or all(s.name != "square" for s in det.ShapeDetector().detect(img))
    assert "square" in {s.name for s in det.ShapeDetector().detect(img, invert=True)}


def test_pipeline_scales_large_image_back_to_original_coords():
    big = cv2.resize(tag_image("tag36h11", 9, px=40), (2448, 2048), interpolation=cv2.INTER_NEAREST)
    res = det.Pipeline(backend="opencv", max_width=1024).run(big, tags=True, shapes=False)
    assert res.backend == "opencv-aruco" and [t.id for t in res.tags] == [9]
    cx, cy = res.tags[0].center
    assert abs(cx - 1224) < 12 and abs(cy - 1024) < 12  # 원본 좌표로 복원
    assert "tags 1 [9]" in res.summary()
    canvas = cv2.cvtColor(cv2.resize(big, (612, 512)), cv2.COLOR_GRAY2BGR)
    det.draw(canvas, res, 0.25)  # 그리기만 되면 됨
    assert det.Pipeline().run(big, tags=False, shapes=False).summary().endswith("0ms")
