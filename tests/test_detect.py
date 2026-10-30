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
    img = cv2.resize(cells, None, fx=px, fy=px, interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(img, px * margin, px * margin, px * margin, px * margin, cv2.BORDER_CONSTANT, value=255)
    return 255 - img if inverted else img


def regular_polygon(center, r, n, rot=0.0):
    return np.array([(center[0] + r * np.cos(a + rot), center[1] + r * np.sin(a + rot))
                     for a in np.linspace(0, 2 * np.pi, n + 1)[:-1]], np.int32)


def cross_polygon(center, L, w, rot=0.0):
    """팔 길이 L(전체 폭), 팔 두께 w 인 십자 12각형."""
    a, b = L / 2, w / 2
    pts = [(-b, -a), (b, -a), (b, -b), (a, -b), (a, b), (b, b), (b, a), (-b, a), (-b, b), (-a, b), (-a, -b), (-b, -b)]
    c, s = np.cos(rot), np.sin(rot)
    return np.array([(center[0] + x * c - y * s, center[1] + x * s + y * c) for x, y in pts], np.int32)


def shapes_image() -> np.ndarray:
    img = np.full((600, 800), 230, np.uint8)  # 밝은 종이
    cv2.rectangle(img, (60, 60), (220, 220), 20, -1)  # square
    cv2.rectangle(img, (300, 80), (560, 200), 20, -1)  # rect
    cv2.circle(img, (680, 150), 80, 20, -1)  # circle
    cv2.fillPoly(img, [regular_polygon((120, 450), 90, 3, -np.pi / 2)], 20)  # 정삼각형
    cv2.fillPoly(img, [regular_polygon((400, 430), 90, 6)], 20)  # 정육각형
    cv2.fillPoly(img, [cross_polygon((650, 430), 180, 60, 0.3)], 20)  # 정십자 (기울임)
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


def test_shapes_all_types():
    found, rejected = det.ShapeDetector().detect(shapes_image())
    by = {s.name: s for s in found}
    assert sorted(by) == ["circle", "cross", "hexagon", "rect", "square", "triangle"] and rejected == 0
    assert abs(by["square"].center[0] - 140) < 3 and abs(by["square"].center[1] - 140) < 3
    for name in ("triangle", "square", "hexagon", "cross", "circle"):
        assert by[name].dev < 0.1, (name, by[name].dev)  # 정형
    assert by["square"].size_px == pytest.approx(160, abs=4)


def test_regular_only_rejects_irregular():
    img = np.full((400, 800), 230, np.uint8)
    cv2.fillPoly(img, [np.array([[60, 300], [380, 300], [90, 80]])], 20)  # 찌그러진 삼각형
    cv2.fillPoly(img, [regular_polygon((600, 200), 100, 3)], 20)  # 정삼각형
    cv2.rectangle(img, (420, 320), (780, 380), 20, -1)  # 가늘고 긴 사각형
    o = det.ShapeOptions(regular_only=True, tol=0.1)
    found, rejected = det.ShapeDetector(o).detect(img)
    assert [s.name for s in found] == ["triangle"] and rejected == 2
    found, _ = det.ShapeDetector(det.ShapeOptions(regular_only=False)).detect(img)
    assert sorted(s.name for s in found) == ["rect", "triangle", "triangle"]
    assert max(s.dev for s in found if s.name == "triangle") > 0.2  # 찌그러진 쪽
    # 종류 선택
    found, rejected = det.ShapeDetector(det.ShapeOptions(types="rect")).detect(img)
    assert [s.name for s in found] == ["rect"] and rejected == 2


def test_cross_irregular_has_large_dev():
    img = np.full((400, 400), 230, np.uint8)
    poly = cross_polygon((200, 200), 200, 60)
    poly[0:2, 1] -= 60  # 위쪽 팔만 길게
    cv2.fillPoly(img, [poly], 20)
    found, _ = det.ShapeDetector().detect(img)
    assert [s.name for s in found] == ["cross"] and found[0].dev > 0.15


def test_min_size_filter():
    img = np.full((300, 300), 230, np.uint8)
    cv2.rectangle(img, (20, 20), (50, 50), 20, -1)  # 30px
    cv2.rectangle(img, (100, 100), (260, 260), 20, -1)  # 160px
    found, _ = det.ShapeDetector(det.ShapeOptions(min_px=80)).detect(img)
    assert len(found) == 1 and found[0].size_px == pytest.approx(160, abs=4)


def test_shapes_bright_on_dark_with_invert():
    img = 255 - shapes_image()
    assert "square" not in {s.name for s in det.ShapeDetector().detect(img)[0]}
    assert "square" in {s.name for s in det.ShapeDetector().detect(img, invert=True)[0]}


def outline_image(thickness: int = 1) -> np.ndarray:
    """펜으로 그린 종이: 테두리만 있는 정사각형·정삼각형·원과 두 획 십자. 배경은 어두운 방."""
    img = np.full((576, 640), 12, np.uint8)
    cv2.rectangle(img, (120, 100), (560, 480), 150, -1)  # 종이
    cv2.rectangle(img, (160, 140), (260, 240), 85, thickness)
    cv2.polylines(img, [regular_polygon((360, 190), 55, 3, -np.pi / 2)], True, 85, thickness)
    cv2.circle(img, (470, 190), 50, 85, thickness)
    cv2.line(img, (200, 330), (200, 430), 85, thickness)  # 십자 세로 획
    cv2.line(img, (150, 380), (250, 380), 85, thickness)  # 십자 가로 획
    return img


@pytest.mark.parametrize("mode", ["outline", "both"])
def test_outline_shapes_and_stroke_cross(mode):
    found, _ = det.ShapeDetector(det.ShapeOptions(mode=mode)).detect(outline_image())
    names = sorted(s.name for s in found)
    assert names == ["circle", "cross", "square", "triangle"], names  # both 여도 중복 없음
    by = {s.name: s for s in found}
    for name in ("square", "triangle", "circle", "cross"):
        assert by[name].dev < 0.1, (name, by[name].dev)
    assert abs(by["cross"].center[0] - 200) < 3 and abs(by["cross"].center[1] - 380) < 3
    assert abs(by["square"].center[0] - 210) < 3 and abs(by["square"].center[1] - 190) < 3
    # Otsu(filled) 만으로는 종이가 통째로 잡힐 뿐 선 도형은 못 찾는다
    names = {s.name for s in det.ShapeDetector(det.ShapeOptions(mode="filled")).detect(outline_image())[0]}
    assert "cross" not in names and "circle" not in names


def test_outline_mode_ignores_bright_filled_shapes():
    """밝은 채움 도형의 가장자리를 선 도형으로 오인하지 않는다 (invert 를 켜야 잡힌다)."""
    img = 255 - shapes_image()
    assert det.ShapeDetector(det.ShapeOptions(mode="outline")).detect(img)[0] == []
    assert "square" in {s.name for s in det.ShapeDetector(det.ShapeOptions(mode="outline")).detect(img, invert=True)[0]}


def test_shape_tracker_rate_and_hold():
    def sq(cx):
        return det.ShapeDet("square", np.array([[cx - 20, 80], [cx + 20, 80], [cx + 20, 120], [cx - 20, 120]], float),
                            (cx, 100.0), 1600.0, 40.0, 0.02)
    tr = det.ShapeTracker(window=10, hold=2)
    out = tr.update([sq(100)])
    assert out[0].rate == 1.0 and not out[0].held
    out = tr.update([])  # 1프레임 놓침 → 유지
    assert len(out) == 1 and out[0].held and out[0].rate == 0.5
    out = tr.update([sq(102)])  # 같은 자리로 다시
    assert len(out) == 1 and not out[0].held and out[0].rate == pytest.approx(2 / 3)
    for _ in range(3):
        out = tr.update([])
    assert out == []  # hold 를 넘기면 화면에서 사라진다
    out = tr.update([sq(100)])
    assert out[0].rate == pytest.approx(3 / 7)  # 트랙은 window 안이라 아직 살아 있다
    out = tr.update([sq(400)])  # 다른 자리 → 새 트랙, 기존 것은 유지
    assert sorted((s.name, s.held, round(s.rate, 2)) for s in out) == [("square", False, 1.0), ("square", True, 0.38)]
    for _ in range(12):
        tr.update([])
    assert tr.tracks == []


def test_pipeline_track_adds_rate_and_summary():
    p = det.Pipeline(backend="opencv", track=True)
    res = p.run(shapes_image(), tags=False, shapes=True)
    assert all(s.rate == 1.0 for s in res.shapes) and "hit 100%" in res.summary()
    res = p.run(np.full((600, 800), 230, np.uint8), tags=False, shapes=True)  # 빈 종이 → 전부 유지(held)
    assert res.shapes and all(s.held and s.rate == 0.5 for s in res.shapes)
    assert det.Pipeline(backend="opencv").run(shapes_image(), tags=False, shapes=True).shapes[0].rate is None


def test_pipeline_scales_large_image_back_to_original_coords():
    big = cv2.resize(tag_image("tag36h11", 9, px=40), (2448, 2048), interpolation=cv2.INTER_NEAREST)
    res = det.Pipeline(backend="opencv", max_width=1024).run(big, tags=True, shapes=False)
    assert res.backend == "opencv-aruco" and [t.id for t in res.tags] == [9]
    cx, cy = res.tags[0].center
    assert abs(cx - 1224) < 12 and abs(cy - 1024) < 12  # 원본 좌표로 복원
    assert "tags 1 [9]" in res.summary()
    res2 = det.Pipeline(backend="opencv", max_width=1024).run(big, tags=False, shapes=True,
                                                             shape_options=det.ShapeOptions(min_px=100))
    assert [s.name for s in res2.shapes] == ["square"] and res2.shapes[0].size_px > 1000  # 태그 전체 = 큰 정사각형
    canvas = cv2.cvtColor(cv2.resize(big, (612, 512)), cv2.COLOR_GRAY2BGR)
    det.draw(canvas, res, 0.25)  # 그리기만 되면 됨
    assert det.Pipeline().run(big, tags=False, shapes=False).summary().endswith("0ms")
