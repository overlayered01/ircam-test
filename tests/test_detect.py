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


def test_filled_shapes_on_paper_in_dark_room():
    """어두운 방에서 밝은 종이 위의 검정 도형: 배경 덩어리의 구멍 안에 있어도 전부 잡혀야 한다."""
    img = np.full((576, 640), 12, np.uint8)
    paper = shapes_image()
    paper = cv2.resize(paper, (480, 360), interpolation=cv2.INTER_AREA)
    img[100:460, 80:560] = (paper.astype(np.float32) * 0.6 + 10).astype(np.uint8)
    found, _ = det.ShapeDetector(det.ShapeOptions(mode="filled")).detect(img)
    assert sorted(s.name for s in found) == ["circle", "cross", "hexagon", "rect", "square", "triangle"]
    found, _ = det.ShapeDetector(det.ShapeOptions(mode="both")).detect(img)
    assert sorted(s.name for s in found) == ["circle", "cross", "hexagon", "rect", "square", "triangle"]  # 중복 없음


def test_same_spot_requires_similar_size():
    big = det.ShapeDet("rect", np.zeros((4, 2)), (300.0, 300.0), 90000.0, 300.0, 0.4)
    small = det.ShapeDet("square", np.zeros((4, 2)), (320.0, 310.0), 900.0, 30.0, 0.02)
    near = det.ShapeDet("square", np.zeros((4, 2)), (325.0, 312.0), 1000.0, 32.0, 0.03)
    assert not det._same_spot(big, small) and det._same_spot(small, near)


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


def test_detect_worker_sync_and_threaded():
    img = shapes_image()
    w = det.DetectWorker(det.Pipeline(backend="opencv"), threaded=False)
    assert w.submit(img, False, True, max_width=0) and w.result.shapes and w.skipped == 0
    assert w.pipeline.max_width == 10**9  # max_width 0 = 원본

    w = det.DetectWorker(det.Pipeline(backend="opencv"))
    try:
        assert w.result.summary().endswith("0ms")
        w.submit(img, False, True)
        w.wait()
        assert not w.busy and [s.name for s in w.result.shapes]
        # 바쁜 동안 들어온 프레임은 버린다: 큰 영상을 넣고 바로 하나 더
        big = cv2.resize(img, None, fx=4, fy=4)
        n = sum(w.submit(big, False, True, max_width=0) for _ in range(3))
        w.wait()
        assert n >= 1 and w.skipped == 3 - n
    finally:
        w.close()


def test_tag_decimate_and_sharpen_recreate_detector():
    pytest.importorskip("pupil_apriltags")
    p = det.Pipeline(backend="pupil", tag_decimate=1.0)
    img = tag_image("tag36h11", 3, px=40)
    assert [t.id for t in p.run(img, tags=True, shapes=False).tags] == [3]
    d1 = p._tag
    p.tag_decimate = 2.0
    assert [t.id for t in p.run(img, tags=True, shapes=False).tags] == [3]
    assert p._tag is not d1 and p._tag.decimate == 2.0 and p._tag.sharpen == 1.0
    p.tag_sharpen = 0.25
    p.run(img, tags=True, shapes=False)
    assert p._tag.sharpen == 0.25


def test_prepare_upscale_and_coords():
    img = tag_image("tag36h11", 5, px=12)  # 작은 태그
    small, scale = det.prepare(img, 1280, upscale=2.0)
    assert small.shape == (img.shape[0] * 2, img.shape[1] * 2) and scale == 0.5
    res = det.Pipeline(backend="opencv", upscale=2.0).run(img, tags=True, shapes=False)
    assert [t.id for t in res.tags] == [5]
    cx, cy = res.tags[0].center
    assert abs(cx - img.shape[1] / 2) < 3 and abs(cy - img.shape[0] / 2) < 3  # 원본 좌표


def test_tag_hamming_filter_accepts_one_wrong_bit():
    pytest.importorskip("pupil_apriltags")
    px, margin = 20, 4
    img = tag_image("tag36h11", 7, px=px, margin=margin).copy()
    r0 = (margin + 1) * px  # 테두리 안쪽 첫 데이터 셀 (0,0)
    cell = img[r0:r0 + px, r0:r0 + px]
    img[r0:r0 + px, r0:r0 + px] = 255 - cell  # 비트 하나 뒤집기
    p = det.Pipeline(backend="pupil")
    res = p.run(img, tags=True, shapes=False)
    assert res.tags == [] and res.rejected == 1
    p.tag_hamming = 1
    res = p.run(img, tags=True, shapes=False)
    assert [(t.id, t.hamming) for t in res.tags] == [(7, 1)]


def test_tag_tracker_confirms_after_consecutive_frames():
    def tag(i, x, size=30):
        c = np.array([[x, 100], [x + size, 100], [x + size, 100 + size], [x, 100 + size]], np.float64)
        return det.TagDet(i, c, 0, 100.0)
    tr = det.TagTracker()
    assert tr.update([tag(1, 50)], confirm=3) == ([], 1)
    assert tr.update([tag(1, 55)], confirm=3) == ([], 1)  # 조금 움직여도 같은 자리
    out, pending = tr.update([tag(1, 60), tag(7, 400)], confirm=3)
    assert [t.id for t in out] == [1] and pending == 1  # 1 은 3프레임째, 7 은 처음
    out, _ = tr.update([tag(1, 60)], confirm=3)  # 7 이 한 프레임 빠짐 → 트랙 유지
    assert [t.id for t in out] == [1] and {t.last.id for t in tr.tracks} == {1, 7}
    tr.update([tag(1, 60)], confirm=3)  # 7 두 프레임째 없음 → 삭제
    assert {t.last.id for t in tr.tracks} == {1}
    assert tr.update([tag(1, 300)], confirm=3) == ([], 1)  # 멀리 뛰면 새 트랙
    assert tr.update([], confirm=1) == ([], 0)


def test_pipeline_tag_confirm_and_margin():
    pytest.importorskip("pupil_apriltags")
    img = tag_image("tag36h11", 4, px=30)
    p = det.Pipeline(backend="pupil", track=True, tag_confirm=2)
    r1 = p.run(img, tags=True, shapes=False)
    assert r1.tags == [] and r1.tags_pending == 1 and "(+1 new)" in r1.summary()
    r2 = p.run(img, tags=True, shapes=False)
    assert [t.id for t in r2.tags] == [4] and r2.tags_pending == 0
    p.tag_margin = 1000.0  # 어떤 태그도 못 넘는 margin
    r3 = p.run(img, tags=True, shapes=False)
    assert r3.tags == [] and r3.rejected == 1



# ---------------------------------------------------------------- 점 마커 ----------------------------------------------------------------
def dots_image(diam: int = 10, paper: int = 220, ink: int = 25, noise: float = 0.0, blur: float = 0.0,
               rot_deg: float = 0.0) -> tuple[np.ndarray, dict]:
    """1점 / 같은 크기 2점 / 큰+작은 2점 / 1:2 일직선 3점 / L 자 3점 마커와 잡음(글자 획·큰 사각형·가는 선)을 그린 종이 영상.
    각 마커의 기대값 (kind, 중심, 방향) 을 함께 돌려준다."""
    img = np.full((600, 800), paper, np.uint8)
    r = diam // 2
    gap = int(diam * 2.5)  # 점 중심 간격 (지름 2.5배 — 기본 묶기 3.5배 안, 마커 간 거리는 10배 이상)

    def rot(pts, c):
        a = np.radians(rot_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return [(tuple((R @ (np.array(p, float) - c)) + c)) for p in pts]

    def put(pts, radii):
        for (x, y), rr in zip(pts, radii):
            cv2.circle(img, (int(round(x)), int(round(y))), rr, ink, -1, cv2.LINE_AA)

    exp = {}
    c = np.array([120.0, 120.0]);  put(rot([c], c), [r]);  exp["1"] = (c, None)
    c = np.array([400.0, 120.0]);  pts = rot([c - (gap / 2, 0), c + (gap / 2, 0)], c);  put(pts, [r, r])
    exp["2"] = (c, rot_deg % 180)
    c = np.array([650.0, 120.0]);  pts = rot([c - (gap / 2, 0), c + (gap / 2, 0)], c);  put(pts, [int(round(r * 1.7)), r])
    exp["2s"] = (c, rot_deg % 360)  # 큰 점(왼쪽) → 작은 점(오른쪽) = +x 방향
    s = diam * 1.5  # 1:2 직선 마커의 짧은 간격: 긴 간격(3 지름)이 기본 묶기 거리(3.5 지름) 안에 들어야 한다
    c = np.array([150.0, 400.0]);  pts = rot([c - (s * 1.5, 0), c - (s * 0.5, 0), c + (s * 1.5, 0)], c)
    put(pts, [r, r, r]);  exp["3I"] = (np.mean(pts, axis=0), (rot_deg + 180) % 360)  # 외톨이(오른쪽) → 붙은 둘(왼쪽) = -x
    c = np.array([450.0, 400.0]);  pts = rot([c, c + (gap, 0), c + (0, gap)], c);  put(pts, [r, r, r])
    exp["3L"] = (np.mean(pts, axis=0), (45 + rot_deg) % 360)  # 꼭짓점 → 나머지 중점 = 45°
    # 잡음: 글자 획 비슷한 가는 선, 큰 어두운 사각형, 작은 점 하나짜리 잡음은 1점 마커로 읽히므로 넣지 않는다
    cv2.line(img, (560, 330), (760, 340), ink, max(1, diam // 4))
    cv2.rectangle(img, (600, 420), (780, 560), ink, -1)
    cv2.putText(img, "ABC 123", (30, 570), cv2.FONT_HERSHEY_SIMPLEX, 2.0, ink, 4)  # 속이 보이는 글자: solidity 로 떨어져야 한다
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img = np.clip(img.astype(np.float32) + np.random.default_rng(0).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img, exp


def _ang_close(a: float, b: float, tol: float = 6.0, period: float = 360.0) -> bool:
    d = (a - b) % period
    return min(d, period - d) <= tol


@pytest.mark.parametrize("diam", [6, 10, 20])
def test_dot_markers_count_and_direction(diam):
    img, exp = dots_image(diam)
    markers, rejected = det.DotDetector().detect(img)
    found = {m.kind: m for m in markers}
    assert sorted(found) == sorted(exp), [(m.kind, m.center) for m in markers]
    assert len(markers) == 5
    for kind, (c, ang) in exp.items():
        m = found[kind]
        assert np.hypot(m.center[0] - c[0], m.center[1] - c[1]) < 1.5, (kind, m.center, c)
        drawn = 2 * (diam // 2) + 1  # cv2.circle(r) 은 지름 2r+1
        assert abs(min(d.diameter for d in m.dots) - drawn) < max(1.5, drawn * 0.12), (kind, [d.diameter for d in m.dots])
        if ang is None:
            assert m.angle is None
        elif kind == "2":
            assert not m.sure and _ang_close(m.angle, ang, period=180)
        else:
            assert m.sure and _ang_close(m.angle, ang), (kind, m.angle, ang)


def test_dot_markers_rotated_blurred_noisy():
    img, exp = dots_image(10, rot_deg=37.0, blur=1.2, noise=4.0)
    markers, _ = det.DotDetector().detect(img)
    found = {m.kind: m for m in markers}
    assert sorted(found) == sorted(exp)
    for kind in ("2s", "3I", "3L"):
        assert found[kind].sure and _ang_close(found[kind].angle, exp[kind][1], tol=8.0), (kind, found[kind].angle, exp[kind][1])
    assert _ang_close(found["2"].angle, exp["2"][1], tol=8.0, period=180)


def test_dot_min_px_and_gap_options():
    img, _ = dots_image(6)
    assert len(det.DotDetector(det.DotOptions(min_px=4)).detect(img)[0]) == 5
    assert det.DotDetector(det.DotOptions(min_px=16)).detect(img)[0] == []  # 7~11 px 점은 최소 16 px 에 못 미친다
    two_plus = det.DotDetector(det.DotOptions(min_px=4, min_count=2)).detect(img)[0]
    assert sorted(m.kind for m in two_plus) == ["2", "2s", "3I", "3L"]  # 1점 마커 무시
    # 묶기 거리를 점 간격(2.5배)보다 작게 하면 같은 크기 점들은 낱개로 흩어진다. 남는 것: 1:2 직선의 붙은 두 점(1.5배)과
    # 큰+작은 점(평균 지름이 커서 2.0배 안에 든다)
    loose = det.DotDetector(det.DotOptions(gap=2.0)).detect(img)[0]
    assert sorted(m.kind for m in loose) == ["1"] * 7 + ["2", "2s"]


def test_dot_markers_bright_on_dark_need_invert():
    img, exp = dots_image(10)
    inv = 255 - img
    # 반전 영상에서는 글자의 속(B 의 두 구멍)이 어두운 점 마커로 보일 수 있다 — 밝은 점 자리에는 아무것도 없어야 한다
    wrong = det.DotDetector().detect(inv)[0]
    assert not any(np.hypot(m.center[0] - c[0], m.center[1] - c[1]) < 5 for m in wrong for c, _ in exp.values())
    assert sorted(m.kind for m in det.DotDetector().detect(inv, invert=True)[0]) == sorted(exp)


def test_dot_markers_in_dark_room_on_small_paper():
    """어두운 방 + 작은 종이: 전역 Otsu 는 종이/배경 사이에 걸리지만 국소 이진화는 점을 본다."""
    img = np.full((600, 800), 15, np.uint8)
    paper, exp = dots_image(8, paper=200, ink=30)
    img[150:450, 200:600] = cv2.resize(paper, (400, 300), interpolation=cv2.INTER_AREA)
    markers, _ = det.DotDetector(det.DotOptions(min_px=3)).detect(img)
    assert sorted(m.kind for m in markers) == sorted(exp)


def test_pipeline_dots_scaled_and_confirmed():
    img, exp = dots_image(24)
    big = cv2.resize(img, (2400, 1800), interpolation=cv2.INTER_CUBIC)  # 점 72 px → max_width 1280 에서 38 px
    p = det.Pipeline(backend="opencv", max_width=1280, track=True, tag_confirm=2)
    res = p.run(big, tags=False, shapes=False, dots=True)
    assert res.dots == [] and res.dots_pending == 5 and "(+5 new)" in res.summary()
    res = p.run(big, tags=False, shapes=False, dots=True)
    assert len(res.dots) == 5 and res.dots_pending == 0 and "dots 5 [" in res.summary()
    m = {d.kind: d for d in res.dots}["3L"]
    c = exp["3L"][0] * 3  # 원본 좌표로 돌아와야 한다
    assert np.hypot(m.center[0] - c[0], m.center[1] - c[1]) < 4 and abs(m.diameter - 72) < 8
    res = p.run(big, tags=False, shapes=False, dots=False)  # 끄면 트랙 리셋
    assert res.dots == [] and p.dot_tracker.tracks == []


def test_detect_worker_passes_dots():
    img, exp = dots_image(10)
    w = det.DetectWorker(det.Pipeline(backend="opencv"), threaded=False)
    w.submit(img, tags=False, shapes=False, dots=True, dot_options=det.DotOptions(min_px=4), max_width=0)
    assert sorted(m.kind for m in w.result.dots) == sorted(exp)
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    det.draw(canvas, w.result)  # 그리기가 죽지 않는지
    assert (canvas != cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)).any()


def test_classify_dots_equilateral_and_even_line_are_axis_only():
    def dot(x, y, d=10.0):
        return det.Dot((x, y), d, np.pi * d * d / 4, 1.0, 100.0)
    tri = det.classify_dots([dot(0, 0), dot(100, 0), dot(50, 86.6)])
    assert tri.kind == "3L" and not tri.sure
    line = det.classify_dots([dot(0, 0), dot(50, 0), dot(100, 0)])
    assert line.kind == "3I" and not line.sure and _ang_close(line.angle, 0, period=180)
    line = det.classify_dots([dot(0, 0), dot(30, 0), dot(100, 0)])
    assert line.kind == "3I" and line.sure and _ang_close(line.angle, 180)  # 외톨이(100) → 붙은 둘(0,30) 쪽
    big_small = det.classify_dots([dot(0, 0, 16), dot(0, 40, 10)])
    assert big_small.kind == "2s" and big_small.sure and _ang_close(big_small.angle, 90)


def test_estimate_noise_and_denoise_modes():
    rng = np.random.default_rng(0)
    clean, exp = dots_image(12)
    noisy = np.clip(clean.astype(np.float32) + rng.normal(0, 12, clean.shape), 0, 255).astype(np.uint8)
    assert det.estimate_noise(clean) < 2 and 9 < det.estimate_noise(noisy) < 15
    # auto: 잡음 σ≈12, 점 13 px (≥ 12) → 7x7 블러
    p = det.Pipeline(backend="opencv", max_width=10**9, denoise="auto")
    res = p.run(noisy, tags=False, shapes=False, dots=True, dot_options=det.DotOptions(min_px=12))
    assert res.denoise == "blur7" and res.noise is not None and "noise" in res.summary()
    # 점이 작으면(Dot min px 4) 자동 블러를 더 얹지 않는다
    res = p.run(noisy, tags=False, shapes=False, dots=True, dot_options=det.DotOptions(min_px=4))
    assert res.denoise == ""
    # avg4: 서로 다른 잡음 프레임 4장을 평균하면 잡음이 절반 아래로, 마커는 그대로
    p = det.Pipeline(backend="opencv", max_width=10**9, denoise="avg4")
    for i in range(4):
        frame = np.clip(clean.astype(np.float32) + np.random.default_rng(i).normal(0, 12, clean.shape), 0, 255).astype(np.uint8)
        res = p.run(frame, tags=False, shapes=False, dots=True)
    assert res.denoise == "avg4" and sorted(m.kind for m in res.dots) == sorted(exp)
    assert det.estimate_noise(p._denoise(frame, 0.0, None, 1.0)[0]) < 8  # 4장 평균 → σ 12 → 6 (가장자리 몫 포함)
    # 크기가 바뀌면 누적을 버린다; off 로 돌아가면 비운다
    res = p.run(cv2.resize(frame, (400, 300)), tags=False, shapes=False, dots=True)
    assert res.denoise == "avg1"
    p.denoise = "off"
    assert p.run(frame, tags=False, shapes=False, dots=True).denoise == "" and not p._avg
