"""april_tags/ 의 태그 PNG 가 기대 ID 로 디코딩되는지 (장치 불필요)."""

from pathlib import Path

import cv2
import pytest

from ir_detect.tags import AprilTagDetector, OpenCVDetector, degrade, identify, ids_of, load_tag, render, warp

TAG_DIR = Path(__file__).resolve().parent.parent / "april_tags"
FILES = sorted(TAG_DIR.glob("*.png"))
pytestmark = pytest.mark.skipif(not FILES, reason="april_tags/ 에 PNG 없음")


@pytest.fixture(params=FILES, ids=lambda p: p.name)
def tag(request):
    return load_tag(request.param)


def prepared(tag):
    """검출기에 넣을 셀 영상: 여백 제거(테두리 1셀), 반전 태그면 반전."""
    cells = tag.tag_only(1)
    return 255 - cells if tag.is_inverted() else cells


def test_bits_match_expected_id(tag):
    assert tag.expected_id in {i for i, _, _ in identify(tag)}


@pytest.mark.parametrize("px", [40, 6, 3])
def test_opencv_detects(tag, px):
    det = OpenCVDetector(tag.family)
    assert ids_of(det.detect(render(prepared(tag), px_per_cell=px))) == [tag.expected_id]


def test_opencv_detects_degraded(tag):
    det = OpenCVDetector(tag.family)
    assert tag.expected_id in ids_of(det.detect(degrade(warp(render(prepared(tag))))))


@pytest.mark.parametrize("px", [40, 6, 3])
def test_apriltag_detects(tag, px):
    pytest.importorskip("pupil_apriltags")
    det = AprilTagDetector(tag.family)
    assert ids_of(det.detect(render(prepared(tag), px_per_cell=px))) == [tag.expected_id]


def test_apriltag_rotations(tag):
    pytest.importorskip("pupil_apriltags")
    det = AprilTagDetector(tag.family)
    img = render(prepared(tag))
    for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        assert ids_of(det.detect(cv2.rotate(img, rot))) == [tag.expected_id]
