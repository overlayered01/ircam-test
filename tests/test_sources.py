import sys

import numpy as np
import pytest

from ir_detect.preprocess import to_u8
from ir_detect.sources import Mode, SourceUnavailable
from ir_detect.sources.orbbec_ir import IR_MODE_PASSIVE, OrbbecIRSource, build_modes

# Femto Bolt FW 1.0.9 IR 프로파일 일부
BOLT_IR = [
    (640, 576, 30, "Y16"),
    (1024, 1024, 15, "Y16"),
    (1024, 1024, 5, "Y16"),
    (640, 576, 5, "Y16"),
    (320, 288, 30, "Y16"),
    (512, 512, 30, "Y16"),
]


def test_build_modes_order_and_labels():
    modes = build_modes(BOLT_IR, passive_supported=False)
    assert [m.res for m in modes] == ["320x288", "512x512", "640x576", "640x576", "1024x1024", "1024x1024"]
    assert [m.fps for m in modes if m.res == "640x576"] == [30, 5]
    assert modes[0].label == "NFOV 2x2 binned"
    assert modes[-1].label == "WFOV unbinned"
    assert all(m.param("ir_mode") is None for m in modes)


def test_build_modes_passive_only_1024():
    modes = build_modes(BOLT_IR, passive_supported=True)
    passive = [m for m in modes if m.param("ir_mode") == IR_MODE_PASSIVE]
    assert [(m.res, m.fps) for m in passive] == [("1024x1024", 15), ("1024x1024", 5)]
    assert all(m.label == "Passive IR" for m in passive)
    assert modes[-2:] == passive  # 액티브 뒤에 온다


def test_build_modes_dedup():
    assert len(build_modes(BOLT_IR + BOLT_IR, passive_supported=False)) == len(BOLT_IR)


def test_mode_str_and_hashable():
    m = Mode(1024, 1024, 15, "Y16", "Passive IR", (("ir_mode", 1),))
    assert str(m) == "1024x1024@15 Y16 [Passive IR]"
    assert len({m, Mode(1024, 1024, 15, "Y16", "Passive IR", (("ir_mode", 1),))}) == 1


def test_orbbec_without_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)  # import 시 ImportError
    ok, why = OrbbecIRSource.available()
    assert not ok and "pyorbbecsdk" in why
    assert OrbbecIRSource.discover() == []
    with pytest.raises(SourceUnavailable):
        OrbbecIRSource().open()


def test_to_u8_ignores_saturated_pixels():
    raw = np.full((100, 100), 400, np.uint16)
    raw[:2, :2] = 65535  # 가까운 물체 포화
    img = to_u8(raw)
    assert img.dtype == np.uint8
    assert img[50, 50] == 255  # 포화 픽셀 기준이면 거의 0 이 된다
    assert to_u8(np.zeros((4, 4), np.uint8)).dtype == np.uint8
