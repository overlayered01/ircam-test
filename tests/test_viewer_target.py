"""fov_viewer 설정 메뉴 로직 (장치 불필요)."""

import pytest

pytest.importorskip("pyorbbecsdk")

from ir_detect import detect as det  # noqa: E402
from fov_viewer import (  # noqa: E402
    DeviceChoice,
    MenuRow,
    MenuState,
    Opened,
    Target,
    ViewFlags,
    build_orbbec_menu,
    detect_flags,
    describe,
    orbbec_menu_result,
    sort_devices,
    step_value,
    EXPOSURE_MS,
    GAIN_DB,
)

IR_MODES = [(320, 288), (640, 576), (512, 512), (1024, 1024)]
COLOR_MODES = [(1280, 720), (1280, 960), (1920, 1080), (2560, 1440), (3840, 2160)]
FPS = {(320, 288): {5, 15, 25, 30}, (640, 576): {5, 15, 25, 30}, (512, 512): {5, 15, 25, 30}, (1024, 1024): {5, 15}}
DEVICES = ["Femto Bolt USB3.1 CL8764100GL", "Femto Mega Ethernet CL2P661007P 192.168.1.10", "DMK 33GX264 36814688 GigE"]


def opened(dev=0):
    return Opened(dev, "orbbec", "", ir_modes=IR_MODES, color_modes=COLOR_MODES, fps_by_mode=FPS)


def menu(target=Target(ires=(640, 576, 0)), flags=ViewFlags(), dev=0):
    return build_orbbec_menu(DEVICES, opened(dev), target, flags)


def test_rows_reflect_state():
    rows = dict(menu().rows())
    assert rows["Device"] == DEVICES[0]
    assert rows["IR mode"].startswith("640x576 NFOV")
    assert rows["FPS"] == "auto" and rows["Depth"] == "off" and rows["D2C align"] == "on"


def test_toggle_depth_and_color_then_target():
    m = menu()
    m.sel = [r.key for r in m._rows].index("depth")
    m.adjust(1)
    m.sel = [r.key for r in m._rows].index("color")
    m.adjust(1)
    t, flags = orbbec_menu_result(m, opened())
    assert t.depth and t.color and t.ires == (640, 576, 0) and t.cres == (1280, 720, 0)
    assert flags == ViewFlags()
    assert not m.changed("dev") and m.changed("depth")


def test_color_res_turns_color_on_and_wraps():
    m = menu()
    m.sel = [r.key for r in m._rows].index("cres")
    m.adjust(-1)  # 처음에서 이전 → 마지막
    t, _ = orbbec_menu_result(m, opened())
    assert t.color and t.cres == (3840, 2160, 0)


def test_ir_mode_change_resets_unavailable_fps():
    m = menu(Target(ires=(640, 576, 30)))
    assert m.row("fps").value == "30"
    m.sel = [r.key for r in m._rows].index("ir")
    m.adjust(1)  # 512x512 (30 지원) → 유지
    assert m.row("fps").value == "30"
    m.adjust(1)  # 1024x1024 (15 까지) → auto
    assert m.row("fps").value == "auto"
    assert orbbec_menu_result(m, opened())[0].ires == (1024, 1024, 0)


def test_fps_cycles_through_mode_options():
    m = menu(Target(ires=(1024, 1024, 0)))
    m.sel = [r.key for r in m._rows].index("fps")
    m.adjust(1)
    assert m.row("fps").value == "5"
    m.adjust(1)
    assert m.row("fps").value == "15"
    m.adjust(1)
    assert m.row("fps").value == "auto"  # 순환


def test_device_change_and_flags():
    m = menu(flags=ViewFlags(grid=False, align=True, auto=False))
    m.sel = 0
    m.adjust(1)
    assert m.changed("dev") and m.get("dev") == 1
    m.adjust(2)
    assert m.get("dev") == 0  # 순환 (3개)
    m.sel = [r.key for r in m._rows].index("grid")
    m.adjust(1)
    assert orbbec_menu_result(m, opened())[1] == ViewFlags(grid=True, align=True, auto=False)
    m.move(-1)
    assert m._rows[m.sel].key == "cres"
    m.move(len(m._rows))
    assert m._rows[m.sel].key == "cres"


def test_generic_menu_state():
    m = MenuState([MenuRow("a", "A", ["x", "y"]), MenuRow("b", "B", ["off", "on"], 1)])
    assert m.rows() == [("A", "x"), ("B", "on")]
    m.adjust(-1)
    assert m.row("a").value == "y" and m.changed("a") and not m.changed("b")
    assert not m.has("zzz")


def test_describe():
    assert describe(Target()) == "IR default / depth off / color off"
    s = describe(Target(ires=(1024, 1024, 15), depth=True, color=True, cres=(1920, 1080, 0)))
    assert s.startswith("IR 1024x1024 WFOV") and "@15" in s and s.endswith("depth on / color 1920x1080")


def test_detect_rows_round_trip():
    flags = ViewFlags(detect="apriltag", detect_width=0, tag_decimate=2.0, tag_sharpen=0.5, tag_hamming=1, shape_mode="filled",
                      tag_margin=20.0, tag_confirm=3)
    m = menu(flags=flags)
    assert m.row("dwidth").value == "full" and m.row("tdec").value == "2" and m.row("tsharp").value == "0.5"
    assert m.row("thamm").value == "1" and m.row("tmarg").value == "20" and m.row("tconf").value == "3"
    assert menu(flags=ViewFlags()).row("tconf").value == "off"
    assert menu(flags=ViewFlags(detect_width=-2)).row("dwidth").value == "2x"
    assert detect_flags(m, ViewFlags()) == flags
    md = menu(flags=ViewFlags(detect="dots", dot_min=6, dot_gap=5.0, dot_min_count=2))
    assert md.row("detect").value == "dots" and md.row("dmin").value == "6" and md.row("dgap").value == "5x"
    assert md.row("dcnt").value == "2"
    f2 = detect_flags(md, ViewFlags())
    assert f2.dots and not f2.tags and not f2.shapes and f2.dot_options == det.DotOptions(6, 5.0, 2)
    m.sel = [r.key for r in m._rows].index("tdec")
    m.adjust(1)
    assert detect_flags(m, ViewFlags()).tag_decimate == 4.0


def test_sort_devices_prefers_dmk_then_mega_then_bolt():
    found = [DeviceChoice("orbbec", "Femto Bolt USB3.1 CL8764100GL"),
             DeviceChoice("orbbec", "Femto Mega Ethernet CL2P661007P 192.168.1.10"),
             DeviceChoice("orbbec", "Gemini 2 USB3.1 X"),
             DeviceChoice("tis", "DMK 33GX264 36814688 GigE", tis_name="DMK 33GX264 36814688"),
             DeviceChoice("tis", "DMK 33GX264 99 GigE", tis_name="DMK 33GX264 99")]
    assert [c.label.split()[0] + c.label.split()[-2] for c in sort_devices(found)] == [
        "DMK36814688", "DMK99", "FemtoCL2P661007P", "FemtoUSB3.1", "GeminiUSB3.1"]  # DMK(발견 순) → Mega → Bolt → 나머지
    assert sort_devices([]) == []


def test_step_value_moves_one_step_and_clamps():
    assert step_value(EXPOSURE_MS, 10.0, 1) == 15 and step_value(EXPOSURE_MS, 10.0, -1) == 7
    assert step_value(EXPOSURE_MS, 41.67, 1) == 50 and step_value(EXPOSURE_MS, 41.67, -1) == 30  # 가장 가까운 단계(40) 기준
    assert step_value(EXPOSURE_MS, 1000, 1) == 1000 and step_value(EXPOSURE_MS, 0.05, -1) == 0.05
    assert step_value(GAIN_DB, None, 1) == 3 and step_value(GAIN_DB, 46.4, 1) == 48
