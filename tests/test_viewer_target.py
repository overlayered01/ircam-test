"""fov_viewer 설정 메뉴 로직 (장치 불필요)."""

import pytest

pytest.importorskip("pyorbbecsdk")

from fov_viewer import (  # noqa: E402
    MenuRow,
    MenuState,
    Opened,
    Target,
    ViewFlags,
    build_orbbec_menu,
    describe,
    orbbec_menu_result,
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
