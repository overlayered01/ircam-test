"""(b) 라이브 뷰어: IR 을 기본으로 띄우고, Depth / RGB 는 켜고 끌 수 있다. 장치는 메뉴에서 바꾼다.

장치
  - Femto Bolt / Femto Mega (Orbbec SDK): IR + depth + color, 공장 캘리브레이션으로 화각 경계 오버레이
  - DMK 33GX264 (The Imaging Source, GigE): IR 한 장만. 노출·게인을 메뉴에서 조절. 캘리브레이션 파일이 있으면 각도 격자

패널 구성 (IR 이 기준 센서)
  [IR + color 화각 경계]        [Color + IR/depth 화각 경계]
     (magenta = color FOV)        (cyan = IR/depth 센서 사각형 화각, green = depth 값이 실제로 나오는 영역)
  [Depth + color 화각 경계]     [Depth→Color 정렬 블렌드]
  꺼진 스트림의 패널은 비워 둔다.

  - 경계선은 지정 거리(Z) 평면 기준이다. 시차(baseline) 때문에 거리마다 겹침 영역이 달라진다.
  - 'auto' 모드에서는 화면 중앙의 실제 depth 중앙값을 거리로 사용한다 (depth 가 켜져 있을 때).
  - Depth 패널의 'valid' FOV 는 실제로 깊이 값이 나온 픽셀의 각도 범위 (WFOV 원형 영역 확인용).

    python fov_viewer.py                                   # 첫 번째 장치, IR 만 (장치 기본 모드)
    python fov_viewer.py --serial CL8764100GL              # Orbbec 장치 지정
    python fov_viewer.py --ip 192.168.1.10                 # Femto Mega 이더넷 (자동 탐색 안 될 때)
    python fov_viewer.py --device dmk                      # 이름에 'dmk' 가 들어간 장치로 시작
    python fov_viewer.py --ir 1024x1024@15 --depth --color # 모드·스트림 지정해서 시작

키:  e 설정 메뉴   메뉴 안: ↑↓(w/s) 항목   ←→(a/d) 값 변경   Enter 적용   Esc/e 닫기
     +/- 거리 ±100mm   [ ] 거리 ±1000mm   s 스냅샷 저장   q/ESC 종료
  스트림 구성이나 장치를 바꾸면 재시작 (Bolt USB ~4초, Mega 이더넷 ~2초, DMK ~1초). 그동안 창은 계속 반응한다.
  fps 는 켜진 스트림이 공통으로 지원하는 최댓값으로 자동 선택된다 (메뉴에서 고정 가능).
  Femto 의 IR 과 depth 는 같은 센서라 항상 같은 모드로 켜진다.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from femto_fov import viz
from femto_fov.device import (
    DEFAULT_NET_PORT,
    AlignFilter,
    Config,
    Context,
    DeviceEntry,
    OBError,
    OBStreamType,
    Pipeline,
    color_to_bgr,
    depth_to_mm,
    device_summary,
    enumerate_devices,
    extrinsic_Rt,
    find_profile,
    get_device,
    ir_to_u16,
    list_profiles,
    mode_name,
    parse_res,
    quiet_sdk_log,
    to_intrinsics,
    unique_resolutions,
)
from femto_fov.fov_math import (
    Intrinsics,
    fov_pinhole,
    fov_undistorted,
    footprint_at_distance,
    overlap_ratio,
    project_between,
    project_border,
    valid_region_fov,
)
from ir_detect import detect as det
from ir_detect.preprocess import to_u8
from ir_detect.sources import CameraSource, Mode
from ir_detect.sources.tis_dmk import TisDmkSource

WIN = "IR camera viewer"

# cv2.waitKeyEx 의 화살표 코드 (Windows / Linux-GTK)
KEY_UP = {2490368, 65362}
KEY_DOWN = {2621440, 65364}
KEY_LEFT = {2424832, 65361}
KEY_RIGHT = {2555904, 65363}
KEY_ENTER = {13, 10}
KEY_ESC = 27

EXPOSURE_MS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
GAIN_DB = [0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48]


@dataclass(frozen=True)
class Target:
    """원하는 스트림 구성. ires/cres = (w, h, fps), 0 은 장치 기본값. IR 은 항상 켠다."""

    ires: tuple = (0, 0, 0)  # IR 모드 (depth 도 같은 모드). fps 0 = 자동
    cres: tuple = (0, 0, 0)  # color 해상도 (꺼져 있어도 기억해 둔다)
    depth: bool = False
    color: bool = False


@dataclass
class ViewFlags:
    grid: bool = False  # 각도 격자
    align: bool = True  # D2C 정렬
    auto: bool = False  # 화면 중앙 depth 를 거리로
    detect: str = "off"  # off / apriltag / shapes / both
    family: str = "tag36h11"
    invert: bool = False  # 검출 전 영상 반전 (흰 테두리 태그, 어두운 바탕의 밝은 도형)
    shape_type: str = "all"  # det.SHAPE_TYPES
    shape_regular: bool = True  # 정형(정삼각형·정사각형·정십자·원)만
    shape_tol: float = 0.1  # 정형 허용 편차
    shape_min: int = 20  # 최소 크기 (원본 픽셀)
    shape_mode: str = "both"  # det.SHAPE_MODES: filled / outline / both
    detect_width: int = 1280  # 검출용 축소 폭 (0 = 원본 그대로). DMK 2448px 에서 작은 도형은 full 로

    @property
    def tags(self) -> bool:
        return self.detect in ("apriltag", "both")

    @property
    def shapes(self) -> bool:
        return self.detect in ("shapes", "both")

    @property
    def shape_options(self) -> det.ShapeOptions:
        return det.ShapeOptions(self.shape_type, self.shape_regular, self.shape_tol, self.shape_min, self.shape_mode)


DETECT_OPTS = ["off", "apriltag", "shapes", "both"]
DETECT_WIDTHS = ["640", "1280", "full"]  # 검출용 축소 폭
DETECT_WIDTH_VALUES = (640, 1280, 0)


# ---------------------------------------------------------------------------
# 설정 메뉴 (로직만, 그리기는 draw_menu)
# ---------------------------------------------------------------------------


@dataclass
class MenuRow:
    key: str
    label: str
    options: list  # 표시 문자열 목록
    index: int = 0
    group: str = "camera"  # 패널: "camera" (왼쪽) / "detect" (오른쪽)

    @property
    def value(self) -> str:
        return self.options[self.index]


PANELS = (("camera", "Camera"), ("detect", "Detection"))


class MenuState:
    """↑↓ 로 항목, ←→ 로 값을 바꾸고 Enter 로 한꺼번에 적용한다 (재시작은 한 번만).
    on_adjust(menu, key) 로 항목 간 연동(모드 바꾸면 fps 목록 갱신 등)을 건다."""

    def __init__(self, rows: list[MenuRow], on_adjust: Callable[["MenuState", str], None] | None = None):
        self._rows = rows
        self._on_adjust = on_adjust
        self.sel = 0
        self.initial = {r.key: r.index for r in rows}

    def row(self, key: str) -> MenuRow:
        return next(r for r in self._rows if r.key == key)

    def has(self, key: str) -> bool:
        return any(r.key == key for r in self._rows)

    def get(self, key: str) -> int:
        return self.row(key).index

    def set(self, key: str, index: int) -> None:
        self.row(key).index = index

    def changed(self, key: str) -> bool:
        return self.has(key) and self.get(key) != self.initial[key]

    def rows(self) -> list[tuple[str, str]]:
        return [(r.label, r.value) for r in self._rows]

    def move(self, d: int) -> None:
        self.sel = (self.sel + d) % len(self._rows)

    def adjust(self, d: int) -> None:
        r = self._rows[self.sel]
        r.index = (r.index + d) % len(r.options)
        if self._on_adjust:
            self._on_adjust(self, r.key)


def _onoff(v: bool) -> int:
    return 1 if v else 0


def _nearest(options: list[float], v: float | None) -> int:
    if v is None:
        return 0
    return min(range(len(options)), key=lambda i: abs(options[i] - v))


def detect_rows(flags: ViewFlags) -> list[MenuRow]:
    """모든 장치에 공통인 인식 항목."""
    fams = list(det.FAMILIES)
    types = list(det.SHAPE_TYPES)
    tols = [f"{int(t * 100)}%" for t in det.SHAPE_TOLS]
    mins = [str(v) for v in det.SHAPE_MIN_PX]
    modes = list(det.SHAPE_MODES)
    g = "detect"
    return [
        MenuRow("detect", "Detect", DETECT_OPTS, DETECT_OPTS.index(flags.detect), g),
        MenuRow("dwidth", "Detect width", DETECT_WIDTHS, _nearest(list(DETECT_WIDTH_VALUES), flags.detect_width), g),
        MenuRow("family", "Tag family", fams, fams.index(flags.family) if flags.family in fams else 0, g),
        MenuRow("invert", "Invert image", ["off", "on"], _onoff(flags.invert), g),
        MenuRow("stype", "Shape type", types, types.index(flags.shape_type) if flags.shape_type in types else 0, g),
        MenuRow("smode", "Shape mode", modes, modes.index(flags.shape_mode) if flags.shape_mode in modes else 0, g),
        MenuRow("sreg", "Regular only", ["off", "on"], _onoff(flags.shape_regular), g),
        MenuRow("stol", "Shape tolerance", tols, _nearest(list(det.SHAPE_TOLS), flags.shape_tol), g),
        MenuRow("smin", "Shape min px", mins, _nearest(list(det.SHAPE_MIN_PX), flags.shape_min), g),
    ]


def detect_flags(m: MenuState, flags: ViewFlags) -> ViewFlags:
    return replace(
        flags,
        detect=m.row("detect").value,
        detect_width=DETECT_WIDTH_VALUES[m.get("dwidth")],
        family=m.row("family").value,
        invert=bool(m.get("invert")),
        shape_type=m.row("stype").value,
        shape_mode=m.row("smode").value,
        shape_regular=bool(m.get("sreg")),
        shape_tol=det.SHAPE_TOLS[m.get("stol")],
        shape_min=det.SHAPE_MIN_PX[m.get("smin")],
    )


def build_orbbec_menu(labels: list[str], op: "Opened", target: Target, flags: ViewFlags) -> MenuState:
    ir_opts = [f"{w}x{h} {mode_name('ir', w, h)}" for w, h in op.ir_modes]
    ir_i = op.ir_modes.index(target.ires[:2]) if target.ires[:2] in op.ir_modes else 0
    c_opts = [f"{w}x{h} ({mode_name('color', w, h)})" for w, h in op.color_modes]
    c_i = op.color_modes.index(target.cres[:2]) if target.cres[:2] in op.color_modes else 0

    def fps_options(i: int) -> list[str]:
        return ["auto"] + [str(f) for f in sorted(op.fps_by_mode.get(op.ir_modes[i], ()))]

    fps_o = fps_options(ir_i)
    fps_i = fps_o.index(str(target.ires[2])) if str(target.ires[2]) in fps_o else 0
    rows = [
        MenuRow("dev", "Device", labels, op.dev),
        MenuRow("ir", "IR mode", ir_opts, ir_i),
        MenuRow("fps", "FPS", fps_o, fps_i),
        MenuRow("depth", "Depth", ["off", "on"], _onoff(target.depth)),
        MenuRow("color", "Color", ["off", "on"], _onoff(target.color)),
        MenuRow("cres", "Color res", c_opts, c_i),
        MenuRow("grid", "Angle grid", ["off", "on"], _onoff(flags.grid)),
        MenuRow("align", "D2C align", ["off", "on"], _onoff(flags.align)),
        MenuRow("auto", "Auto distance", ["off", "on"], _onoff(flags.auto)),
        *detect_rows(flags),
    ]

    def on_adjust(m: MenuState, key: str) -> None:
        if key == "ir":  # 모드가 바뀌면 그 모드의 fps 목록으로. 없는 값이면 auto
            r = m.row("fps")
            cur = r.value
            r.options = fps_options(m.get("ir"))
            r.index = r.options.index(cur) if cur in r.options else 0
        elif key == "cres":  # 해상도를 고르면 color 를 켠다
            m.set("color", 1)

    return MenuState(rows, on_adjust)


def orbbec_menu_result(m: MenuState, op: "Opened") -> tuple[Target, ViewFlags]:
    fps = m.row("fps").value
    t = Target(
        (*op.ir_modes[m.get("ir")], 0 if fps == "auto" else int(fps)),
        (*op.color_modes[m.get("cres")], 0),
        bool(m.get("depth")),
        bool(m.get("color")),
    )
    return t, detect_flags(m, ViewFlags(bool(m.get("grid")), bool(m.get("align")), bool(m.get("auto"))))


def build_tis_menu(labels: list[str], op: "Opened", flags: ViewFlags) -> MenuState:
    src: TisDmkSource = op.source
    mode_opts = [f"{m.pixel_format} {m.res}" for m in op.modes]
    mode_i = op.modes.index(src.mode) if src.mode in op.modes else 0
    exp = src.get_value("Exposure")
    rows = [
        MenuRow("dev", "Device", labels, op.dev),
        MenuRow("mode", "Mode", mode_opts, mode_i),
        MenuRow("exp_auto", "Exposure auto", ["off", "on"], _onoff(src.get_auto("Exposure"))),
        MenuRow("exp", "Exposure (ms)", [f"{v:g}" for v in EXPOSURE_MS], _nearest(EXPOSURE_MS, exp * 1000 if exp else None)),
        MenuRow("gain_auto", "Gain auto", ["off", "on"], _onoff(src.get_auto("Gain"))),
        MenuRow("gain", "Gain (dB)", [str(v) for v in GAIN_DB], _nearest(GAIN_DB, src.get_value("Gain"))),
    ]
    if op.intr is not None:
        rows.append(MenuRow("grid", "Angle grid", ["off", "on"], _onoff(flags.grid)))
    rows += detect_rows(flags)

    def on_adjust(m: MenuState, key: str) -> None:
        if key == "exp":
            m.set("exp_auto", 0)
        elif key == "gain":
            m.set("gain_auto", 0)

    return MenuState(rows, on_adjust)


@dataclass
class MenuLayout:
    """draw_menu 가 그린 위치. 마우스 클릭을 항목/화살표/버튼에 대응시킨다. rows 순서 = MenuState 의 행 순서."""

    rows: list[tuple[tuple[int, int, int, int], int, int]]  # 행마다 (행 사각형, '<' 오른끝 x, '>' 왼끝 x)
    apply: tuple[int, int, int, int]
    close: tuple[int, int, int, int]

    @staticmethod
    def inside(r: tuple[int, int, int, int], x: int, y: int) -> bool:
        return r[0] <= x <= r[2] and r[1] <= y <= r[3]

    def row_at(self, x: int, y: int) -> int:
        return next((i for i, (r, _, _) in enumerate(self.rows) if self.inside(r, x, y)), -1)

    def arrow(self, i: int, x: int) -> int:
        """행 i 안의 x 가 어느 쪽인지: -1 '<', +1 '>' 또는 값 글자, 0 라벨."""
        r, left_end, right_start = self.rows[i]
        if x >= right_start:
            return 1
        if x <= left_end:
            return -1 if x > left_end - 34 else 0
        return 1


ROW_H = 32
PANEL_GAP = 16


def draw_menu(canvas: np.ndarray, menu: MenuState) -> MenuLayout:
    """패널 두 개: 왼쪽 Camera (장치·스트림·표시), 오른쪽 Detection (태그·도형). 행 순서는 MenuState 그대로."""
    rows = menu._rows
    label_w = 170
    panels = []  # (group, title, [행 인덱스…], 폭)
    for group, title in PANELS:
        idx = [i for i, r in enumerate(rows) if r.group == group]
        if not idx:
            continue
        value_w = max(220, max(len(rows[i].value) for i in idx) * 11 + 80)
        panels.append((group, title, idx, label_w + value_w + 40))
    n_max = max(len(p[2]) for p in panels)
    h = ROW_H * (n_max + 1) + 24
    x0, y0 = 30, 30
    total_w = sum(p[3] for p in panels) + PANEL_GAP * (len(panels) - 1)
    layout_rows: dict[int, tuple] = {}
    px = x0
    for group, title, idx, w in panels:
        box = canvas[y0 : y0 + h, px : px + w]
        box[:] = (box * 0.25).astype(np.uint8)
        cv2.rectangle(canvas, (px, y0), (px + w, y0 + h), viz.YELLOW, 1)
        viz.put_text(canvas, title, (px + 16, y0 + 24), 0.65, viz.CYAN)
        vx = px + label_w  # 값 영역 시작
        for k, i in enumerate(idx):
            y = y0 + ROW_H * (k + 2)
            r = (px + 6, y - 22, px + w - 6, y + 8)
            color = viz.YELLOW if i == menu.sel else viz.WHITE
            if i == menu.sel:
                cv2.rectangle(canvas, (r[0], r[1]), (r[2], r[3]), (60, 60, 60), -1)
            viz.put_text(canvas, rows[i].label, (px + 16, y), 0.6, color)
            viz.put_text(canvas, "<", (vx, y), 0.6, color)
            viz.put_text(canvas, rows[i].value, (vx + 28, y), 0.6, color)
            viz.put_text(canvas, ">", (px + w - 34, y), 0.6, color)
            layout_rows[i] = (r, vx + 24, px + w - 44)
        px += w + PANEL_GAP
    # 아래 바: 안내 + Apply / Close
    by0, by1 = y0 + h + 8, y0 + h + 44
    bar = canvas[by0:by1, x0 : x0 + total_w]
    bar[:] = (bar * 0.25).astype(np.uint8)
    cv2.rectangle(canvas, (x0, by0), (x0 + total_w, by1), viz.YELLOW, 1)
    apply = (x0 + total_w - 220, by0 + 6, x0 + total_w - 124, by1 - 6)
    close = (x0 + total_w - 108, by0 + 6, x0 + total_w - 12, by1 - 6)
    for r, text in ((apply, "Apply"), (close, "Close")):
        cv2.rectangle(canvas, (r[0], r[1]), (r[2], r[3]), viz.YELLOW, 1)
        viz.put_text(canvas, text, (r[0] + 14, r[3] - 8), 0.55, viz.YELLOW)
    viz.put_text(canvas, "up/down or click: select   left/right, < >, wheel: change   Enter: apply   Esc: close",
                 (x0 + 12, by1 - 12), 0.45, viz.YELLOW)
    return MenuLayout([layout_rows[i] for i in range(len(rows))], apply, close)


def draw_menu_button(canvas: np.ndarray) -> tuple[int, int, int, int]:
    """하단 오른쪽 'MENU' 버튼. 클릭하면 메뉴가 열린다."""
    h, w = canvas.shape[:2]
    r = (w - 110, h - 34, w - 14, h - 6)
    cv2.rectangle(canvas, (r[0], r[1]), (r[2], r[3]), (40, 40, 40), -1)
    cv2.rectangle(canvas, (r[0], r[1]), (r[2], r[3]), viz.YELLOW, 1)
    viz.put_text(canvas, "MENU (e)", (r[0] + 10, r[3] - 9), 0.5, viz.YELLOW)
    return r


# ---------------------------------------------------------------------------
# 장치 / 스트림
# ---------------------------------------------------------------------------


@dataclass
class Session:
    target: Target  # 실제로 켜진 구성 (해상도는 확정값)
    ir_p: object
    depth_p: object | None
    color_p: object | None
    fps: int
    ii: Intrinsics
    di: Intrinsics | None
    ci: Intrinsics | None
    R_ic: np.ndarray | None = None  # IR → color
    t_ic: np.ndarray | None = None
    R_ci: np.ndarray | None = None  # color → IR
    t_ci: np.ndarray | None = None
    R_dc: np.ndarray | None = None  # depth → color
    t_dc: np.ndarray | None = None
    R_cd: np.ndarray | None = None  # color → depth
    t_cd: np.ndarray | None = None
    static: dict = field(default_factory=dict)


@dataclass
class DeviceChoice:
    kind: str  # "orbbec" | "tis"
    label: str
    entry: DeviceEntry | None = None  # orbbec
    tis_name: str = ""  # tis unique name


@dataclass
class Opened:
    """열린 장치 하나. Orbbec 은 pipeline + st, TIS 는 source + modes."""

    dev: int
    kind: str
    summary: str
    pipeline: Pipeline | None = None
    ir_modes: list = field(default_factory=list)
    color_modes: list = field(default_factory=list)
    fps_by_mode: dict = field(default_factory=dict)
    st: Session | None = None
    source: CameraSource | None = None
    modes: list = field(default_factory=list)
    intr: Intrinsics | None = None

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        if self.source is not None:
            self.source.close()
        self.pipeline = self.source = None


def device_label(e: DeviceEntry) -> str:
    s = f"{e.name.replace('Orbbec ', '')} {e.connection} {e.serial}"
    return f"{s} {e.ip}" if e.ip else s


def find_devices(ctx: Context, ip: str) -> list[DeviceChoice]:
    out = []
    if ip:
        out.append(DeviceChoice("orbbec", f"Femto Mega Ethernet {ip}", DeviceEntry("Femto Mega", "", "Ethernet", ip)))
    out += [DeviceChoice("orbbec", device_label(e), e) for e in enumerate_devices(ctx)]
    ok, why = TisDmkSource.available()
    if ok:
        out += [DeviceChoice("tis", s.unique_name + " GigE", tis_name=s.unique_name) for s in TisDmkSource.discover()]
    else:
        print(f"[tis] {why}")
    return out


def open_device(ctx: Context, choices: list[DeviceChoice], i: int, target: Target, port: int,
                mode: Mode | None = None) -> Opened:
    """장치를 열고 스트림 시작. target 의 해상도가 이 장치에 없으면 기본값으로."""
    c = choices[i]
    if c.kind == "tis":
        src = TisDmkSource(c.label, c.tis_name)
        src.open()
        modes = src.modes()
        if mode is None:  # 기본: 100 Mbps 링크에서도 볼 만한 1280x960 Y800, 없으면 첫 모드
            mode = next((m for m in modes if m.res == "1280x960" and m.pixel_format == "Y800"), modes[0])
        src.start(mode)
        info = src.info()
        summary = "  ".join(f"{k}={v}" for k, v in info.items())
        print(summary)
        return Opened(i, "tis", summary, source=src, modes=modes, intr=src.intrinsics(mode))

    e = c.entry
    dev = get_device(ctx, ip=e.ip, port=port) if e.serial == "" else get_device(ctx, serial=e.serial)
    pipeline = Pipeline(dev)
    ir_modes = sorted(
        (p.key for p in unique_resolutions(list_profiles(pipeline, "ir"))),
        key=lambda r: (mode_name("ir", *r).startswith("WFOV"), r[0] * r[1]),
    )  # NFOV binned → NFOV → WFOV binned → WFOV
    color_modes = [p.key for p in unique_resolutions(list_profiles(pipeline, "color"))]
    fps_by_mode: dict = {}
    for p in list_profiles(pipeline, "ir"):
        fps_by_mode.setdefault(p.key, set()).add(p.fps)
    if target.ires[:2] not in ir_modes:
        target = replace(target, ires=(0, 0, target.ires[2]))
    if target.cres[:2] not in color_modes:
        target = replace(target, cres=(0, 0, 0))
    summary = device_summary(pipeline)
    print(summary)
    st = start_streams(pipeline, target)
    return Opened(i, "orbbec", summary, pipeline, ir_modes, color_modes, fps_by_mode, st)


def describe(t: Target) -> str:
    ir = f"IR {t.ires[0]}x{t.ires[1]} {mode_name('ir', t.ires[0], t.ires[1])}" if t.ires[0] else "IR default"
    if t.ires[2]:
        ir += f"@{t.ires[2]}"
    depth = "depth on" if t.depth else "depth off"
    color = (f"color {t.cres[0]}x{t.cres[1]}" if t.cres[0] else "color default") if t.color else "color off"
    return f"{ir} / {depth} / {color}"


def resolve_target(pipeline, t: Target) -> Target:
    """해상도 0 (기본값) 을 장치 기본 프로파일의 해상도로 확정."""
    ires, cres = t.ires, t.cres
    if ires[0] == 0:
        p = find_profile(pipeline, "ir")
        ires = (p.get_width(), p.get_height(), ires[2])
    if t.color and cres[0] == 0:
        p = find_profile(pipeline, "color")
        cres = (p.get_width(), p.get_height(), cres[2])
    return replace(t, ires=ires, cres=cres)


def common_fps(pipeline, t: Target) -> int:
    """켜진 스트림이 모두 지원하는 가장 높은 fps (예: Bolt WFOV 1024x1024 는 15fps 까지)."""
    def fps_set(sensor, w, h):
        return {p.fps for p in list_profiles(pipeline, sensor) if p.key == (w, h)}

    sets = [fps_set("ir", *t.ires[:2])]
    if t.depth:
        sets.append(fps_set("depth", *t.ires[:2]))
    if t.color:
        sets.append(fps_set("color", *t.cres[:2]))
    both = set.intersection(*sets)
    want = t.ires[2] or (t.cres[2] if t.color else 0)
    if want and want in both:
        return want
    if not both:
        raise OBError(f"공통 fps 없음: {describe(t)}")
    return max(both)


def start_streams(pipeline, target: Target, running: bool = False) -> Session:
    """스트림 프로파일 선택 → 파이프라인 (재)시작. IR 은 항상, depth/color 는 target 에 따라."""
    if running:
        pipeline.stop()
    t = resolve_target(pipeline, target)
    fps = common_fps(pipeline, t)
    w, h = t.ires[:2]
    ir_p = find_profile(pipeline, "ir", w, h, fps)
    depth_p = find_profile(pipeline, "depth", w, h, fps) if t.depth else None  # IR 과 같은 센서/모드
    color_p = find_profile(pipeline, "color", t.cres[0], t.cres[1], fps) if t.color else None

    config = Config()
    for p, name in ((ir_p, "ir"), (depth_p, "depth"), (color_p, "color")):
        if p is not None:
            config.enable_stream(p)
            print(f"  {name:5}: {p.get_width()}x{p.get_height()}@{p.get_fps()} {p.get_format().name}")

    ii = to_intrinsics(ir_p)
    di = to_intrinsics(depth_p) if depth_p is not None else None
    ci = to_intrinsics(color_p) if color_p is not None else None
    s = Session(t, ir_p, depth_p, color_p, ir_p.get_fps(), ii, di, ci)
    if color_p is not None:
        s.R_ic, s.t_ic = extrinsic_Rt(ir_p, color_p)
        s.R_ci, s.t_ci = extrinsic_Rt(color_p, ir_p)
        print(f"  IR→color baseline t = {np.round(s.t_ic, 2)} mm")
        if depth_p is not None:
            s.R_dc, s.t_dc = extrinsic_Rt(depth_p, color_p)
            s.R_cd, s.t_cd = extrinsic_Rt(color_p, depth_p)
    s.static = {"ir": (fov_pinhole(ii), fov_undistorted(ii))}
    if di is not None:
        s.static["depth"] = (fov_pinhole(di), fov_undistorted(di))
    if ci is not None:
        s.static["color"] = (fov_pinhole(ci), fov_undistorted(ci))
    try:
        pipeline.enable_frame_sync()
    except Exception:
        pass
    pipeline.start(config)
    return s


class Switcher:
    """재시작/장치 전환을 백그라운드 스레드에서 수행 (그동안 UI 는 계속 반응)."""

    def __init__(self):
        self.thread: threading.Thread | None = None
        self.result: tuple | None = None
        self.label = ""
        self.started = 0.0

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, fn, label: str) -> None:
        self.label, self.started, self.result = label, time.time(), None
        self.thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self.thread.start()

    def _run(self, fn) -> None:
        try:
            self.result = (fn(), None)
        except Exception as e:  # noqa: BLE001
            self.result = (None, e)

    def take_result(self) -> tuple:
        r, self.result = self.result, None
        return r

    def join(self) -> None:
        if self.thread is not None:
            self.thread.join()


def show_message(w: int, h: int, lines: list[str], base: np.ndarray | None = None) -> None:
    """안내 화면 (base 가 있으면 어둡게 깔고 그 위에)."""
    img = (base * 0.35).astype(np.uint8) if base is not None else np.zeros((h, w, 3), np.uint8)
    viz.put_lines(img, lines, 0.8, viz.YELLOW, origin=(30, h // 2 - 40))
    cv2.imshow(WIN, img)
    cv2.waitKey(1)


# ---------------------------------------------------------------------------
# 렌더링
# ---------------------------------------------------------------------------


class OrbbecView:
    """Femto 프레임 → 4 패널. depth 유효 영역 이동평균 등 프레임 간 상태를 들고 있다."""

    def __init__(self, cell_w: int, cell_h: int, detp: det.Pipeline):
        self.cell_w, self.cell_h = cell_w, cell_h
        self.blank = np.zeros((cell_h, cell_w, 3), np.uint8)
        self.detp = detp
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.reset()

    def reset(self) -> None:
        self.last = {"color": None, "ir": None, "depth": None}
        self.det = det.DetResult()
        self.valid_score = None  # depth 유효 픽셀 이동평균 (노이즈로 깜빡이는 픽셀 제거)
        self.hull = None  # 유효 영역 볼록 껍질 (depth 픽셀)
        self.vfov = None  # 유효 영역 FOV (10프레임마다 갱신)
        self.frame_i = 0
        self.cov = {"c": float("nan"), "d": float("nan"), "vc": float("nan")}
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)  # 프로파일 캐시 초기화

    def ingest(self, st: Session, frames, do_align: bool) -> bool:
        """프레임셋에서 영상을 꺼낸다. 켜진 스트림이 다 모였으면 True."""
        has_color, has_depth = st.color_p is not None, st.depth_p is not None
        irf = frames.get_ir_frame()
        cf = frames.get_color_frame() if has_color else None
        df = frames.get_depth_frame() if has_depth else None
        if irf is not None:
            self.last["ir"] = ir_to_u16(irf)
        if cf is not None:
            img = color_to_bgr(cf)
            if img is not None:
                self.last["color"] = img
        if df is not None:
            self.last["depth"] = depth_to_mm(df)
        self.aligned = None
        if do_align and cf is not None and df is not None:
            try:
                out = self.align.process(frames)
                if out is not None:
                    fs = out.as_frame_set() if hasattr(out, "as_frame_set") else out
                    adf = fs.get_depth_frame()
                    if adf is not None:
                        self.aligned = depth_to_mm(adf)
            except Exception:
                self.aligned = None
        return (self.last["ir"] is not None and (not has_color or self.last["color"] is not None)
                and (not has_depth or self.last["depth"] is not None))

    def auto_distance(self, fallback: float) -> float:
        depth = self.last["depth"]
        if depth is None:
            return fallback
        h, w = depth.shape
        c = depth[h // 2 - 10 : h // 2 + 10, w // 2 - 10 : w // 2 + 10]
        c = c[c > 0]
        return float(np.median(c)) if c.size else fallback

    def panels(self, st: Session, distance: float, flags: ViewFlags) -> list:
        ii, ci, di, static, last = st.ii, st.ci, st.di, st.static, self.last
        has_color, has_depth = st.color_p is not None, st.depth_p is not None
        cell_w, cell_h = self.cell_w, self.cell_h

        # --- depth 실제 유효 영역 (센서 사각형이 아니라 값이 나오는 영역) ---
        if has_depth:
            depth = last["depth"]
            cur = (depth > 0).astype(np.float32)
            if self.valid_score is None or self.valid_score.shape != cur.shape:
                self.valid_score = cur
            else:
                cv2.accumulateWeighted(cur, self.valid_score, 0.1)  # 이동평균 (in-place)
            self.frame_i += 1
            if self.hull is None or self.frame_i % 10 == 0:  # 무거운 계산은 10프레임마다
                valid_mask = self.valid_score > 0.2
                nz = cv2.findNonZero(valid_mask.astype(np.uint8))
                self.hull = cv2.convexHull(nz).reshape(-1, 2).astype(np.float64) if nz is not None else None
                self.vfov = valid_region_fov(valid_mask, di)
        hull = self.hull

        # --- 경계 투영 (원본 픽셀 좌표). IR 이 기준 센서, depth 는 같은 센서 ---
        i_in_c = c_in_i = c_in_d = v_in_c = None
        cov_c = cov_d = cov_vc = float("nan")
        if has_color:
            i_in_c = project_border(ii, ci, st.R_ic, st.t_ic, distance)
            c_in_i = project_border(ci, ii, st.R_ci, st.t_ci, distance)
            cov_c, _ = overlap_ratio(i_in_c, ci.width, ci.height)
            if has_depth:
                c_in_d = project_border(ci, di, st.R_cd, st.t_cd, distance)
                cov_d, _ = overlap_ratio(c_in_d, di.width, di.height)
                if hull is not None and len(hull) >= 3:
                    v_in_c = project_between(hull, distance, di, ci, st.R_dc, st.t_dc)
                    cov_vc, _ = overlap_ratio(v_in_c, ci.width, ci.height)
        self.cov = {"c": cov_c, "d": cov_d, "vc": cov_vc}

        # 각 패널은 표시 크기로 먼저 줄인 뒤 그린다 (원본 해상도에서 그리면 WFOV/4K 에서 매우 느림)
        si = viz.fit_scale(ii.width, ii.height, cell_w, cell_h)
        ir = viz.normalize_ir(viz.shrink(last["ir"], si))
        if flags.grid:
            viz.draw_angle_grid(ir, ii.scaled(si))
        fw, fh = footprint_at_distance(static["ir"][1], distance)
        ir_lines = [
            f"IR {ii.width}x{ii.height} ({mode_name('ir', ii.width, ii.height)}) @{st.fps}fps",
            f"pinhole {static['ir'][0].short()}",
            f"undist  {static['ir'][1].short()}",
            f"@{distance:.0f}mm view = {fw:.0f} x {fh:.0f} mm",
        ]
        if c_in_i is not None:
            viz.draw_polygon(ir, c_in_i * si, viz.MAGENTA, 2)
            ir_lines.append("magenta: color FOV")
        if flags.detect != "off":
            self.detp.max_width = flags.detect_width or 10**9
            self.det = self.detp.run(to_u8(last["ir"]), flags.tags, flags.shapes, flags.family, flags.invert, flags.shape_options)
            det.draw(ir, self.det, si)
            ir_lines.append(f"{flags.family if flags.tags else ''} {'inv' if flags.invert else ''} {self.det.summary()}")

        if has_color:
            sc = viz.fit_scale(ci.width, ci.height, cell_w, cell_h)
            color_small = viz.shrink(last["color"], sc)
            color = color_small.copy()
            if flags.grid:
                viz.draw_angle_grid(color, ci.scaled(sc))
            viz.draw_polygon(color, i_in_c * sc, viz.CYAN, 2)
            if v_in_c is not None:
                viz.draw_polygon(color, v_in_c * sc, viz.GREEN, 2)
            fw, fh = footprint_at_distance(static["color"][1], distance)
            color_lines = [
                f"COLOR {ci.width}x{ci.height} ({mode_name('color', ci.width, ci.height)})",
                f"pinhole {static['color'][0].short()}",
                f"undist  {static['color'][1].short()}",
                f"@{distance:.0f}mm view = {fw:.0f} x {fh:.0f} mm",
                f"cyan : IR/depth sensor FOV, covers {cov_c * 100:.1f}% of color",
            ]
            if has_depth:
                color_lines.append(f"green: depth valid area, covers {cov_vc * 100:.1f}% of color")
        else:
            color, color_lines = self.blank, ["COLOR off   (e: menu)"]

        if has_depth:
            sd = viz.fit_scale(di.width, di.height, cell_w, cell_h)
            dimg = viz.colorize_depth(viz.shrink(depth, sd, nearest=True))
            if flags.grid:
                viz.draw_angle_grid(dimg, di.scaled(sd))
            if hull is not None:
                viz.draw_polygon(dimg, hull * sd, viz.GREEN, 1)
            depth_lines = [
                f"DEPTH {di.width}x{di.height} ({mode_name('depth', di.width, di.height)})",
                f"pinhole {static['depth'][0].short()}",
                f"undist  {static['depth'][1].short()}",
                f"valid   {self.vfov.short()}",
            ]
            if c_in_d is not None:
                viz.draw_polygon(dimg, c_in_d * sd, viz.MAGENTA, 2)
                depth_lines.append(f"magenta: color FOV, covers {cov_d * 100:.1f}% of depth")
        else:
            dimg, depth_lines = self.blank, ["DEPTH off   (e: menu)"]

        if has_color and has_depth and self.aligned is not None and self.aligned.shape[:2] == last["color"].shape[:2]:
            small_d = viz.shrink(self.aligned, sc, nearest=True)
            blend = cv2.addWeighted(color_small, 0.5, viz.colorize_depth(small_d), 0.5, 0)
            blend_lines = ["Depth aligned to Color (D2C)", f"color pixels with depth: {(small_d > 0).mean() * 100:.1f}%"]
        elif not (has_color and has_depth):
            blend, blend_lines = self.blank, ["D2C needs color + depth"]
        else:
            blend = np.zeros_like(color)
            blend_lines = ["D2C align off" if not flags.align else "D2C align unavailable"]
        return [(ir, ir_lines), (color, color_lines), (dimg, depth_lines), (blend, blend_lines)]

    def snapshot_meta(self, st: Session, distance: float) -> dict:
        ii, ci, di, static = st.ii, st.ci, st.di, st.static
        meta = {
            "distance_mm": distance,
            "fps": st.fps,
            "ir": {"res": [ii.width, ii.height], "pinhole": static["ir"][0].as_dict(),
                   "undistorted": static["ir"][1].as_dict()},
        }
        if ci is not None:
            meta["color"] = {"res": [ci.width, ci.height], "pinhole": static["color"][0].as_dict(),
                             "undistorted": static["color"][1].as_dict()}
            meta["ir_covers_color_pct"] = round(self.cov["c"] * 100, 2)
            meta["ir_to_color_t_mm"] = np.round(st.t_ic, 3).tolist()
        if di is not None:
            meta["depth"] = {"res": [di.width, di.height], "pinhole": static["depth"][0].as_dict(),
                             "undistorted": static["depth"][1].as_dict(), "valid": self.vfov.as_dict()}
        if ci is not None and di is not None:
            meta["depth_valid_covers_color_pct"] = round(self.cov["vc"] * 100, 2)
            meta["color_covers_depth_pct"] = round(self.cov["d"] * 100, 2)
        meta.update(detection_meta(self.det))
        return meta


def detection_meta(res: det.DetResult) -> dict:
    if not res.backend and not res.shapes:
        return {}
    return {
        "detection": {
            "backend": res.backend,
            "ms": round(res.ms, 1),
            "tags": [{"id": t.id, "center": [round(v, 1) for v in t.center], "size_px": round(t.size_px, 1),
                      "hamming": t.hamming, "margin": None if t.margin is None else round(t.margin, 1),
                      "corners": np.round(t.corners, 1).tolist()} for t in res.tags],
            "shapes": [{"name": s.name, "center": [round(v, 1) for v in s.center], "area_px": round(s.area)} for s in res.shapes],
        }
    }


class TisView:
    """DMK 프레임 → IR 패널 하나 (나머지는 비움)."""

    def __init__(self, cell_w: int, cell_h: int, detp: det.Pipeline):
        self.cell_w, self.cell_h = cell_w, cell_h
        self.blank = np.zeros((cell_h, cell_w, 3), np.uint8)
        self.detp = detp
        self.reset()

    def reset(self) -> None:
        self.last = {"color": None, "ir": None, "depth": None}
        self.det = det.DetResult()
        self.frame = None

    def ingest(self, frame) -> bool:
        self.frame = frame
        self.last["ir"] = frame.raw
        return True

    def panels(self, op: Opened, flags: ViewFlags, fps: float) -> list:
        f = self.frame
        src: TisDmkSource = op.source
        s = viz.fit_scale(f.raw.shape[1], f.raw.shape[0], self.cell_w, self.cell_h)
        ir = cv2.cvtColor(viz.shrink(f.image, s), cv2.COLOR_GRAY2BGR)
        exp, gain = src.get_value("Exposure"), src.get_value("Gain")
        lines = [
            f"{op.summary.split('  ')[0].replace('name=', '')}",
            f"{f.mode.pixel_format} {f.mode.res}  set {f.meta.get('fps_set', 0):.1f}fps / actual {fps:.1f}fps",
            f"exposure {exp * 1000:.2f}ms {'(auto)' if src.get_auto('Exposure') else ''}   gain {gain:.1f}dB {'(auto)' if src.get_auto('Gain') else ''}"
            if exp is not None and gain is not None else "",
            f"raw max {int(f.raw.max())} / {np.iinfo(f.raw.dtype).max}   sat {(f.image >= 250).mean() * 100:.1f}%"
            f"   focus {cv2.Laplacian(viz.shrink(f.image, s), cv2.CV_64F).var():.0f} (higher = sharper)",
        ]
        if op.intr is not None:
            if flags.grid:
                viz.draw_angle_grid(ir, op.intr.scaled(s))
            lines.append(f"undist  {fov_undistorted(op.intr).short()}  (calib file)")
        else:
            lines.append("no calibration: FOV overlay unavailable (calib/dmk_<serial>.json)")
        if flags.detect != "off":
            self.detp.max_width = flags.detect_width or 10**9
            self.det = self.detp.run(f.image, flags.tags, flags.shapes, flags.family, flags.invert, flags.shape_options)
            det.draw(ir, self.det, s)
            lines.append(f"{flags.family if flags.tags else ''} {'inv' if flags.invert else ''} {self.det.summary()}")
        na = ["n/a for this device"]
        return [(ir, lines), (self.blank, na), (self.blank, na), (self.blank, na)]

    def snapshot_meta(self, op: Opened, fps: float) -> dict:
        f = self.frame
        src: TisDmkSource = op.source
        meta = {"mode": str(f.mode), "fps_set": f.meta.get("fps_set"), "fps_measured": round(fps, 2),
                "exposure_s": src.get_value("Exposure"), "gain_db": src.get_value("Gain")}
        meta.update(detection_meta(self.det))
        return meta


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default="", help="Orbbec 장치 S/N")
    ap.add_argument("--ip", default="", help="Orbbec 네트워크 장치 IP (자동 탐색에 안 잡힐 때)")
    ap.add_argument("--port", type=int, default=DEFAULT_NET_PORT)
    ap.add_argument("--device", default="", help="장치 이름에 포함된 문자열로 시작 장치 선택 (예: dmk, mega, bolt)")
    ap.add_argument("--ir", help="IR(=depth) 모드 WxH[@fps] (Femto: 640x576 / 320x288 / 1024x1024 / 512x512)")
    ap.add_argument("--depth", nargs="?", const="on", help="depth 켜고 시작. 해상도를 주면 --ir 과 같음")
    ap.add_argument("--color", nargs="?", const="on", help="color 켜고 시작. 해상도 WxH[@fps] 지정 가능")
    ap.add_argument("--distance", type=float, default=1000.0, help="경계선 투영 거리 mm (기본 1000)")
    ap.add_argument("--out", type=Path, default=Path("results"), help="스냅샷 저장 폴더")
    ap.add_argument("--cell", default="800x600", help="패널 하나 크기 WxH")
    ap.add_argument("--detect", default="off", choices=DETECT_OPTS, help="시작 시 인식 모드")
    ap.add_argument("--family", default="tag36h11", choices=det.FAMILIES, help="AprilTag 계열")
    ap.add_argument("--invert", action="store_true", help="검출 전 영상 반전")
    ap.add_argument("--detect-width", type=int, default=1280, choices=DETECT_WIDTH_VALUES,
                    help="검출용 축소 폭 (0 = 원본). DMK 에서 작은 도형은 0")
    args = ap.parse_args()

    ires = parse_res(args.ir)
    if args.depth and args.depth != "on":
        ires = parse_res(args.depth)
    cres = parse_res(args.color) if args.color and args.color != "on" else (0, 0, 0)
    target = Target(ires, cres, depth=args.depth is not None, color=args.color is not None)

    cell_w, cell_h = map(int, args.cell.lower().split("x"))
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, cell_w * 2, cell_h * 2)
    mouse_events: list[tuple[int, int, int, int]] = []  # (event, x, y, flags) — 메인 루프에서 처리
    cv2.setMouseCallback(WIN, lambda e, x, y, f, _p=None: mouse_events.append((e, x, y, f)))
    show_message(cell_w * 2, cell_h * 2, ["Connecting...", "(device open ~3s + stream start ~2s)"])

    quiet_sdk_log()
    ctx = Context()
    choices = find_devices(ctx, args.ip)
    if not choices:
        raise SystemExit("장치를 찾을 수 없습니다. USB/네트워크 연결을 확인하거나 (gige_discover.py) --ip 로 지정하세요.")
    labels = [c.label for c in choices]
    print("장치:", *labels, sep="\n  ")
    dev_i = 0
    if args.serial:
        dev_i = next((i for i, c in enumerate(choices) if c.entry is not None and c.entry.serial == args.serial), -1)
        if dev_i < 0:
            raise SystemExit(f"S/N {args.serial} 장치가 없습니다. 연결된 장치: {', '.join(labels)}")
    elif args.device:
        dev_i = next((i for i, l in enumerate(labels) if args.device.lower() in l.lower()), -1)
        if dev_i < 0:
            raise SystemExit(f"'{args.device}' 에 맞는 장치가 없습니다. 연결된 장치: {', '.join(labels)}")

    op = open_device(ctx, choices, dev_i, target, args.port)
    detp = det.Pipeline(track=True)  # 도형은 프레임 간 추적해 검출률 [xx%] 표시
    views = {"orbbec": OrbbecView(cell_w, cell_h, detp), "tis": TisView(cell_w, cell_h, detp)}
    switcher = Switcher()
    canvas = None
    menu: MenuState | None = None
    distance = args.distance
    flags = ViewFlags(detect=args.detect, family=args.family, invert=args.invert, detect_width=args.detect_width)
    fps_t, fps_n, fps = time.time(), 0, 0.0
    toast, toast_until = "", 0.0

    def apply_menu(m: MenuState) -> None:
        nonlocal flags, toast, toast_until
        if m.changed("dev"):
            new_i = m.get("dev")
            old_pipe, old_src = op.pipeline, op.source
            op.pipeline = op.source = None  # 닫히는 중 표시 (실패 시 이전 장치 다시 열기)

            def job():
                if old_pipe is not None:
                    old_pipe.stop()
                if old_src is not None:
                    old_src.close()
                return open_device(ctx, choices, new_i, target, args.port)

            switcher.start(job, f"Opening {labels[new_i]}")
            return
        if op.kind == "orbbec":
            new_t, flags = orbbec_menu_result(m, op)
            if new_t != op.st.target:
                switcher.start(lambda p=op.pipeline, t=new_t: start_streams(p, t, running=True),
                               f"Switching to {describe(new_t)}")
        else:
            src: TisDmkSource = op.source
            flags = detect_flags(m, flags)
            if m.has("grid"):
                flags = replace(flags, grid=bool(m.get("grid")))
            if m.changed("exp_auto") and m.get("exp_auto"):
                src.set_auto("Exposure", True)
            elif m.changed("exp") or (m.changed("exp_auto") and not m.get("exp_auto")):
                src.set_value("Exposure", EXPOSURE_MS[m.get("exp")] / 1000.0)
            if m.changed("gain_auto") and m.get("gain_auto"):
                src.set_auto("Gain", True)
            elif m.changed("gain") or (m.changed("gain_auto") and not m.get("gain_auto")):
                src.set_value("Gain", float(GAIN_DB[m.get("gain")]))
            if m.changed("mode"):
                mode = op.modes[m.get("mode")]

                def job(mode=mode):
                    src.stop()
                    src.start(mode)
                    op.intr = src.intrinsics(mode)
                    return "restarted"

                switcher.start(job, f"Switching to {mode}")

    try:
        while True:
            # --- 재시작 / 장치 전환 처리 (드라이버 때문에 수 초 걸려 백그라운드에서) ---
            if switcher.busy:
                show_message(cell_w * 2, cell_h * 2,
                             [f"{switcher.label}  ({time.time() - switcher.started:.1f}s)",
                              "camera driver needs a few seconds to restart streams", "q: quit"], base=canvas)
                k = cv2.waitKeyEx(30)
                if k != -1 and (k & 0xFF) in (ord("q"), KEY_ESC):
                    switcher.join()
                    break
                continue
            if switcher.result is not None:
                res, err = switcher.take_result()
                if err is not None:
                    print(f"전환 실패: {err}")
                    toast = f"failed: {str(err)[:70]}"
                    if op.pipeline is None and op.source is None:  # 장치 전환 중 실패 → 이전 장치 다시 열기
                        op = open_device(ctx, choices, op.dev, target, args.port)
                elif isinstance(res, Opened):
                    op = res
                    toast = f"{labels[op.dev]}  ({time.time() - switcher.started:.1f}s)"
                elif isinstance(res, Session):
                    op.st = res
                    toast = f"{describe(op.st.target)} @ {op.st.fps}fps  ({time.time() - switcher.started:.1f}s)"
                else:
                    toast = f"{op.source.mode}  ({time.time() - switcher.started:.1f}s)"
                toast_until = time.time() + 3
                views[op.kind].reset()
                fps_t, fps_n, fps = time.time(), 0, 0.0

            view = views[op.kind]
            if op.kind == "orbbec":
                frames = op.pipeline.wait_for_frames(1000)
                ready = frames is not None and view.ingest(op.st, frames, flags.align)
            else:
                frame = op.source.read(1500)
                ready = frame is not None and view.ingest(frame)
            if not ready:
                k = cv2.waitKeyEx(1)
                if k != -1 and (k & 0xFF) in (ord("q"), KEY_ESC) and menu is None:
                    break
                continue

            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
            if op.kind == "orbbec":
                if flags.auto:
                    distance = view.auto_distance(distance)
                panels = view.panels(op.st, distance, flags)
                footer = f"Z={distance:.0f}mm {'(auto)' if flags.auto else ''}  {fps:.1f}fps   "
            else:
                panels = view.panels(op, flags, fps)
                footer = f"{fps:.1f}fps   "
            canvas = viz.grid(panels, 2, cell_w, cell_h)
            viz.put_text(canvas, footer + "e menu   +/- [ ] distance   s save   q quit",
                         (10, canvas.shape[0] - 10), 0.5, viz.YELLOW)
            if time.time() < toast_until:
                viz.put_text(canvas, toast, (10, canvas.shape[0] - 40), 0.7, viz.YELLOW)
            menu_btn = draw_menu_button(canvas)
            menu_layout = draw_menu(canvas, menu) if menu is not None else None
            cv2.imshow(WIN, canvas)

            k = cv2.waitKeyEx(1)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            # --- 마우스: 하단 MENU 버튼, 메뉴 행 클릭(선택), < > 클릭·휠(값), Apply / Close ---
            while mouse_events:
                e, mx, my, mf = mouse_events.pop(0)
                if menu is None:
                    if e == cv2.EVENT_LBUTTONDOWN and MenuLayout.inside(menu_btn, mx, my):
                        menu = (build_orbbec_menu(labels, op, op.st.target, flags) if op.kind == "orbbec"
                                else build_tis_menu(labels, op, flags))
                    continue
                if menu_layout is None:
                    continue
                row = menu_layout.row_at(mx, my)
                if e == cv2.EVENT_LBUTTONDOWN:
                    if MenuLayout.inside(menu_layout.apply, mx, my):
                        apply_menu(menu)
                        menu = None
                    elif MenuLayout.inside(menu_layout.close, mx, my):
                        menu = None
                    elif row >= 0:
                        menu.sel = row
                        d = menu_layout.arrow(row, mx)  # '<' = -1, '>' 또는 값 글자 = +1, 라벨 = 0
                        if d:
                            menu.adjust(d)
                elif e == cv2.EVENT_MOUSEWHEEL and row >= 0:
                    menu.sel = row
                    delta = (mf >> 16) & 0xFFFF  # 휠 방향은 flags 상위 16비트 (signed)
                    menu.adjust(1 if delta < 0x8000 else -1)

            if k == -1:
                continue
            ch = k & 0xFF if k < 0x10000 else -1

            if menu is not None:  # --- 메뉴 열린 상태 ---
                if k in KEY_UP or ch == ord("w"):
                    menu.move(-1)
                elif k in KEY_DOWN or ch == ord("s"):
                    menu.move(1)
                elif k in KEY_LEFT or ch == ord("a"):
                    menu.adjust(-1)
                elif k in KEY_RIGHT or ch == ord("d"):
                    menu.adjust(1)
                elif ch in (KEY_ESC, ord("e")):
                    menu = None
                elif ch in KEY_ENTER:
                    apply_menu(menu)
                    menu = None
                continue

            # --- 일반 키 ---
            if ch in (ord("q"), KEY_ESC):
                break
            if ch == ord("e"):
                menu = (build_orbbec_menu(labels, op, op.st.target, flags) if op.kind == "orbbec"
                        else build_tis_menu(labels, op, flags))
            elif ch in (ord("+"), ord("=")):
                distance += 100
            elif ch == ord("-"):
                distance = max(100, distance - 100)
            elif ch == ord("]"):
                distance += 1000
            elif ch == ord("["):
                distance = max(100, distance - 1000)
            elif ch == ord("s"):
                meta = {"device": op.summary}
                meta.update(view.snapshot_meta(op.st, distance) if op.kind == "orbbec" else view.snapshot_meta(op, fps))
                stem = save_snapshot(args.out, canvas, view.last, meta)
                toast, toast_until = f"saved {stem.name}.*", time.time() + 3
    finally:
        switcher.join()
        if switcher.result is not None and isinstance(switcher.result[0], Opened):
            op = switcher.result[0]  # 장치 전환 중 종료 → 새로 열린 쪽을 닫는다
        op.close()
        cv2.destroyAllWindows()


def save_snapshot(out: Path, canvas: np.ndarray, last: dict, meta: dict) -> Path:
    """results/viewer_<시각>.* 로 저장하고 공통 경로(stem)를 돌려준다. 같은 초에 두 번 누르면 _2, _3 ... 을 붙인다."""
    out.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp, n = base, 1
    while (out / f"viewer_{stamp}.json").exists():
        n += 1
        stamp = f"{base}_{n}"
    cv2.imwrite(str(out / f"viewer_{stamp}.png"), canvas)
    cv2.imwrite(str(out / f"viewer_{stamp}_ir{last['ir'].dtype.itemsize * 8}.png"), last["ir"])
    if last["color"] is not None:
        cv2.imwrite(str(out / f"viewer_{stamp}_color.png"), last["color"])
    if last["depth"] is not None:
        np.save(out / f"viewer_{stamp}_depth_mm.npy", last["depth"])
    (out / f"viewer_{stamp}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"스냅샷 저장: {out / f'viewer_{stamp}'}.*")
    return out / f"viewer_{stamp}"


if __name__ == "__main__":
    main()
