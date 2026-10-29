"""Femto Bolt / Femto Mega IR 스트림 (USB, Femto Mega 는 이더넷도).

IR 만 단독으로 켠다 (depth 불필요). 패시브 IR 은 OB_PROP_SWITCH_IR_MODE_INT=1 로 자체 조명을 끈 모드.
Femto Bolt FW 1.0.9 실측: 1024x1024 에서만 패시브 IR 영상이 나오고, 다른 해상도에서 1 로 두면
IR 대신 depth 같은 값이 나온다 → 패시브는 1024x1024 모드로만 내놓는다 (Azure Kinect PASSIVE_IR 과 동일).

같은 해상도로 액티브 ↔ 패시브를 바꾸면 SDK 가 처리 엔진을 다시 만들지 않아 이전 모드가 남는다
(액티브→패시브: 프레임 없음 + RawPhaseStreamer 오류, 패시브→액티브: 전부 0). 다른 해상도 스트림을
잠깐 start/stop 하면 풀린다 (~1.3초, 프레임 대기 불필요). 프로세스 안에서만 남는 상태라 종료 시엔 불필요.
"""

from __future__ import annotations

from typing import Iterable

from femto_fov.fov_math import Intrinsics
from femto_fov.specs import FEMTO_DEPTH_MODES

from ..preprocess import to_u8
from .base import CameraSource, Frame, Mode, SourceUnavailable

IR_MODE_ACTIVE, IR_MODE_PASSIVE = 0, 1
PASSIVE_RES = {(1024, 1024)}


def build_modes(profiles: Iterable[tuple[int, int, int, str]], passive_supported: bool) -> list[Mode]:
    """(w, h, fps, fmt) 프로파일 목록 → Mode 목록. 액티브(해상도↑, fps↓ 순) 뒤에 패시브."""
    keys = sorted(set(profiles), key=lambda p: (p[0] * p[1], p[0], -p[2], p[3]))
    active = [Mode(w, h, fps, fmt, FEMTO_DEPTH_MODES.get((w, h), "")) for w, h, fps, fmt in keys]
    passive = []
    if passive_supported:
        passive = [
            Mode(w, h, fps, fmt, "Passive IR", (("ir_mode", IR_MODE_PASSIVE),))
            for w, h, fps, fmt in keys
            if (w, h) in PASSIVE_RES
        ]
    return active + passive


class OrbbecIRSource(CameraSource):
    kind = "orbbec"

    def __init__(self, name: str = "Femto", serial: str = "", ip: str = "", port: int = 0):
        super().__init__(name)
        self.serial, self.ip, self.port = serial, ip, port
        self._ctx = self._dev = self._pipe = None
        self._profiles: dict = {}
        self._passive_supported = False
        self._intr: Intrinsics | None = None
        self._last_started: Mode | None = None

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            import pyorbbecsdk  # noqa: F401
        except ImportError as e:
            return False, f"pyorbbecsdk 미설치 ({e})"
        return True, ""

    @classmethod
    def discover(cls) -> list["OrbbecIRSource"]:
        """연결된 장치마다 소스 하나 (S/N 으로 지정). SDK 가 없으면 빈 목록."""
        if not cls.available()[0]:
            return []
        from femto_fov import device as fd

        fd.quiet_sdk_log()
        out = []
        for e in fd.enumerate_devices(fd.Context()):
            name = e.name.replace("Orbbec ", "")
            out.append(cls(f"{name} ({e.connection})", serial=e.serial))
        return out

    def open(self) -> None:
        ok, why = self.available()
        if not ok:
            raise SourceUnavailable(why)
        import pyorbbecsdk as ob

        from femto_fov import device as fd

        fd.quiet_sdk_log()
        self._ctx = fd.Context()
        try:
            self._dev = fd.get_device(self._ctx, self.serial, self.ip, self.port or fd.DEFAULT_NET_PORT)
        except LookupError as e:
            raise SourceUnavailable(str(e)) from e
        except fd.OBError as e:
            where = f"{self.ip}:{self.port or fd.DEFAULT_NET_PORT}" if self.ip else self.serial
            raise SourceUnavailable(f"장치 {where} 를 찾을 수 없습니다 ({e})") from e
        try:
            self._pipe = fd.Pipeline(self._dev)
        except (fd.OBError, UnicodeDecodeError, RuntimeError) as e:
            raise SourceUnavailable(f"{fd.OPEN_HINT}\n  (원본 오류: {type(e).__name__}: {e})") from e

        self._ir_mode_prop = ob.OBPropertyID.OB_PROP_SWITCH_IR_MODE_INT
        self._passive_supported = self._dev.is_property_supported(
            self._ir_mode_prop, ob.OBPermissionType.PERMISSION_READ_WRITE
        )
        self._profiles = {(p.width, p.height, p.fps, p.fmt): p.profile for p in fd.list_profiles(self._pipe, "ir")}

    def close(self) -> None:
        if self._pipe is None:
            return
        self.stop()
        # 패시브로 두고 나가면 다른 도구(fov_viewer 등)의 IR 이 이상하게 나오므로 액티브로 되돌린다
        self._set_ir_mode(IR_MODE_ACTIVE)
        self._ctx = self._dev = self._pipe = None
        self._profiles = {}
        self._last_started = None

    def modes(self) -> list[Mode]:
        return build_modes(self._profiles, self._passive_supported)

    def _profile(self, mode: Mode):
        return self._profiles[(mode.width, mode.height, int(mode.fps), mode.pixel_format)]

    def _set_ir_mode(self, value: int) -> None:
        if self._passive_supported:
            self._dev.set_int_property(self._ir_mode_prop, value)

    def start(self, mode: Mode) -> None:
        from femto_fov import device as fd

        if self.mode is not None:
            self.stop()
        profile = self._profile(mode)
        ir_mode = mode.param("ir_mode", IR_MODE_ACTIVE)
        self._set_ir_mode(ir_mode)
        last = self._last_started
        if last and (last.width, last.height) == (mode.width, mode.height) and last.param("ir_mode", IR_MODE_ACTIVE) != ir_mode:
            self._flush_ir_mode(mode)
        cfg = fd.Config()
        cfg.enable_stream(profile)
        self._pipe.start(cfg)
        self._intr = fd.to_intrinsics(profile)
        self.mode = self._last_started = mode

    def _flush_ir_mode(self, mode: Mode) -> None:
        """같은 해상도 IR 모드 전환 시 남는 이전 모드를 다른 해상도 스트림을 잠깐 열어 비운다 (모듈 docstring)."""
        from femto_fov import device as fd

        other = min((k for k in self._profiles if k[:2] != (mode.width, mode.height)), key=lambda k: k[0] * k[1], default=None)
        if other is None:
            return
        cfg = fd.Config()
        cfg.enable_stream(self._profiles[other])
        self._pipe.start(cfg)
        self._pipe.stop()

    def stop(self) -> None:
        if self.mode is None:
            return
        self._pipe.stop()
        self.mode = None

    def read(self, timeout_ms: int = 1000) -> Frame | None:
        from femto_fov import device as fd

        fs = self._pipe.wait_for_frames(timeout_ms)
        f = fs.get_ir_frame() if fs is not None else None
        if f is None:
            return None
        raw = fd.ir_to_u16(f)
        return Frame(raw, to_u8(raw), self.mode, self._intr, f.get_timestamp_us() / 1e6)

    def info(self) -> dict[str, str]:
        if self._dev is None:
            return {"name": self.name}
        i = self._dev.get_device_info()
        out = {
            "name": self.name,
            "model": i.get_name(),
            "serial": i.get_serial_number(),
            "firmware": i.get_firmware_version(),
            "connection": i.get_connection_type(),
        }
        if self.ip:
            out["ip"] = self.ip
        return out

    def intrinsics(self, mode: Mode) -> Intrinsics | None:
        from femto_fov import device as fd

        return fd.to_intrinsics(self._profile(mode))
