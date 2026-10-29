"""The Imaging Source DMK 33GX264 (GigE, 2448x2048 모노) — IC Imaging Control 3.x `tisgrabber` 경로.

이 PC 에는 구형 "GigECam Driver and Tools 3.7" 이 설치되어 있어 (DirectShow 필터 + GigECamDevServer 서비스)
IC Imaging Control 4 (imagingcontrol4 pip) 는 GenTL 드라이버가 없어 장치를 못 찾는다. 대신 같은 드라이버를
쓰는 IC 3.x 의 C 래퍼 tisgrabber_x64.dll 을 third_party/tisgrabber/ 에 두고 ctypes 로 부른다
(출처: github.com/TheImagingSource/IC-Imaging-Control-Samples, Python/tisgrabber/samples).

- 비디오 포맷은 "Y800 (2448x2048)" 꼴 문자열. Y800 = 8비트, Y16 = 16비트 (12비트 ADC 를 상위 비트에).
- fps 는 카메라가 받아 주는 값으로 설정되지만 실제는 링크 대역폭이 결정한다: 2448x2048 Y800 은 프레임당 5MB
  → 100 Mbps 링크에서 약 2.5fps, 1 Gbps 에서 카메라 최대(약 20fps 급).
- 카메라는 기본 link-local(169.254.x.x) + DHCP 설정이라 PC 어댑터에 같은 서브넷 주소가 있어야 드라이버가
  찾는다. 안 보이면 `python gige_discover.py` 로 확인하고 `--force-ip` 로 임시 주소를 줄 수 있다.
- 내부 파라미터는 공장값이 없다. calib/<serial>.json (Intrinsics.as_dict 형식) 이 있으면 그것을 쓴다.
"""

from __future__ import annotations

import ctypes
import json
import re
import sys
from pathlib import Path

import numpy as np

from femto_fov.fov_math import Intrinsics
from ir_detect.preprocess import to_u8

from .base import CameraSource, Frame, Mode, SourceUnavailable

TIS_DIR = Path(__file__).resolve().parents[2] / "third_party" / "tisgrabber"
CALIB_DIR = Path(__file__).resolve().parents[2] / "calib"
IC_SUCCESS = 1
SINK_Y800, SINK_Y16 = 0, 4  # tisgrabber.SinkFormats
FPS_CANDIDATES = (60.0, 30.0, 20.0, 15.0, 10.0, 5.0)

_ic = None  # DLL 핸들 (프로세스당 하나)


def parse_video_format(text: str) -> tuple[str, int, int] | None:
    """'Y800 (2448x2048)' → ('Y800', 2448, 2048). 모노 포맷이 아니면 None."""
    m = re.fullmatch(r"(Y800|Y16)\s*\((\d+)x(\d+)\)", text.strip())
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def build_modes(formats: list[str], fps: float = 0.0) -> list[Mode]:
    """모노 포맷만, 해상도 큰 순. 극단적으로 작은 라인 포맷(528x4 등)은 뺀다. fps 0 = 가능한 최대."""
    out = []
    for f in formats:
        p = parse_video_format(f)
        if p is None or p[2] < 64:
            continue
        fmt, w, h = p
        out.append(Mode(w, h, fps, fmt, "", (("video_format", f),)))
    out.sort(key=lambda m: (m.pixel_format != "Y800", -(m.width * m.height)))
    return out


def _lib():
    global _ic
    if _ic is not None:
        return _ic
    if sys.platform != "win32":
        raise ImportError("tisgrabber 는 Windows 전용")
    dll = TIS_DIR / "tisgrabber_x64.dll"
    if not dll.exists():
        raise ImportError(f"{dll} 없음")
    sys.path.insert(0, str(TIS_DIR))
    import tisgrabber as tis  # noqa: WPS433

    ic = ctypes.cdll.LoadLibrary(str(dll))
    tis.declareFunctions(ic)
    ic.IC_GetVideoFormat.restype = ctypes.c_char_p
    ic.IC_GetFrameRate.restype = ctypes.c_float
    ic.IC_InitLibrary(0)
    ic._tis = tis
    _ic = ic
    return ic


class TisDmkSource(CameraSource):
    kind = "tis"

    def __init__(self, name: str = "DMK", unique_name: str = ""):
        super().__init__(name)
        self.unique_name = unique_name  # 'DMK 33GX264 36814688' (모델 + S/N). 비우면 첫 장치
        self._h = None
        self._formats: list[str] = []
        self._mode_fps = 0.0
        self._intr: Intrinsics | None = None
        self._buf = (ctypes.c_long(), ctypes.c_long(), ctypes.c_int(), ctypes.c_int())  # w, h, bits, colorformat

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            _lib()
        except (ImportError, OSError) as e:
            return False, f"tisgrabber 를 쓸 수 없습니다 ({e}). third_party/tisgrabber/ 와 GigECam 드라이버가 필요합니다."
        return True, ""

    @classmethod
    def discover(cls) -> list["TisDmkSource"]:
        """드라이버가 찾은 TIS 장치마다 소스 하나. 네트워크 설정이 안 맞으면 카메라가 있어도 비어 있다."""
        ok, _ = cls.available()
        if not ok:
            return []
        ic = _lib()
        names = [ic._tis.D(ic.IC_GetUniqueNamefromList(i)) for i in range(ic.IC_GetDeviceCount())]
        return [cls(f"{n.rsplit(' ', 1)[0]} (GigE)", n) for n in names]

    def open(self) -> None:
        ok, why = self.available()
        if not ok:
            raise SourceUnavailable(why)
        ic = _lib()
        T = ic._tis.T
        if not self.unique_name:
            if ic.IC_GetDeviceCount() == 0:
                raise SourceUnavailable(
                    "TIS 장치를 찾을 수 없습니다. 카메라와 PC 어댑터가 같은 서브넷인지 "
                    "`python gige_discover.py` 로 확인하세요 (카메라 기본은 169.254.x.x link-local)."
                )
            self.unique_name = ic._tis.D(ic.IC_GetUniqueNamefromList(0))
        self._h = ic.IC_CreateGrabber()
        if ic.IC_OpenDevByUniqueName(self._h, T(self.unique_name)) != IC_SUCCESS or not ic.IC_IsDevValid(self._h):
            ic.IC_ReleaseGrabber(self._h)
            self._h = None
            raise SourceUnavailable(f"TIS 장치 '{self.unique_name}' 를 열 수 없습니다 (다른 프로그램이 쓰는 중이거나 네트워크 불일치)")
        n = ic.IC_GetVideoFormatCount(self._h)
        self._formats = [ic._tis.D(ic.IC_GetVideoFormat(self._h, i)) for i in range(n)]
        self._intr = self._load_calib()

    def close(self) -> None:
        if self._h is None:
            return
        self.stop()
        _lib().IC_ReleaseGrabber(self._h)
        self._h = None

    def modes(self) -> list[Mode]:
        return build_modes(self._formats)

    def start(self, mode: Mode) -> None:
        ic = _lib()
        T = ic._tis.T
        if self.mode is not None:
            self.stop()
        fmt = mode.param("video_format") or f"{mode.pixel_format} ({mode.width}x{mode.height})"
        if ic.IC_SetVideoFormat(self._h, T(fmt)) != IC_SUCCESS:
            raise SourceUnavailable(f"비디오 포맷 설정 실패: {fmt}")
        ic.IC_SetFormat(self._h, SINK_Y16 if mode.pixel_format == "Y16" else SINK_Y800)
        for fps in ((mode.fps,) if mode.fps else FPS_CANDIDATES):
            if ic.IC_SetFrameRate(self._h, ctypes.c_float(fps)) == IC_SUCCESS:
                break
        self._mode_fps = float(ic.IC_GetFrameRate(self._h))
        if ic.IC_StartLive(self._h, 0) != IC_SUCCESS:
            raise SourceUnavailable(f"스트림 시작 실패: {fmt}")
        self.mode = mode

    def stop(self) -> None:
        if self.mode is None:
            return
        _lib().IC_StopLive(self._h)
        self.mode = None

    def read(self, timeout_ms: int = 1000) -> Frame | None:
        ic = _lib()
        if ic.IC_SnapImage(self._h, int(timeout_ms)) != IC_SUCCESS:
            return None
        w, h, bits, _ = self._buf
        ic.IC_GetImageDescription(self._h, w, h, bits, self._buf[3])
        ptr = ic.IC_GetImagePtr(self._h)
        if not ptr:
            return None
        nbytes = w.value * h.value * (bits.value // 8)
        flat = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte)), (nbytes,))
        if bits.value == 16:
            raw = flat.view(np.uint16).reshape(h.value, w.value).copy()
        else:
            raw = flat.reshape(h.value, w.value).copy()
        # tisgrabber 의 IC_GetImagePtr 버퍼는 이미 위→아래 순서다 (DirectShow 의 bottom-up 이 아님).
        # 실측: 뒤집으면 거울상이 되어 AprilTag 디코딩이 전부 실패한다 (사각형은 찾지만 hamming 2).
        return Frame(raw, to_u8(raw), self.mode, self._intr, 0.0, {"fps_set": self._mode_fps})

    # --- 노출 / 게인 (스트리밍 중에도 바뀐다) ---
    def get_auto(self, prop: str) -> bool | None:
        """prop = 'Exposure' | 'Gain'. 지원하지 않으면 None."""
        ic = _lib()
        v = ctypes.c_int()
        return bool(v.value) if ic.IC_GetPropertySwitch(self._h, ic._tis.T(prop), ic._tis.T("Auto"), v) == IC_SUCCESS else None

    def set_auto(self, prop: str, on: bool) -> bool:
        ic = _lib()
        return ic.IC_SetPropertySwitch(self._h, ic._tis.T(prop), ic._tis.T("Auto"), int(on)) == IC_SUCCESS

    def get_value(self, prop: str) -> float | None:
        """절대값: Exposure 는 초, Gain 은 dB."""
        ic = _lib()
        v = ctypes.c_float()
        ok = ic.IC_GetPropertyAbsoluteValue(self._h, ic._tis.T(prop), ic._tis.T("Value"), v) == IC_SUCCESS
        return float(v.value) if ok else None

    def value_range(self, prop: str) -> tuple[float, float] | None:
        ic = _lib()
        lo, hi = ctypes.c_float(), ctypes.c_float()
        ok = ic.IC_GetPropertyAbsoluteValueRange(self._h, ic._tis.T(prop), ic._tis.T("Value"), lo, hi) == IC_SUCCESS
        return (float(lo.value), float(hi.value)) if ok else None

    def set_value(self, prop: str, value: float) -> bool:
        """자동이 켜져 있으면 끄고 값을 넣는다."""
        ic = _lib()
        self.set_auto(prop, False)
        return ic.IC_SetPropertyAbsoluteValue(self._h, ic._tis.T(prop), ic._tis.T("Value"), ctypes.c_float(value)) == IC_SUCCESS

    def info(self) -> dict[str, str]:
        out = {"name": self.name, "model": "DMK 33GX264", "connection": "GigE"}
        if self.unique_name:
            parts = self.unique_name.rsplit(" ", 1)
            out["model"] = parts[0]
            if len(parts) == 2:
                out["serial"] = parts[1]
        return out

    def intrinsics(self, mode: Mode) -> Intrinsics | None:
        if self._intr is None:
            return None
        s = mode.width / self._intr.width
        return self._intr.scaled(s) if abs(s - 1) > 1e-6 else self._intr

    def _load_calib(self) -> Intrinsics | None:
        serial = self.info().get("serial", "")
        for name in (f"dmk_{serial}.json", "dmk.json"):
            p = CALIB_DIR / name
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                return Intrinsics(d["width"], d["height"], d["fx"], d["fy"], d["cx"], d["cy"],
                                  tuple(d.get("dist", (0.0,) * 8)), d.get("model", "BROWN_CONRADY_K6"))
        return None
