"""pyorbbecsdk 래퍼: 장치 열기, 스트림 프로파일 열거, 프레임 → numpy 변환."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np

from .fov_math import Intrinsics
from .specs import FEMTO_DEPTH_MODES

# cp949 콘솔/파이프에서 특수문자(°, × 등) 출력 시 죽지 않도록
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

try:
    from pyorbbecsdk import (
        AlignFilter,
        Config,
        Context,
        Device,
        OBError,
        OBFormat,
        OBLogLevel,
        OBSensorType,
        OBStreamType,
        Pipeline,
        VideoStreamProfile,
    )
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "pyorbbecsdk 를 불러올 수 없습니다. Python 3.12 venv 에서 `pip install -r requirements.txt` 를 실행하세요.\n"
        f"원인: {e}"
    )

SENSORS = {
    "color": OBSensorType.COLOR_SENSOR,
    "depth": OBSensorType.DEPTH_SENSOR,
    "ir": OBSensorType.IR_SENSOR,
}

# Femto Mega 네트워크 장치 기본 포트
DEFAULT_NET_PORT = 8090

OPEN_HINT = (
    "장치를 열 수 없습니다 (Windows: 액세스 거부 0x80070005 등).\n"
    "  - OrbbecViewer / k4aviewer / 다른 스크립트가 카메라를 쓰고 있지 않은지 확인\n"
    "  - 설정 > 개인정보 > 카메라 > '데스크톱 앱이 카메라에 액세스하도록 허용' 켜기\n"
    "  - USB 를 다시 꽂고 재시도"
)


def mode_name(sensor: str, width: int, height: int) -> str:
    if sensor in ("depth", "ir"):
        name = FEMTO_DEPTH_MODES.get((width, height), "")
        if sensor == "ir" and (width, height) == (1024, 1024) and not name:
            return "Passive IR"
        return name
    g = np.gcd(width, height)
    return f"{width // g}:{height // g}"


def to_intrinsics(profile: VideoStreamProfile) -> Intrinsics:
    i = profile.get_intrinsic()
    d = profile.get_distortion()
    dist = (d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6)
    return Intrinsics(i.width, i.height, i.fx, i.fy, i.cx, i.cy, dist, d.model.name)


def extrinsic_Rt(src: VideoStreamProfile, dst: VideoStreamProfile) -> tuple[np.ndarray, np.ndarray]:
    """src → dst 외부 파라미터 (R 3x3, t mm)."""
    e = src.get_extrinsic_to(dst)
    R = np.asarray(e.rot, dtype=np.float64).reshape(3, 3)
    t = np.asarray(e.transform, dtype=np.float64).reshape(3)
    return R, t


def quiet_sdk_log() -> None:
    try:
        Context.set_logger_to_console(OBLogLevel.ERROR)
    except Exception:
        pass


def require_device() -> Context:
    quiet_sdk_log()
    ctx = Context()
    devs = ctx.query_devices()
    if devs.get_count() == 0:
        print("장치를 찾을 수 없습니다. Femto 카메라를 USB 3 포트에 연결했는지 확인하세요.")
        sys.exit(1)
    return ctx


def open_pipeline() -> Pipeline:
    """장치 확인 후 Pipeline 생성. 장치 점유 시 SDK 가 한글 오류 메시지를 디코딩하지 못해
    UnicodeDecodeError 로 터지는 경우가 있어 원인 안내로 바꿔 준다."""
    require_device()
    try:
        return Pipeline()
    except (OBError, UnicodeDecodeError, RuntimeError) as e:
        print(f"{OPEN_HINT}\n  (원본 오류: {type(e).__name__}: {e})")
        sys.exit(1)


@dataclass
class DeviceEntry:
    name: str
    serial: str
    connection: str  # "USB3.1", "Ethernet" 등 SDK 문자열 그대로
    ip: str = ""


def enumerate_devices(ctx: Context) -> list[DeviceEntry]:
    """연결된 Orbbec 장치 목록 (USB + 자동 탐색된 네트워크 장치)."""
    dl = ctx.query_devices()
    out = []
    for i in range(dl.get_count()):
        conn = dl.get_device_connection_type_by_index(i)
        ip = ""
        if not conn.upper().startswith("USB"):
            try:
                ip = dl.get_device_ip_address_by_index(i)
            except OBError:
                pass
        out.append(DeviceEntry(dl.get_device_name_by_index(i), dl.get_device_serial_number_by_index(i), conn, ip))
    return out


def get_device(ctx: Context, serial: str = "", ip: str = "", port: int = DEFAULT_NET_PORT) -> Device:
    """ip 가 있으면 네트워크 장치(Femto Mega 이더넷), 아니면 serial 로, 둘 다 없으면 첫 번째 장치.
    ctx 가 사라지면 장치도 무효가 되므로 호출자가 ctx 를 들고 있어야 한다."""
    if ip:
        return ctx.create_net_device(ip, port)
    dl = ctx.query_devices()
    if dl.get_count() == 0:
        raise LookupError("장치를 찾을 수 없습니다. 카메라를 USB 3 포트에 연결했는지 확인하세요.")
    if not serial:
        return dl.get_device_by_index(0)
    serials = [dl.get_device_serial_number_by_index(i) for i in range(dl.get_count())]
    if serial not in serials:
        raise LookupError(f"S/N {serial} 장치가 없습니다. 연결된 장치: {', '.join(serials)}")
    return dl.get_device_by_serial_number(serial)


def device_summary(pipeline: Pipeline) -> str:
    info = pipeline.get_device().get_device_info()
    return f"{info.get_name()}  SN={info.get_serial_number()}  PID=0x{info.get_pid():04X}  FW={info.get_firmware_version()}"


@dataclass
class ProfileInfo:
    sensor: str
    width: int
    height: int
    fps: int
    fmt: str
    profile: VideoStreamProfile

    @property
    def key(self) -> tuple[int, int]:
        return self.width, self.height


def list_profiles(pipeline: Pipeline, sensor: str) -> list[ProfileInfo]:
    try:
        plist = pipeline.get_stream_profile_list(SENSORS[sensor])
    except OBError:
        return []
    out = []
    for i in range(plist.get_count()):
        p = plist.get_stream_profile_by_index(i)
        if not p.is_video_stream_profile():
            continue
        vp = p.as_video_stream_profile()
        out.append(ProfileInfo(sensor, vp.get_width(), vp.get_height(), vp.get_fps(), vp.get_format().name, vp))
    return out


def unique_resolutions(profiles: list[ProfileInfo]) -> list[ProfileInfo]:
    """해상도별 대표 프로파일 하나씩 (FOV 는 해상도/모드에만 의존하고 fps·포맷과 무관)."""
    best: dict[tuple[int, int], ProfileInfo] = {}
    for p in profiles:
        cur = best.get(p.key)
        if cur is None or p.fps > cur.fps:
            best[p.key] = p
    return sorted(best.values(), key=lambda p: (p.width * p.height, p.width))


def find_profile(pipeline: Pipeline, sensor: str, width: int = 0, height: int = 0, fps: int = 0) -> VideoStreamProfile:
    plist = pipeline.get_stream_profile_list(SENSORS[sensor])
    if width == 0 and height == 0 and fps == 0:
        return plist.get_default_video_stream_profile()
    fmt = OBFormat.UNKNOWN_FORMAT
    if sensor == "color":
        # 디코딩 비용이 적은 포맷 우선
        for f in (OBFormat.BGRA, OBFormat.RGB, OBFormat.YUYV, OBFormat.NV12, OBFormat.MJPG):
            try:
                return plist.get_video_stream_profile(width, height, f, fps)
            except OBError:
                continue
    return plist.get_video_stream_profile(width, height, fmt, fps)


def parse_res(text: str | None) -> tuple[int, int, int]:
    """'640x576' 또는 '640x576@30' → (w, h, fps). None → (0,0,0)=기본값."""
    if not text:
        return 0, 0, 0
    fps = 0
    if "@" in text:
        text, f = text.split("@", 1)
        fps = int(f)
    w, h = text.lower().split("x")
    return int(w), int(h), fps


# ---------------------------------------------------------------------------
# 프레임 변환
# ---------------------------------------------------------------------------


def color_to_bgr(frame) -> np.ndarray | None:
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == OBFormat.BGRA:
        return cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
    if fmt == OBFormat.RGBA:
        return cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
    if fmt in (OBFormat.YUYV, OBFormat.YUY2):
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    if fmt == OBFormat.NV12:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.NV21:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV21)
    if fmt == OBFormat.I420:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    print(f"지원하지 않는 color 포맷: {fmt}")
    return None


def depth_to_mm(frame) -> np.ndarray:
    w, h = frame.get_width(), frame.get_height()
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(h, w)
    scale = frame.get_depth_scale()
    if scale == 1.0:  # Femto 는 1 unit = 1mm → uint16 그대로 (float 변환 생략)
        return raw.copy()
    return raw.astype(np.float32) * scale


def ir_to_u16(frame) -> np.ndarray:
    w, h = frame.get_width(), frame.get_height()
    data = np.asanyarray(frame.get_data())
    if frame.get_format() in (OBFormat.Y8, OBFormat.GRAY):
        return data.reshape(h, w).astype(np.uint16)
    return np.frombuffer(data, dtype=np.uint16).reshape(h, w)


__all__ = [
    "AlignFilter",
    "Config",
    "DEFAULT_NET_PORT",
    "DeviceEntry",
    "OBError",
    "OPEN_HINT",
    "OBStreamType",
    "Pipeline",
    "SENSORS",
    "ProfileInfo",
    "color_to_bgr",
    "depth_to_mm",
    "device_summary",
    "enumerate_devices",
    "get_device",
    "extrinsic_Rt",
    "find_profile",
    "ir_to_u16",
    "list_profiles",
    "mode_name",
    "open_pipeline",
    "parse_res",
    "require_device",
    "to_intrinsics",
    "unique_resolutions",
]
