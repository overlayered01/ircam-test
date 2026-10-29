"""DMK(tisgrabber) 소스의 모드 파싱과 GigE Vision 탐색 패킷 파싱 (장치 불필요)."""

import socket
import struct

from ir_detect.gige import _parse_ack
from ir_detect.sources.tis_dmk import build_modes, parse_video_format

FORMATS = ["Y16 (528x4)", "Y16 (640x480)", "Y16 (2448x2048)", "Y800 (528x4)", "Y800 (1280x960)",
           "Y800 (2448x2048)", "RGB24 (640x480)", "RGB32 (2448x2048)"]


def test_parse_video_format():
    assert parse_video_format("Y800 (2448x2048)") == ("Y800", 2448, 2048)
    assert parse_video_format("Y16 (640x480)") == ("Y16", 640, 480)
    assert parse_video_format("RGB24 (640x480)") is None
    assert parse_video_format("garbage") is None


def test_build_modes_mono_only_sorted_y800_first():
    modes = build_modes(FORMATS)
    assert [(m.pixel_format, m.res) for m in modes] == [
        ("Y800", "2448x2048"), ("Y800", "1280x960"), ("Y16", "2448x2048"), ("Y16", "640x480"),
    ]  # 528x4 라인 포맷과 RGB 는 제외
    assert modes[0].param("video_format") == "Y800 (2448x2048)"
    assert modes[0].fps == 0  # 0 = 가능한 최대


def _ack(model=b"DMK 33GX264", ip="169.254.91.98", mask="255.255.0.0", cfg=0x6):
    p = bytearray(248)
    p[10:16] = bytes.fromhex("000748A1A980")
    p[20:24] = struct.pack(">I", cfg)
    p[36:40] = socket.inet_aton(ip)
    p[52:56] = socket.inet_aton(mask)
    manu = b"The Imaging Source Europe GmbH"
    p[72:72 + len(manu)] = manu
    p[104:104 + len(model)] = model
    p[216:224] = b"36814688"
    return b"\0" * 8 + bytes(p)


def test_parse_discovery_ack():
    d = _parse_ack(_ack(), "192.168.1.100")
    assert d.model == "DMK 33GX264" and d.serial == "36814688"
    assert d.mac == "00:07:48:A1:A9:80" and d.ip == "169.254.91.98" and d.mask == "255.255.0.0"
    assert d.ip_config_str == "DHCP+persistent" and d.seen_from == "192.168.1.100"
    assert _parse_ack(b"\0" * 20, "x") is None
