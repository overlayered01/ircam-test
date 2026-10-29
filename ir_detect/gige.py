"""GigE Vision 제어 채널(GVCP) 최소 구현: 카메라 탐색과 임시 IP 부여. SDK 없이 네트워크 상태를 확인할 때 쓴다.

- discover(): UDP 3956 브로드캐스트 DISCOVERY_CMD → 같은 L2 에 있는 GigE Vision 장치 전부 (서브넷이 달라도 응답).
  DMK 33GX264 뿐 아니라 Femto Mega 도 GigE Vision 을 말하므로 같이 잡힌다.
- force_ip(): FORCEIP_CMD 로 카메라 IP 를 바꾼다. 전원을 끄면 원래 설정(link-local/DHCP/persistent)으로 돌아간다.
  PC 어댑터에 카메라와 같은 서브넷 주소를 줄 관리자 권한이 없을 때 반대로 카메라를 PC 쪽으로 옮기는 용도.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass

GVCP_PORT = 3956
_MAGIC = 0x42


@dataclass
class GigEDevice:
    manufacturer: str
    model: str
    version: str
    serial: str
    user_name: str
    mac: str
    ip: str
    mask: str
    gateway: str
    ip_config: int  # bit0 LLA, bit1 DHCP, bit2 persistent (현재 설정)
    seen_from: str  # 응답이 들어온 PC 쪽 주소

    @property
    def ip_config_str(self) -> str:
        bits = [n for b, n in ((1, "LLA"), (2, "DHCP"), (4, "persistent")) if self.ip_config & b]
        return "+".join(bits) or "none"


def _txt(b: bytes) -> str:
    return b.split(b"\0")[0].decode(errors="replace")


def _parse_ack(data: bytes, seen_from: str) -> GigEDevice | None:
    if len(data) < 8 + 248:
        return None
    p = data[8:]
    ip = lambda a: socket.inet_ntoa(p[a : a + 4])  # noqa: E731
    return GigEDevice(
        manufacturer=_txt(p[72:104]),
        model=_txt(p[104:136]),
        version=_txt(p[136:168]),
        serial=_txt(p[216:232]),
        user_name=_txt(p[232:248]),
        mac=":".join(f"{x:02X}" for x in p[10:16]),
        ip=ip(36),
        mask=ip(52),
        gateway=ip(68),
        ip_config=struct.unpack(">I", p[20:24])[0],
        seen_from=seen_from,
    )


def local_ipv4() -> list[str]:
    """PC 의 IPv4 주소 전부 (루프백 제외). 브로드캐스트는 인터페이스마다 따로 보내야 한다."""
    out = []
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        a = info[4][0]
        if not a.startswith("127.") and a not in out:
            out.append(a)
    return out


def discover(bind_ips: list[str] | None = None, wait_s: float = 1.5) -> list[GigEDevice]:
    pkt = struct.pack(">BBHHH", _MAGIC, 0x11, 0x0002, 0, 1)  # flags: ack 요구 + 브로드캐스트 ack
    found: dict[str, GigEDevice] = {}
    for bind_ip in bind_ips or local_ipv4():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            s.bind((bind_ip, 0))
            s.sendto(pkt, ("255.255.255.255", GVCP_PORT))
        except OSError:
            s.close()
            continue
        s.settimeout(wait_s)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            try:
                data, _ = s.recvfrom(2048)
            except socket.timeout:
                break
            d = _parse_ack(data, bind_ip)
            if d is not None and d.mac not in found:
                found[d.mac] = d
        s.close()
    return list(found.values())


def force_ip(mac: str, new_ip: str, mask: str, gateway: str = "0.0.0.0", bind_ip: str = "") -> bool:
    """FORCEIP_CMD. mac 은 'AA:BB:..' 꼴. 응답(ack)이 오면 True. 카메라가 바로 재설정하며 ack 를 생략하기도 한다."""
    m = bytes(int(x, 16) for x in mac.split(":"))
    a = socket.inet_aton
    payload = b"\0\0" + m[:2] + m[2:6] + b"\0" * 12 + a(new_ip) + b"\0" * 12 + a(mask) + b"\0" * 12 + a(gateway)
    pkt = struct.pack(">BBHHH", _MAGIC, 0x01, 0x0004, len(payload), 2) + payload
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.bind((bind_ip, 0))
    s.settimeout(2.0)
    s.sendto(pkt, ("255.255.255.255", GVCP_PORT))
    try:
        data, _ = s.recvfrom(1024)
        status = struct.unpack(">H", data[:2])[0]
        return status == 0
    except socket.timeout:
        return False
    finally:
        s.close()
