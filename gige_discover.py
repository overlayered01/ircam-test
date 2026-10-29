"""GigE Vision 카메라(DMK 33GX264, Femto Mega) 네트워크 탐색. SDK 가 장치를 못 찾을 때 원인 확인용.

    python gige_discover.py                                    # 모든 어댑터에서 브로드캐스트 탐색
    python gige_discover.py --force-ip 192.168.1.20 --mac 00:07:48:A1:A9:80   # 카메라에 임시 IP (전원 끄면 원복)

카메라와 PC 어댑터가 같은 서브넷이어야 드라이버(GigECamDevServer / Orbbec SDK)가 연결할 수 있다.
탐색에는 보이는데 SDK 가 못 열면 서브넷 불일치다. 해결은 둘 중 하나:
  - PC 어댑터에 카메라와 같은 서브넷 주소 추가 (관리자 PowerShell):
      New-NetIPAddress -InterfaceAlias "<어댑터>" -IPAddress 169.254.1.1 -PrefixLength 16
  - 카메라를 PC 서브넷으로 옮김: --force-ip (임시) 또는 카메라 도구에서 persistent IP 설정
"""

from __future__ import annotations

import argparse
import ipaddress
import sys

from ir_detect.gige import discover, force_ip, local_ipv4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force-ip", metavar="IP", help="카메라에 줄 임시 IP")
    ap.add_argument("--mac", help="--force-ip 대상 MAC (탐색 결과가 하나면 생략 가능)")
    ap.add_argument("--mask", default="255.255.255.0")
    ap.add_argument("--gateway", default="0.0.0.0")
    a = ap.parse_args()

    ips = local_ipv4()
    print("PC 주소:", ", ".join(ips) or "(없음)")
    devs = discover(ips)
    if not devs:
        print("GigE Vision 장치 없음 (케이블/링크, 방화벽 UDP 3956 확인)")
        return 1
    for d in devs:
        same = any(ipaddress.ip_address(d.ip) in ipaddress.ip_network(f"{d.ip}/{d.mask}", strict=False)
                   and ipaddress.ip_address(p) in ipaddress.ip_network(f"{d.ip}/{d.mask}", strict=False) for p in ips)
        print(f"{d.model:20s} SN={d.serial:12s} ip={d.ip}/{d.mask} mac={d.mac} cfg={d.ip_config_str}"
              f"  {'같은 서브넷 OK' if same else '!! PC 와 서브넷 불일치 (SDK 가 못 엽니다)'}")

    if a.force_ip:
        targets = [d for d in devs if not a.mac or d.mac.upper() == a.mac.upper()]
        if len(targets) != 1:
            print("--mac 으로 대상을 하나만 지정하세요:", ", ".join(f"{d.model}={d.mac}" for d in devs), file=sys.stderr)
            return 2
        t = targets[0]
        ok = force_ip(t.mac, a.force_ip, a.mask, a.gateway, bind_ip=t.seen_from)
        print(f"FORCEIP {t.model} → {a.force_ip}/{a.mask}: {'ack' if ok else 'no ack'}")
        for d in discover([t.seen_from]):
            if d.mac == t.mac:
                print(f"  지금 주소: {d.ip}/{d.mask}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
