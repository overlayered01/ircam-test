"""데이터시트 공칭 FOV (비교 기준용). 모델/펌웨어별로 다를 수 있으니 필요하면 수정하세요.

키: (sensor, mode) → (H, V) degree. mode 는 device.mode_name() 결과.
SDK 를 import 하지 않으므로 장치/SDK 없이도 쓸 수 있다.
"""

# Femto Bolt / Femto Mega (Azure Kinect 호환) depth 모드 이름. 해상도로 모드가 결정된다.
FEMTO_DEPTH_MODES = {
    (320, 288): "NFOV 2x2 binned",
    (640, 576): "NFOV unbinned",
    (512, 512): "WFOV 2x2 binned",
    (1024, 1024): "WFOV unbinned",
}

NOMINAL_FOV = {
    # Femto Bolt / Femto Mega 데이터시트 공칭값
    ("depth", "NFOV unbinned"): (75.0, 65.0),
    ("depth", "NFOV 2x2 binned"): (75.0, 65.0),
    ("depth", "WFOV unbinned"): (120.0, 120.0),
    ("depth", "WFOV 2x2 binned"): (120.0, 120.0),
    ("ir", "NFOV unbinned"): (75.0, 65.0),
    ("ir", "NFOV 2x2 binned"): (75.0, 65.0),
    ("ir", "WFOV unbinned"): (120.0, 120.0),
    ("ir", "WFOV 2x2 binned"): (120.0, 120.0),
    ("color", "16:9"): (80.0, 51.0),
}


def nominal(sensor: str, mode: str) -> tuple[float, float] | None:
    return NOMINAL_FOV.get((sensor, mode))
