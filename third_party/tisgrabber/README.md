# tisgrabber (IC Imaging Control 3.x C 래퍼)

The Imaging Source 카메라(DMK 33GX264)를 구형 GigECam 드라이버(DirectShow)로 여는 데 쓴다.
`ir_detect/sources/tis_dmk.py` 가 ctypes 로 `tisgrabber_x64.dll` 을 불러온다.

| 파일 | 출처 |
|---|---|
| `tisgrabber.py` | https://github.com/TheImagingSource/IC-Imaging-Control-Samples/tree/master/Python/tisgrabber/samples |
| `tisgrabber_x64.dll` | 같은 폴더 (IC Imaging Control 3.5 런타임 래퍼) |
| `TIS_UDSHL11_x64.dll` | 같은 폴더 (IC Imaging Control 3.5 코어) |

2026-10-29 master 에서 받음. 세 파일은 The Imaging Source 가 샘플과 함께 재배포용으로 제공하는 런타임이다.

필요 조건: Windows 64비트, "GigECam Driver and Tools" (GigECamDevServer 서비스) 설치.
IC Imaging Control 4 (`imagingcontrol4` pip) 로 바꾸려면 IC4 용 GigE Vision 드라이버(GenTL producer)를 설치해야 한다.
