# femto-test — Femto RGB / IR / Depth 화각(FOV) 테스트

Orbbec Femto Bolt / Femto Mega 의 color, IR, depth 스트림 화각을 세 가지 방법으로 확인한다.

| 스크립트 | 무엇을 | 장치 필요 |
|---|---|---|
| `fov_table.py` | **(a) 이론 FOV** — 모든 해상도/모드의 공장 캘리브레이션(내부 파라미터 + 렌즈 왜곡)으로 FOV 계산 | O (스트리밍 X) |
| `fov_viewer.py` | **(b) 라이브 비교** — 세 스트림을 동시에 띄우고 서로의 화각 경계, 실제 depth 유효 영역, D2C 정렬을 오버레이 | O |
| `fov_measure.py` | **(c) 실측** — 체커보드 / depth 유효 영역 / 줄자로 공장값과 독립적으로 측정 | O (`manual` 계산만은 X) |

## 설치

```powershell
.\setup.ps1          # Python 3.12 .venv 생성 + pyorbbecsdk2, opencv, numpy 설치
```

- `pyorbbecsdk2` 는 Python 3.13 이하 휠만 있다 (이 PC 의 기본 3.14 에서는 설치 불가) → `py -3.12` 사용.
- 이후 모든 명령은 `.\.venv\Scripts\python <script>` 로 실행.
- 카메라는 USB 3 포트에 연결. OrbbecViewer 등 다른 프로그램이 카메라를 잡고 있으면 열리지 않는다.

## (a) 이론 FOV 표

```powershell
.\.venv\Scripts\python fov_table.py --out results
```

이 장치(Femto Bolt, FW 1.0.9) 실행 결과:

| 스트림 | 해상도 (모드) | pinhole H×V | **undistorted H×V** | 데이터시트 |
|---|---|---|---|---|
| color | 1280x720 ~ 3840x2160 (16:9) | 81.0 × 51.3 | **80.1 × 50.8** | 80 × 51 |
| color | 1280x960 (4:3) | 65.3 × 51.3 | **64.5 × 50.8** | - |
| depth / IR | 640x576, 320x288 (NFOV) | 64.8 × 59.2 | **73.0 × 65.6** | 75 × 65 |
| depth / IR | 1024x1024, 512x512 (WFOV) | 90.9 × 90.8 | **120.1 × 120.0** | 120 × 120 |

- **pinhole**: `2·atan(W / 2fx)` — 렌즈 왜곡을 무시한 값. 많이 쓰이지만 depth 처럼 왜곡이 큰 렌즈에서는 크게 과소평가된다.
- **undistorted**: 이미지 끝 픽셀을 왜곡 보정해 실제 광선 사이 각도를 구한 값. 데이터시트와 일치하는 쪽이 이 값이다.
- 같은 모드 안에서 binned/unbinned, color 16:9 해상도끼리는 FOV 가 같다 (fx 만 해상도에 비례).
- depth 와 IR 은 같은 센서라 내부 파라미터가 동일하다.

## (b) 라이브 뷰어

IR 을 기본으로 띄우고, 장치·모드·depth/color 켜기 끄기는 `e` 메뉴에서 바꾼다 (IR 카메라 인식 작업에서 IR 만 보다가 필요할 때 나머지를 켜는 용도).
장치는 Femto Bolt / Femto Mega (Orbbec) 와 DMK 33GX264 (The Imaging Source GigE) 를 지원하며 메뉴에서 서로 바꿀 수 있다.

```powershell
.\.venv\Scripts\python fov_viewer.py                                   # 첫 번째 장치, IR 만
.\.venv\Scripts\python fov_viewer.py --device dmk                      # 이름에 dmk 가 들어간 장치로 시작 (mega / bolt 도 됨)
.\.venv\Scripts\python fov_viewer.py --serial CL8764100GL              # Orbbec 장치 S/N 지정
.\.venv\Scripts\python fov_viewer.py --ip 192.168.1.10                 # Femto Mega 이더넷 (자동 탐색 안 될 때)
.\.venv\Scripts\python fov_viewer.py --ir 1024x1024@15 --depth --color # 모드·스트림 지정해서 시작
```

DMK 는 IR 패널 하나만 쓰고(depth/color 없음) 메뉴에 모드·노출·게인 항목이 대신 나온다. 공장 캘리브레이션이 없어 화각 오버레이는 안 그리며,
`calib/dmk_<serial>.json` (Intrinsics 필드: width, height, fx, fy, cx, cy, dist, model) 을 두면 각도 격자만 그린다.

| 패널 | 오버레이 |
|---|---|
| IR (기준) | **magenta** = color 화각 (color 켜졌을 때) |
| Color | **cyan** = IR/depth 센서 사각형 화각, **green** = depth 값이 실제로 나오는 영역 (depth 켜졌을 때). 각각 color 화면을 몇 % 덮는지 표시 |
| Depth | **magenta** = color 화각, green = 유효 영역 외곽. `valid` = 실제 유효 픽셀의 각도 범위 |
| D2C | depth 를 color 에 정렬해 블렌드 (둘 다 켜졌을 때). color 픽셀 중 depth 가 있는 비율 |

| 키 | 동작 |
|---|---|
| `e` 또는 하단 `MENU` 버튼 | 설정 메뉴. Orbbec: 장치 / IR 모드 / fps / depth on-off / color on-off / color 해상도 / 각도 격자 / D2C 정렬 / 자동 거리. DMK: 장치 / 모드 / 노출 auto·값 / 게인 auto·값. 공통: Detect / Tag family / Invert image / 도형 조건. 키보드 `↑↓` 항목, `←→` 값, `Enter` 적용, `Esc` 닫기. 마우스: 행 클릭 선택, `<` `>` 클릭·휠로 값, `Apply` / `Close` 버튼 |
| `+/-`, `[ ]` | 경계선 투영 거리 ±100mm / ±1000mm |
| `s` | 스냅샷 (PNG + 켜진 스트림의 depth npy / color PNG + JSON) 저장 |
| `q` / `ESC` | 종료 |

전체 키 목록: [docs/SHORTCUTS.md](docs/SHORTCUTS.md)

- Femto 의 IR 과 depth 는 같은 센서라 항상 같은 모드로 켜진다. fps 는 켜진 스트림이 공통으로 지원하는 최댓값 (Bolt WFOV unbinned → 15fps, Mega 는 30fps). 메뉴에서 고정할 수 있다.
- 메뉴에서 바꾼 것은 `Enter` 에 한꺼번에 적용되어 재시작은 한 번만 한다. 재시작은 Bolt USB **약 4초**, Mega 이더넷 약 2초 (카메라 드라이버의 stop 이 color ~2초, depth ~0.8초 — 실행 중 재설정은 드라이버가 거부). 백그라운드에서 진행되어 창은 계속 반응한다.
- 장치를 바꾸면 이전 장치를 닫고 새로 연다. 시작 시 찾은 장치(Orbbec USB + 자동 탐색된 이더넷 + `--ip`, TIS 드라이버가 찾은 DMK)만 목록에 나온다.
- 프로그램 시작에도 약 6초 걸린다 (장치 열기 ~3.4초 + 스트림 시작/첫 프레임 ~2.4초). 그동안 창에 연결 중 표시가 나온다.
- 딜레이 원인과 측정값 상세: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- 지원하지 않는 조합이면 이전 구성이 유지된다.
- Femto Mega 이더넷: IR/depth 는 Y16 비압축이라 640x576@30 하나에 약 177 Mbps 가 필요하다. 100 Mbps 링크에서는 IR 만 ~13fps, IR+depth ~7fps 로 떨어진다 (color 는 H.264/MJPG 압축이라 영향 없음). 1 Gbps 포트를 쓸 것.

- 카메라 간 기준선(depth→color 약 32mm) 때문에 겹침 영역은 거리마다 달라진다. 거리를 바꿔 가며 확인할 것.
- NFOV depth 의 실제 유효 영역은 육각형이라 센서 사각형(cyan)보다 좁다. 1080p color 기준 1m 에서 센서 사각형은 color 의 약 99.7% 를 덮지만, 실제 유효 영역(green)은 그보다 작다.

## (c) 실측

### 체커보드 — 초점거리 직접 추정 + 캘리브레이션

```powershell
# color 스트림. 보드 거리는 D2C 정렬된 depth 로 자동 측정
.\.venv\Scripts\python fov_measure.py checker --stream color --color 1920x1080 --board 9x6 --square 25
# IR 스트림 = depth FOV (같은 센서). IR 영상에서 보드가 잘 보인다
.\.venv\Scripts\python fov_measure.py checker --stream ir --depth 640x576 --board 9x6 --square 25
```

- `--board` 는 **내부 코너 수** (칸 수 - 1), `--square` 는 자로 잰 한 칸 크기(mm).
- 원리: 정면을 향한 보드에서 `f = 픽셀 거리 × Z / 실제 거리` → `FOV = 2·atan(W / 2f)`. 공장 캘리브레이션을 쓰지 않는다.
- 보드를 카메라 정면에 평행하게 두고 `SPACE` 로 샘플을 추가한다. 기울기가 3% 를 넘으면 초점 추정에서 제외된다 (`--max-tilt`).
- 보드를 화면 여러 위치/각도에 두고 5장 이상 모은 뒤 `c` 를 누르면 OpenCV `calibrateCamera` 로 왜곡까지 포함한 독립 캘리브레이션을 하고, 공장값과 비교한다.
- 결과는 `results/checker_*.json` 에 저장된다 (`s` 또는 종료 시).
- ToF depth 는 흑백 칸에서 값이 조금 다를 수 있다. 거리를 줄자로 쟀다면 `--distance` 로 고정값을 쓴다.

### depth 유효 영역

```powershell
.\.venv\Scripts\python fov_measure.py valid --depth 1024x1024@15 --frames 60
```

- 넓은 실내(벽/바닥/천장이 0.5~4m 안)를 향하게 두고 실행. 프레임을 누적해 절반 이상에서 값이 나온 픽셀을 유효로 본다.
- 광학 중심을 지나는 행/열에서 유효 영역 양 끝의 각도 = 실제로 쓸 수 있는 depth FOV.
- 이 장치 NFOV 결과: 공장 undistorted 73.0 × 65.6 → 유효 영역 **69.8 × 65.2**, 센서 픽셀의 61.5% 만 유효 (육각형 마스크).
- 카메라 바로 앞에 물체가 있으면(WFOV 최소 거리 미만 등) 유효 영역이 작게 잡히니 주의.

### 줄자

```powershell
# 한 거리에서
.\.venv\Scripts\python fov_measure.py manual --distance 1000 --width 1680 --height 1070
# 두 거리 차분 (광학 중심 위치 오차가 상쇄되어 더 정확)
.\.venv\Scripts\python fov_measure.py manual --distance 1000 --width 1680 --distance2 2000 --width2 3360
# 가이드 화면: 화면 끝 표시 + 중앙 depth, 종료 후 폭/높이 입력, 공장값과 비교
.\.venv\Scripts\python fov_measure.py manual --live --stream ir --depth 640x576
```

## IR 인식 비교 (작업 중)

IR 카메라(Femto Bolt / Femto Mega / DMK 33GX264)로 AprilTag·문양·도형 인식을 비교하는 도구. 현재는 카메라 소스와 모드 확인, 태그 파일 검증까지.

```powershell
.\.venv\Scripts\python list_modes.py                         # 연결된 장치 전부의 IR 모드 (Orbbec + DMK)
.\.venv\Scripts\python list_modes.py --source tis            # DMK 만 (orbbec / tis / all)
.\.venv\Scripts\python list_modes.py --ip 192.168.1.10       # Femto Mega 이더넷
.\.venv\Scripts\python list_modes.py --probe --out results   # 모드마다 실제로 열어 첫 프레임 시간·실측 fps·포화율 저장
.\.venv\Scripts\python gige_discover.py                      # GigE Vision 카메라 네트워크 탐색 (DMK, Mega)
```

### DMK 33GX264 (The Imaging Source, GigE)

- 이 PC 의 드라이버는 구형 **GigECam Driver and Tools 3.7** (DirectShow). IC Imaging Control 4 (`imagingcontrol4`) 는 IC4 용 GigE Vision 드라이버가
  없으면 장치를 못 찾으므로, 같은 드라이버를 쓰는 IC 3.x 래퍼 `tisgrabber_x64.dll` 을 `third_party/tisgrabber/` 에 두고 ctypes 로 쓴다 (pip 설치 불필요).
- 모드: Y800(8비트) / Y16(16비트, 12비트 ADC) × 2448x2048 … 640x240. fps 는 카메라 설정값과 무관하게 **링크 대역폭이 결정**한다:
  2448x2048 Y800 은 프레임당 5MB → 100 Mbps 링크에서 약 1.7fps, 1280x960 약 5fps. 1 Gbps 포트에서는 카메라 최대(2448x2048 약 24fps).
- 노출·게인은 뷰어 메뉴에서 바꾼다 (`TisDmkSource.set_value / set_auto`). 기본 자동 노출은 어두운 IR 장면에서 짧게 잡히는 편이라 수동 20ms + 12dB 정도부터 보는 게 빠르다.
- 네트워크: 카메라 기본은 link-local(169.254.x.x) + DHCP 라 PC 어댑터에 같은 서브넷 주소가 있어야 드라이버가 찾는다.
  어댑터에 고정 IP 를 넣으면 DHCP 가 꺼지면서 169.254 주소가 사라져 카메라가 안 보이게 된다. `gige_discover.py` 로 확인하고,
  관리자 권한이 없으면 `gige_discover.py --force-ip 192.168.1.20` 으로 카메라를 PC 서브넷으로 옮긴다 (카메라 전원을 끄면 원복).

Femto Bolt (FW 1.0.9) IR 모드: 320x288 / 640x576 (NFOV), 512x512 / 1024x1024 (WFOV) 각각 30·25·15·5fps (1024x1024 는 15·5fps), 그리고 **Passive IR** 1024x1024 15·5fps. 모두 Y16, depth 없이 IR 만 단독으로 켠다.

- **Passive IR** = Femto 자체 IR 조명을 끈 모드 (`OB_PROP_SWITCH_IR_MODE_INT=1`). 외부 IR 조명이 없으면 거의 검게 나온다 (실내 median 0).
- 1024x1024 외의 해상도에서 패시브로 두면 IR 이 아닌 depth 같은 값이 나와 모드 목록에서 뺐다.
- 같은 1024x1024 에서 액티브 ↔ 패시브를 바로 바꾸면 SDK 가 이전 모드를 유지한다 (프레임 없음 / 전부 0). 소스가 다른 해상도를 잠깐 열었다 닫아 자동으로 풀어 주며, 그만큼 전환이 ~1.3초 더 걸린다.

### 라이브 인식 테스트 (AprilTag · 도형)

뷰어 `e` 메뉴의 **Detect** 를 `apriltag` / `shapes` / `both` 로 두면 IR 패널에 검출 결과가 겹쳐 그려지고
(초록 = 태그, 빨간 점 = 첫 코너, 주황 = 도형) 패널 글자에 `tags N [id…] shapes … ms` 요약이 나온다. 장치를 바꿔도 설정은 유지된다.

| 항목 | 값 | 비고 |
|---|---|---|
| Detect | off / apriltag / shapes / both | 재시작 없이 즉시 적용 |
| Tag family | tag36h11 / tag25h9 / tag16h5 / tag36h10 | 36h11 권장. 16h5 는 오검출이 많아 hamming 0, margin ≥ 10 으로 거른다 |
| Invert image | off / on | 흰 테두리(inv) 태그, 어두운 바탕의 밝은 도형일 때 켠다 |
| Shape type | all / triangle / square / cross / circle / rect / pentagon / hexagon | 찾을 도형 종류 |
| Regular only | on / off | 정형만: 정삼각형 · 정사각형 · 정십자 · 원 · 정오각형 · 정육각형. rect 는 제외 |
| Shape tolerance | 5 / 10 / 15 / 20 / 30 % | 정형 허용 편차. 변 길이·대각선·팔 길이·종횡비가 이상값에서 벗어난 최대 비율 |
| Shape min px | 10 / 20 / 40 / 80 / 160 | 외접 사각형 짧은 변 최소 (원본 픽셀) |

- 태그: `pupil-apriltags` 가 있으면 그것(hamming / decision_margin 표시), 없으면 OpenCV ArUco. 두 검출기 모두 **검은 테두리** 를 찾는다.
- 도형: Otsu 이진화 → 외곽선 → 다각형 근사. triangle / square / rect / cross(십자) / pentagon / hexagon / circle. 기본은 밝은 종이 위의 어두운 도형.
  도형마다 정형 편차(dev %)를 같이 표시한다 (주황 = 허용 안, 노랑 = 허용 밖). 어두운 장면에서는 큰 어두운 영역이 rect 로 잡히니 노출을 올리고 보자.
- 검출은 폭 1280px 로 줄인 영상에서 하고 좌표는 원본으로 돌린다 (DMK 2448x2048 도 약 50ms).
- `s` 스냅샷의 JSON 에 태그 id·중심·한 변 픽셀·코너, 도형 이름·중심·면적이 들어간다.
- 저장된 영상으로 다시 돌리려면:

```powershell
.\.venv\Scripts\python detect_image.py resultsiewer_*_ir16.png --shapes          # 16비트 IR 스냅샷
.\.venv\Scripts\python detect_image.py photo.png --family tag16h5 --invert           # 오버레이 *_det.png + *_det.json
```

### AprilTag 파일 검증

```powershell
.\.venv\Scripts\python check_tags.py                     # april_tags/*.png 전부
.\.venv\Scripts\python check_tags.py --dir tags --family tag36h11
```

`april_tags/` 의 태그 PNG(apriltag-imgs 형식, 파일명 `tag16_05_00001.png` → 16h5, ID 1)가 사전의 올바른 비트인지 대조하고,
확대·회전·원근·노이즈·축소(셀당 2px) 영상에서 OpenCV ArUco 와 pupil-apriltags(설치 시)가 실제로 그 ID 로 읽는지 표로 보여 준다.
같은 검사를 `tests/test_april_tags.py` 가 pytest 로 돌린다.

- 두 검출기 모두 **검은 테두리 1셀** 을 찾는다. 흰 테두리(`_inv`) 태그는 영상을 먼저 반전(`255 - img`)해야 하고,
  바깥 흰 여백이 테두리와 붙어 2셀 두께가 되면 AprilTag 라이브러리는 디코딩에 실패한다 → 인쇄 시 여백 1셀을 잘라내거나 검게.
- 16h5 는 코드가 30개뿐이라 노이즈 영상에서 hamming 1~2 짜리 오검출이 매 프레임 여러 개 나온다.
  `hamming == 0` 과 `decision_margin` 으로 거르거나(`ir_detect.tags.ids_of`), 가능하면 36h11 을 쓴다.

## 구조

```
femto_fov/
  fov_math.py   FOV 계산, 왜곡 보정(Brown-Conrady K6 / Kannala-Brandt), 스트림 간 투영, 체커보드 초점 추정 — SDK 비의존
  device.py     pyorbbecsdk 래퍼 (장치 선택, 프로파일 열거, 내부/외부 파라미터, 프레임 변환)
  viz.py        컬러맵, 경계/각도 격자 오버레이, 패널 배치
  specs.py      데이터시트 공칭 FOV, Femto depth 모드 이름 (비교 기준, 필요 시 수정)
ir_detect/
  sources/      카메라 소스 공통 인터페이스 (base.py), Femto Bolt / Mega IR (orbbec_ir.py), DMK 33GX264 (tis_dmk.py)
  preprocess.py 센서 원본 → 8비트 변환
  detect.py     AprilTag(pupil-apriltags / OpenCV ArUco) · 도형 검출, 오버레이 그리기
  gige.py       GigE Vision GVCP 탐색 / FORCEIP (SDK 없이 네트워크 확인)
  tags.py       AprilTag PNG 비트 대조, 합성 영상 검출 (OpenCV ArUco / pupil-apriltags)
list_modes.py   카메라별 IR 모드 목록 / 실측
gige_discover.py GigE Vision 카메라 탐색 / 임시 IP
third_party/tisgrabber/  IC Imaging Control 3.x 래퍼 DLL (DMK 용, 출처는 안의 README)
check_tags.py   april_tags/ 태그 파일 검증
detect_image.py 저장된 영상에서 태그·도형 검출
april_tags/     사용할 AprilTag PNG (apriltag-imgs 형식)
tests/          단위 테스트 (장치 불필요): .\.venv\Scripts\python -m pytest
results/        출력 (gitignore)
```
