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
.\.venv\Scripts\python fov_viewer.py                                   # 연결된 것 중 DMK → Femto Mega → Femto Bolt 순으로 시작, IR 만
.\.venv\Scripts\python fov_viewer.py --device bolt                     # 이름에 bolt 가 들어간 장치로 시작 (dmk / mega 도 됨)
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
| `e` 또는 하단 `MENU` 버튼 | 설정 메뉴 (왼쪽 Camera / 오른쪽 Detection 패널). Orbbec: 장치 / IR 모드 / fps / depth on-off / color on-off / color 해상도 / 각도 격자 / D2C 정렬 / 자동 거리. DMK: 장치 / 모드 / 노출 auto·값 / 게인 auto·값. 공통: Detect / Tag family / Invert image / 도형 조건. 키보드 `↑↓` 항목, `←→` 값, `Enter` 적용, `Esc` 닫기. 마우스: 행 클릭 선택, `<` `>` 클릭·휠로 값, `Apply` / `Close` 버튼 |
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
- 노출·게인은 뷰어 메뉴에서 바꾸거나 `--exposure 10 --gain 20` (ms, dB) 으로 시작부터 고정한다 (자동 off, 장치를 다시 열거나 모드를
  바꿔도 다시 적용). 자동 노출은 어두운 IR 장면에서 fps 한계(24 fps 면 41.67 ms)까지 올라가고 게인도 46 dB 근처에 붙는다 —
  움직이는 태그는 흐려져 검출이 들쑥날쑥하고 노이즈가 커서 작은 태그를 놓친다. 10~20 ms 로 고정하고 모자라면 IR 조명을 더 준다.
  패널의 `exposure … (fixed)` 로 고정 여부를 본다. 메뉴 없이 `,` `.` (노출) `;` `'` (게인) 키로도 즉시 바꾼다.
  카메라 노출 범위는 0.02 ms ~ 30 s (메뉴는 0.05 ~ 1000 ms). 프레임 간격보다 긴 노출은 카메라가 fps 를 낮춰 받아 준다
  (100 ms → 10 fps). 자동 노출의 상한은 메뉴 `Exposure auto max` (카메라 "Auto Max Value") 로 올린다.
- 프레임 드랍: 순수 수신은 2448x2048 Y800 에서 23.3~23.6 fps, 20초에 1~4장 손실(USB GbE 어댑터, 점보 프레임 없음)로 무시할 수준.
  뷰어에서 보이던 드랍은 검출이 프레임 간격보다 길어 생긴 것이라 검출 스레드 분리로 사라진다. 그래도 끊기면 Tag decimate 2.
- 네트워크: 카메라 기본은 link-local(169.254.x.x) + DHCP 라 PC 어댑터에 같은 서브넷 주소가 있어야 드라이버가 찾는다.
  어댑터에 고정 IP 를 넣으면 DHCP 가 꺼지면서 169.254 주소가 사라져 카메라가 안 보이게 된다. `gige_discover.py` 로 확인하고,
  관리자 권한이 없으면 `gige_discover.py --force-ip 192.168.1.20` 으로 카메라를 PC 서브넷으로 옮긴다 (카메라 전원을 끄면 원복).

Femto Bolt (FW 1.0.9) IR 모드: 320x288 / 640x576 (NFOV), 512x512 / 1024x1024 (WFOV) 각각 30·25·15·5fps (1024x1024 는 15·5fps), 그리고 **Passive IR** 1024x1024 15·5fps. 모두 Y16, depth 없이 IR 만 단독으로 켠다.

- **Passive IR** = Femto 자체 IR 조명을 끈 모드 (`OB_PROP_SWITCH_IR_MODE_INT=1`). 외부 IR 조명이 없으면 거의 검게 나온다 (실내 median 0).
- 1024x1024 외의 해상도에서 패시브로 두면 IR 이 아닌 depth 같은 값이 나와 모드 목록에서 뺐다.
- 같은 1024x1024 에서 액티브 ↔ 패시브를 바로 바꾸면 SDK 가 이전 모드를 유지한다 (프레임 없음 / 전부 0). 소스가 다른 해상도를 잠깐 열었다 닫아 자동으로 풀어 주며, 그만큼 전환이 ~1.3초 더 걸린다.

### 라이브 인식 테스트 (AprilTag · 도형 · 점 마커)

뷰어 `e` 메뉴의 **Detect** 를 `apriltag` / `shapes` / `both` / `dots` 로 두면 IR 패널에 검출 결과가 겹쳐 그려지고
(초록 = 태그, 빨간 점 = 첫 코너, 주황 = 도형, 자홍 = 점 마커) 패널 글자에 `tags N [id…] shapes … ms` 요약이 나온다. 장치를 바꿔도 설정은 유지된다.

| 항목 | 값 | 비고 |
|---|---|---|
| Detect | off / apriltag / shapes / both / dots | 재시작 없이 즉시 적용. dots = 인쇄한 점 1·2·3개 마커 (아래) |
| Detect width | 640 / 1280 / full / 2x | 검출용 축소 폭. DMK 2448px 에서 작은 도형·태그를 보려면 full (outline 포함 약 90 ms). 2x 는 원본을 2배로 키워 검출 (셀 2~3 px 짜리 태그용, 시간 4배 — Mega 640x576 급에서) |
| Tag family | tag36h11 / tag25h9 / tag16h5 / tag36h10 | 36h11 권장. 16h5 는 오검출이 많아 hamming 0, margin ≥ 10 으로 거른다 |
| Tag decimate | 1 / 2 / 4 | 태그 후보 탐색 해상도 1/n. 2 면 4~6배 빠르고(DMK 1280 폭 46 → 11~18 ms) 최소 태그 크기가 2배(36h11 18 → 36 px) |
| Tag sharpen | 0.25 / 0.5 / 1 | 비트 읽기 전 선명화 (pupil decode_sharpening). 1 이 기본: 18 px 36h11 0/10 → 8/10, 회색 20 px 5/10 → 9/10, 시간 동일 |
| Tag max hamming | 0 / 1 / 2 | 허용 비트 오류. 흐린 작은 태그는 한 비트가 틀려 hamming 1 로 떨어진다. 36h11·25h9 는 1 까지 안전, 16h5 는 0 |
| Tag min margin | 10 / 20 / 40 / 80 | decision_margin 최소. 제대로 보이는 태그는 100 이상(sharpen 1), 오검출은 대개 10~40 이라 20~40 이면 대부분 걸러진다 |
| Confirm frames | off / 2 / 3 / 5 | 같은 ID(점 마커는 개수)가 같은 자리(중심 거리 ≤ 한 변)에 연속 n 프레임 나와야 표시. 한 프레임 놓친 건 봐준다. 요약의 `(+n new)` 가 대기 중인 수 |
| Invert image | off / on | 흰 테두리(inv) 태그, 어두운 바탕의 밝은 도형일 때 켠다 |
| Shape type | all / triangle / square / cross / circle / rect / pentagon / hexagon | 찾을 도형 종류 |
| Shape mode | both / filled / outline | filled = 속이 찬 도형, outline = 선으로 그린 도형(펜 선, 두 획 십자). both 는 둘 다 |
| Regular only | on / off | 정형만: 정삼각형 · 정사각형 · 정십자 · 원 · 정오각형 · 정육각형. rect 는 제외 |
| Shape tolerance | 5 / 10 / 15 / 20 / 30 % | 정형 허용 편차. 변 길이·대각선·팔 길이·종횡비가 이상값에서 벗어난 최대 비율 |
| Shape min px | 10 / 20 / 40 / 80 / 160 | 외접 사각형 짧은 변 최소 (원본 픽셀) |
| Dot min px | 3 / 4 / 6 / 8 / 12 | 점 지름 최소 (원본 픽셀). 4 px 아래는 잡음과 구분이 어렵다 |
| Dot group gap | 2.5x / 3.5x / 5x | 같은 마커로 묶는 점 중심 간 거리 (점 지름 배수). 시트는 3 d 이내로 인쇄되어 있다 |
| Dot min count | 1 / 2 | 2 면 1점 마커를 무시한다 (멀리서 글자 조각·얼룩이 1점으로 읽힐 때) |

- 작은 태그 (예: 2 m 에서 2 cm): 태그 한 변이 검출 영상에서 36h11 기준 18 px(셀 2.2 px), 회색·흐린 태그면 22 px 은 돼야 한다.
  2 cm 가 몇 px 로 보이는지는 렌즈가 정한다 — DMK(3.45 µm 픽셀) 2 m 에서 8 mm 렌즈 23 px, 12 mm 35 px, 16 mm 46 px, 25 mm 72 px;
  Mega NFOV 640x576 은 5 px 라 불가. 순서대로 볼 것: ① Detect width = full (1280 이면 DMK 는 절반 크기로 줄어 11 px),
  ② Tag decimate = 1, Tag sharpen = 1, ③ 노출 고정 + 게인 낮춰 노이즈 줄이기, 포커스 (패널 focus 값이 최대가 되게 렌즈 링),
  ④ 인쇄: 레이저 토너는 IR 에서 회색으로 보이는 경우가 많다 — 잉크젯 검정(안료) 이나 무광 검정지에 흰 테두리 한 칸 이상,
  ⑤ 그래도 모자라면 16h5 (14 px, ID 30개, 오검출 주의) 또는 더 긴 렌즈. Mega 에서는 Detect width 2x.
  DMK 의 1280x960 등 작은 포맷은 **중앙 잘라내기(ROI)** 라 픽셀 밀도는 2448x2048 과 같다 (실측: 같은 태그가 두 모드 모두 30 px).
  화각만 좁아지고 fps 가 오르므로, 태그를 더 크게 보려면 모드가 아니라 렌즈(초점거리)나 거리를 바꿔야 한다.
- 오검출(엉뚱한 곳에 태그): 16h5 는 코드가 30개뿐이라 배경 무늬가 가끔 한 프레임짜리 가짜 ID 로 읽힌다. 순서대로:
  ① Tag min margin 20~40 (진짜 태그는 100 넘는다), ② Tag confirm 2~3 (가짜는 자리·ID 가 이어지지 않는다), ③ 가능하면 36h11.
- 검출은 별도 스레드에서 돈다 (`det.DetectWorker`). 검출이 프레임 간격보다 오래 걸리면 그동안 들어온 프레임은 검출만 건너뛰고
  영상은 카메라 fps 그대로 흐른다 (결과는 한두 프레임 늦게 따라온다). 같은 스레드에서 돌리던 때는 DMK 2448px 에서 검출 46 ms 가
  프레임 41 ms 를 넘어 정확히 한 장 걸러 한 장씩 버려졌고(11.7 fps), 장면 따라 검출 시간이 달라져 간격이 튀었다.
- 태그: `pupil-apriltags` 가 있으면 그것(hamming / decision_margin 표시), 없으면 OpenCV ArUco. 두 검출기 모두 **검은 테두리** 를 찾는다.
- 도형: triangle / square / rect / cross(십자) / pentagon / hexagon / circle. 기본은 밝은 종이 위의 어두운 도형.
  도형마다 정형 편차(dev %)를 같이 표시한다 (주황 = 허용 안, 노랑 = 허용 밖). 어두운 장면에서는 큰 어두운 영역이 rect 로 잡히니 노출을 올리고 보자.
  - filled: 영상 전체 Otsu 이진화 → 바깥 외곽선 → 다각형 근사. 인쇄한 검정 도형용. 640x576 에서 약 1 ms.
  - outline: 국소(adaptive) 이진화로 가는 선만 뽑아 닫힘·팽창으로 잇고, 닫힌 윤곽은 안쪽 구멍을, 구멍 없는 획(십자 두 줄)은 팽창된 바깥을
    분류한다. 획이 주변보다 어두운 후보만 받는다 (invert 면 밝은 쪽). 손으로 그린 도형, 또는 종이가 밝아 Otsu 임계값이 종이/배경 사이로
    잡혀 펜 선이 사라지는 장면용. 640x576 약 5 ms, 1280px 축소 영상 약 22 ms.
  - 작은 도형: 모서리가 뭉개져 꼭짓점이 남거나(사각형 → 육각형) 잘리면(삼각형 꼭대기 → 사다리꼴) 2배·3배 거친 근사로 다시 보고
    편차가 작은 해석을 고른다 (넓이가 0.7~1.3배로 맞고, 볼록 도형은 solidity ≥ 0.85 일 때만).
  - 최소 크기 (어두운 방 + 인쇄 시트 합성, 실측 IR 노이즈, Regular only 10%, 5회 중 5회 성공 기준, 외접 사각형 한 변):

    | 도형 | filled | outline |
    |---|---|---|
    | circle | 8 px | 20 px |
    | square | 10 px | 20 px |
    | cross | 12 px | 15 px |
    | triangle | 20 px | 25 px |
    | pentagon | 30 px | 30 px |
    | hexagon | 40 px (작으면 원과 혼동) | 불안정 |

    Femto Mega NFOV 640x576 은 2.7 m 에서 1 px ≈ 5.4 mm 다 (초점거리 ≈ 505 px). 즉 사각형 ≥ 55 mm, 십자 ≥ 65 mm,
    삼각형 ≥ 110 mm, 오각형 ≥ 160 mm 가 하한이고 여유를 두려면 1.5배. DMK 2448x2048 은 같은 화각이면 픽셀이 3.8배 촘촘해
    (렌즈 8 mm 기준 1 px ≈ 1.2 mm) 60 mm 시트가 50 px 로 보인다 — 단, Detect width 를 full 로 두어야 한다 (1280 이면 26 px).
  - 종이에 선으로 그린 도형은 640x576 에서 한 변 30px 이상(1 m 에서 약 7 cm)은 돼야 안정적으로 잡힌다. 작은 손그림 삼각형은 선이 끊기면 못 잡는다.
  - 정형 편차는 픽셀 양자화 몫(1px / 변 길이)을 빼고 센다. 20px 변은 1px 이 이미 5% 라 그만큼은 봐준다.
  - 뷰어는 도형을 프레임 간 추적한다: 라벨의 `[xx%]` 가 최근 30프레임 중 검출된 비율(검출률), 요약의 `hit xx%` 는 그 평균.
    이번 프레임에 안 잡혀도 직전 10프레임 안에 보였으면 회색으로 유지해 깜빡이지 않는다. 검출률을 올리려면 도형을 키우거나
    가까이 두거나(DMK 는 픽셀이 4배라 유리) Shape tolerance 를 15% 로 올린다.
- 인쇄용 시트: `python shape_sheet.py` → `shapes/shapes_filled.png`, `shapes/shapes_outline.png` (A4 300 dpi, 여섯 도형 60 mm, 선 2 mm).
  `--size`, `--stroke` 로 바꾼다. 100% 배율로 레이저 인쇄.
- **점 마커** (Detect `dots`): 인쇄한 검정 점 1·2·3개를 한 마커로 읽는다. 점 하나는 지름 **6 px 이상**이면 안정적으로 잡히고(합성 검증:
  6·9·12 px 점 잡음·흐림 아래 5/5, 3~5 px 는 0~1/5) 태그는 한 변 18~22 px 이 필요하므로, 같은 크기 마커로 태그보다 약 3배 먼 거리까지 간다.
  DMK 2448 전체(full)에서 2 m·4 mm 점이 약 6 px (Detect width 1280 이면 3 px 라 안 된다 — dots 는 full 로).
  - 검출: 국소 이진화로 어두운 덩어리를 찾고 → 덩어리마다 (안쪽 밝기 + 바깥 밝기)/2 로 다시 잘라 붙은 점을 가른 뒤 → 종횡비·채움비(원 0.785)·
    원형도(≥ 0.75)·볼록도(≥ 0.88, 글자 B·고리 O 탈락)·대비(≥ 20)로 점만 남기고 → 밝기 가중 중심(서브픽셀) → 중심 간 거리 ≤ gap × 지름이고
    지름 비 2.4 이내인 점끼리 묶는다. 4개 이상 뭉친 그룹은 버린다. 2448x2048 full 에서 깨끗한 장면 약 22 ms, 아주 지저분한 장면 ~100 ms (검출 스레드가 프레임을 건너뛴다).
  - 종류와 방향 (라벨 `D…`, 화살표 = 방향 확정, 양쪽 선 = 축만):

    | 라벨 | 배치 | 방향 |
    |---|---|---|
    | D1 | 점 하나 | 없음 (위치만) |
    | D2 | 같은 크기 두 점 | 축만 (앞뒤 모름, 연보라) |
    | D2s | 큰 점(1.7 d) + 작은 점 | 큰 점 → 작은 점 |
    | D3I | 일직선, 간격 1 : 2 | 외톨이 점 → 붙은 두 점 쪽 (등간격이면 축만) |
    | D3L | 직각 L | 꼭짓점(내각 최대) → 나머지 두 점의 중점 (정삼각형이면 축만) |

  - 오검출: 1점 마커가 가장 약하다 (멀리서 글자 조각·얼룩이 점으로 보인다). Confirm frames 2~3 을 켜고, 1점이 필요 없으면 Dot min count 2.
    글자가 있는 종이는 글자를 점에서 5 d 이상 떨어뜨린다 (가까우면 한 마커로 묶인다). 반전(흰 점)은 Invert image.
  - 인쇄용 시트 (`dot_sheet.py`, A4 300 dpi, 100% 배율, 아래 100 mm 선으로 배율 확인; 글자는 얇은 회색):
    - `python dot_sheet.py markers [--dots 4 6 8]` → `shapes/dots_markers.png` (_p2 …): 다섯 마커 × 지름.
    - `python dot_sheet.py sizes [--kinds 3L] [--sizes 2 3 4 5 6 8 10 12] [--grays 0 90 150]` → `shapes/dots_sizes.png` (_p2 …):
      한 마커를 크기·잉크 농도별로. 거리별 최소 크기와 회색 잉크 한계를 재는 용도. 합성 2 m 검증: 검정·회색 150 모두 3 mm(4.5 px)까지, 2 mm 는 안 됨.
    - `python dot_sheet.py cube [--face 80 --dot 4 --picture 38 --edge 2 --faces 1 2 2s 3I 3L 3L,1,3L,1]` → `shapes/cube_faces.png` (_p2)
      + 면마다 `cube_face_N_<spec>.png` (점만, 정확히 80 mm → 인쇄소용). 8 cm 큐브 면: 네 모서리에 마커(L 꼭짓점·큰 점·외톨이 점이 면
      모서리 쪽), 가운데 그림 자리(점선). spec 이 하나면 네 모서리 같은 마커, `TL,TR,BR,BL` 로 모서리별 지정(`-` 는 비움).
      **그림의 어두운 부분이 점에서 묶기 거리(3.5 d = 14 mm) 안에 오면 점과 한 마커로 묶여 깨진다** — 스크립트가 그림 상자 ↔ 점 최소 거리를
      출력하고 모자라면 경고한다 (기본값 14.6 mm). 그림을 키우려면 점을 줄이거나 그림 모서리를 밝게/둥글게. 시트의 면 사이는 6 d 이상
      (이웃 면 모서리 마커끼리 묶이지 않게), 재단 표시는 변 중앙 눈금(테두리 선은 2 mm 옆 점과 붙는다).
    - 2 m 에서 DMK 는 4 mm (6 px), Mega(640px) 는 8 mm 부터.
- 검출은 폭 1280px 로 줄인 영상에서 하고 좌표는 원본으로 돌린다 (DMK 2448x2048 도 약 50ms).
- `s` 스냅샷의 JSON 에 태그 id·중심·한 변 픽셀·코너, 도형 이름·중심·면적, 점 마커 종류·개수·중심·방향·점 좌표가 들어간다.
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
detect_image.py 저장된 영상에서 태그·도형·점 마커 검출 (--dots)
shape_sheet.py  인쇄용 도형 테스트 시트 생성 → shapes/
dot_sheet.py    인쇄용 점 마커 시트: markers / sizes / cube(8 cm 큐브 면) → shapes/dots_*.png, cube_face*.png
april_tags/     사용할 AprilTag PNG (apriltag-imgs 형식)
shapes/         인쇄용 도형 시트 (filled / outline) · 점 마커 시트 (markers / sizes / cube), A4 300 dpi
tests/          단위 테스트 (장치 불필요): .\.venv\Scripts\python -m pytest
results/        출력 (gitignore)
```
