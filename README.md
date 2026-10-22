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

## (b) 라이브 FOV 뷰어

```powershell
.\.venv\Scripts\python fov_viewer.py                                   # 기본 프로파일
.\.venv\Scripts\python fov_viewer.py --depth 640x576 --color 1920x1080
.\.venv\Scripts\python fov_viewer.py --depth 1024x1024@15 --color 3840x2160@15
```

| 패널 | 오버레이 |
|---|---|
| Color | **cyan** = depth 센서 사각형 화각, **green** = depth 값이 실제로 나오는 영역 (지정 거리 Z 평면 기준). 각각 color 화면을 몇 % 덮는지 표시 |
| IR | **magenta** = color 화각 |
| Depth | **magenta** = color 화각, green = 유효 영역 외곽. `valid` = 실제 유효 픽셀의 각도 범위 |
| D2C | depth 를 color 에 정렬해 블렌드. color 픽셀 중 depth 가 있는 비율 |

| 키 | 동작 |
|---|---|
| `m` / `M` | depth 모드 다음/이전: NFOV binned → NFOV → WFOV binned → WFOV (IR 도 같이 전환) |
| `c` / `C` | color 해상도 다음/이전: 1280x720 → 1280x960(4:3) → 1920x1080 → 2560x1440 → 3840x2160 |
| `+/-`, `[ ]` | 경계선 투영 거리 ±100mm / ±1000mm |
| `a` | 화면 중앙 depth 를 거리로 자동 사용 |
| `g` | 각도 격자 (10° 간격, 왜곡 반영) |
| `d` | D2C 정렬 on/off (4K 에서 느리면 끄기) |
| `s` | 스냅샷 (PNG + depth npy + JSON) 저장 |
| `q` / `ESC` | 종료 |

- 모드 전환은 스트림을 재시작해야 해서 **약 4초** 걸린다 (카메라 드라이버의 stop 이 color ~2초, depth ~0.8초 — 실행 중 재설정은 드라이버가 거부). 전환은 백그라운드에서 진행되어 창은 계속 반응하고, 키를 연달아 누르면 마지막에 고른 모드로 **한 번만** 재시작한다 (마지막 키 후 0.5초 대기).
- fps 는 두 스트림이 공통으로 지원하는 최댓값으로 자동 선택된다 (WFOV unbinned → 15fps).
- 프로그램 시작에도 약 6초 걸린다 (장치 열기 ~3.4초 + 스트림 시작/첫 프레임 ~2.4초). 그동안 창에 연결 중 표시가 나온다.
- 딜레이 원인과 측정값 상세: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- 지원하지 않는 조합이면 이전 모드로 되돌아간다.

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

## 구조

```
femto_fov/
  fov_math.py   FOV 계산, 왜곡 보정(Brown-Conrady K6 / Kannala-Brandt), 스트림 간 투영, 체커보드 초점 추정 — SDK 비의존
  device.py     pyorbbecsdk 래퍼 (프로파일 열거, 내부/외부 파라미터, 프레임 변환)
  viz.py        컬러맵, 경계/각도 격자 오버레이, 패널 배치
  specs.py      데이터시트 공칭 FOV (비교 기준, 필요 시 수정)
tests/          단위 테스트 (장치 불필요): .\.venv\Scripts\python -m pytest
results/        출력 (gitignore)
```
