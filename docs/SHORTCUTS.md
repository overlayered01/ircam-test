# 단축키 정리

창이 뜨는 스크립트 세 개의 키 목록. 모든 창은 `q` 또는 `ESC` 로 닫고, 창의 X 버튼을 눌러도 종료된다.
키는 OpenCV 창에 포커스가 있을 때만 먹는다 (터미널에 치면 안 됨). 한/영 전환은 영문 상태로.

## fov_viewer.py — 라이브 FOV 뷰어

IR 이 기본으로 켜져 있다. 장치·모드·스트림 설정은 전부 `e` 메뉴에서 바꾼다.

### 일반 키

| 키 | 동작 | 비고 |
|---|---|---|
| `e` | 설정 메뉴 열기 / 닫기 | 아래 "메뉴" 참고. 하단 오른쪽 `MENU (e)` 버튼을 클릭해도 열린다 |
| `+` 또는 `=` | 경계선 투영 거리 +100 mm | 시차 때문에 거리마다 겹침이 달라진다 |
| `-` | 경계선 투영 거리 −100 mm | 최소 100 mm |
| `]` | 경계선 투영 거리 +1000 mm | |
| `[` | 경계선 투영 거리 −1000 mm | |
| `s` | 스냅샷 저장 | 화면 왼쪽 아래에 `saved viewer_...` 알림이 3초 뜨고 터미널에도 경로가 찍힌다. `results/viewer_<시각>.png` (화면), `_ir16.png`, 켜져 있으면 `_color.png` 와 `_depth_mm.npy`, `.json` (FOV·커버리지 수치) |
| `q` / `ESC` | 종료 | 재시작 중에도 `q` 로 종료 가능 |

하단 노란 줄에 현재 거리, fps, 키 요약이 항상 표시된다.

### 메뉴 (`e` 또는 `MENU` 버튼)

메뉴는 두 패널이다. 왼쪽 **Camera** = 장치·스트림·표시 설정, 오른쪽 **Detection** = AprilTag·도형 설정. `↑↓` 는 왼쪽 패널을 다 지나면 오른쪽으로 이어진다. 아래 바에 `Apply` / `Close` 버튼.

| 키 / 마우스 | 동작 |
|---|---|
| `↑` / `↓` (또는 `w` / `s`), 행 클릭 | 항목 이동 |
| `←` / `→` (또는 `a` / `d`), `<` `>` 클릭, 값 글자 클릭(다음), 행 위에서 휠 | 값 바꾸기 (목록은 순환, on/off 는 토글) |
| `Enter`, `Apply` 버튼 | 적용. 스트림·장치가 바뀌면 재시작 한 번 (Bolt USB 약 4초, Mega 이더넷 약 2초) |
| `ESC` / `e`, `Close` 버튼 | 적용하지 않고 닫기 |

메뉴 안에서는 일반 키(`s` 저장 등)가 동작하지 않는다.

| 항목 | 값 | 비고 |
|---|---|---|
| Device | 시작 시 찾은 Orbbec 장치 목록 (USB + 자동 탐색된 이더넷, `--ip` 로 준 장치) | 바꾸면 파이프라인을 새로 연다. 새 장치에 없는 해상도는 기본값으로 |
| IR mode | NFOV binned → NFOV → WFOV binned → WFOV | depth 도 같은 모드 (같은 센서) |
| FPS | auto / 그 모드가 지원하는 값 | auto = 켜진 스트림의 공통 최댓값. 모드를 바꿔 그 fps 가 없으면 auto 로 |
| Depth | on / off | |
| Color | on / off | |
| Color res | 1280x720 → 1280x960 → 1920x1080 → 2560x1440 → 3840x2160 | 고르면 Color 가 on 이 된다 |
| Angle grid | on / off | 10° 간격, 렌즈 왜곡 반영. 즉시 적용 |
| D2C align | on / off | color + depth 가 켜져 있을 때만 의미 있음. 4K 에서 느리면 끈다. 즉시 적용 |
| Auto distance | on / off | 화면 중앙 depth 의 중앙값을 거리로 사용 (depth 켜졌을 때). 즉시 적용 |
| Detect | off / apriltag / shapes / both | IR 패널에 검출 결과 오버레이 + 요약. 즉시 적용, 장치를 바꿔도 유지 |
| Detect width | 640 / 1280 / full | 검출용 축소 폭. DMK 에서 작은 도형은 full |
| Tag family | tag36h11 / tag25h9 / tag16h5 / tag36h10 | |
| Invert image | on / off | 흰 테두리(inv) 태그, 어두운 바탕의 밝은 도형 |
| Shape type | all / triangle / square / cross / circle / rect / pentagon / hexagon | 찾을 도형 종류 |
| Shape mode | both / filled / outline | 속이 찬 도형 / 선으로 그린 도형(두 획 십자 포함) / 둘 다 |
| Regular only | on / off | 정삼각형·정사각형·정십자·원·정오각형·정육각형만 (rect 제외) |
| Shape tolerance | 5 / 10 / 15 / 20 / 30 % | 정형 허용 편차 |
| Shape min px | 10 / 20 / 40 / 80 / 160 | 도형 최소 크기 (외접 사각형 짧은 변) |

DMK 33GX264 를 고르면 메뉴 항목이 아래로 바뀐다.

| 항목 | 값 | 비고 |
|---|---|---|
| Device | 위와 같음 | |
| Mode | Y800 2448x2048 → … → Y16 640x240 | 픽셀 포맷 × 해상도. 실제 fps 는 링크 대역폭이 결정 (100 Mbps: 전체 해상도 ~1.7fps) |
| Exposure auto | on / off | 값을 바꾸면 자동으로 off |
| Exposure (ms) | 0.05 … 200 | 즉시 적용 (재시작 없음) |
| Gain auto | on / off | |
| Gain (dB) | 0 … 48 | 즉시 적용 |
| Angle grid | on / off | `calib/dmk_<serial>.json` 이 있을 때만 나온다 |
| Detect / Tag family / Invert image / Shape … | 위와 같음 | |

- 지원하지 않는 조합이면 "failed: ..." 가 표시되고 이전 구성이 유지된다.
- 장치 전환에 실패하면 이전 장치를 다시 연다.

## fov_measure.py checker — 체커보드 실측

| 키 | 동작 | 비고 |
|---|---|---|
| `SPACE` | 샘플 추가 | 보드가 검출된 상태에서만. 기울기가 `--max-tilt`(기본 3%) 를 넘거나 거리를 못 재면 캘리브레이션용 뷰로만 추가되고 초점 추정에서는 빠진다 |
| `c` | OpenCV 캘리브레이션 실행 | 5장 이상 모은 뒤. 왜곡까지 포함해 공장값과 비교 |
| `r` | 샘플 초기화 | 모은 샘플·캘리브레이션 결과 모두 삭제 |
| `s` | 결과 저장 | `results/checker_*.json`. 종료 시에도 자동 저장된다 |
| `q` / `ESC` | 종료 | |

## fov_measure.py valid — depth 유효 영역

| 키 | 동작 |
|---|---|
| `q` / `ESC` | 누적 중단 (그때까지 모은 프레임으로 계산) |
| 아무 키 | 결과 창 닫기 |

## fov_measure.py manual --live — 줄자 실측 가이드

| 키 | 동작 |
|---|---|
| `q` / `ESC` | 가이드 창 닫기. 이어서 터미널에서 폭/높이를 입력한다 |

## 키가 없는 스크립트

`fov_table.py`, `list_modes.py`, `check_tags.py`, `detect_image.py`, `gige_discover.py` 는 창 없이 터미널에 결과만 출력한다. 중단은 `Ctrl+C`.
