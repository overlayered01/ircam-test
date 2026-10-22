# 뷰어 딜레이 원인 분석

`fov_viewer.py` 사용 중 겪은 느림/딜레이의 원인, 측정값, 조치 내용을 정리한다.

- 측정 환경: Orbbec Femto Bolt (FW 1.0.9, USB 3.1), Windows 11, pyorbbecsdk2 2.1.2 (OrbbecSDK 2.9.3), OpenCV 5.0, Python 3.12
- 측정일: 2026-10-22
- 모든 수치는 실제 장치에서 측정. 장면/PC 에 따라 다를 수 있다.

## 요약

| # | 증상 | 원인 | 줄일 수 있나 | 조치 |
|---|---|---|---|---|
| 1 | 전체적으로 프레임이 낮음 (NFOV 8.6fps, WFOV 5.5fps) | 뷰어가 원본 해상도로 그린 뒤 축소 | O | 렌더링 최적화 → 센서 최대치 도달 |
| 2 | WFOV 에서 프레임이 절반 | WFOV unbinned 센서 최대 15fps | X (하드웨어) | 30fps 가 필요하면 WFOV binned 512x512 사용 |
| 3 | 모드 전환 키를 누르면 ~4.5초 멈춤 | 카메라 드라이버의 스트림 종료(`pipeline.stop`)가 ~2.9초 | X (드라이버) | 백그라운드 전환 + 연타 묶기로 체감 개선 |
| 4 | 실행 후 화면이 뜨기까지 ~6초 | 장치 열기 3.4초 + 스트림 시작/첫 프레임 2.4초 | X (SDK/드라이버) | 창을 먼저 띄우고 "Connecting..." 표시 |
| 5 | 4K + WFOV 에서 12fps | D2C 정렬 비용 | 일부 | `d` 키로 정렬 끄기 |

평상시 화면 지연(프레임 도착 → 표시)은 **중앙값 35ms** 로 문제없다. 사람이 느끼는 "반응 느림"은 거의 전부 #3, #4 에서 나온다.

---

## 1. 렌더링 비용 (해결됨)

### 증상
NFOV 에서도 30fps 가 안 나오고, WFOV 에서는 5fps 수준으로 끊김.

### 원인
매 프레임 원본 해상도(color 1920x1080, depth 1024x1024)에서 컬러맵·블렌드·오버레이를 모두 그린 뒤, 마지막에 800x600 패널로 축소하고 있었다. 화면에 보이지도 않을 픽셀을 처리하느라 시간을 썼다.

단계별 측정 (WFOV 1024x1024 + color 1080p, 프레임당):

| 단계 | 시간 |
|---|---|
| depth 컬러맵 (`colorize_depth`, float 연산) — depth + 정렬 depth 2회 | **48 ms** |
| D2C 정렬 (`AlignFilter`) | 27 ms |
| IR 정규화 (`np.percentile` 1M 픽셀) | 7 ms |
| 프레임 변환 | 3 ms |

15fps 의 프레임 예산은 66ms 인데 컬러맵과 정렬만으로 이를 넘는다.

### 조치
- 패널을 **표시 크기로 먼저 축소한 뒤** 그린다. 경계선/격자 좌표는 같은 배율로 스케일 (`Intrinsics.scaled()`, `viz.shrink()`).
- `colorize_depth` 를 **룩업 테이블** 방식으로 교체: 1024² 기준 25ms → **2.4ms** (색 차이 최대 5/255).
- depth 를 uint16 그대로 사용 (Femto 는 1 unit = 1mm 라 float 변환 불필요).
- IR 정규화의 백분위를 1/16 샘플로 계산: 7ms → **2.8ms**.
- 유효 영역 FOV·외곽선(convex hull) 계산을 10프레임마다 한 번으로.

### 결과 (첫 화면 이후 루프 fps)

| 모드 | 이전 | 이후 | 센서 최대 |
|---|---|---|---|
| NFOV 640x576 + 1080p | 8.6 | **31** | 30 |
| WFOV 1024x1024 + 1080p | 5.5 | **16** | 15 |
| WFOV binned 512x512 + 1080p | - | **31** | 30 |
| WFOV 1024x1024 + 4K | 2.5 | **12** | 15 |

---

## 2. WFOV unbinned 의 15fps 한계 (하드웨어)

`fov_table.py` / 프로파일 목록에서 depth·IR 1024x1024 는 최대 15fps 만 제공된다. D2C 정렬을 꺼도 14.9fps 로 측정되어 소프트웨어 문제가 아님을 확인했다.

- 30fps 가 필요하면 **WFOV 2x2 binned (512x512)** — 화각은 120° 로 동일하고 해상도만 절반.
- 뷰어는 color/depth 가 공통으로 지원하는 최대 fps 를 자동 선택하므로, WFOV unbinned 선택 시 color 도 15fps 로 내려간다.

---

## 3. 모드 전환 시 ~4.5초 멈춤 (드라이버, 완화함)

### 증상
`m` / `c` 키로 모드를 바꾸면 화면이 4초 넘게 멈춘다. 여러 번 누르면 누른 횟수만큼 누적된다.

### 원인
모드를 바꾸려면 파이프라인을 멈췄다가 새 설정으로 다시 시작해야 하는데, **`pipeline.stop()` 자체가 매우 느리다.**

전환 1회의 단계별 시간:

| 단계 | 시간 |
|---|---|
| `pipeline.stop()` | **2,810 ~ 3,070 ms** |
| 프로파일 조회·선택 (`list_profiles`, `find_profile`) | ~10 ms |
| `pipeline.start()` | ~620 ms |
| 첫 프레임 도착 | 390 ~ 800 ms |
| **합계** | **~4.0 ~ 4.5 s** |

`stop()` 이 느린 이유를 스트림 조합별로 측정:

| 켜 둔 스트림 | `stop()` 시간 |
|---|---|
| depth 만 | 791 ms |
| depth + IR | 807 ms |
| color 만 (BGRA) | **1,971 ms** |
| color 만 (MJPG) | **2,111 ms** |
| color + depth + IR (frame sync on) | 2,669 ms |
| color + depth + IR (frame sync off) | 2,786 ms |

→ **color 스트림 종료에만 ~2초** (Windows Media Foundation 의 UVC 소스 해제), depth 가 ~0.8초. 포맷(BGRA/MJPG)이나 frame sync 와는 무관하다.

`stop()` 없이 실행 중에 새 설정으로 `start()` 를 다시 호출해 보았으나 드라이버가 거부했다:

```
spSourceReaderEx->SetNativeMediaType(...) returned: HResult 0xc00d36b2: "현재 상태에서 잘못된 요청입니다."
```

따라서 이 지연은 애플리케이션에서 줄일 수 없다.

### 조치 (체감 개선)
- **백그라운드 전환**: 재시작을 별도 스레드(`Switcher`)에서 수행. 그동안 창은 계속 반응하고 `Switching to … (2.1s)` 와 경과 시간을 보여 준다.
- **연타 묶기 (debounce)**: 마지막 키 입력 후 0.5초 기다렸다가 최종 목표로 한 번만 재시작. 예: NFOV 에서 `m` 2번 → 중간 모드(512x512)를 건너뛰고 1024x1024 로 1회만 재시작 (이전: 2회 × 4.5초).
- 전환 중에도 `m`/`c` 로 목표를 바꿀 수 있고, `q` 로 종료 가능.
- 전환 중에는 메인 스레드가 `wait_for_frames` 를 호출하지 않는다 (stop/start 와 동시 호출 방지).

### 대안 (미적용)
센서 단위 API(`Sensor.switch_profile`)로 depth 만 바꾸고 color 는 계속 켜 두면 color 의 2초를 피할 수 있을 가능성이 있다. 다만 Pipeline 의 frame sync 와 `AlignFilter`(FrameSet 필요)를 쓸 수 없게 되어 구조를 크게 바꿔야 하므로 적용하지 않았다.

---

## 4. 시작 시 ~6초 (SDK/드라이버)

| 단계 | 누적 시간 |
|---|---|
| `open_pipeline()` — 장치 탐색 + 열기 | 3.44 s |
| 스트림 시작 + 첫 프레임 표시 | 5.87 s |

장치 열기는 SDK 내부(장치 열거, 펌웨어 통신)라 줄일 수 없다. 창을 먼저 띄워 "Connecting to Femto..." 를 표시하도록 바꿨다.

첫 프레임 직후 10프레임 정도는 지연이 170ms → 83ms 로 줄어들며 안정되는데, 시작 직후 SDK 내부 큐가 비워지는 과정으로 보인다. 이후 중앙값 35ms.

---

## 5. 기타

### 카메라 점유로 인한 실행 실패
뷰어가 실행 중인 상태에서 다른 스크립트를 실행하면 장치를 열 수 없다 (Windows: `0x80070005 액세스가 거부되었습니다`). Femto 는 한 프로세스만 열 수 있다.

SDK 가 이 한글 오류 메시지를 UTF-8 로 디코딩하지 못해 원인 대신 `UnicodeDecodeError` 가 나는 문제가 있어, `open_pipeline()` 에서 잡아 원인 안내로 바꿨다.

### OpenCV 5 텍스트 겹침
OpenCV 5 의 새 폰트 렌더러는 두께에 따라 글자 간격이 달라져, 두꺼운 외곽선 + 얇은 글자 방식이 어긋나 보였다. 외곽선을 같은 두께로 위치만 옮겨 그리는 방식으로 변경. (성능과는 무관)

---

## 측정 방법

- **루프 fps / 렌더링 비용**: `cv2.imshow` / `cv2.waitKey` 를 대체하는 하네스로 뷰어를 GUI 없이 실행하고, 첫 화면 이후 `imshow` 호출 간격으로 fps 를 계산. 단계별 시간은 `time.perf_counter()` 로 측정 (cProfile 은 pybind 확장 호출 — `wait_for_frames`, `AlignFilter.process` — 을 잡지 못함).
- **화면 지연**: color 프레임의 `get_system_timestamp_us()` (호스트 도착 시각) 와 표시 시각의 차이.
- **`stop()` 시간**: 스트림 조합별로 `pipeline.start()` → 30프레임 수신 → `pipeline.stop()` 소요 시간.

재현용 최소 코드 (`stop()` 측정):

```python
import time
from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline

pipeline = Pipeline()
profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
cfg = Config()
cfg.enable_stream(profiles.get_video_stream_profile(1920, 1080, OBFormat.BGRA, 30))
pipeline.start(cfg)
for _ in range(30):
    pipeline.wait_for_frames(1000)
t = time.perf_counter()
pipeline.stop()
print(f"stop: {(time.perf_counter() - t) * 1000:.0f} ms")   # Femto Bolt 에서 ~2000 ms
```
