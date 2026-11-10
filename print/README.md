# print/ — 인쇄용 시트

모두 A4 300 dpi. **100 % 배율**(실제 크기)로 레이저 인쇄하고, 아래쪽 100 mm 선을 자로 재서 배율을 확인한다.
글자와 보조선은 얇은 회색이라 카메라에서는 거의 안 보인다. 다시 만들 때의 명령은 각 항목에 적었다.

| 폴더 | 내용 | 용도 |
|---|---|---|
| `shapes/` | `shapes_filled.png` 속이 찬 도형 6개, `shapes_outline.png` 선 도형 6개 (60 mm) | 도형 인식(Detect `shapes`) 시험. `python shape_sheet.py` |
| `dots/` | `dots_markers.png`(+`_p2`) 다섯 가지 점 마커 × 지름 4/6/8 mm | 점 마커 종류별 인식 확인. `python dot_sheet.py markers` |
| | `dots_sizes.png`(~`_p4`) L 마커를 지름 2~12 mm × 잉크 농도 0/90/150 으로 | 거리별 최소 점 크기, 회색 잉크 한계 재기. `python dot_sheet.py sizes --grays 0 90 150` |
| `cube/faces_5/` | **실제 쓸 큐브 5면** (80 mm, 점 4 mm, 그림 자리 38 mm). 모서리 점 개수 조합 1-1-1-1 / 1-1-1-2 / 1-1-2-1 / 1-2-1-1 / 2-1-1-1 | `faces_5.png`, `_p2.png` 는 A4 시험·재단용, `single/face_N_*.png` 는 점만 있는 80 mm 낱장 → 인쇄소에 그대로 |
| `cube/test_2m_for_2.7m/` | 위 5면을 **2.7 m 용을 2 m 에서 시험하도록 2/2.7 배로 줄인** 시트. 장마다 실제 점 지름 3 / 4 / 5 / 6 / 8 mm | 점 크기 결정용. 2 m 에서 찍어 어느 장까지 20 마커가 다 잡히는지 본다 (합성 예측: 4 mm 부터) |
| `cube/faces_kinds/` | 다섯 마커 종류(1, 2, 2s, 3I, 3L)를 각각 네 모서리에 둔 면 + 3L/1 섞은 면 | 모서리 마커 종류 비교용 |
| `cube/combos12/` | 모서리마다 점 1개/2개인 16가지 조합 전부 | 면 ID 를 더 늘릴 때 |

큐브 폴더 공통: `<이름>.png` (A4, 면 사이 6 d 이상 띄움, 변 중앙 눈금에 자를 대고 재단) 와 `single/face_N_<TL-TR-BR-BL>.png`
(점만, 정확히 80 mm). 이름의 숫자는 모서리별 점 개수이고 순서는 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래.
가운데 점선 상자(38 mm)에 그림을 넣는다. 그림의 어두운 부분은 점에서 묶기 거리(3.5 × 점 지름 = 14 mm) 안에 들어오면 안 된다.

```
python dot_sheet.py cube --faces 1,1,1,1 1,1,1,2 1,1,2,1 1,2,1,1 2,1,1,1 --stem faces_5
python dot_sheet.py cube --faces 1,1,1,1 1,1,1,2 1,1,2,1 1,2,1,1 2,1,1,1 --dot 3 4 5 6 8 --distance 2.7 2 --no-singles --stem test_2m_for_2.7m
python dot_sheet.py cube --stem faces_kinds
python dot_sheet.py cube --faces combos12 --stem combos12
```

뷰어: `python fov_viewer.py --device dmk --detect dots --detect-width 0 --tag-confirm 3` (축소 시험 시트는 Dot min px 3).
