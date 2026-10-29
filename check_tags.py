"""april_tags/ 의 AprilTag PNG 가 올바른 ID 인지, 검출기가 실제로 읽는지 확인 (장치 불필요).

    python check_tags.py                                  # april_tags/*.png
    python check_tags.py april_tags/tag16_05_00001_inv.png
    python check_tags.py --dir other_dir --family tag36h11

파일명 tag16_05_00001*.png 에서 계열(16h5)과 기대 ID(1) 를 읽는다. OpenCV ArUco 는 항상,
pupil-apriltags 는 설치되어 있으면 함께 돌린다 (requirements.txt 에 포함).

반전(inv) 태그는 검출 전에 영상을 반전하고, 바깥 흰 여백은 떼고(테두리 1셀) 검출한다.
"파일 그대로" / "여백 포함" 줄은 참고용이라 실패해도 결과에 포함하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from ir_detect.tags import AprilTagDetector, OpenCVDetector, degrade, identify, ids_of, load_tag, render, warp

INFO_ONLY = ("파일 그대로", "반전, 여백 포함")  # inv 태그에서는 실패가 정상인 경우


def make_detectors(family: str) -> list:
    dets = [OpenCVDetector(family)]
    try:
        dets.append(AprilTagDetector(family))
    except ImportError:
        print(" (pupil-apriltags 미설치 → OpenCV 만 사용)")
    return dets


def check_file(path: Path, family: str | None, cache: dict) -> bool:
    tag = load_tag(path, family)
    n = tag.n_data
    print(f"\n=== {path.name}  [{tag.family}, {tag.cells.shape[0]}x{tag.cells.shape[1]} cells, expected id {tag.expected_id}]")
    for row in tag.cells:
        print("   " + "".join("#" if v else "." for v in row))

    matches = identify(tag)
    print(" bit pattern:", ", ".join(f"id {i} {pol} rot{r}" for i, pol, r in matches) or "일치하는 ID 없음")
    ok_bits = tag.expected_id is None or any(i == tag.expected_id for i, _, _ in matches)
    inv = tag.is_inverted()

    tag1 = tag.tag_only(1)  # 테두리 1셀
    full = tag.cells  # 여백 포함
    if inv:
        tag1, full = 255 - tag1, 255 - full
    std = render(tag1)

    if tag.family not in cache:
        cache[tag.family] = make_detectors(tag.family)

    ok_detect = True
    for det in cache[tag.family]:
        print(f" --- {det.name}")
        pre = "반전, " if inv else ""
        cases = [
            ("파일 그대로 (반전 없음)", render(tag.cells, bg=255 - int(tag.cells[0, 0]))),
            (f"{pre}여백 포함 (테두리 2셀)" if inv else "여백 포함 (흰 여백 1셀)", render(full)),
            (f"{pre}테두리 1셀", std),
            ("  + 회전 90", cv2.rotate(std, cv2.ROTATE_90_CLOCKWISE)),
            ("  + 원근", warp(std)),
            ("  + 원근·50% 대비·노이즈", degrade(warp(std))),
        ] + [(f"  셀당 {c}px (태그 {(n + 2) * c}px)", render(tag1, px_per_cell=c)) for c in (6, 4, 3, 2)]
        for label, img in cases:
            r = det.detect(img)
            clean = ids_of(r)
            hit = tag.expected_id in clean
            info = label.startswith(INFO_ONLY)
            mark = "O" if hit else ("-" if info else "X")
            if not hit and not info:
                ok_detect = False
            extra = f"  (hamming>0 {len(r) - len(clean)}개 제외)" if len(r) != len(clean) else ""
            print(f"  {mark} {label:30s} {clean}{extra}")

    ok = ok_bits and ok_detect
    note = "  (inv 태그: 검출 전 영상 반전, 바깥 흰 여백은 잘라내거나 검게)" if inv else ""
    print(" 결과:", ("OK" if ok else "FAIL") + note)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="태그 PNG (생략하면 --dir 의 *.png)")
    ap.add_argument("--dir", default="april_tags")
    ap.add_argument("--family", default=None, help="파일명에서 못 읽을 때: tag16h5 / tag25h9 / tag36h10 / tag36h11")
    a = ap.parse_args()
    files = [Path(f) for f in a.files] or sorted(Path(a.dir).glob("*.png"))
    if not files:
        print("태그 파일 없음", file=sys.stderr)
        return 2
    cache: dict = {}
    results = [check_file(f, a.family, cache) for f in files]
    print(f"\n{sum(results)}/{len(results)} OK")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
