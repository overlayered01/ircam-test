"""저장된 영상(뷰어 스냅샷, 사진)에서 AprilTag / 도형을 찾아 결과를 출력하고 오버레이 PNG 를 저장한다 (장치 불필요).

    python detect_image.py results/viewer_20261029_180000_ir16.png
    python detect_image.py photo.png --family tag16h5 --invert --shapes
    python detect_image.py *.png --no-tags --shapes --out results/detect

16비트 PNG(Femto IR 스냅샷)는 ir_detect.preprocess.to_u8 로 8비트로 바꿔 검출한다.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2

from ir_detect import detect as det
from ir_detect.preprocess import to_u8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="영상 파일 (와일드카드 가능)")
    ap.add_argument("--family", default="tag36h11", choices=det.FAMILIES)
    ap.add_argument("--backend", default="auto", choices=["auto", "opencv", "pupil"])
    ap.add_argument("--no-tags", action="store_true")
    ap.add_argument("--shapes", action="store_true", help="도형 검출도 수행")
    ap.add_argument("--shape-type", default="all", choices=det.SHAPE_TYPES, help="찾을 도형 종류")
    ap.add_argument("--regular", action="store_true", help="정형(정삼각형·정사각형·정십자·원…)만")
    ap.add_argument("--tol", type=float, default=0.1, help="정형 허용 편차 비율 (기본 0.1 = 10%%)")
    ap.add_argument("--min-px", type=int, default=20, help="도형 최소 크기 (외접 사각형 짧은 변, 픽셀)")
    ap.add_argument("--shape-mode", default="both", choices=det.SHAPE_MODES, help="filled = 채워진 도형, outline = 선 도형, both")
    ap.add_argument("--invert", action="store_true", help="검출 전 영상 반전 (흰 테두리 태그 / 어두운 바탕의 밝은 도형)")
    ap.add_argument("--max-width", type=int, default=1280, help="검출용 축소 폭 (0 = 원본 그대로)")
    ap.add_argument("--out", type=Path, default=None, help="오버레이 PNG + JSON 저장 폴더 (기본: 입력 옆 *_det.png)")
    a = ap.parse_args()

    files = [Path(f) for pat in a.files for f in (glob.glob(pat) or [pat])]
    pipe = det.Pipeline(a.backend, a.max_width or 10**9)
    bad = 0
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"{f}: 읽기 실패", file=sys.stderr)
            bad += 1
            continue
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.dtype != "uint8":
            gray = to_u8(gray)
        opts = det.ShapeOptions(a.shape_type, a.regular, a.tol, a.min_px, a.shape_mode)
        res = pipe.run(gray, tags=not a.no_tags, shapes=a.shapes, family=a.family, invert=a.invert, shape_options=opts)
        print(f"{f.name}: {res.summary()}")
        for t in res.tags:
            print(f"   tag {t.id:4d}  center=({t.center[0]:.0f},{t.center[1]:.0f})  size={t.size_px:.0f}px"
                  + (f"  margin={t.margin:.0f}" if t.margin is not None else ""))
        for s in res.shapes:
            print(f"   {s.name:9s} center=({s.center[0]:.0f},{s.center[1]:.0f})  size={s.size_px:.0f}px  dev={s.dev * 100:.0f}%")
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        det.draw(canvas, res, 1.0, thickness=max(2, gray.shape[1] // 800))
        out_dir = a.out or f.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{f.stem}_det.png"), canvas)
        meta = {"file": str(f), "family": a.family, "invert": a.invert, "backend": res.backend, "ms": round(res.ms, 1),
                "tags": [{"id": t.id, "center": t.center, "size_px": t.size_px, "margin": t.margin} for t in res.tags],
                "shapes": [{"name": s.name, "center": s.center, "area_px": s.area} for s in res.shapes]}
        (out_dir / f"{f.stem}_det.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
