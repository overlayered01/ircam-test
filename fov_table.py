"""(a) 이론 FOV 표: 장치가 지원하는 모든 color / depth / IR 해상도의 내부 파라미터로 FOV 계산.

스트리밍 없이 공장 캘리브레이션 값만 읽는다.

    python fov_table.py                 # 표 출력
    python fov_table.py --out results   # results/fov_table.csv, .json 저장
    python fov_table.py --all-fps       # fps/포맷별 프로파일까지 모두 출력
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from femto_fov.device import (
    device_summary,
    list_profiles,
    mode_name,
    open_pipeline,
    to_intrinsics,
    unique_resolutions,
)
from femto_fov.fov_math import fov_pinhole, fov_undistorted
from femto_fov.specs import nominal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="CSV/JSON 저장 폴더")
    ap.add_argument("--all-fps", action="store_true", help="해상도별 대표 1개가 아닌 모든 프로파일 출력")
    args = ap.parse_args()

    pipeline = open_pipeline()
    print(device_summary(pipeline))
    print()

    rows = []
    for sensor in ("color", "ir", "depth"):
        profiles = list_profiles(pipeline, sensor)
        if not profiles:
            print(f"[{sensor}] 지원 프로파일 없음\n")
            continue
        if not args.all_fps:
            profiles = unique_resolutions(profiles)
        for p in profiles:
            intr = to_intrinsics(p.profile)
            if intr.fx <= 0 or intr.fy <= 0:
                print(f"  {sensor} {p.width}x{p.height}: 내부 파라미터 없음 (fx=0)")
                continue
            ph = fov_pinhole(intr)
            ud = fov_undistorted(intr)
            mode = mode_name(sensor, p.width, p.height)
            nom = nominal(sensor, mode)
            rows.append(
                {
                    "sensor": sensor,
                    "width": p.width,
                    "height": p.height,
                    "fps": p.fps,
                    "format": p.fmt,
                    "mode": mode,
                    "fx": round(intr.fx, 3),
                    "fy": round(intr.fy, 3),
                    "cx": round(intr.cx, 3),
                    "cy": round(intr.cy, 3),
                    "dist_model": intr.model,
                    "dist": [round(v, 6) for v in intr.dist],
                    "pinhole_h": round(ph.h, 2),
                    "pinhole_v": round(ph.v, 2),
                    "pinhole_d": round(ph.d, 2),
                    "undist_h": round(ud.h, 2),
                    "undist_v": round(ud.v, 2),
                    "undist_d": round(ud.d, 2),
                    "nominal_h": nom[0] if nom else None,
                    "nominal_v": nom[1] if nom else None,
                }
            )

    print_table(rows)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jpath = args.out / f"fov_table_{stamp}.json"
        cpath = args.out / f"fov_table_{stamp}.csv"
        jpath.write_text(
            json.dumps({"device": device_summary(pipeline), "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with cpath.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({**r, "dist": " ".join(map(str, r["dist"]))})
        print(f"\n저장: {jpath}\n      {cpath}")


def print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'sensor':6} {'resolution':>11} {'fps':>3} {'fmt':6} {'mode':16} "
        f"{'fx':>8} {'fy':>8} | {'pinhole H×V (D)':>22} | {'undistorted H×V (D)':>22} | {'nominal H×V':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    last = None
    for r in rows:
        if last and last != r["sensor"]:
            print()
        last = r["sensor"]
        nom = f"{r['nominal_h']:.0f}×{r['nominal_v']:.0f}" if r["nominal_h"] else "-"
        print(
            f"{r['sensor']:6} {str(r['width']) + 'x' + str(r['height']):>11} {r['fps']:>3} {r['format'][:6]:6} "
            f"{r['mode'][:16]:16} {r['fx']:8.2f} {r['fy']:8.2f} | "
            f"{fmt_fov(r['pinhole_h'], r['pinhole_v'], r['pinhole_d']):>22} | "
            f"{fmt_fov(r['undist_h'], r['undist_v'], r['undist_d']):>22} | {nom:>12}"
        )
    print(
        "\n* pinhole: fx/fy 만으로 계산 (왜곡 무시)   * undistorted: 렌즈 왜곡을 반영한 이미지 끝 광선 사이 각도"
        "\n* nominal: femto_fov/specs.py 의 데이터시트 공칭값"
    )


def fmt_fov(h: float, v: float, d: float) -> str:
    if any(math.isnan(x) for x in (h, v, d)):
        return "n/a"
    return f"{h:6.2f}×{v:6.2f} ({d:6.2f})"


if __name__ == "__main__":
    main()
