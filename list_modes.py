"""카메라별 IR 모드 전체 목록. --probe 로 각 모드를 실제로 열어 동작 여부·첫 프레임 시간·실측 fps 확인.

    python list_modes.py                         # 연결된 Orbbec 장치 전부 (Femto Bolt / Mega USB)
    python list_modes.py --serial CL8764100GL    # 특정 장치
    python list_modes.py --ip 192.168.1.10       # Femto Mega 이더넷
    python list_modes.py --probe --out results   # 모드별 실측 후 results/modes_*.csv, .json 저장
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from femto_fov.fov_math import fov_undistorted
from ir_detect.sources import CameraSource, Mode, SourceUnavailable
from ir_detect.sources.orbbec_ir import OrbbecIRSource


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["orbbec"], default="orbbec")
    ap.add_argument("--serial", default="", help="장치 S/N (생략하면 연결된 장치 전부)")
    ap.add_argument("--ip", default="", help="네트워크 장치 IP (Femto Mega 이더넷)")
    ap.add_argument("--port", type=int, default=0, help="네트워크 장치 포트 (기본 8090)")
    ap.add_argument("--probe", action="store_true", help="모드마다 스트림을 열어 실측")
    ap.add_argument("--frames", type=int, default=15, help="--probe 때 모드당 받을 프레임 수")
    ap.add_argument("--out", type=Path, help="CSV/JSON 저장 폴더")
    args = ap.parse_args()

    ok, why = OrbbecIRSource.available()
    if not ok:
        sys.exit(why)
    if args.serial or args.ip:
        sources: list[CameraSource] = [OrbbecIRSource("Femto", args.serial, args.ip, args.port)]
    else:
        sources = OrbbecIRSource.discover()
        if not sources:
            sys.exit("Orbbec 장치를 찾을 수 없습니다. USB 연결을 확인하거나 --ip 로 네트워크 장치를 지정하세요.")

    for src in sources:
        try:
            src.open()
        except SourceUnavailable as e:
            print(f"[{src.name}] {e}\n")
            continue
        try:
            info = src.info()
            print("  ".join(f"{k}={v}" for k, v in info.items()))
            rows = [describe(src, m) for m in src.modes()]
            if args.probe:
                for r, m in zip(rows, src.modes()):
                    print(f"  probe {m} ...", end="", flush=True)
                    r.update(probe(src, m, args.frames))
                    print(" ok" if r["ok"] else f" 실패: {r['error']}")
            print()
            print_table(rows, args.probe)
            print()
            if args.out:
                save(args.out, info, rows)
        finally:
            src.close()


def describe(src: CameraSource, m: Mode) -> dict:
    row = {
        "width": m.width,
        "height": m.height,
        "fps": m.fps,
        "format": m.pixel_format,
        "label": m.label,
        "params": dict(m.params),
        "fov_h": math.nan,
        "fov_v": math.nan,
    }
    intr = src.intrinsics(m)
    if intr is not None and intr.fx > 0:
        f = fov_undistorted(intr)
        row.update(fov_h=f.h, fov_v=f.v, fx=intr.fx, fy=intr.fy)
    return row


def probe(src: CameraSource, m: Mode, frames: int) -> dict:
    out = {"ok": False, "error": "", "first_frame_s": math.nan, "measured_fps": math.nan,
           "raw_median": math.nan, "raw_p99": math.nan, "saturated_pct": math.nan}
    t0 = time.perf_counter()
    deadline = t0 + 3.0 + frames / max(m.fps, 1)
    try:
        src.start(m)
        got, t_first, t_last, last = 0, None, None, None
        while got < frames and time.perf_counter() < deadline:
            f = src.read(1000)
            if f is None:
                continue
            t_last = time.perf_counter()
            if t_first is None:
                t_first = t_last
            got, last = got + 1, f
        if last is None:
            out["error"] = "프레임 없음"
            return out
        raw = last.raw
        sat = np.iinfo(raw.dtype).max
        out.update(
            ok=True,
            first_frame_s=t_first - t0,
            measured_fps=(got - 1) / (t_last - t_first) if got > 1 else math.nan,
            raw_median=float(np.median(raw)),
            raw_p99=float(np.percentile(raw, 99)),
            saturated_pct=float((raw >= sat).mean() * 100),
        )
    except Exception as e:  # 모드 하나 실패해도 나머지는 계속
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        src.stop()
    return out


def print_table(rows: list[dict], probed: bool) -> None:
    hdr = f"{'#':>3} {'resolution':>11} {'fps':>4} {'fmt':6} {'label':16} {'FOV H×V':>14}"
    if probed:
        hdr += f" | {'first':>6} {'fps(실측)':>9} {'median':>7} {'p99':>7} {'sat%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows):
        fov = "-" if math.isnan(r["fov_h"]) else f"{r['fov_h']:6.1f}×{r['fov_v']:5.1f}"
        line = (
            f"{i:>3} {str(r['width']) + 'x' + str(r['height']):>11} {r['fps']:>4g} {r['format'][:6]:6} "
            f"{r['label'][:16]:16} {fov:>14}"
        )
        if probed:
            if r["ok"]:
                line += (
                    f" | {r['first_frame_s']:5.2f}s {r['measured_fps']:9.1f} {r['raw_median']:7.0f} "
                    f"{r['raw_p99']:7.0f} {r['saturated_pct']:6.2f}"
                )
            else:
                line += f" | 실패: {r['error']}"
        print(line)
    print("\n* FOV: 공장 캘리브레이션 + 렌즈 왜곡 반영 (undistorted)")
    if probed:
        print("* first: start() 부터 첫 프레임까지   * median / p99: 마지막 프레임 원본값   * sat%: 포화 픽셀 비율")


def save(out: Path, info: dict, rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stem = f"modes_{info.get('serial') or info['name']}_{datetime.now():%Y%m%d_%H%M%S}"
    clean = [{k: None if isinstance(v, float) and math.isnan(v) else v for k, v in r.items()} for r in rows]
    with open(out / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump({"device": info, "modes": clean}, f, ensure_ascii=False, indent=2, default=str)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {out / stem}.csv / .json")


if __name__ == "__main__":
    main()
