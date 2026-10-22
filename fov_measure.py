"""(c) 실측 FOV: 공장 캘리브레이션 값과 독립적으로 화각을 측정해 비교한다.

서브커맨드
  checker  체커보드 + 거리로 초점거리를 직접 추정 → FOV.  여러 장 모으면 OpenCV 캘리브레이션도 수행.
           (depth 는 IR 과 같은 센서/광학계라 --stream ir 결과가 depth FOV 이다)
  valid    depth 유효 픽셀 영역의 실제 각도 범위 (WFOV 원형/육각 영역, 가장자리 손실 확인)
  manual   줄자 실측: 거리와 화면 끝~끝 폭/높이를 재서 FOV 계산 (--live 로 가이드 화면 제공)

    python fov_measure.py checker --stream color --board 9x6 --square 25
    python fov_measure.py checker --stream ir --depth 1024x1024@15 --board 9x6 --square 25
    python fov_measure.py valid --depth 1024x1024@15 --frames 60
    python fov_measure.py manual --distance 1000 --width 1680 --height 1070
    python fov_measure.py manual --distance 1000 --width 1680 --distance2 2000 --width2 3360
    python fov_measure.py manual --live --stream ir --depth 640x576

checker 키:  SPACE 샘플 추가   c 캘리브레이션 실행   r 초기화   s 결과 저장   q/ESC 종료
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from femto_fov import viz
from femto_fov.device import (
    AlignFilter,
    Config,
    OBStreamType,
    color_to_bgr,
    depth_to_mm,
    device_summary,
    find_profile,
    ir_to_u16,
    mode_name,
    open_pipeline,
    parse_res,
    to_intrinsics,
)
from femto_fov.fov_math import (
    Fov,
    Intrinsics,
    focal_from_checkerboard,
    fov_from_focal,
    fov_from_physical,
    fov_pinhole,
    fov_undistorted,
    valid_region_fov,
)

WIN = "Femto FOV measure"


# ---------------------------------------------------------------------------
# 공통: 스트림 열기
# ---------------------------------------------------------------------------


class Streams:
    """측정 대상 스트림(color 또는 ir) + 거리용 depth 를 함께 연다."""

    def __init__(self, target: str, color_res: str | None, depth_res: str | None, need_depth: bool = True):
        self.pipeline = open_pipeline()
        self.target = target
        print(device_summary(self.pipeline))
        dw, dh, dfps = parse_res(depth_res)
        self.depth_p = find_profile(self.pipeline, "depth", dw, dh, dfps)
        config = Config()
        if target == "color":
            cw, ch, cfps = parse_res(color_res)
            self.target_p = find_profile(self.pipeline, "color", cw, ch, cfps)
            config.enable_stream(self.target_p)
        elif target == "ir":
            # Femto 의 IR 은 depth 와 같은 모드여야 동시 스트리밍 가능
            self.target_p = find_profile(
                self.pipeline, "ir", self.depth_p.get_width(), self.depth_p.get_height(), self.depth_p.get_fps()
            )
            config.enable_stream(self.target_p)
        else:
            self.target_p = self.depth_p
        if need_depth or target == "depth":
            config.enable_stream(self.depth_p)
        self.intr = to_intrinsics(self.target_p)
        self.depth_intr = to_intrinsics(self.depth_p)
        p = self.target_p
        print(f"  target {target}: {p.get_width()}x{p.get_height()}@{p.get_fps()} {p.get_format().name}")
        print(f"  depth       : {self.depth_p.get_width()}x{self.depth_p.get_height()}@{self.depth_p.get_fps()}")
        try:
            self.pipeline.enable_frame_sync()
        except Exception:
            pass
        self.pipeline.start(config)
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM) if target == "color" else None

    def read(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """(표시용 BGR, 검출용 gray, target 픽셀 격자에 맞춘 depth mm)"""
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            return None, None, None
        depth = None
        if self.target == "color":
            cf = frames.get_color_frame()
            if cf is None:
                return None, None, None
            bgr = color_to_bgr(cf)
            if bgr is None:
                return None, None, None
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if frames.get_depth_frame() is not None:
                try:
                    out = self.align.process(frames)
                    fs = out.as_frame_set() if hasattr(out, "as_frame_set") else out
                    adf = fs.get_depth_frame() if fs is not None else None
                    if adf is not None:
                        depth = depth_to_mm(adf)
                        if depth.shape != gray.shape:
                            depth = None
                except Exception:
                    depth = None
            return bgr, gray, depth
        # IR / depth: 같은 센서 → 같은 픽셀 격자
        df = frames.get_depth_frame()
        if df is not None:
            depth = depth_to_mm(df)
        if self.target == "depth":
            if depth is None:
                return None, None, None
            return viz.colorize_depth(depth), None, depth
        irf = frames.get_ir_frame()
        if irf is None:
            return None, None, None
        bgr = viz.normalize_ir(ir_to_u16(irf))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if depth is not None and depth.shape != gray.shape:
            depth = None
        return bgr, gray, depth

    def stop(self) -> None:
        self.pipeline.stop()


def factory_fovs(intr: Intrinsics) -> tuple[Fov, Fov]:
    return fov_pinhole(intr), fov_undistorted(intr)


def sample_depth(depth: np.ndarray, pts: np.ndarray, r: int = 3) -> np.ndarray:
    h, w = depth.shape
    out = np.zeros(len(pts))
    for i, (x, y) in enumerate(np.round(pts).astype(int)):
        patch = depth[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)]
        v = patch[patch > 0]
        out[i] = np.median(v) if v.size else 0
    return out


# ---------------------------------------------------------------------------
# checker
# ---------------------------------------------------------------------------


def detect_board(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    h, w = gray.shape
    scale = min(1.0, 1280 / w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else gray
    ok, corners = cv2.findChessboardCornersSB(small, (cols, rows), cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    corners = corners.reshape(-1, 2) / scale
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
    win = max(3, int(5 / scale))
    refined = cv2.cornerSubPix(gray, corners.reshape(-1, 1, 2).astype(np.float32), (win, win), (-1, -1), crit)
    return refined.reshape(-1, 2)


def cmd_checker(args) -> None:
    cols, rows = map(int, args.board.lower().split("x"))
    st = Streams(args.stream, args.color, args.depth)
    intr = st.intr
    f_ph, f_ud = factory_fovs(intr)
    rational = any(abs(v) > 1e-9 for v in intr.dist[5:8])

    samples: list[dict] = []
    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    calib: dict | None = None
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 800)
    try:
        while True:
            bgr, gray, depth = st.read()
            if bgr is None:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue
            corners = detect_board(gray, cols, rows)
            view = bgr.copy()
            live = None
            if corners is not None:
                cv2.drawChessboardCorners(view, (cols, rows), corners.reshape(-1, 1, 2).astype(np.float32), True)
                z = _corner_depth(args, depth, corners)
                if z is not None:
                    try:
                        fx, fy, tilt = focal_from_checkerboard(corners, cols, rows, args.square, z)
                        live = (fx, fy, tilt, float(np.median(z[z > 0])))
                    except ValueError:
                        live = None

            s = 0.6 * intr.width / 1280
            lines = [
                f"{args.stream.upper()} {intr.width}x{intr.height} ({mode_name(args.stream, intr.width, intr.height)})"
                f"  board {cols}x{rows} @ {args.square}mm",
                f"factory pinhole  {f_ph.short()}   (fx {intr.fx:.1f}, fy {intr.fy:.1f})",
                f"factory undist   {f_ud.short()}",
            ]
            if live:
                fx, fy, tilt, zm = live
                lf = fov_from_focal(intr.width, intr.height, fx, fy)
                warn = "  <-- board tilted, face camera" if tilt > args.max_tilt else ""
                lines += [
                    f"live: Z {zm:.0f}mm  fx {fx:.1f} fy {fy:.1f}  tilt {tilt * 100:.1f}%{warn}",
                    f"live pinhole     {lf.short()}",
                ]
            elif corners is not None:
                lines.append("board found but no depth on it (use --distance)")
            else:
                lines.append("board not found")
            if samples:
                fx = np.median([x["fx"] for x in samples])
                fy = np.median([x["fy"] for x in samples])
                mf = fov_from_focal(intr.width, intr.height, fx, fy)
                lines += [
                    f"samples {len(samples)}: median fx {fx:.1f} fy {fy:.1f}",
                    f"measured pinhole {mf.short()}   dH {mf.h - f_ph.h:+.2f}  dV {mf.v - f_ph.v:+.2f}",
                ]
            if calib:
                lines += [
                    f"calib ({calib['n']} views, rms {calib['rms']:.3f}px) fx {calib['fx']:.1f} fy {calib['fy']:.1f}",
                    f"calib undist     {calib['fov_undist'].short()}",
                ]
            lines.append("SPACE add  c calibrate  r reset  s save  q quit")
            viz.put_lines(view, lines, scale=s)
            cv2.imshow(WIN, view)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == ord(" ") and corners is not None:
                obj_pts.append(objp.copy())
                img_pts.append(corners.astype(np.float32).reshape(-1, 1, 2))
                if live and live[2] <= args.max_tilt:
                    samples.append({"fx": live[0], "fy": live[1], "tilt": live[2], "z": live[3]})
                    print(f"샘플 {len(samples)}: Z={live[3]:.0f}mm fx={live[0]:.2f} fy={live[1]:.2f}")
                else:
                    print(f"캘리브레이션용 뷰만 추가 ({len(img_pts)}장) — 거리/기울기 조건 미달로 초점 추정 제외")
            elif k == ord("c"):
                calib = run_calibration(obj_pts, img_pts, intr, rational)
            elif k == ord("r"):
                samples.clear()
                obj_pts.clear()
                img_pts.clear()
                calib = None
            elif k == ord("s"):
                save_checker(args, st, intr, f_ph, f_ud, samples, calib)
    finally:
        st.stop()
        cv2.destroyAllWindows()
    if samples or calib:
        save_checker(args, st, intr, f_ph, f_ud, samples, calib)


def _corner_depth(args, depth: np.ndarray | None, corners: np.ndarray) -> np.ndarray | None:
    if args.distance:
        return np.full(len(corners), float(args.distance))
    if depth is None:
        return None
    z = sample_depth(depth, corners)
    if (z > 0).mean() < 0.5:
        return None
    return z


def run_calibration(obj_pts, img_pts, intr: Intrinsics, rational: bool) -> dict | None:
    if len(img_pts) < 5:
        print(f"캘리브레이션에는 최소 5장이 필요합니다 (현재 {len(img_pts)}장). 보드를 화면 여러 위치/각도로 두고 SPACE.")
        return None
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    print(f"캘리브레이션 중... ({len(img_pts)}장, {'rational' if rational else 'standard'} 모델)")
    rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (intr.width, intr.height), None, None, flags=flags)
    d = D.ravel().tolist() + [0.0] * 8
    ci = Intrinsics(intr.width, intr.height, K[0, 0], K[1, 1], K[0, 2], K[1, 2], tuple(d[:8]))
    res = {
        "n": len(img_pts),
        "rms": float(rms),
        "fx": ci.fx,
        "fy": ci.fy,
        "cx": ci.cx,
        "cy": ci.cy,
        "dist": [round(v, 6) for v in ci.dist],
        "fov_pinhole": fov_pinhole(ci),
        "fov_undist": fov_undistorted(ci),
    }
    print(f"  rms={rms:.3f}px  fx={ci.fx:.2f} fy={ci.fy:.2f} cx={ci.cx:.2f} cy={ci.cy:.2f}")
    print(f"  calib pinhole   {res['fov_pinhole']}")
    print(f"  calib undist    {res['fov_undist']}")
    print(f"  factory         fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.cx:.2f} cy={intr.cy:.2f}")
    return res


def save_checker(args, st: Streams, intr, f_ph, f_ud, samples, calib) -> None:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "device": device_summary(st.pipeline),
        "stream": args.stream,
        "resolution": [intr.width, intr.height],
        "mode": mode_name(args.stream, intr.width, intr.height),
        "board": args.board,
        "square_mm": args.square,
        "factory": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
                    "pinhole": f_ph.as_dict(), "undistorted": f_ud.as_dict()},
        "samples": samples,
    }
    if samples:
        fx = float(np.median([x["fx"] for x in samples]))
        fy = float(np.median([x["fy"] for x in samples]))
        data["measured"] = {"fx": fx, "fy": fy, "pinhole": fov_from_focal(intr.width, intr.height, fx, fy).as_dict()}
    if calib:
        data["calibration"] = {
            **{k: v for k, v in calib.items() if not k.startswith("fov")},
            "pinhole": calib["fov_pinhole"].as_dict(),
            "undistorted": calib["fov_undist"].as_dict(),
        }
    p = out / f"checker_{args.stream}_{intr.width}x{intr.height}_{stamp}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {p}")


# ---------------------------------------------------------------------------
# valid
# ---------------------------------------------------------------------------


def cmd_valid(args) -> None:
    st = Streams("depth", None, args.depth)
    intr = st.depth_intr
    f_ph, f_ud = factory_fovs(intr)
    acc = np.zeros((intr.height, intr.width), np.uint16)
    n = 0
    print(f"{args.frames} 프레임 누적 중... 넓은 실내(벽/바닥/천장이 2~4m 안에 있는 곳)를 향하게 두세요.")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    try:
        while n < args.frames:
            _, _, depth = st.read()
            if depth is None:
                continue
            acc += depth > 0
            n += 1
            view = viz.colorize_depth(depth)
            viz.put_lines(view, [f"{n}/{args.frames}"], 0.5 * intr.width / 640)
            cv2.imshow(WIN, view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        st.stop()
    if n == 0:
        print("프레임을 받지 못했습니다.")
        return
    ratio = acc / n
    mask = ratio >= args.min_ratio
    vf = valid_region_fov(mask, intr)
    print()
    print(f"DEPTH {intr.width}x{intr.height} ({mode_name('depth', intr.width, intr.height)}), {n} frames")
    print(f"  factory pinhole  {f_ph}")
    print(f"  factory undist   {f_ud}")
    print(f"  valid region     {vf}   (픽셀이 {args.min_ratio * 100:.0f}% 이상 프레임에서 유효)")
    print(f"  유효 픽셀 비율   {mask.mean() * 100:.1f}% of sensor")

    view = viz.colorize_depth(np.where(mask, 1500.0, 0.0).astype(np.float32))
    viz.draw_angle_grid(view, intr)
    viz.put_lines(view, [f"valid  {vf.short()}", f"undist {f_ud.short()}"], 0.5 * intr.width / 640)
    cv2.imshow(WIN, view)
    print("결과 창에서 아무 키나 누르면 종료합니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.out / f"valid_{intr.width}x{intr.height}_{stamp}"
    cv2.imwrite(str(base) + "_mask.png", (mask * 255).astype(np.uint8))
    cv2.imwrite(str(base) + "_view.png", view)
    Path(str(base) + ".json").write_text(
        json.dumps(
            {
                "resolution": [intr.width, intr.height],
                "mode": mode_name("depth", intr.width, intr.height),
                "frames": n,
                "min_ratio": args.min_ratio,
                "factory_pinhole": f_ph.as_dict(),
                "factory_undistorted": f_ud.as_dict(),
                "valid_region": vf.as_dict(),
                "valid_pixel_pct": round(float(mask.mean() * 100), 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"저장: {base}.*")


# ---------------------------------------------------------------------------
# manual
# ---------------------------------------------------------------------------


def cmd_manual(args) -> None:
    factory = None
    distance = args.distance
    if args.live:
        distance, factory = manual_live(args)
    if not distance:
        distance = float(input("카메라 ~ 측정 평면 거리 (mm): "))
    width = args.width if args.width else float(input("화면 왼쪽 끝 ~ 오른쪽 끝 폭 (mm): "))
    height = args.height
    if height is None and args.live:
        txt = input("화면 위쪽 끝 ~ 아래쪽 끝 높이 (mm, 생략은 Enter): ").strip()
        height = float(txt) if txt else None
    if args.distance2 and args.width2:
        # 두 거리 차분: 광학 중심 ~ 측정 기준면 오프셋이 상쇄된다
        dz = args.distance2 - distance
        height = (args.height2 - height) if (height and args.height2) else None
        m = fov_from_physical(dz, args.width2 - width, height)
        print()
        print(f"차분 측정: Z {distance:.0f} -> {args.distance2:.0f}mm, 폭 {width:.0f} -> {args.width2:.0f}mm")
    else:
        m = fov_from_physical(distance, width, height)
        print()
        print(f"거리 {distance:.0f}mm, 폭 {width:.0f}mm" + (f", 높이 {height:.0f}mm" if height else ""))
    print(f"  measured  H {m.h:6.2f}°" + (f"  V {m.v:6.2f}°  D {m.d:6.2f}°" if height else ""))
    if factory:
        f_ph, f_ud = factory
        print(f"  factory pinhole  {f_ph}")
        print(f"  factory undist   {f_ud}")
        print(f"  차이 (measured - undist)  dH {m.h - f_ud.h:+.2f}°" + (f"  dV {m.v - f_ud.v:+.2f}°" if height else ""))
    print(
        "\n※ 거리는 카메라 앞면이 아니라 광학 중심 기준입니다. 정확도를 높이려면 두 거리(예: 1m, 2m)에서 재고"
        "\n   폭 차이 / 거리 차이 로 계산하면 광학 중심 오프셋이 상쇄됩니다:"
        "\n   FOV = 2·atan((W2 - W1) / (2·(Z2 - Z1)))   → --distance2 --width2 [--height2]"
    )


def manual_live(args) -> tuple[float | None, tuple[Fov, Fov]]:
    """가이드 화면: 화면 끝/중앙 표시 + 중앙 depth. 화면 가장자리에 마커를 맞춘 뒤 q."""
    st = Streams(args.stream, args.color, args.depth)
    intr = st.intr
    factory = factory_fovs(intr)
    last_z = None
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print("화면 좌/우(상/하) 끝 선에 물체(테이프 등)를 맞춘 뒤 q 를 누르세요.")
    try:
        while True:
            bgr, _, depth = st.read()
            if bgr is None:
                continue
            view = bgr.copy()
            h, w = view.shape[:2]
            t = max(2, w // 400)
            cv2.rectangle(view, (0, 0), (w - 1, h - 1), viz.YELLOW, t * 2)
            cx, cy = int(intr.cx), int(intr.cy)
            cv2.line(view, (cx, 0), (cx, h), viz.GREEN, t)
            cv2.line(view, (0, cy), (w, cy), viz.GREEN, t)
            for x in (0, w - 1):
                cv2.line(view, (x, cy - h // 10), (x, cy + h // 10), viz.MAGENTA, t * 3)
            for y in (0, h - 1):
                cv2.line(view, (cx - w // 10, y), (cx + w // 10, y), viz.MAGENTA, t * 3)
            if depth is not None:
                z = sample_depth(depth, np.array([[intr.cx, intr.cy]]), r=max(3, w // 200))[0]
                if z > 0:
                    last_z = float(z)
            viz.put_lines(
                view,
                [
                    f"{args.stream.upper()} {w}x{h}  factory undist {factory[1].short()}",
                    f"center depth: {last_z:.0f} mm" if last_z else "center depth: -",
                    "align markers to magenta edge ticks, then q",
                ],
                0.6 * w / 1280,
            )
            cv2.imshow(WIN, view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        st.stop()
        cv2.destroyAllWindows()
    distance = args.distance or last_z
    if distance and not args.distance:
        print(f"중앙 depth {distance:.0f}mm 를 거리로 사용합니다 (--distance 로 덮어쓰기 가능).")
    return distance, factory


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, stream=True):
        if stream:
            p.add_argument("--stream", choices=["color", "ir"], default="color", help="측정할 스트림")
            p.add_argument("--color", help="color 해상도 WxH[@fps]")
        p.add_argument("--depth", help="depth(=IR) 해상도 WxH[@fps]")
        p.add_argument("--out", type=Path, default=Path("results"))

    p = sub.add_parser("checker", help="체커보드 실측 + 캘리브레이션")
    common(p)
    p.add_argument("--board", default="9x6", help="내부 코너 수 COLSxROWS (기본 9x6)")
    p.add_argument("--square", type=float, required=True, help="한 칸 크기 mm (자로 실측한 값)")
    p.add_argument("--distance", type=float, help="depth 대신 사용할 고정 거리 mm (광학 중심 ~ 보드)")
    p.add_argument("--max-tilt", type=float, default=0.03, help="허용 보드 기울기 (Z 편차/평균, 기본 0.03)")
    p.set_defaults(func=cmd_checker)

    p = sub.add_parser("valid", help="depth 유효 영역 FOV")
    common(p, stream=False)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--min-ratio", type=float, default=0.5, help="이 비율 이상 프레임에서 유효해야 유효 픽셀로 판정")
    p.set_defaults(func=cmd_valid)

    p = sub.add_parser("manual", help="줄자 실측 계산")
    common(p)
    p.add_argument("--live", action="store_true", help="가이드 화면을 띄우고 공장값과 비교")
    p.add_argument("--distance", type=float, help="광학 중심 ~ 평면 거리 mm")
    p.add_argument("--width", type=float, help="화면 끝~끝 폭 mm")
    p.add_argument("--height", type=float, help="화면 끝~끝 높이 mm")
    p.add_argument("--distance2", type=float, help="두 번째 거리 mm (차분 측정)")
    p.add_argument("--width2", type=float, help="두 번째 거리에서의 폭 mm")
    p.add_argument("--height2", type=float, help="두 번째 거리에서의 높이 mm")
    p.set_defaults(func=cmd_manual)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
