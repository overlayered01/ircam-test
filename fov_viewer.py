"""(b) 라이브 FOV 비교 뷰어: RGB / IR / Depth 를 동시에 띄우고 서로의 화각 경계를 겹쳐 그린다.

패널 구성
  [Color + depth 화각 경계]   [IR + color 화각 경계]
     (cyan = depth 센서 사각형 화각, green = depth 값이 실제로 나오는 영역)
  [Depth + color 화각 경계]   [Depth→Color 정렬 블렌드]

  - 경계선은 지정 거리(Z) 평면 기준이다. 시차(baseline) 때문에 거리마다 겹침 영역이 달라진다.
  - 'auto' 모드에서는 화면 중앙의 실제 depth 중앙값을 거리로 사용한다.
  - Depth 패널의 'valid' FOV 는 실제로 깊이 값이 나온 픽셀의 각도 범위 (WFOV 원형 영역 확인용).

    python fov_viewer.py
    python fov_viewer.py --depth 1024x1024@15 --color 1920x1080
    python fov_viewer.py --depth 640x576 --color 3840x2160@15 --distance 1500

키:  +/- 거리 ±100mm   [ ] 거리 ±1000mm   a auto 거리   g 각도 격자   d D2C 정렬 on/off
     m / M  depth 모드 다음/이전 (NFOV binned → NFOV → WFOV binned → WFOV, IR 도 같이 바뀜)
     c / C  color 해상도 다음/이전 (16:9 ↔ 4:3 포함)
     s 스냅샷 저장   q/ESC 종료
  fps 는 두 스트림이 공통으로 지원하는 최댓값으로 자동 선택된다 (WFOV unbinned 는 15fps).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from femto_fov import viz
from femto_fov.device import (
    AlignFilter,
    Config,
    OBError,
    OBStreamType,
    color_to_bgr,
    depth_to_mm,
    device_summary,
    extrinsic_Rt,
    find_profile,
    ir_to_u16,
    list_profiles,
    mode_name,
    open_pipeline,
    parse_res,
    to_intrinsics,
    unique_resolutions,
)
from femto_fov.fov_math import (
    Intrinsics,
    fov_pinhole,
    fov_undistorted,
    footprint_at_distance,
    overlap_ratio,
    project_between,
    project_border,
    valid_region_fov,
)

WIN = "Femto FOV viewer"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--color", help="color 해상도 WxH[@fps] (기본: 장치 기본값)")
    ap.add_argument("--depth", help="depth 해상도 WxH[@fps] (Femto: 640x576 / 320x288 / 1024x1024 / 512x512)")
    ap.add_argument("--no-ir", action="store_true", help="IR 스트림 끄기")
    ap.add_argument("--distance", type=float, default=1000.0, help="경계선 투영 거리 mm (기본 1000)")
    ap.add_argument("--out", type=Path, default=Path("results"), help="스냅샷 저장 폴더")
    ap.add_argument("--cell", default="800x600", help="패널 하나 크기 WxH")
    args = ap.parse_args()

    cell_w, cell_h = map(int, args.cell.lower().split("x"))
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, cell_w * 2, cell_h * 2)
    show_message(cell_w * 2, cell_h * 2, ["Connecting to Femto...", "(device open ~3s + stream start ~2s)"])

    pipeline = open_pipeline()
    summary = device_summary(pipeline)
    print(summary)

    # 전환 가능한 모드 목록 (해상도별 1개)
    depth_modes = sorted(
        (p.key for p in unique_resolutions(list_profiles(pipeline, "depth"))),
        key=lambda r: (mode_name("depth", *r).startswith("WFOV"), r[0] * r[1]),
    )  # NFOV binned → NFOV → WFOV binned → WFOV
    color_modes = [p.key for p in unique_resolutions(list_profiles(pipeline, "color"))]

    st = start_streams(pipeline, parse_res(args.color), parse_res(args.depth), not args.no_ir)
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    switcher = Switcher(pipeline)
    pending, pending_t = None, 0.0  # 연타 묶기: 마지막 키 후 SWITCH_DEBOUNCE 초 지나면 전환 시작
    canvas = None

    distance = args.distance
    auto = False
    show_grid = False
    do_align = True
    last = {"color": None, "ir": None, "depth": None}
    valid_score = None  # depth 유효 픽셀 이동평균 (노이즈로 깜빡이는 픽셀 제거)
    hull = None  # 유효 영역 볼록 껍질 (depth 픽셀)
    vfov = None  # 유효 영역 FOV (10프레임마다 갱신)
    frame_i = 0
    fps_t, fps_n, fps = time.time(), 0, 0.0
    toast, toast_until = "", 0.0  # 모드 전환 안내 메시지

    try:
        while True:
            # --- 모드 전환 처리 (파이프라인 재시작은 드라이버 때문에 ~4초 걸려 백그라운드에서) ---
            if switcher.busy:
                target = pending or switcher.target
                show_message(
                    cell_w * 2, cell_h * 2,
                    [f"Switching to {describe(*target)}  ({time.time() - switcher.started:.1f}s)",
                     "camera driver needs ~4s to restart streams",
                     "m/M c/C: change target   q: quit"],
                    base=canvas,
                )
                k = cv2.waitKey(30) & 0xFF
                if k in (ord("q"), 27):
                    switcher.join()
                    break
                if k in MODE_KEYS:
                    pending = next_mode(k, target, depth_modes, color_modes)
                    pending_t = time.time()
                continue
            if switcher.result is not None:
                st, toast = switcher.take_result()
                if pending == (st.cres, st.dres):
                    pending = None
                align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)  # 프로파일 캐시 초기화
                toast_until = time.time() + 3
                last = {"color": None, "ir": None, "depth": None}
                valid_score, hull = None, None
            if pending is not None and time.time() - pending_t >= SWITCH_DEBOUNCE:
                if pending != (st.cres, st.dres):
                    switcher.start(st, *pending)
                pending = None
                continue

            ci, di, ii, ir_p = st.ci, st.di, st.ii, st.ir_p
            R_dc, t_dc, R_cd, t_cd, R_ci, t_ci = st.R_dc, st.t_dc, st.R_cd, st.t_cd, st.R_ci, st.t_ci
            static = st.static
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                if _key_loop() == "quit":
                    break
                continue

            cf, df = frames.get_color_frame(), frames.get_depth_frame()
            irf = frames.get_ir_frame() if ir_p is not None else None
            if cf is not None:
                img = color_to_bgr(cf)
                if img is not None:
                    last["color"] = img
            if df is not None:
                last["depth"] = depth_to_mm(df)
            if irf is not None:
                last["ir"] = ir_to_u16(irf)

            aligned = None
            if do_align and cf is not None and df is not None:
                try:
                    out = align.process(frames)
                    if out is not None:
                        fs = out.as_frame_set() if hasattr(out, "as_frame_set") else out
                        adf = fs.get_depth_frame()
                        if adf is not None:
                            aligned = depth_to_mm(adf)
                except Exception:
                    aligned = None

            if last["color"] is None or last["depth"] is None:
                continue

            depth = last["depth"]
            if auto:
                h, w = depth.shape
                c = depth[h // 2 - 10 : h // 2 + 10, w // 2 - 10 : w // 2 + 10]
                c = c[c > 0]
                if c.size:
                    distance = float(np.median(c))

            # --- 실제 depth 유효 영역 (센서 사각형이 아니라 값이 나오는 영역) ---
            cur = (depth > 0).astype(np.float32)
            if valid_score is None or valid_score.shape != cur.shape:
                valid_score = cur
            else:
                cv2.accumulateWeighted(cur, valid_score, 0.1)  # 이동평균 (in-place)
            frame_i += 1
            if hull is None or frame_i % 10 == 0:  # 무거운 계산은 10프레임마다
                valid_mask = valid_score > 0.2
                nz = cv2.findNonZero(valid_mask.astype(np.uint8))
                hull = cv2.convexHull(nz).reshape(-1, 2).astype(np.float64) if nz is not None else None
                vfov = valid_region_fov(valid_mask, di)

            # --- 경계 투영 (원본 픽셀 좌표) ---
            d_in_c = project_border(di, ci, R_dc, t_dc, distance)
            c_in_d = project_border(ci, di, R_cd, t_cd, distance)
            c_in_i = project_border(ci, ii, R_ci, t_ci, distance)
            cov_c, _ = overlap_ratio(d_in_c, ci.width, ci.height)
            cov_d, _ = overlap_ratio(c_in_d, di.width, di.height)
            v_in_c, cov_vc = None, float("nan")
            if hull is not None and len(hull) >= 3:
                v_in_c = project_between(hull, distance, di, ci, R_dc, t_dc)
                cov_vc, _ = overlap_ratio(v_in_c, ci.width, ci.height)

            # 각 패널은 표시 크기로 먼저 줄인 뒤 그린다 (원본 해상도에서 그리면 WFOV/4K 에서 매우 느림)
            sc = viz.fit_scale(ci.width, ci.height, cell_w, cell_h)
            sd = viz.fit_scale(di.width, di.height, cell_w, cell_h)
            si = viz.fit_scale(ii.width, ii.height, cell_w, cell_h)

            # --- color 패널 ---
            color_small = viz.shrink(last["color"], sc)
            color = color_small.copy()
            if show_grid:
                viz.draw_angle_grid(color, ci.scaled(sc))
            viz.draw_polygon(color, d_in_c * sc, viz.CYAN, 2)
            if v_in_c is not None:
                viz.draw_polygon(color, v_in_c * sc, viz.GREEN, 2)
            fw, fh = footprint_at_distance(static["color"][1], distance)
            color_lines = [
                f"COLOR {ci.width}x{ci.height} ({mode_name('color', ci.width, ci.height)})",
                f"pinhole {static['color'][0].short()}",
                f"undist  {static['color'][1].short()}",
                f"@{distance:.0f}mm view = {fw:.0f} x {fh:.0f} mm",
                f"cyan : depth sensor FOV, covers {cov_c * 100:.1f}% of color",
                f"green: depth valid area, covers {cov_vc * 100:.1f}% of color",
            ]

            # --- IR 패널 ---
            if last["ir"] is not None:
                ir = viz.normalize_ir(viz.shrink(last["ir"], si))
                if show_grid:
                    viz.draw_angle_grid(ir, ii.scaled(si))
                viz.draw_polygon(ir, c_in_i * si, viz.MAGENTA, 2)
                ir_lines = [
                    f"IR {ii.width}x{ii.height} ({mode_name('ir', ii.width, ii.height)})",
                    f"pinhole {static['ir'][0].short()}",
                    f"undist  {static['ir'][1].short()}",
                    "magenta: color FOV",
                ]
            else:
                ir = np.zeros((480, 640, 3), np.uint8)
                ir_lines = ["IR disabled"]

            # --- depth 패널 ---
            dimg = viz.colorize_depth(viz.shrink(depth, sd, nearest=True))
            if show_grid:
                viz.draw_angle_grid(dimg, di.scaled(sd))
            viz.draw_polygon(dimg, c_in_d * sd, viz.MAGENTA, 2)
            if hull is not None:
                viz.draw_polygon(dimg, hull * sd, viz.GREEN, 1)
            depth_lines = [
                f"DEPTH {di.width}x{di.height} ({mode_name('depth', di.width, di.height)})",
                f"pinhole {static['depth'][0].short()}",
                f"undist  {static['depth'][1].short()}",
                f"valid   {vfov.short()}",
                f"magenta: color FOV, covers {cov_d * 100:.1f}% of depth",
            ]

            # --- 정렬 블렌드 ---
            if aligned is not None and aligned.shape[:2] == last["color"].shape[:2]:
                small_d = viz.shrink(aligned, sc, nearest=True)
                blend = cv2.addWeighted(color_small, 0.5, viz.colorize_depth(small_d), 0.5, 0)
                blend_lines = [
                    "Depth aligned to Color (D2C)",
                    f"color pixels with depth: {(small_d > 0).mean() * 100:.1f}%",
                ]
            else:
                blend = np.zeros_like(color)
                blend_lines = ["D2C align off (d)" if not do_align else "D2C align unavailable"]

            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (time.time() - fps_t), 0, time.time()
            canvas = viz.grid(
                [(color, color_lines), (ir, ir_lines), (dimg, depth_lines), (blend, blend_lines)],
                2,
                cell_w,
                cell_h,
            )
            viz.put_text(
                canvas,
                f"Z={distance:.0f}mm {'(auto)' if auto else ''}  {fps:.1f}fps   "
                "+/- [ ] dist  a auto  g grid  d align  m/M depth mode  c/C color res  s save  q quit",
                (10, canvas.shape[0] - 10),
                0.5,
                viz.YELLOW,
            )
            if pending is not None:
                viz.put_text(canvas, f"next: {describe(*pending)}", (10, canvas.shape[0] - 40), 0.7, viz.YELLOW)
            elif time.time() < toast_until:
                viz.put_text(canvas, toast, (10, canvas.shape[0] - 40), 0.7, viz.YELLOW)
            cv2.imshow(WIN, canvas)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k in (ord("+"), ord("=")):
                distance += 100
            elif k == ord("-"):
                distance = max(100, distance - 100)
            elif k == ord("]"):
                distance += 1000
            elif k == ord("["):
                distance = max(100, distance - 1000)
            elif k == ord("a"):
                auto = not auto
            elif k == ord("g"):
                show_grid = not show_grid
            elif k == ord("d"):
                do_align = not do_align
            elif k in MODE_KEYS:
                base = pending or (st.cres, st.dres)
                pending = next_mode(k, base, depth_modes, color_modes)
                pending_t = time.time()
            elif k == ord("s"):
                save_snapshot(
                    args.out,
                    canvas,
                    last,
                    {
                        "device": summary,
                        "distance_mm": distance,
                        "color": {"res": [ci.width, ci.height], "pinhole": static["color"][0].as_dict(),
                                  "undistorted": static["color"][1].as_dict()},
                        "depth": {"res": [di.width, di.height], "pinhole": static["depth"][0].as_dict(),
                                  "undistorted": static["depth"][1].as_dict(), "valid": vfov.as_dict()},
                        "depth_valid_covers_color_pct": round(cov_vc * 100, 2),
                        "ir": {"res": [ii.width, ii.height], "pinhole": static["ir"][0].as_dict(),
                               "undistorted": static["ir"][1].as_dict()},
                        "depth_covers_color_pct": round(cov_c * 100, 2),
                        "color_covers_depth_pct": round(cov_d * 100, 2),
                        "depth_to_color_t_mm": np.round(t_dc, 3).tolist(),
                    },
                )
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


@dataclass
class Session:
    color_p: object
    depth_p: object
    ir_p: object
    fps: int
    ci: Intrinsics
    di: Intrinsics
    ii: Intrinsics
    R_dc: np.ndarray
    t_dc: np.ndarray
    R_cd: np.ndarray
    t_cd: np.ndarray
    R_ci: np.ndarray
    t_ci: np.ndarray
    static: dict

    @property
    def cres(self) -> tuple:
        return (self.ci.width, self.ci.height, 0)

    @property
    def dres(self) -> tuple:
        return (self.di.width, self.di.height, 0)


SWITCH_DEBOUNCE = 0.5
MODE_KEYS = (ord("m"), ord("M"), ord("c"), ord("C"))


def describe(cres: tuple, dres: tuple) -> str:
    return f"depth {dres[0]}x{dres[1]} {mode_name('depth', dres[0], dres[1])} / color {cres[0]}x{cres[1]}"


def next_mode(k: int, base: tuple, depth_modes: list, color_modes: list) -> tuple:
    """base=(cres, dres) 에서 키에 따라 다음/이전 모드."""
    cres, dres = base
    is_depth = k in (ord("m"), ord("M"))
    modes = depth_modes if is_depth else color_modes
    cur = (dres if is_depth else cres)[:2]
    i = modes.index(cur) if cur in modes else 0
    nxt = modes[(i + (1 if k in (ord("m"), ord("c")) else -1)) % len(modes)]
    return (cres, (*nxt, 0)) if is_depth else ((*nxt, 0), dres)


class Switcher:
    """파이프라인 재시작을 백그라운드 스레드에서 수행 (그동안 UI 는 계속 반응)."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.thread: threading.Thread | None = None
        self.result: tuple | None = None
        self.target: tuple | None = None
        self.started = 0.0

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, cur: Session, cres: tuple, dres: tuple) -> None:
        self.target = (cres, dres)
        self.started = time.time()
        self.thread = threading.Thread(target=self._run, args=(cur, cres, dres), daemon=True)
        self.thread.start()

    def _run(self, cur: Session, cres: tuple, dres: tuple) -> None:
        use_ir = cur.ir_p is not None
        try:
            new = start_streams(self.pipeline, cres, dres, use_ir, running=True)
            toast = f"{describe(new.cres, new.dres)} @ {new.fps}fps  ({time.time() - self.started:.1f}s)"
        except Exception as e:  # 지원하지 않는 조합이면 이전 모드로 복구
            print(f"모드 전환 실패: {e}")
            new = start_streams(self.pipeline, cur.cres, cur.dres, use_ir, running=False)
            toast = "switch failed - reverted"
        self.result = (new, toast)

    def take_result(self) -> tuple:
        r, self.result = self.result, None
        return r

    def join(self) -> None:
        if self.thread is not None:
            self.thread.join()


def show_message(w: int, h: int, lines: list[str], base: np.ndarray | None = None) -> None:
    """안내 화면 (base 가 있으면 어둡게 깔고 그 위에)."""
    img = (base * 0.35).astype(np.uint8) if base is not None else np.zeros((h, w, 3), np.uint8)
    viz.put_lines(img, lines, 0.8, viz.YELLOW, origin=(30, h // 2 - 40))
    cv2.imshow(WIN, img)
    cv2.waitKey(1)


def common_fps(pipeline, cres: tuple, dres: tuple) -> int:
    """color/depth 해상도가 둘 다 지원하는 가장 높은 fps (예: WFOV 1024x1024 는 15fps 까지)."""
    def fps_set(sensor, w, h):
        return {p.fps for p in list_profiles(pipeline, sensor) if (w == 0 or p.key == (w, h))}

    want = cres[2] or dres[2]
    both = fps_set("color", *cres[:2]) & fps_set("depth", *dres[:2])
    if want and want in both:
        return want
    return max(both) if both else 0


def start_streams(pipeline, cres: tuple, dres: tuple, use_ir: bool, running: bool = False) -> Session:
    """스트림 프로파일 선택 → 파이프라인 (재)시작. cres/dres = (w, h, fps), 0 은 기본값."""
    if running:
        pipeline.stop()
    fps = common_fps(pipeline, cres, dres) if (cres[0] or dres[0]) else 0
    color_p = find_profile(pipeline, "color", cres[0], cres[1], fps if cres[0] else cres[2])
    depth_p = find_profile(pipeline, "depth", dres[0], dres[1], fps if dres[0] else dres[2])
    ir_p = None
    if use_ir:
        try:
            # Femto 의 IR 은 depth 와 같은 센서/모드여야 동시 스트리밍 가능
            ir_p = find_profile(pipeline, "ir", depth_p.get_width(), depth_p.get_height(), depth_p.get_fps())
        except OBError as e:
            print(f"IR 프로파일을 찾지 못해 IR 없이 진행합니다: {e}")

    config = Config()
    config.enable_stream(color_p)
    config.enable_stream(depth_p)
    if ir_p is not None:
        config.enable_stream(ir_p)
    for p, name in ((color_p, "color"), (depth_p, "depth"), (ir_p, "ir")):
        if p is not None:
            print(f"  {name:5}: {p.get_width()}x{p.get_height()}@{p.get_fps()} {p.get_format().name}")

    ci = to_intrinsics(color_p)
    di = to_intrinsics(depth_p)
    ii = to_intrinsics(ir_p) if ir_p is not None else di
    R_dc, t_dc = extrinsic_Rt(depth_p, color_p)  # depth → color
    R_cd, t_cd = extrinsic_Rt(color_p, depth_p)  # color → depth
    R_ci, t_ci = extrinsic_Rt(color_p, ir_p) if ir_p is not None else (R_cd, t_cd)
    print(f"  depth→color baseline t = {np.round(t_dc, 2)} mm")
    static = {
        "color": (fov_pinhole(ci), fov_undistorted(ci)),
        "depth": (fov_pinhole(di), fov_undistorted(di)),
        "ir": (fov_pinhole(ii), fov_undistorted(ii)),
    }
    try:
        pipeline.enable_frame_sync()
    except Exception:
        pass
    pipeline.start(config)
    return Session(color_p, depth_p, ir_p, depth_p.get_fps(), ci, di, ii,
                   R_dc, t_dc, R_cd, t_cd, R_ci, t_ci, static)


def _key_loop() -> str | None:
    k = cv2.waitKey(1) & 0xFF
    return "quit" if k in (ord("q"), 27) else None


def save_snapshot(out: Path, canvas: np.ndarray, last: dict, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(out / f"viewer_{stamp}.png"), canvas)
    cv2.imwrite(str(out / f"viewer_{stamp}_color.png"), last["color"])
    np.save(out / f"viewer_{stamp}_depth_mm.npy", last["depth"])
    if last["ir"] is not None:
        cv2.imwrite(str(out / f"viewer_{stamp}_ir16.png"), last["ir"])
    (out / f"viewer_{stamp}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"스냅샷 저장: {out / f'viewer_{stamp}'}.*")


if __name__ == "__main__":
    main()
