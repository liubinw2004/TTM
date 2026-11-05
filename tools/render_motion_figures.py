"""从 Motion JSON 渲染骨架动作图（不依赖浏览器）。

用途：README 里的「人物怎么动」效果图。输入是 ``ttm-motion/v1`` 导出物，
按 ``skeleton.bones`` 的 parent + rest_offset_cm 做正向运动学，再取 3/4 视角做
正交投影，画成静帧分解图与 GIF。

为什么不用前端截图：本机四套 FBX 在 three.js 里解析 morph target 时会把渲染进程
打崩（见 README 已知限制 8），浏览器只剩兜底线框。这里绕开资产，直接画动作本身。

用法::

    ./.venv/Scripts/python.exe tools/render_motion_figures.py
    ./.venv/Scripts/python.exe tools/render_motion_figures.py --azimuth 35 --json <path>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

SKY = (22, 25, 32)
GRID = (36, 40, 50)
GROUND = (78, 86, 100)
BONE = (48, 232, 196)
BONE_BACK = (24, 120, 104)   # 远离镜头的一侧压暗，给出前后层次
JOINT = (236, 250, 246)
TEXT = (154, 164, 180)
ACCENT = (255, 196, 84)

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]

# 每段骨骼的粗细（纯观感分层）
THICK = {
    "pelvis": 10, "spine_01": 10, "spine_02": 9, "spine_03": 8, "spine_04": 7,
    "neck_01": 6, "head": 8,
    "clavicle_l": 6, "upperarm_l": 7, "lowerarm_l": 6, "hand_l": 5,
    "clavicle_r": 6, "upperarm_r": 7, "lowerarm_r": 6, "hand_r": 5,
    "thigh_l": 8, "calf_l": 7, "foot_l": 5, "ball_l": 4,
    "thigh_r": 8, "calf_r": 7, "foot_r": 5, "ball_r": 4,
}


def load_font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton 积，四元数按 [x, y, z, w] 排列。"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = q[:3]
    t = 2.0 * np.cross(qv, v)
    return v + q[3] * t + np.cross(qv, t)


def qnorm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])


def solve_frame(bones: list[dict], rots: list[np.ndarray], root_pos: np.ndarray):
    """单帧正向运动学 → 每个关节的世界坐标。"""
    n = len(bones)
    world_q: list[np.ndarray | None] = [None] * n
    world_p: list[np.ndarray | None] = [None] * n
    for i, bone in enumerate(bones):
        parent = int(bone["parent"])
        q_local = qnorm(rots[i])
        if parent < 0:
            world_q[i] = q_local
            world_p[i] = root_pos.astype(float).copy()
        else:
            pq = world_q[parent]
            offset = np.array(bone["rest_offset_cm"], dtype=float)
            world_q[i] = qmul(pq, q_local)
            world_p[i] = world_p[parent] + qrot(pq, offset)
    return world_p


def collect(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    bones = data["skeleton"]["bones"]
    names = [b["name"] for b in bones]
    index = {nm: i for i, nm in enumerate(names)}
    track_by_bone = {t["bone"]: t["rotations_xyzw"] for t in data["tracks"]}
    n_frames = int(data["n_frames"])
    fps = float(data["fps"])

    ident = [0.0, 0.0, 0.0, 1.0]
    rots_per_frame = []
    for f in range(n_frames):
        rots = []
        for nm in names:
            seq = track_by_bone.get(nm)
            rots.append(np.array(seq[f] if seq and f < len(seq) else ident, dtype=float))
        rots_per_frame.append(rots)
    root_pos = np.array(data.get("root_positions_cm") or [[0, 0, 0]] * n_frames, dtype=float)
    return data, bones, names, index, rots_per_frame, root_pos, fps


def project(points: list[np.ndarray], azimuth_deg: float):
    """绕 Y 轴转 azimuth 后正交投影到 (u, v)，并返回深度用于排序。"""
    a = np.deg2rad(azimuth_deg)
    ca, sa = np.cos(a), np.sin(a)
    uv, depth = [], []
    for p in points:
        u = ca * p[0] + sa * p[2]
        v = p[1]
        d = -sa * p[0] + ca * p[2]
        uv.append((u, v))
        depth.append(d)
    return uv, depth


def global_frame(all_frames: list[list[np.ndarray]], azimuth: float, size, margin=0.10):
    """所有帧共用一套取景：同一缩放、同一水平中心、地面固定在 y=0。"""
    W, H = size
    us, vs = [], []
    for pts in all_frames:
        uv, _ = project(pts, azimuth)
        us += [u for u, _ in uv]
        vs += [v for _, v in uv]
    umin, umax = min(us), max(us)
    vmin, vmax = max(min(vs), 0.0), max(vs)
    span_u = max(umax - umin, 1e-3)
    span_v = max(vmax - min(vmin, 0.0), 1e-3)
    avail_w = W * (1 - 2 * margin)
    avail_h = H * (1 - 2 * margin)
    scale = min(avail_w / span_u, avail_h / span_v)
    cu = (umin + umax) / 2
    return {
        "scale": scale,
        "cu": cu,
        "v_top": vmax,
        "v_bottom": min(vmin, 0.0),
        "W": W,
        "H": H,
    }


def to_px(u, v, fr):
    """世界 (u,v) → 像素。v_top 对齐顶部留白，地面 y=0 自然落位。"""
    W, H = fr["W"], fr["H"]
    scale = fr["scale"]
    x = W / 2 + (u - fr["cu"]) * scale
    y = H * 0.10 + (fr["v_top"] - v) * scale
    return x, y


def draw_frame(frame, bones, names, index, fr, title="", subtitle="", azimuth=30.0, size=None):
    W, H = size or (fr["W"], fr["H"])
    ss = 2
    img = Image.new("RGB", (W * ss, H * ss), SKY)
    dr = ImageDraw.Draw(img)

    fr_s = dict(fr)
    fr_s["W"], fr_s["H"] = W * ss, H * ss
    fr_s["scale"] = fr["scale"] * ss  # 画布放大 ss 倍，比例必须同比放大，否则人像缩成小图

    uv, depth = project(frame, azimuth)

    # 地面与参考横线（用世界 y=0, 10, 20… 定位，跨帧一致）
    yg = to_px(0, 0, fr_s)[1]
    if 0 <= yg < H * ss:
        dr.line([(0, yg), (W * ss, yg)], fill=GROUND, width=2 * ss)
    step = 50.0
    k = 1
    while True:
        yk = to_px(0, k * step, fr_s)[1]
        if yk < 0:
            break
        dr.line([(0, yk), (W * ss, yk)], fill=GRID, width=1)
        k += 1

    dmin, dmax = min(depth), max(depth)
    dspan = max(dmax - dmin, 1e-6)

    for i in sorted(range(len(bones)), key=lambda j: depth[j]):
        parent = int(bones[i]["parent"])
        if parent < 0:
            continue
        x1, y1 = to_px(uv[parent][0], uv[parent][1], fr_s)
        x2, y2 = to_px(uv[i][0], uv[i][1], fr_s)
        t = (depth[i] - dmin) / dspan
        col = tuple(int(a + (b - a) * t) for a, b in zip(BONE_BACK, BONE))
        w = THICK.get(names[i], 5)
        dr.line([(x1, y1), (x2, y2)], fill=col, width=max(1, int(w * ss)))
        r = 2.4 * ss
        dr.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=col)

    if "head" in index:
        hx, hy = to_px(uv[index["head"]][0], uv[index["head"]][1], fr_s)
        r = 7.5 * ss
        dr.ellipse([hx - r, hy - r, hx + r, hy + r], fill=JOINT)

    img = img.resize((W, H), Image.LANCZOS)

    dr = ImageDraw.Draw(img)
    if title:
        dr.text((18, 16), title, fill=ACCENT, font=load_font(20))
    if subtitle:
        dr.text((18, 44), subtitle, fill=TEXT, font=load_font(15))
    return img


def build_sheet(cells, out: Path, cols: int = 3):
    w, h = cells[0][1].size
    rows = (len(cells) + cols - 1) // cols
    gap = 3
    canvas = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (12, 13, 17))
    dr = ImageDraw.Draw(canvas)
    font = load_font(16)
    for i, (t, im) in enumerate(cells):
        r, c = divmod(i, cols)
        x, y = c * (w + gap), r * (h + gap)
        canvas.paste(im, (x, y))
        dr.rectangle([x + 10, y + h - 34, x + 92, y + h - 8], fill=(30, 33, 42))
        dr.text((x + 18, y + h - 30), f"{t:.2f}s", fill=ACCENT, font=font)
    canvas.save(out, optimize=True)
    print(f"  [sheet] {out.name}  {canvas.size[0]}x{canvas.size[1]}  ({len(cells)} 帧)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "deliverables/generated/00_打一套长拳第七段.json"))
    ap.add_argument("--json2", default=str(ROOT / "deliverables/generated/02_慢慢走上楼梯.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "images"))
    ap.add_argument("--azimuth", type=float, default=32.0)
    ap.add_argument("--panel", type=int, default=6)
    ap.add_argument("--panel-size", default="480x620")
    ap.add_argument("--gif-size", default="320x420")
    ap.add_argument("--range", default="", help="只渲染某段帧，形如 0:96（默认全片）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pw, ph = (int(v) for v in args.panel_size.lower().split("x"))
    gw, gh = (int(v) for v in args.gif_size.lower().split("x"))

    def render_set(json_path: Path, tag: str, prompt: str, gif_name: str, frange: str = ""):
        print(f"[{tag}] {json_path.name}")
        data, bones, names, index, rpf, rootp, fps = collect(json_path)
        frames = [solve_frame(bones, r, rootp[f]) for f, r in enumerate(rpf)]
        if frange:
            a, _, b = frange.partition(":")
            frames = frames[int(a or 0):int(b or len(frames))]
        n = len(frames)
        print(f"   {n} 帧 / {len(bones)} 关节 / {fps:g} fps / prompt={data.get('prompt')!r}")

        fr = global_frame(frames, args.azimuth, (pw, ph), margin=0.12)
        picks = [int(round(k * (n - 1) / (args.panel - 1))) for k in range(args.panel)]
        cells = []
        for f in picks:
            im = draw_frame(
                frames[f], bones, names, index, fr,
                title=prompt if f == picks[0] else "",
                subtitle=(f"帧 {f} / {n} · 3/4 视角 · 关节 23" if f == picks[0] else ""),
                azimuth=args.azimuth, size=(pw, ph),
            )
            cells.append((f / fps, im))
        build_sheet(cells, out_dir / f"motion-{tag}.png", cols=3)

        fr_gif = global_frame(frames, args.azimuth, (gw, gh), margin=0.10)
        step = max(1, n // 60)
        gif_frames = [
            draw_frame(frames[f], bones, names, index, fr_gif, azimuth=args.azimuth, size=(gw, gh))
            for f in range(0, n, step)
        ]
        gif_frames[0].save(
            out_dir / gif_name, save_all=True, append_images=gif_frames[1:],
            duration=max(int(round(1000 / fps * step)), 30), loop=0, optimize=True,
        )
        print(f"  [gif  ] {gif_name}  ({len(gif_frames)} 帧, {step} 帧/采样)")

    render_set(Path(args.json), "wushu", "打一套长拳第七段（前 3.2s）", "motion-wushu.gif", frange=args.range)
    p2 = Path(args.json2)
    if p2.is_file():
        render_set(p2, "stairs", "慢慢走上楼梯", "motion-stairs.gif")

    # 清掉上一版浏览器截图脚本留下的中间帧
    stale = out_dir / "_frames"
    if stale.is_dir():
        for f in stale.glob("*.png"):
            f.unlink()
        stale.rmdir()

    print("\n产出：")
    for p in sorted(out_dir.glob("*.png")) + sorted(out_dir.glob("*.gif")):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
