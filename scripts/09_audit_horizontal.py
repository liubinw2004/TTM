"""盲评包水平位移审计（只读）。

背景：工程侧审计 1415 份已交付 Motion JSON 发现水平速度 p99=4.45 / max=5.77 m/s，
「搬起一个箱子」约 1/3 尝试以 ~5.77 m/s 前进（5s 走 28.9m），且饱和样本在不同 seed
之间逐帧步长完全相同（采样器塌缩）。盲评包在交给评分人之前必须确认没有被污染。

指标（与已交付产物的 h_official 口径一致）：
    mean_speed_mps   : 逐帧水平位移均值 / 帧时长
    max_step_cm      : 单帧最大水平位移（cm）
    n_distinct_steps : 逐帧水平位移「不同取值」个数（塌缩指纹，<=2 视为塌缩）

判定：
    FAST       mean_speed_mps > 3.0
    COLLAPSED  n_distinct_steps <= 2
    OK         其余

用法：
    .venv/Scripts/python.exe scripts/09_audit_horizontal.py \
        --clips deliverables/blind_eval/clips \
        --prompts deliverables/blind_eval/prompts.csv \
        --out deliverables/blind_eval/quality_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPEED_LIMIT_MPS = 3.0
DISTINCT_FLOOR = 2
ROUND_NDIGITS = 4


def parse_bvh_root_motion(path: Path) -> tuple[list[tuple[float, float]], float]:
    """返回 (每帧 root 的 (x, z) 列表, 帧时长秒)。

    只解析 ROOT 的三个 position 通道（约定为前 6 个 channel 里的 X/Y/Z position）。
    """
    xs: list[float] = []
    zs: list[float] = []
    frame_time = 0.0
    motion_started = False
    root_pos_slots: list[int] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("Frame Time:"):
                frame_time = float(line.split(":", 1)[1].strip())
                continue
            if not motion_started:
                # ROOT 的 CHANNELS 行是全局第一行 CHANNELS，取其 position 槽位即可
                if line.startswith("CHANNELS") and root_pos_slots is None:
                    names = line.split()[2:]
                    root_pos_slots = [
                        i for i, n in enumerate(names) if n.endswith("position")
                    ]
                if line.upper() == "MOTION":
                    motion_started = True
                continue

            if line.startswith("Frames:"):
                continue
            if not line:
                continue
            parts = line.split()
            if root_pos_slots is None or len(parts) <= max(root_pos_slots):
                continue
            xs.append(float(parts[root_pos_slots[0]]))
            zs.append(float(parts[root_pos_slots[-1]]))

    if frame_time <= 0.0:
        frame_time = 1.0 / 30.0
    return list(zip(xs, zs)), frame_time


def audit_clip(path: Path) -> dict:
    frames, frame_time = parse_bvh_root_motion(path)
    steps_cm = []
    for (x0, z0), (x1, z1) in zip(frames, frames[1:]):
        steps_cm.append(math.hypot(x1 - x0, z1 - z0))

    if not steps_cm:
        return {
            "mean_speed_mps": 0.0,
            "max_step_cm": 0.0,
            "n_distinct_steps": 0,
            "n_frames": len(frames),
        }

    mean_step_cm = sum(steps_cm) / len(steps_cm)
    distinct = {round(s, ROUND_NDIGITS) for s in steps_cm}
    return {
        "mean_speed_mps": mean_step_cm / 100.0 / frame_time,
        "max_step_cm": max(steps_cm),
        "n_distinct_steps": len(distinct),
        "n_frames": len(frames),
    }


def classify(mean_speed_mps: float, n_distinct_steps: int) -> str:
    if mean_speed_mps > SPEED_LIMIT_MPS:
        return "FAST"
    if n_distinct_steps <= DISTINCT_FLOOR:
        return "COLLAPSED"
    return "OK"


def read_prompt_index(prompts_csv: Path) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    if not prompts_csv.is_file():
        return index
    with prompts_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            index[row["clip_file"]] = (row.get("command_no", ""), row.get("command_text", ""))
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="盲评包水平位移审计（只读）")
    ap.add_argument("--clips", default=str(PROJECT_ROOT / "deliverables/blind_eval/clips"))
    ap.add_argument(
        "--prompts", default=str(PROJECT_ROOT / "deliverables/blind_eval/prompts.csv")
    )
    ap.add_argument(
        "--out", default=str(PROJECT_ROOT / "deliverables/blind_eval/quality_audit.csv")
    )
    args = ap.parse_args(argv)

    clips_dir = Path(args.clips)
    files = sorted(clips_dir.glob("*.bvh"))
    if not files:
        print(f"未找到 BVH: {clips_dir}", file=sys.stderr)
        return 1

    prompt_index = read_prompt_index(Path(args.prompts))
    rows = []
    for path in files:
        stats = audit_clip(path)
        flag = classify(stats["mean_speed_mps"], stats["n_distinct_steps"])
        cmd_no, cmd_text = prompt_index.get(path.name, ("", ""))
        rows.append(
            {
                "clip_file": path.name,
                "command_no": cmd_no,
                "command_text": cmd_text,
                "mean_speed_mps": f"{stats['mean_speed_mps']:.4f}",
                "max_step_cm": f"{stats['max_step_cm']:.4f}",
                "n_distinct_steps": stats["n_distinct_steps"],
                "n_frames": stats["n_frames"],
                "flag": flag,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_fast = sum(1 for r in rows if r["flag"] == "FAST")
    n_collapsed = sum(1 for r in rows if r["flag"] == "COLLAPSED")
    speeds = sorted(float(r["mean_speed_mps"]) for r in rows)

    def pct(q: float) -> float:
        return speeds[min(len(speeds) - 1, int(round(q * (len(speeds) - 1))))]

    print(f"clip 数: {len(rows)}  ->  {out_path}")
    print(
        f"平均速度 m/s: p50={pct(0.50):.3f}  p90={pct(0.90):.3f}  "
        f"max={speeds[-1]:.3f}  阈值 {SPEED_LIMIT_MPS}"
    )
    print(f"FAST={n_fast}  COLLAPSED={n_collapsed}  OK={len(rows) - n_fast - n_collapsed}")
    for r in rows:
        if r["flag"] != "OK":
            print(
                f"  [{r['flag']}] {r['clip_file']} {r['command_no']} {r['command_text']} "
                f"speed={r['mean_speed_mps']} max_step={r['max_step_cm']} "
                f"distinct={r['n_distinct_steps']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
