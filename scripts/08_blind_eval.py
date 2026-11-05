"""08_blind_eval —— 生成人工盲评执行包（QA-PLAN §5.3 / 验收裁决 v1.3.1）。

人工盲评是整个项目**唯一**无法机器化的验收环节（语义正确性：动作像不像这句中文）。
本脚本把「评分人需要的东西」一次性做齐：

* 指令集 = 10 条冻结指令（``deliverables/software-company/e2e_commands.json``）
  + 13 条按窗口数分档抽样的补充条件（覆盖武术 ≥5、类别 ≥4 类），共 **23 条**；
* 每条 **3 个 seed**（101/202/303，刻意避开 E-01 的 42/43/44 以保证独立性）
  → 共 69 个 clip；
* **打乱顺序 + 脱敏文件名**（``clip_001.bvh`` …）：评分人只看到
  「文件 ↔ 随机指令序号」的配对，看不到 seed、看不到生成顺序；
* 产出 ``clips/``、``prompts.csv``、``score_sheet.csv``、``README.md``；
* 指令/seed/文件名的完整映射写入 ``answer_key.json``（**只给主理人/QA，不给评分人**）。

生成走**服务端真实链路**（``TestClient`` + ``POST /generate`` + 下载 ``bvh_url``），
保证盲评对象就是用户能拿到的那份产物（ground_origin / trim_head 全部生效）。

用法::

    .venv/Scripts/python.exe scripts/08_blind_eval.py                # 全量 69 clip
    .venv/Scripts/python.exe scripts/08_blind_eval.py --seeds 101    # 只跑一个 seed（试跑）
    .venv/Scripts/python.exe scripts/08_blind_eval.py --dry-run      # 只打印抽样结果不生成
    .venv/Scripts/python.exe scripts/08_blind_eval.py --keep         # 保留已有 clips 不重生成
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(_ROOT / "src"), str(_ROOT)):  # src 用于 ttm.*，根目录用于 server.*
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

PROJECT_ROOT = _ROOT
E2E_COMMANDS_JSON = PROJECT_ROOT / "deliverables/software-company/e2e_commands.json"
LABELS_JSONL = PROJECT_ROOT / "data/tier_S/labels_merged.jsonl"
OUT_DIR = PROJECT_ROOT / "deliverables/blind_eval"
CLIPS_DIR = OUT_DIR / "clips"

#: 抽样随机种子（固定 → 结果可复现；改它会换掉 13 条补充指令）。
SAMPLE_SEED = 20261029
#: 每条指令的 seed（避开 E-01 的 42/43/44，避免与验收矩阵产物重复）。
DEFAULT_SEEDS = (101, 202, 303)
#: 补充条件条数与分档配额（窗口数：thin ≤10 / mid 11–30 / thick >30）。
EXTRA_TOTAL = 13
STRATA_QUOTA = (("thin", 3), ("mid", 4), ("thick", 6))
#: 补充条件里必须凑够的武术条数（主理人要求 ≥5）。
EXTRA_WUSHU_MIN = 5
#: 补充条件必须覆盖的 category_l1 类别数下限。
EXTRA_MIN_CATEGORIES = 4

REQUIRED_DIMS = (
    ("category_correct", "类别正确：动作与该中文指令是否同类（1=完全无关 … 5=一眼认出）"),
    ("amplitude_ok", "幅度合理：关节幅度是否像真人做这个动作（1=僵直/夸张变形 … 5=自然）"),
    ("no_float_penetration", "无穿模漂浮：无脚踩空/身体穿模/整体漂移（1=明显 … 5=完全无）"),
    ("duration_ok", "时长符合：5 秒内动作是否完整收尾（1=半截/循环重放 … 5=自然起止）"),
)


# ---------------------------------------------------------------------------
# 指令集：10 条冻结 + 13 条分层抽样
# ---------------------------------------------------------------------------
def load_labels() -> list[dict[str, Any]]:
    """读窗口级标签（13696 行）。只在 train 里统计窗口数（可学条件 178 个）。"""
    rows: list[dict[str, Any]] = []
    for line in LABELS_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def pick_extra_conditions(labels: list[dict[str, Any]], frozen_prompts: list[str]) -> list[dict[str, Any]]:
    """从 178 个可学条件里按窗口数分档抽 13 条（武术 ≥5、类别 ≥4）。

    分档口径（train 窗口数）：thin ≤10（11 个条件）/ mid 11–30（21 个）/ thick >30（146 个）。
    已出现在冻结指令里的条件会被排除，避免同一条件出两份。
    """
    train = [row for row in labels if row.get("split") == "train"]
    stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        stats[row["zh_label"]].append(row)

    def is_frozen(name: str) -> bool:
        return any(name in prompt or prompt in name for prompt in frozen_prompts)

    pooled: dict[str, list[dict[str, Any]]] = {"thin": [], "mid": [], "thick": []}
    for name, rows in stats.items():
        if is_frozen(name):
            continue
        n = len(rows)
        strata = "thin" if n <= 10 else ("mid" if n <= 30 else "thick")
        texts = rows[0].get("zh_texts") or [name]
        pooled[strata].append(
            {
                "condition": name,
                "stratum": strata,
                "windows": n,
                "is_wushu": bool(rows[0].get("is_wushu")),
                "category_l1": rows[0].get("category_l1") or "",
                "category_l2": rows[0].get("category_l2") or "",
                "texts": list(texts),
            }
        )

    rng = random.Random(SAMPLE_SEED)
    for bucket in pooled.values():
        bucket.sort(key=lambda item: (-item["windows"], item["condition"]))

    picked: list[dict[str, Any]] = []
    for stratum, quota in STRATA_QUOTA:
        candidates = pooled[stratum]
        wushu = [item for item in candidates if item["is_wushu"]]
        other = [item for item in candidates if not item["is_wushu"]]
        rng.shuffle(wushu)
        rng.shuffle(other)
        # thin/mid 先保证武术覆盖（这两档武术占比高），thick 先保证日常动作多样性
        take_wushu = 2 if quota >= 3 else 1
        chosen = wushu[:take_wushu] + other[: max(0, quota - take_wushu)]
        if len(chosen) < quota:  # 该档武术不够，用同档其它条件补满
            chosen += other[len(chosen) - take_wushu :][: quota - len(chosen)]
        picked.extend(chosen[:quota])

    # 硬约束兜底：武术 ≥5、类别 ≥4
    def topup(predicate, count: int, reason: str) -> None:
        already = {item["condition"] for item in picked}
        pool = [
            item
            for bucket in pooled.values()
            for item in bucket
            if item["condition"] not in already and predicate(item)
        ]
        rng.shuffle(pool)
        for item in pool[:count]:
            picked.append(item)
            already.add(item["condition"])
        if count:
            print(f"[08] 兜底补入 {reason}")

    wushu_now = sum(1 for item in picked if item["is_wushu"])
    if wushu_now < EXTRA_WUSHU_MIN:
        topup(lambda item: item["is_wushu"], EXTRA_WUSHU_MIN - wushu_now, "武术条件")
    cats_now = len({item["category_l1"] for item in picked})
    if cats_now < EXTRA_MIN_CATEGORIES:
        topup(
            lambda item: item["category_l1"] not in {i["category_l1"] for i in picked},
            EXTRA_MIN_CATEGORIES - cats_now,
            "不同类别",
        )

    # 兜底可能超额，按窗口数降序裁到 13 条（保证 thick 档不被打散）
    picked.sort(key=lambda item: (-item["windows"], item["condition"]))
    return picked[:EXTRA_TOTAL]


def build_commands(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """23 条指令：10 条冻结 + 13 条抽样；返回**已打乱顺序**并编 S01..S23 的列表。"""
    frozen = json.loads(E2E_COMMANDS_JSON.read_text(encoding="utf-8"))
    commands: list[dict[str, Any]] = []
    for entry in frozen["commands"]:
        commands.append(
            {
                "source": "frozen",
                "frozen_id": entry["id"],
                "command": entry["cmd"],
                "condition": "",
                "cat": entry.get("cat", ""),
                "windows": None,
                "is_wushu": ("拳" in entry["cmd"] or "剑" in entry["cmd"] or "武术" in entry["cmd"]),
            }
        )
    extras = pick_extra_conditions(labels, [entry["cmd"] for entry in frozen["commands"]])
    for item in extras:
        texts = item["texts"]
        # 武术优先用「打一套X」这类自然命令句，日常动作直接用第一条原始文本
        command = next((t for t in texts if t.startswith("打一套")), texts[0])
        commands.append(
            {
                "source": "sampled",
                "frozen_id": "",
                "command": command,
                "condition": item["condition"],
                "cat": f"{item['category_l1']} / {item['category_l2']}".strip(" /"),
                "windows": item["windows"],
                "is_wushu": item["is_wushu"],
            }
        )

    rng = random.Random(SAMPLE_SEED + 1)  # 指令编号单独一条随机流，便于复现
    rng.shuffle(commands)
    for index, item in enumerate(commands, start=1):
        item["command_no"] = f"S{index:02d}"
    return commands


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def _count_bvh_frames(raw: bytes) -> int:
    """从 BVH 文本里读 ``Frames:`` 行（``--keep`` 复用旧文件时补 n_frames）。"""
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        if line.startswith("Frames:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return 0
    return 0

def make_client(cfg):
    """服务端 TestClient（真实 /generate 链路）。"""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from server.app import create_app  # noqa: PLC0415

    return TestClient(create_app(cfg))


def generate_clips(commands: list[dict[str, Any]], seeds: tuple[int, ...], cfg, keep: bool) -> list[dict[str, Any]]:
    """按「指令 × seed」批量生成，落盘成 clip_XXX.bvh（打乱顺序）。

    Returns:
        每个 clip 的记录：``file / command_no / command / seed / n_frames / ms / sha256``。
    """
    plan_pairs = [(item, seed) for item in commands for seed in seeds]
    rng = random.Random(SAMPLE_SEED + 2)  # 文件名编号的随机流
    rng.shuffle(plan_pairs)

    client = make_client(cfg)
    guidance = float(cfg.runtime.guidance)
    clips: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (item, seed) in enumerate(plan_pairs, start=1):
        name = f"clip_{index:03d}.bvh"
        target = CLIPS_DIR / name
        record: dict[str, Any] = {
            "file": name,
            "command_no": item["command_no"],
            "command": item["command"],
            "seed": int(seed),
            "status": 0,
            "n_frames": 0,
            "ms": 0.0,
            "sha256": "",
            "error": "",
        }
        if keep and target.is_file() and target.stat().st_size > 0:
            raw = target.read_bytes()
            record.update(
                {
                    "status": 200,
                    "n_frames": _count_bvh_frames(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "error": "skipped(keep)",
                }
            )
            clips.append(record)
            continue

        mark = time.perf_counter()
        try:
            resp = client.post(
                "/generate",
                json={
                    "prompt": item["command"],
                    "duration_s": 5.0,
                    "seed": int(seed),
                    "guidance": guidance,
                    "export": "bvh",
                },
            )
            body = resp.json() if resp.status_code == 200 else {}
            record["status"] = resp.status_code
            if resp.status_code != 200 or not body.get("bvh_url"):
                record["error"] = f"HTTP {resp.status_code}: {str(body)[:200]}"
            else:
                download = client.get(body["bvh_url"])
                if download.status_code != 200:
                    record["error"] = f"下载 BVH 失败 HTTP {download.status_code}"
                else:
                    payload = download.content
                    target.write_bytes(payload)
                    record["sha256"] = hashlib.sha256(payload).hexdigest()
                    record["n_frames"] = int(body.get("n_frames") or 0)
                    record["ground_origin_applied"] = bool(body.get("ground_origin_applied"))
        except Exception as exc:  # noqa: BLE001 - 记录后继续，报告里体现
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["ms"] = round((time.perf_counter() - mark) * 1000.0, 1)
        clips.append(record)
        flag = "OK " if not record["error"] else "ERR"
        print(f"[08] {flag} {name}  {item['command_no']} seed={seed}  {item['command'][:16]}  "
              f"{record['n_frames']}f  {record['ms']:.0f}ms  {record['error'][:60]}")
    print(f"[08] 生成完毕：{len(clips)} 个 clip，总耗时 {time.perf_counter() - started:.1f}s")
    return clips


# ---------------------------------------------------------------------------
# 落盘：prompts.csv / score_sheet.csv / answer_key.json / README.md
# ---------------------------------------------------------------------------
def write_prompts_csv(commands: list[dict[str, Any]], clips: list[dict[str, Any]]) -> Path:
    """给评分人看的清单：只有「文件名 ↔ 随机指令序号 ↔ 指令文本」，无 seed/无顺序信息。"""
    path = OUT_DIR / "prompts.csv"
    ordered = sorted(clips, key=lambda clip: clip["file"])
    cmd_by_no = {item["command_no"]: item for item in commands}
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_file", "command_no", "command_text"])
        for clip in ordered:
            writer.writerow([clip["file"], clip["command_no"], cmd_by_no[clip["command_no"]]["command"]])
    return path


def write_score_sheet(clips: list[dict[str, Any]]) -> Path:
    """空白评分表：前两列已填，四个维度留空，底部附评分说明。"""
    path = OUT_DIR / "score_sheet.csv"
    ordered = sorted(clips, key=lambda clip: clip["file"])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["clip_file", "command_no", *(name for name, _ in REQUIRED_DIMS), "rater", "notes"]
        )
        for clip in ordered:
            writer.writerow([clip["file"], clip["command_no"], "", "", "", "", "", ""])
    return path


def write_answer_key(commands, clips, seeds, cfg, elapsed) -> Path:
    """完整映射 + 生成配置（**不给评分人**）。"""
    path = OUT_DIR / "answer_key.json"
    failed = [clip for clip in clips if clip["error"] and not clip["error"].startswith("skipped")]
    payload = {
        "format": "ttm-blind-eval-answer-key/v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "owner": "software-qa-engineer（评分人不得查看本文件）",
        "protocol": {
            "n_commands": len(commands),
            "seeds_per_command": len(seeds),
            "seeds": list(seeds),
            "n_clips": len(clips),
            "shuffle_seed": SAMPLE_SEED,
            "guidance": float(cfg.runtime.guidance),
            "steps": int(getattr(cfg.diffusion, "ddim_steps", 50)),
            "duration_s": 5.0,
            "trim_head_frames": int(getattr(cfg.runtime, "trim_head_frames", 0)),
            "export": "bvh（服务端 /generate → bvh_url 下载，ground_origin 已平移）",
        },
        "generation": {
            "elapsed_s": round(elapsed, 1),
            "failed_clips": len(failed),
            "avg_ms_per_clip": round(sum(c["ms"] for c in clips) / max(len(clips), 1), 1),
        },
        "commands": [
            {
                "command_no": item["command_no"],
                "command": item["command"],
                "condition": item["condition"],
                "source": item["source"],
                "frozen_id": item["frozen_id"],
                "category": item["cat"],
                "train_windows": item["windows"],
                "is_wushu": item["is_wushu"],
            }
            for item in commands
        ],
        "clips": clips,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_readme(commands, clips, seeds) -> Path:
    """人工执行说明（评分标准 / Blender 操作 / 两人独立评 / 一致率算法）。"""
    path = OUT_DIR / "README.md"
    n_wushu = sum(1 for item in commands if item["is_wushu"])
    cats = len({item["cat"].split("/")[0].strip() for item in commands})
    text = f"""# TTM 人工盲评执行包

> 由 `scripts/08_blind_eval.py` 生成（{time.strftime("%Y-%m-%d %H:%M:%S")}）。
> 这是 TTM 项目**唯一**无法机器化的验收环节：机器只能验「合法性/多样性/可分性」，
> 「动作像不像这句中文」必须由人判断（PRD §S1）。

## 1. 包里有什么

| 路径 | 内容 | 谁可以看 |
|---|---|---|
| `clips/clip_001.bvh` … `clip_{len(clips):03d}.bvh` | {len(clips)} 个动作（{len(commands)} 条指令 × {len(seeds)} 个 seed），**文件名已脱敏、顺序已打乱** | 评分人 |
| `prompts.csv` | 文件名 ↔ 随机指令序号 ↔ 指令原文 | 评分人 |
| `score_sheet.csv` | 空白评分表（前两列已填，四个维度待填） | 评分人（各存一份） |
| `README.md` | 本文件 | 评分人 |
| `answer_key.json` | 文件名 ↔ 指令 ↔ **seed** 的完整映射 + 生成配置 | **仅主理人 / QA，禁止给评分人** |

- 指令集 = 10 条冻结指令 + {len(commands) - 10} 条按窗口数分档抽样的补充条件，共 **{len(commands)} 条**；
  其中武术类 **{n_wushu} 条**、覆盖 **{cats} 个一级类别**。
- 每条指令 3 个 seed（{", ".join(str(s) for s in seeds)}）。
- 生成参数：guidance 7.5、DDIM 50 步、时长 5 秒、`ground_origin` 已生效（BVH 首帧贴地）。

## 2. 怎么播（Blender 4.x，约 3 分钟看完 1 个）

1. `File → Import → Motion Capture (.bvh)`，选中 `clips/clip_XXX.bvh`；
2. 导入对话框保持默认（Scale 1.0、Y 轴向上、30 fps）；
3. 按 `空格` 播放，`Shift+←` 回到第 1 帧逐帧看起手；
4. 视角建议：默认前视 + 侧视（小键盘 `1` / `3`）各看一遍——穿模在侧视最容易发现；
5. 看完一个就填一行 `score_sheet.csv`，**不要在文件名或备注里推测 seed / 指令顺序**。

> 没有 Blender 时的等价方案：`web/index.html` 起服务后逐个加载同名 JSON（观感略差，
> 但足以判断类别与幅度；穿模/漂浮判断请以 Blender 为准）。

## 3. 四个维度怎么打分（每项 1–5 分，只打整数）

| 维度 | 5 分 | 3 分（及格线） | 1 分 |
|---|---|---|---|
| **类别正确** | 一眼就能说出这是该指令的动作 | 能看出大致方向，但需要靠指令猜 | 完全无关（说「走」却在原地挥手） |
| **幅度合理** | 关节幅度与真人做该动作一致 | 明显偏小/偏大，但仍认得出来 | 僵直不动或夸张到变形 |
| **无穿模漂浮** | 脚着地、无穿插、无整体漂移 | 轻微滑步或一处短暂穿插 | 明显腾空/穿模/整体漂移 |
| **时长符合** | 5 秒内自然起手→收尾 | 中段有重复或略突兀，起止尚可 | 只有半截动作，或明显循环重放 |

判定口径（沿用 QA-PLAN §5.3）：
- **单项均分 ≥3.5** 视为该维度通过；
- **类别正确**是硬指标：任一条指令的「类别正确」均分 **<3.0** 即判该条件不达 PRD §S1。

## 4. 两人独立评（主理人 + 用户，代理标准）

1. 两人**各自**复制一份 `score_sheet.csv`，文件名加 `_raterA` / `_raterB`；
2. **独立评、不讨论**：同一房间也请先各自填完再对；
3. 顺序建议 A 正序、B 倒序看，减少「同一段记忆」带来的相关性（可选）；
4. 填完把两份表交给 QA（放进 `deliverables/blind_eval/` 即可）。

## 5. 一致率算法（QA 复核时执行）

对每个维度 d，用两份表按 `clip_file` 对齐后计算：

1. **完全一致率** `exact_d = 打平分的条数 / 总条数`；
2. **邻分一致率** `within1_d = |A−B| ≤ 1 的条数 / 总条数`（主判据，≥0.80 视为一致）；
3. **二次加权 kappa**（可选，论文口径）：
   `kappa = 1 − Σ w_ij·O_ij / Σ w_ij·E_ij`，其中 `w_ij = (i−j)²/(K−1)²`，K=5；
4. **分歧处理**：`|A−B| ≥ 2` 的 clip 由两人一起回看，取**较低分**为准（保守原则）；
5. 汇总：每维度报 `均值 ± 标准差`、`within1` 一致率、分歧 clip 清单。

> 若 `within1` 一致率 <0.80，说明评分标准本身有歧义，先对齐标准再重评，**不要**直接平均了事。

## 6. 回收清单

评完请把这三个文件交回 `deliverables/blind_eval/`：

- `score_sheet_raterA.csv`、`score_sheet_raterB.csv`
- 一句话结论（是否认为动作能对上中文指令、最差的 3 个 clip 是哪些）
"""
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成人工盲评执行包")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="逗号分隔的 seed 列表（默认 101,202,303）")
    parser.add_argument("--dry-run", action="store_true", help="只打印抽样结果，不生成")
    parser.add_argument("--keep", action="store_true", help="已有 clips 不重新生成")
    parser.add_argument("--config", default="configs/tier_s.yaml")
    args = parser.parse_args(argv)

    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    labels = load_labels()
    commands = build_commands(labels)

    by_source = Counter(item["source"] for item in commands)
    n_wushu = sum(1 for item in commands if item["is_wushu"])
    print(f"[08] 指令集 {len(commands)} 条（冻结 {by_source['frozen']} + 抽样 {by_source['sampled']}），"
          f"武术 {n_wushu} 条，seed {seeds} → 预计 {len(commands) * len(seeds)} 个 clip")
    for item in commands:
        windows = "-" if item["windows"] is None else f"{item['windows']}窗"
        wushu = "武术" if item["is_wushu"] else "    "
        print(f"[08]   {item['command_no']}  {item['command'][:20]:<22} "
              f"{item['source']:<8} {wushu} {windows}")

    if args.dry_run:
        print("[08] --dry-run：不生成，退出")
        return 0

    from ttm.common.config import load_config  # noqa: PLC0415

    cfg = load_config(args.config)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    clips = generate_clips(commands, seeds, cfg, keep=args.keep)
    elapsed = time.perf_counter() - started

    prompts = write_prompts_csv(commands, clips)
    sheet = write_score_sheet(clips)
    key = write_answer_key(commands, clips, seeds, cfg, elapsed)
    readme = write_readme(commands, clips, seeds)

    ok = sum(1 for clip in clips if not clip["error"] or clip["error"].startswith("skipped"))
    failed = [clip for clip in clips if clip["error"] and not clip["error"].startswith("skipped")]
    print("\n[08] ===== 盲评包已生成 =====")
    print(f"[08] clips      : {len(clips)} 个（成功 {ok} / 失败 {len(failed)}）→ {CLIPS_DIR}")
    print(f"[08] 总耗时     : {elapsed:.1f}s（平均 {elapsed / max(len(clips), 1) * 1000:.0f} ms/clip）")
    print(f"[08] prompts    : {prompts}")
    print(f"[08] score_sheet: {sheet}")
    print(f"[08] answer_key : {key}")
    print(f"[08] README     : {readme}")
    if failed:
        print("[08] 失败清单：")
        for clip in failed:
            print(f"[08]   {clip['file']} {clip['command_no']} seed={clip['seed']}: {clip['error'][:120]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
