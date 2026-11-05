#!/usr/bin/env python
"""07_acceptance.py —— TTM 一键验收入口（QA-PLAN v1.3 / 裁决 v1.3.1 的 35 条验收项）。

用法（**必须前台跑**，后台/沙箱会误杀写盘用例出假 E/F，工程师实测过）::

    .venv/Scripts/python.exe scripts/07_acceptance.py --tier fast   # 快速圈（≤3 分钟，跳过 gpu/slow）
    .venv/Scripts/python.exe scripts/07_acceptance.py --tier full   # 全量圈（35 条全跑）

输出：
* ``runs/eval/acceptance_report.json``：每条 ``{id, name, circle, status, measured, route, notes}``
  —— measured 是实测值，不是只有 PASS 字样；
* 退出码：0 = 无 FAIL；1 = 有 FAIL。

报告补全规则：fast 圈没有收集到的 gpu/slow 用例在 finalize 时补成 SKIP（含原因）；
5 处仅人工项（盲评 / M-02b / Blender / UE / 前端视觉）始终列在 ``manual_items``。

**并发保护**：``acceptance_report.json`` 是单文件「读-改-写」，两个进程同时跑会互相
覆盖条目（实测出现过 tier/PASS/SKIP 三个数凑不齐的假报告）。本脚本用
``runs/eval/.acceptance.lock`` 做独占，撞车时退出码 **2** 并提示等待，不会损坏数据。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "tests"))

#: 正式报告（full 圈默认写这里）；fast 圈默认写 _fast 后缀，避免把 full 结果覆盖掉。
DEFAULT_FULL_REPORT = _PROJECT_ROOT / "runs" / "eval" / "acceptance_report.json"
DEFAULT_FAST_REPORT = _PROJECT_ROOT / "runs" / "eval" / "acceptance_report_fast.json"

ACCEPTANCE_TEST_FILES = [
    "tests/test_acceptance_L1_data.py",
    "tests/test_acceptance_L2_model.py",
    "tests/test_acceptance_L3_function.py",
    "tests/test_acceptance_L4_e2e.py",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TTM 一键验收（35 条，QA-PLAN v1.3）")
    parser.add_argument("--tier", choices=("fast", "full"), default="full", help="full=35 条全量；fast=跳过 gpu/slow")
    parser.add_argument("--pytest-args", default="", help="透传给 pytest 的额外参数")
    parser.add_argument(
        "--out",
        default="",
        help=(
            "报告输出路径。缺省时：full → runs/eval/acceptance_report.json；"
            "fast → runs/eval/acceptance_report_fast.json（避免覆盖正式报告）"
        ),
    )
    args = parser.parse_args(argv)

    # 报告路径必须在 import acceptance_support 之前定好（它在 import 时读环境变量），
    # 且要传给 pytest 子进程，否则子进程会写回默认路径 —— 这正是之前 fast 覆盖 full 的原因。
    default_report = DEFAULT_FULL_REPORT if args.tier == "full" else DEFAULT_FAST_REPORT
    report_path = Path(args.out).resolve() if args.out else default_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TTM_ACCEPTANCE_REPORT"] = str(report_path)

    marker_expr = "not gpu and not slow and not manual" if args.tier == "fast" else "not manual"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *ACCEPTANCE_TEST_FILES,
        "-q",
        "-m",
        marker_expr,
        "--no-header",
    ]
    if args.pytest_args:
        cmd.extend(args.pytest_args.split())

    # 独占锁：acceptance_report.json 是单文件「读-改-写」，两个进程并发跑会互相覆盖条目，
    # 出现 tier/PASS/SKIP 三个数凑不齐的假报告。发现已有锁就快速失败，别去损坏数据。
    lock = _PROJECT_ROOT / "runs/eval/.acceptance.lock"
    stale_after_s = 30 * 60
    if lock.is_file():
        age = time.time() - lock.stat().st_mtime
        if age < stale_after_s:
            print(
                f"[07_acceptance] 另一个验收进程正在运行（锁文件 {lock.name}，{age:.0f}s 前创建）。\n"
                "  acceptance_report.json 不支持并发读改写，请等它结束后再跑；\n"
                f"  若确认已无进程在跑（崩溃残留），删除 {lock} 后重试。",
                file=sys.stderr,
            )
            return 2
        print(f"[07_acceptance] 清理过期锁（{age / 60:.0f} 分钟前）", file=sys.stderr)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"pid={os.getpid()} tier={args.tier} at={time.strftime('%H:%M:%S')}", encoding="utf-8")
    try:
        return _run(args, marker_expr, cmd, lock)
    finally:
        lock.unlink(missing_ok=True)


def _run(args, marker_expr: str, cmd: list[str], lock: Path) -> int:
    """锁内实际执行：reset → pytest → finalize → 打表。"""
    # 必须在这里 import：acceptance_support 在 import 时读 TTM_ACCEPTANCE_REPORT，
    # 而报告路径是 main() 里根据 --tier/--out 才算出来的。
    from acceptance_support import REPORT_PATH, finalize, reset_report

    reset_report(args.tier)
    print(f"[07_acceptance] tier={args.tier} marker={marker_expr!r}（锁: {lock.name}）")
    print(f"[07_acceptance] {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env)
    elapsed = time.perf_counter() - started
    print(f"[07_acceptance] pytest 退出码 {result.returncode}，用时 {elapsed:.0f}s")

    report = finalize(args.tier)
    summary = report["summary"]
    print("=" * 78)
    print(
        f"验收报告: {REPORT_PATH}  |  tier={args.tier}  |  "
        f"PASS={summary.get('PASS', 0)} FAIL={summary.get('FAIL', 0)} "
        f"KNOWN_LIMITATION={summary.get('KNOWN_LIMITATION', 0)} "
        f"WARN={summary.get('WARN', 0)} SKIP={summary.get('SKIP', 0)}"
    )
    for item in report["items"].values():
        flag = {"PASS": "+", "KNOWN_LIMITATION": "~"}.get(item["status"], "-" if item["status"] == "FAIL" else "?")
        route = f" → {item['route']}" if item["route"] != "none" else ""
        print(f" [{flag}] {item['id']:<6} {item['status']:<16} {item['name']}{route}")
    limitations = report.get("limitations") or []
    if limitations:
        print("-" * 78)
        print(f"已知限制 {len(limitations)} 条（既非 PASS 也非 FAIL，永久可见）:")
        for lim in limitations:
            m = lim.get("measured", {})
            print(
                f" [~] {lim['item']} {m.get('cmd', '')} seed={m.get('seed', '')} "
                f"峰峰值={m.get('peak_to_peak_cm', '')}cm ±{m.get('amplitude_cm', '')}cm "
                f"周期={m.get('period_s', '')}s → {lim['title']}"
            )
    print("=" * 78)
    return 0 if summary.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
