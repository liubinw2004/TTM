"""给真实的 Web 界面拍截图（UI 冒烟 + 控制面板效果图）。

用本机 Chrome（playwright channel="chrome"，无需下载浏览器内核）打开真实前端，
调真实 /generate 接口，再把播放进度条拖到不同位置取样。

⚠ 3D 视口受「已知限制 8」影响：四个 FBX 在 three.js 里解析会崩，页面走线框兜底，
所以**视口区截出来只有骨架线**。README 里的动作效果图改由
``tools/render_motion_figures.py`` 从 Motion JSON 渲染，不依赖这里。
本脚本现在的用途是**验证 UI 真的能跑通**（面板、统计、播放条都正常）。

用法::

    ./.venv/Scripts/python.exe tools/capture_screenshots.py --out runs/ui_shots

前置：服务已在 127.0.0.1:8765 启动（见 README §5）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "images"

# 首屏与逐帧取样都等这个超时（首次生成要加载权重，实测 10~20s，留足余量）
GEN_TIMEOUT_MS = 180_000


def wait_ready(page) -> None:
    """等 three.js 与角色加载完（overlay 拿到 hidden 类）。"""
    page.wait_for_selector("#overlay.hidden", timeout=60_000)


def set_scrub(page, value: float) -> None:
    """拖播放进度条。range 控件必须派发 input 事件应用才会响应。"""
    page.evaluate(
        """(v) => {
            const el = document.getElementById('scrub');
            el.value = String(v);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }""",
        value,
    )


def generate(page, prompt: str, seed: int) -> dict:
    page.fill("#prompt", prompt)
    page.fill("#seed", str(seed))
    page.click("#generate")
    # 生成完成后 overlay 会重新隐藏；同时等统计面板出数
    page.wait_for_selector("#overlay.hidden", timeout=GEN_TIMEOUT_MS)
    page.wait_for_function(
        """() => {
            const t = document.getElementById('stat-total');
            return t && t.textContent && t.textContent.trim() !== '–';
        }""",
        timeout=GEN_TIMEOUT_MS,
    )
    time.sleep(0.8)  # 让首帧渲染稳定
    stats = page.evaluate(
        """() => ({
            windows: document.getElementById('stat-windows')?.textContent,
            seam: document.getElementById('stat-seam')?.textContent,
            total: document.getElementById('stat-total')?.textContent,
            sample: document.getElementById('stat-sample')?.textContent,
            decode: document.getElementById('stat-decode')?.textContent,
            blend: document.getElementById('stat-blend')?.textContent,
            frames: document.getElementById('frame-label')?.textContent,
        })"""
    )
    return stats


def capture_viewport(page, path: Path) -> None:
    page.locator("#viewport").screenshot(path=str(path))


def build_grid(frames: list[Path], out: Path, cols: int = 3, label: str = "") -> None:
    imgs = [Image.open(p).convert("RGB") for p in frames]
    w, h = imgs[0].size
    scale = 0.5
    tw, th = int(w * scale), int(h * scale)
    rows = (len(imgs) + cols - 1) // cols
    pad, gap = 8, 6
    canvas = Image.new("RGB", (cols * tw + (cols - 1) * gap, rows * th + (rows - 1) * gap), (18, 18, 22))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(im.resize((tw, th), Image.LANCZOS), (c * (tw + gap), r * (th + gap)))
    canvas.save(out, optimize=True)
    print(f"  [grid] {out.name}  {canvas.size[0]}x{canvas.size[1]}  ({len(imgs)} 帧 {label})")


def build_gif(frames: list[Path], out: Path, width: int = 560, duration: int = 70) -> None:
    imgs = [Image.open(p).convert("RGB") for p in frames]
    w, h = imgs[0].size
    scaled = [im.resize((width, int(h * width / w)), Image.LANCZOS) for im in imgs]
    scaled[0].save(
        out,
        save_all=True,
        append_images=scaled[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    print(f"  [gif ] {out.name}  {scaled[0].size[0]}x{scaled[0].size[1]}  ({len(scaled)} 帧)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--prompt", default="打一套长拳")
    ap.add_argument("--prompt2", default="慢慢走上楼梯")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--frames", type=int, default=24, help="GIF 取样帧数")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_frames"
    tmp.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--enable-unsafe-swiftshader",  # headless 下软件 WebGL
                "--use-angle=swiftshader",
                "--hide-scrollbars",
                "--force-device-scale-factor=1.5",  # 高 DPI，图更清晰
            ],
        )
        page = browser.new_page(viewport={"width": 1560, "height": 940}, device_scale_factor=1.5)
        page.goto(args.url, wait_until="domcontentloaded")
        wait_ready(page)

        # 关掉自动转台，保证各帧机位一致（否则拼图会看起来像抖动）
        if page.is_checked("#toggle-orbit"):
            page.uncheck("#toggle-orbit")
        # 打开骨骼线，让"骨架怎么动"看得见
        if not page.is_checked("#toggle-skeleton"):
            page.check("#toggle-skeleton")
        page.evaluate("() => window.dispatchEvent(new Event('resize'))")

        print(f"[1/4] 生成「{args.prompt}」seed={args.seed}")
        stats = generate(page, args.prompt, args.seed)
        print("      统计:", json.dumps(stats, ensure_ascii=False))

        shot = out_dir / "ui-overview.png"
        page.screenshot(path=str(shot))
        print(f"  [shot] {shot.name}")

        print(f"[2/4] 逐帧取样 {args.frames} 帧")
        frame_paths: list[Path] = []
        for i in range(args.frames):
            v = i / (args.frames - 1)
            set_scrub(page, v)
            page.wait_for_timeout(90)
            p = tmp / f"f{i:03d}.png"
            capture_viewport(page, p)
            frame_paths.append(p)

        print("[3/4] 拼接动作分解图")
        # 取 6 个关键时刻做静态分解图（含起势与收势）
        picks = [frame_paths[int(round(k * (len(frame_paths) - 1) / 5))] for k in range(6)]
        build_grid(picks, out_dir / "motion-phases.png", cols=3, label=args.prompt)

        print("[4/4] 生成循环 GIF")
        build_gif(frame_paths, out_dir / "motion.gif")

        # 第二条指令：上楼（位移型），单独一张，展示指令可换
        print(f"[+] 追加「{args.prompt2}」")
        set_scrub(page, 0.55)
        stats2 = generate(page, args.prompt2, args.seed)
        set_scrub(page, 0.6)
        page.wait_for_timeout(200)
        page.screenshot(path=str(out_dir / "ui-stairs.png"))
        print("      统计:", json.dumps(stats2, ensure_ascii=False))
        print(f"  [shot] ui-stairs.png")

        browser.close()

    print("\n完成，产出：")
    for p in sorted(out_dir.glob("*.png")) + sorted(out_dir.glob("*.gif")):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
