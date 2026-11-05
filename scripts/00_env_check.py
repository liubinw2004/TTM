"""T01 环境自检：解释器 / 缓存重定向 / 写盘守卫 / 依赖 / torch+CUDA。

运行::

    . D:/TTM/scripts/env.ps1          # 先注入环境变量
    python scripts/00_env_check.py    # 退出码 0 即通过

自检项（对应 TTM-ARCH.md §5.4 与 §10 的 T01 验收标准）：

1. **解释器**：Python 版本、可执行文件路径、项目根目录；
2. **缓存重定向**：``PIP_CACHE_DIR`` / ``HF_HOME`` / ``TORCH_HOME`` / ``TMPDIR``
   （以及 Windows 的 ``TEMP`` / ``TMP``）全部指向 ``D:/TTM/.cache``；
   未设置的变量会被**自动补默认值**，指向别处的会被**强制纠正**并告警；
3. **写盘守卫**：``C:`` / ``Z:`` / 项目外的 ``D:`` 路径必须被拒绝，
   ``D:/TTM/**`` 必须放行；
4. **写盘探针**：在 cache / runs / data 三处真实写-读-删一轮，
   并记录所有真实落盘路径，断言 **C 盘零写入**；
5. **依赖**：11 个直接依赖中除 torch 外的包能否 import 及其版本；
6. **torch + CUDA**：``torch.cuda.is_available()``、GPU 名、显存、一次真实矩阵乘。
   torch 未安装或 CUDA 不可用时记为 SKIP（主理人后台安装 torch 期间属正常），
   ``--strict`` 下记 FAIL。

退出码：无 FAIL 时为 0，否则为 1。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# --- sys.path 引导（即便未做 editable install 也能 import ttm）----------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_WARN = "WARN"
STATUS_SKIP = "SKIP"

#: 真实落盘过的路径（用于断言 C 盘零写入）。
CREATED_PATHS: list[str] = []


@dataclass
class CheckResult:
    """单项自检结果。"""

    name: str
    status: str
    detail: str
    extra: dict[str, object] = field(default_factory=dict)

    def line(self) -> str:
        """返回适合控制台打印的一行文本。"""
        return f"[{self.status:<4}] {self.name:<26} {self.detail}"


# ---------------------------------------------------------------------------
# 环境变量重定向
# ---------------------------------------------------------------------------
def _cache_root() -> Path:
    """返回 ``D:/TTM/.cache``（自检期的缓存基准）。"""
    return _PROJECT_ROOT / ".cache"


def default_cache_env() -> dict[str, str]:
    """返回应生效的缓存类环境变量默认值。"""
    cache = _cache_root()
    return {
        "TTM_ROOT": str(_PROJECT_ROOT),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "HF_HOME": str(cache / "hf"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "hf" / "hub"),
        "TRANSFORMERS_CACHE": str(cache / "hf" / "transformers"),
        "TORCH_HOME": str(cache / "torch"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "TMPDIR": str(cache / "tmp"),
        "TEMP": str(cache / "tmp"),
        "TMP": str(cache / "tmp"),
    }


def _is_under(path: str, root: Path) -> bool:
    """判断 ``path`` 是否位于 ``root`` 之内（大小写不敏感）。"""
    if not path:
        return False
    candidate = os.path.normcase(os.path.abspath(os.path.expanduser(path)))
    base = os.path.normcase(os.path.abspath(str(root)))
    return candidate == base or candidate.startswith(base + os.sep)


def apply_cache_env(force: bool = True) -> CheckResult:
    """把缓存类环境变量纠正为指向 ``D:/TTM/.cache``。

    Args:
        force: ``True`` 时强制纠正指向别处的变量（默认行为）。

    Returns:
        自检结果（STATUS_PASS 或 STATUS_WARN）。
    """
    cache = _cache_root()
    defaults = default_cache_env()
    applied: list[str] = []
    corrected: list[str] = []

    for name, value in defaults.items():
        current = os.environ.get(name, "").strip()
        expected_root = _PROJECT_ROOT if name == "TTM_ROOT" else cache
        if current and _is_under(current, expected_root):
            continue
        if not current:
            os.environ[name] = value
            applied.append(name)
        elif force:
            os.environ[name] = value
            corrected.append(f"{name}: {current} -> {value}")
        else:
            corrected.append(f"{name}: {current} (未纠正)")

    # 确保目录存在（全部在 D 盘）
    for name in ("PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME", "TMPDIR"):
        target = Path(os.environ[name])
        target.mkdir(parents=True, exist_ok=True)

    if corrected:
        return CheckResult(
            "cache 环境变量",
            STATUS_WARN,
            f"已纠正 {len(corrected)} 个: " + "; ".join(corrected),
            {"applied": applied, "corrected": corrected},
        )
    detail = "四项缓存变量均指向 D:/TTM/.cache" + (
        f"（本次自动补齐 {len(applied)} 个）" if applied else ""
    )
    return CheckResult("cache 环境变量", STATUS_PASS, detail, {"applied": applied})


def check_cache_env() -> CheckResult:
    """断言缓存类环境变量与 ``tempfile`` 临时目录全部在 D 盘内。"""
    cache = _cache_root()
    required = ("PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME", "TMPDIR")
    offenders = [
        f"{name}={os.environ.get(name, '')!r}"
        for name in required
        if not _is_under(os.environ.get(name, ""), cache)
    ]
    tempdir = tempfile.gettempdir()
    if not _is_under(tempdir, _PROJECT_ROOT):
        offenders.append(f"tempfile.gettempdir()={tempdir!r}")
    if offenders:
        return CheckResult("cache 变量断言", STATUS_FAIL, "不在 D:/TTM 内: " + ", ".join(offenders))
    return CheckResult(
        "cache 变量断言",
        STATUS_PASS,
        f"PIP/HF/TORCH/TMP + tempfile 均落在 {cache.as_posix()}",
        {"tempdir": tempdir},
    )


# ---------------------------------------------------------------------------
# 写盘守卫
# ---------------------------------------------------------------------------
def check_write_guard() -> CheckResult:
    """校验写盘守卫的放行与拒绝行为。"""
    from ttm.common import paths as ttm_paths

    allowed_ok = 0
    denied_ok = 0
    problems: list[str] = []

    allowed_samples = [
        str(_PROJECT_ROOT),
        f"{_PROJECT_ROOT.as_posix()}/data/probe.txt",
        "data/tier_S/index.jsonl",
        "runs/logs/00_env_check.log",
    ]
    for sample in allowed_samples:
        try:
            ttm_paths.assert_write_allowed(sample)
            allowed_ok += 1
        except Exception as exc:  # noqa: BLE001 - 自检需要捕获一切
            problems.append(f"应放行但被拒: {sample} ({exc})")

    denied_samples = [
        "C:/Windows/Temp/ttm_probe.bin",
        "C:/ttm_probe.bin",
        "Z:/1000hours/probe.bin",
        "E:/outside/probe.bin",
        "D:/other_project/probe.bin",
    ]
    for sample in denied_samples:
        try:
            ttm_paths.assert_write_allowed(sample)
            problems.append(f"应拒绝但被放行: {sample}")
        except ttm_paths.WriteNotAllowedError:
            denied_ok += 1
        except Exception as exc:  # noqa: BLE001
            problems.append(f"拒绝时抛错类型不符: {sample} ({type(exc).__name__})")

    if problems:
        return CheckResult("写盘守卫", STATUS_FAIL, "; ".join(problems))
    return CheckResult(
        "写盘守卫",
        STATUS_PASS,
        f"放行 {allowed_ok}/{len(allowed_samples)}，拒绝 {denied_ok}/{len(denied_samples)}（C:/Z:/项目外 D:）",
    )


def check_write_probe() -> CheckResult:
    """在 cache / runs / data / tempfile 四处真实写-读-删一轮。"""
    from ttm.common import paths as ttm_paths

    payload = b"ttm-env-check-payload"
    targets: list[Path] = []

    # 1) 显式探针：cache / runs/logs / data
    explicit = [
        _cache_root() / "tmp" / "probe_env_check.bin",
        _PROJECT_ROOT / "runs" / "logs" / "probe_env_check.bin",
        _PROJECT_ROOT / "data" / "probe_env_check.bin",
    ]
    for raw in explicit:
        try:
            targets.append(ttm_paths.resolve_out(raw, mkparents=True))
        except Exception as exc:  # noqa: BLE001
            return CheckResult("写盘探针", STATUS_FAIL, f"resolve_out 失败 {raw}: {exc}")

    # 2) 隐式探针：tempfile（验证 TMP/TEMP 重定向真的生效）
    try:
        fd, tmp_name = tempfile.mkstemp(prefix="ttm_probe_", suffix=".bin")
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        targets.append(Path(tmp_name))
    except Exception as exc:  # noqa: BLE001
        return CheckResult("写盘探针", STATUS_FAIL, f"tempfile.mkstemp 失败: {exc}")

    # 3) 写入 + 读回校验（tempfile 已写过，其余三个现在写）
    for target in targets[:3]:
        try:
            target.write_bytes(payload)
        except Exception as exc:  # noqa: BLE001
            return CheckResult("写盘探针", STATUS_FAIL, f"写入失败 {target}: {exc}")

    bad: list[str] = []
    for target in targets:
        CREATED_PATHS.append(str(target))
        try:
            content = target.read_bytes()
        except Exception as exc:  # noqa: BLE001
            bad.append(f"读回失败 {target}: {exc}")
            continue
        if content[: len(payload)] != payload:
            bad.append(f"内容不一致 {target}")
        drive = os.path.splitdrive(str(target))[0].replace(":", "").lower()
        if drive != "d":
            bad.append(f"探针落在非 D 盘: {target}")

    # 清理
    for target in targets:
        try:
            target.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - 清理失败不影响结论
            pass

    if bad:
        return CheckResult("写盘探针", STATUS_FAIL, "; ".join(bad))
    return CheckResult(
        "写盘探针", STATUS_PASS, f"{len(targets)} 个探针写-读-删成功，全部在 D 盘", {"probes": len(targets)}
    )


def check_no_c_drive_writes() -> CheckResult:
    """基于真实落盘记录断言 C 盘零写入。"""
    offenders = [
        p for p in CREATED_PATHS if os.path.splitdrive(p)[0].replace(":", "").lower() != "d"
    ]
    if offenders:
        return CheckResult("C 盘零写入", STATUS_FAIL, "发现 C 盘/非 D 盘写入: " + ", ".join(offenders))
    return CheckResult(
        "C 盘零写入",
        STATUS_PASS,
        f"{len(CREATED_PATHS)} 个真实写入点全部位于 D:/TTM（0 个落在 C:）",
        {"created": list(CREATED_PATHS)},
    )


def check_disk_space() -> CheckResult:
    """检查 D 盘剩余空间（PRD 要求总占用 ≤ 20 GB，余量应远大于此）。"""
    try:
        usage = shutil.disk_usage(str(_PROJECT_ROOT))
    except Exception as exc:  # noqa: BLE001
        return CheckResult("磁盘空间", STATUS_WARN, f"无法读取: {exc}")
    free_gb = usage.free / (1024**3)
    status = STATUS_PASS if free_gb >= 20 else STATUS_WARN
    return CheckResult(
        "磁盘空间", status, f"D:/TTM 所在盘剩余 {free_gb:.1f} GB（PRD 预算 20 GB）", {"free_gb": free_gb}
    )


# ---------------------------------------------------------------------------
# 解释器 / 依赖 / torch
# ---------------------------------------------------------------------------
def check_python() -> CheckResult:
    """检查解释器版本与位置是否符合预期。"""
    version = sys.version_info
    detail = (
        f"Python {version.major}.{version.minor}.{version.micro} @ {sys.executable} "
        f"({platform.system()} {platform.machine()})"
    )
    if version < (3, 10):
        return CheckResult("解释器", STATUS_FAIL, detail + "（要求 >= 3.10）")
    venv = os.environ.get("VIRTUAL_ENV", "")
    warnings: list[str] = []
    if ".venv" not in sys.executable.replace("\\", "/") and ".venv" not in venv.replace("\\", "/"):
        warnings.append("当前解释器不在 D:/TTM/.venv 内，请先 source/执行 scripts/env.ps1")
    if warnings:
        return CheckResult("解释器", STATUS_WARN, detail + "；" + "；".join(warnings))
    return CheckResult("解释器", STATUS_PASS, detail)


#: 直接依赖清单（TTM-ARCH.md §6）：(import 名, pip 包名, 用于说明)。
REQUIRED_DEPS: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numpy", "数值主载体"),
    ("scipy", "scipy", "Rotation 欧拉/矩阵/四元数"),
    ("yaml", "pyyaml", "配置加载"),
    ("tqdm", "tqdm", "进度条"),
    ("orjson", "orjson", "JSONL 高速读写"),
    ("fastapi", "fastapi", "推理服务"),
    ("uvicorn", "uvicorn", "ASGI 服务器"),
    ("transformers", "transformers", "bge-small-zh 加载"),
    ("safetensors", "safetensors", "安全权重序列化"),
    ("pytest", "pytest", "测试"),
)


def check_dependencies() -> CheckResult:
    """检查除 torch 外的直接依赖是否可 import，并记录版本。"""
    versions: dict[str, str] = {}
    missing: list[str] = []
    for module_name, pip_name, _purpose in REQUIRED_DEPS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{pip_name}({type(exc).__name__})")
            continue
        versions[pip_name] = str(getattr(module, "__version__", "?"))
    if missing:
        return CheckResult(
            "依赖", STATUS_FAIL, "缺失: " + ", ".join(missing), {"versions": versions}
        )
    summary = ", ".join(f"{name}={ver}" for name, ver in versions.items())
    return CheckResult("依赖", STATUS_PASS, f"{len(versions)}/{len(REQUIRED_DEPS)} 就绪: {summary}", {"versions": versions})


def check_torch(strict: bool) -> list[CheckResult]:
    """检查 torch 与 CUDA，并真实跑一次 GPU 矩阵乘。

    Args:
        strict: ``True`` 时 torch 缺失/CUDA 不可用记 FAIL，否则记 SKIP。

    Returns:
        两个结果项（torch 本体、GPU 计算）。
    """
    try:
        import torch  # noqa: PLC0415 - 延迟导入是有意为之
    except Exception as exc:  # noqa: BLE001
        status = STATUS_FAIL if strict else STATUS_SKIP
        return [
            CheckResult("torch", status, f"未安装或导入失败: {type(exc).__name__}: {exc}"),
            CheckResult("torch CUDA", status, "SKIP_IF_NO_TORCH（等待 torch 安装完成后回归）"),
        ]

    torch_version = getattr(torch, "__version__", "?")
    cuda_available = bool(torch.cuda.is_available())
    torch_result = CheckResult(
        "torch",
        STATUS_PASS,
        f"torch={torch_version}, cuda_build={getattr(torch.version, 'cuda', None)}, "
        f"cuda_available={cuda_available}",
        {"torch": torch_version, "cuda_available": cuda_available},
    )
    if not cuda_available:
        status = STATUS_FAIL if strict else STATUS_SKIP
        return [torch_result, CheckResult("torch CUDA", status, "torch.cuda.is_available() == False")]

    try:
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        vram_gb = properties.total_memory / (1024**3)
        a = torch.arange(1024, dtype=torch.float32, device=device).reshape(32, 32)
        b = torch.eye(32, dtype=torch.float32, device=device)
        c = a @ b
        torch.cuda.synchronize()
        max_err = float((c - a).abs().max().item())
        detail = (
            f"{properties.name}（{vram_gb:.1f} GB）、32x32 矩阵乘 max_err={max_err:.3e}"
        )
        status = STATUS_PASS if max_err < 1e-6 else STATUS_FAIL
        return [
            torch_result,
            CheckResult("torch CUDA", status, detail, {"device": properties.name, "vram_gb": vram_gb}),
        ]
    except Exception as exc:  # noqa: BLE001
        return [torch_result, CheckResult("torch CUDA", STATUS_FAIL, f"GPU 计算失败: {exc}")]


def check_ttm_package() -> CheckResult:
    """检查 ``ttm`` 包可导入且版本契约常量可读。"""
    try:
        import ttm  # noqa: PLC0415
        from ttm.common import logutil, paths as ttm_paths, seed  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return CheckResult("ttm 包", STATUS_FAIL, f"import 失败: {type(exc).__name__}: {exc}")

    try:
        effective_seed = seed.set_seed(123)
        logger = logutil.get_logger("ttm.envcheck", script="00_env_check")
        logger.info("env_check: logger 双写已就绪")
        log_path = logutil.log_file_path("00_env_check")
        root = ttm_paths.project_root()
        layout = getattr(ttm, "LAYOUT_VERSION", None)
        ttm_paths.assert_write_allowed(log_path)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("ttm 包", STATUS_FAIL, f"基础调用失败: {type(exc).__name__}: {exc}")

    CREATED_PATHS.append(str(log_path))
    return CheckResult(
        "ttm 包",
        STATUS_PASS,
        f"ttm {ttm.__version__}, layout_version={layout}, root={root.as_posix()}, "
        f"seed={effective_seed}, log={log_path.name}",
        {"version": ttm.__version__, "layout_version": layout, "root": str(root)},
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_checks(strict: bool = False) -> tuple[list[CheckResult], int]:
    """执行全部自检项。

    Args:
        strict: 是否把 torch 的 SKIP 升级为 FAIL。

    Returns:
        ``(结果列表, 退出码)``。
    """
    results: list[CheckResult] = []
    env_result = apply_cache_env(force=True)
    results.extend(
        [
            check_python(),
            env_result,
            check_cache_env(),
            check_disk_space(),
            check_ttm_package(),
            check_write_guard(),
            check_write_probe(),
            check_no_c_drive_writes(),
            check_dependencies(),
        ]
    )
    results.extend(check_torch(strict))
    exit_code = 0 if all(r.status != STATUS_FAIL for r in results) else 1
    return results, exit_code


def print_report(results: list[CheckResult], as_json: bool) -> None:
    """打印自检报告。

    Args:
        results: 自检结果列表。
        as_json: ``True`` 时输出 JSON（便于机器消费）。
    """
    if as_json:
        payload = {
            "results": [
                {"name": r.name, "status": r.status, "detail": r.detail, "extra": r.extra}
                for r in results
            ],
            "summary": _summary(results),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return

    print("=" * 96)
    print("TTM 环境自检 · T01")
    print("=" * 96)
    for result in results:
        print(result.line())
    summary = _summary(results)
    print("-" * 96)
    print(
        f"合计 {summary['total']} 项：PASS {summary['pass']} / WARN {summary['warn']} / "
        f"SKIP {summary['skip']} / FAIL {summary['fail']}"
    )
    if summary["fail"] == 0:
        print("结论：环境自检通过（退出码 0）。C 盘零写入已验证。")
    else:
        print("结论：存在 FAIL 项，请按上表修复后重跑。")
    print("=" * 96)


def _summary(results: list[CheckResult]) -> dict[str, int]:
    """统计各状态数量。"""
    return {
        "total": len(results),
        "pass": sum(1 for r in results if r.status == STATUS_PASS),
        "warn": sum(1 for r in results if r.status == STATUS_WARN),
        "skip": sum(1 for r in results if r.status == STATUS_SKIP),
        "fail": sum(1 for r in results if r.status == STATUS_FAIL),
    }


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码。
    """
    parser = argparse.ArgumentParser(description="TTM 环境与写盘守卫自检（T01）")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="把 torch 未安装 / CUDA 不可用视为 FAIL（默认视为 SKIP）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = parser.parse_args(argv)

    results, exit_code = run_checks(strict=args.strict)
    print_report(results, as_json=args.json)

    # 收尾：刷盘并关闭日志 handler
    try:
        from ttm.common import logutil  # noqa: PLC0415

        logutil.close_all()
    except Exception:  # noqa: BLE001
        pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
