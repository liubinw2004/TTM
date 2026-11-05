# TTM 环境变量注入（PowerShell）
#
# 用法（必须在当前会话 dot-source，前面带点）：
#   . D:/TTM/scripts/env.ps1
#
# 作用：
#   1. 激活项目内 venv（D:/TTM/.venv），绝不使用系统 Python；
#   2. 把所有缓存/临时目录重定向到 D:/TTM/.cache —— 保证 C: 盘零写入；
#   3. 把 src/ 加入 PYTHONPATH（即使未做 editable install 也能 import ttm）。
#
# 硬约束：只允许写 D:/TTM；Z: 只读；C: 零写入。

$ErrorActionPreference = "Stop"

# --- 项目根（可用外部 TTM_ROOT 覆盖，但必须仍指向 D:/TTM）--------------------
if (-not $env:TTM_ROOT) {
    $env:TTM_ROOT = "D:/TTM"
}
$TTM_ROOT = $env:TTM_ROOT
$TTM_CACHE = Join-Path $TTM_ROOT ".cache"

# --- 缓存重定向 ---------------------------------------------------------------
$env:PIP_CACHE_DIR = Join-Path $TTM_CACHE "pip"
$env:HF_HOME = Join-Path $TTM_CACHE "hf"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $TTM_CACHE "hf/hub"
$env:TRANSFORMERS_CACHE = Join-Path $TTM_CACHE "hf/transformers"
$env:TORCH_HOME = Join-Path $TTM_CACHE "torch"
$env:XDG_CACHE_HOME = Join-Path $TTM_CACHE "xdg"
$env:MPLCONFIGDIR = Join-Path $TTM_CACHE "matplotlib"

# 临时目录：TMPDIR 供 POSIX 工具，TEMP/TMP 供 Windows API 与 pip 子进程
$env:TMPDIR = Join-Path $TTM_CACHE "tmp"
$env:TEMP = $env:TMPDIR
$env:TMP = $env:TMPDIR

# 离线优先：模型权重已在 D 盘缓存后可完全断网
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:TOKENIZERS_PARALLELISM = "false"

# --- 目录准备 -----------------------------------------------------------------
foreach ($d in @($env:PIP_CACHE_DIR, $env:HF_HOME, $env:TORCH_HOME, $env:TMPDIR)) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
    }
}

# --- venv 激活 -----------------------------------------------------------------
$TTM_VENV = Join-Path $TTM_ROOT ".venv"
$TTM_PY = Join-Path $TTM_VENV "Scripts/python.exe"
if (Test-Path $TTM_PY) {
    $env:VIRTUAL_ENV = $TTM_VENV
    $env:PATH = (Join-Path $TTM_VENV "Scripts") + ";" + $env:PATH
}
else {
    Write-Warning "[env.ps1] 未找到 $TTM_PY，请先创建 venv"
}

# --- PYTHONPATH ---------------------------------------------------------------
$env:PYTHONPATH = (Join-Path $TTM_ROOT "src") + $(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" })

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "[env.ps1] TTM_ROOT=$env:TTM_ROOT"
Write-Host "[env.ps1] PIP_CACHE_DIR=$env:PIP_CACHE_DIR"
Write-Host "[env.ps1] HF_HOME=$env:HF_HOME"
Write-Host "[env.ps1] TORCH_HOME=$env:TORCH_HOME"
Write-Host "[env.ps1] TMPDIR=$env:TMPDIR"
Write-Host "[env.ps1] python=$TTM_PY"
