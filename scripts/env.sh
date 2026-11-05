# TTM 环境变量注入（bash / Git-Bash）
#
# 用法：
#   source D:/TTM/scripts/env.sh
#
# 作用：
#   1. 激活项目内 venv（D:/TTM/.venv），绝不使用系统 Python；
#   2. 把所有缓存/临时目录重定向到 D:/TTM/.cache —— 保证 C: 盘零写入；
#   3. 把 src/ 加入 PYTHONPATH（即使未做 editable install 也能 import ttm）。
#
# 硬约束：只允许写 D:/TTM；Z: 只读；C: 零写入。

# --- 项目根（可用外部 TTM_ROOT 覆盖，但必须仍指向 D:/TTM）--------------------
if [ -z "${TTM_ROOT:-}" ]; then
    TTM_ROOT="D:/TTM"
fi
export TTM_ROOT

# --- 缓存重定向 ---------------------------------------------------------------
export PIP_CACHE_DIR="${TTM_ROOT}/.cache/pip"
export HF_HOME="${TTM_ROOT}/.cache/hf"
export HUGGINGFACE_HUB_CACHE="${TTM_ROOT}/.cache/hf/hub"
export TRANSFORMERS_CACHE="${TTM_ROOT}/.cache/hf/transformers"
export TORCH_HOME="${TTM_ROOT}/.cache/torch"
export XDG_CACHE_HOME="${TTM_ROOT}/.cache/xdg"
export MPLCONFIGDIR="${TTM_ROOT}/.cache/matplotlib"

# 临时目录：TMPDIR 供 POSIX 工具，TEMP/TMP 供 Windows API 与 pip 子进程
export TMPDIR="${TTM_ROOT}/.cache/tmp"
export TEMP="${TTM_ROOT}/.cache/tmp"
export TMP="${TTM_ROOT}/.cache/tmp"

# 离线优先：模型权重已在 D 盘缓存后可完全断网
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# --- 目录准备 -----------------------------------------------------------------
mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" 2>/dev/null

# --- venv 激活 -----------------------------------------------------------------
TTM_VENV="${TTM_ROOT}/.venv"
if [ -f "${TTM_VENV}/Scripts/python.exe" ]; then
    export VIRTUAL_ENV="${TTM_VENV}"
    export PATH="${TTM_VENV}/Scripts:${PATH}"
elif [ -f "${TTM_VENV}/bin/python" ]; then
    export VIRTUAL_ENV="${TTM_VENV}"
    export PATH="${TTM_VENV}/bin:${PATH}"
else
    echo "[env.sh] 警告: 未找到 ${TTM_VENV} 下的 Python，请先创建 venv" >&2
fi

# --- PYTHONPATH ---------------------------------------------------------------
if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${TTM_ROOT}/src"
else
    export PYTHONPATH="${TTM_ROOT}/src:${PYTHONPATH}"
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "[env.sh] TTM_ROOT=${TTM_ROOT}"
echo "[env.sh] PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "[env.sh] HF_HOME=${HF_HOME}"
echo "[env.sh] TORCH_HOME=${TORCH_HOME}"
echo "[env.sh] TMPDIR=${TMPDIR}"
echo "[env.sh] python=$(command -v python || echo 'not found')"
