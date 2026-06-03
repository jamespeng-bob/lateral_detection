#!/usr/bin/env bash
# install_torch.sh — install the right CUDA-aware torch wheel for THIS server.
#
# Why a separate script
# ---------------------
# torch + torchvision wheels are CUDA-major-version-specific, and the
# wheel-index URL is the only way to select between them. Different servers
# have different drivers, so a single pinned `--index-url` in requirements.txt
# would break on at least one of them. Servers we currently target:
#
#   bobyard-server-5090   driver 12.8   →   cu128 wheels   (torch 2.9.x)
#   bobyard-server-6000   driver 12.4   →   cu124 wheels   (torch 2.6.x latest)
#
# Run BEFORE `pip install -r requirements.txt`:
#
#   source .venv/bin/activate
#   bash scripts/install_torch.sh
#   pip install -r requirements.txt
#
# Behaviour
# ---------
# 1. Reads the driver's reported CUDA version from `nvidia-smi`.
# 2. Maps it to a PyTorch wheel index ("12.4" → "cu124").
# 3. `pip install torch torchvision --index-url https://download.pytorch.org/whl/<index>`.
# 4. Verifies `torch.cuda.is_available()`.
#
# On macOS or any CPU-only host (no nvidia-smi), falls back to the CPU wheel
# from PyPI — works locally without manual edits.
#
# To force a specific wheel index (e.g. pin reproducibility after a driver
# upgrade), pass it as the first argument:
#
#   bash scripts/install_torch.sh cu124

set -euo pipefail

# -- 1. Pick wheel index ------------------------------------------------------

if [[ $# -ge 1 ]]; then
    INDEX="$1"
    echo "[install_torch] using user-specified index: $INDEX"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[install_torch] nvidia-smi not found — installing CPU-only torch."
    INDEX="cpu"
else
    # `nvidia-smi` header contains e.g. "CUDA Version: 12.4". This is the
    # MAX CUDA the driver supports, which is what wheel-index selection
    # needs to match.
    CUDA_VER=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' \
                          | head -1 \
                          | awk '{print $3}')
    if [[ -z "${CUDA_VER}" ]]; then
        echo "[install_torch] ERROR: could not parse CUDA version from nvidia-smi." >&2
        exit 1
    fi
    MAJOR=${CUDA_VER%.*}
    MINOR=${CUDA_VER#*.}
    INDEX="cu${MAJOR}${MINOR}"
    echo "[install_torch] driver supports CUDA ${CUDA_VER}  →  wheel index ${INDEX}"
fi

if [[ "${INDEX}" == "cpu" ]]; then
    URL="https://download.pytorch.org/whl/cpu"
else
    URL="https://download.pytorch.org/whl/${INDEX}"
fi

# -- 2. Install ---------------------------------------------------------------

echo "[install_torch] pip install torch torchvision --index-url ${URL}"
pip install torch torchvision --index-url "${URL}"

# -- 3. Verify ----------------------------------------------------------------

echo ""
echo "[install_torch] verifying..."
python - <<'PY'
import torch
print(f"  torch       : {torch.__version__}")
print(f"  cuda avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"    cuda:{i}     : {p.name}  sm_{p.major}{p.minor}  "
              f"{p.total_memory / 1e9:.1f} GB")
PY

echo ""
echo "[install_torch] done. Next: pip install -r requirements.txt"
