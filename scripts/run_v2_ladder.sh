#!/usr/bin/env bash
# Run the three v2 backbone-comparison experiments sequentially on a single GPU.
#
# Usage on the server:
#   tmux new -s v2ladder
#   cd ~/james/lateral_detection
#   source .venv/bin/activate
#   bash scripts/run_v2_ladder.sh [cuda:0]
#   # Ctrl-B D to detach; tmux attach -t v2ladder to reattach
#
# Each run lands in runs/v2{a,b,c}_*/, with a separate log file alongside.
# If one run fails, the script continues with the next so you don't waste the
# whole night on a stuck encoder.

set -uo pipefail   # no -e: we WANT to continue on individual-run failure

DEVICE="${1:-cuda:0}"
mkdir -p runs

declare -a VARIANTS=(
    "v2a:configs/train_v2a.yaml"
    "v2b:configs/train_v2b.yaml"
    "v2c:configs/train_v2c.yaml"
)

echo "============================================================"
echo " v2 backbone ladder"
echo "   device:    ${DEVICE}"
echo "   variants:  v2a (efficientnet_b3)  v2b (hgnetv2_b4)  v2c (mit_b2)"
echo "============================================================"
echo

summary=""

for entry in "${VARIANTS[@]}"; do
    tag="${entry%%:*}"
    overlay="${entry##*:}"
    log="runs/${tag}.log"

    echo
    echo "==== [${tag}] starting at $(date '+%F %T') ===="
    echo "     overlay: ${overlay}"
    echo "     log:     ${log}"
    echo

    start=$SECONDS
    if python train.py --overlay "${overlay}" --device "${DEVICE}" 2>&1 | tee "${log}"; then
        status="OK"
    else
        status="FAILED"
    fi
    elapsed=$((SECONDS - start))

    line=$(printf "  %-4s  status=%-6s  elapsed=%02dh%02dm  log=%s" \
        "${tag}" "${status}" $((elapsed / 3600)) $(((elapsed % 3600) / 60)) "${log}")
    summary+="${line}"$'\n'
    echo
    echo "==== [${tag}] ${status} (${elapsed}s) ===="
done

echo
echo "============================================================"
echo " v2 ladder summary"
echo "============================================================"
echo "${summary}"
echo "Compare runs/{v2a,v2b,v2c}*/history.{json,png} to pick a winner."
