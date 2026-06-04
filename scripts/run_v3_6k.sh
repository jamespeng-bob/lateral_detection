#!/usr/bin/env bash
# v3 ladder on bobyard-server-6000 — 4 runs, 2 GPUs, dynamic scheduling.
#
# Runs (all single-GPU, 120 epochs, cosine LR with 5-epoch warmup):
#
#   v3a-b  HGNetv2-B4 + BCE+Dice+clDice (single aux loss)      bs=16
#   v3a-c  MiT-B2     + BCE+Dice+clDice                        bs=8
#   v3b-b  HGNetv2-B4 + BCE+Dice+Lovász+clDice (composite)     bs=16
#   v3b-c  MiT-B2     + BCE+Dice+Lovász+clDice                 bs=8
#
# 2 (encoders) × 2 (loss recipes) factorial — lets us read off whether
# gains are encoder-specific or loss-specific.
#
# Scheduling (uses `wait -n` so we never have an idle GPU between jobs):
#
#   Phase 1:  v3a-b on cuda:0  +  v3a-c on cuda:1   in parallel
#   Phase 2:  whichever finishes first → next job on its GPU
#             whichever GPU frees up next → final remaining job
#
# This keeps both GPUs busy continuously. Wall-clock ≈ ~12-14 h end-to-end
# (each run is ~6-7 h at 120 epochs given v2_6k's 3-4 min/epoch + the
# clDice loss adding ~30% per train step due to skel iterations).
#
# Usage (run inside tmux on bobyard-server-6000):
#   tmux new -s v3_6k
#   cd ~/james/lateral_detection && source .venv/bin/activate
#   bash scripts/run_v3_6k.sh
#   # Ctrl-B then D to detach; `tmux attach -t v3_6k` to reattach.

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs

trap_cleanup() {
    echo "[v3 6k] received signal; killing background train jobs..."
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null
    exit 130
}
trap trap_cleanup INT TERM

run_one() {
    # Args: tag  device  overlay
    local tag="$1"
    local device="$2"
    local overlay="$3"
    local log="runs/${tag}_6k.log"
    echo "[$(date '+%F %T')] [${tag}] starting on ${device} → ${log}"
    python train.py --overlay "${overlay}" --device "${device}" > "${log}" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] [${tag}] exited rc=${rc}"
    return ${rc}
}

# Queue of jobs left to run. Format: "tag device overlay"
# (device is rewritten per-iteration based on which GPU just freed up)
declare -a QUEUE=(
    "v3a_b configs/train_v3a_b.yaml"
    "v3b_b configs/train_v3b_b.yaml"
    "v3a_c configs/train_v3a_c.yaml"
    "v3b_c configs/train_v3b_c.yaml"
)

echo "============================================================"
echo " v3 ladder on bobyard-server-6000   $(date '+%F %T')"
echo "   v3a-b  hgnetv2_b4  bs=16  BCE+Dice+clDice            120 ep cosine"
echo "   v3a-c  mit_b2      bs=8   BCE+Dice+clDice            120 ep cosine"
echo "   v3b-b  hgnetv2_b4  bs=16  BCE+Dice+Lovász+clDice    120 ep cosine"
echo "   v3b-c  mit_b2      bs=8   BCE+Dice+Lovász+clDice    120 ep cosine"
echo "   2 in parallel; remaining 2 launched on freed GPUs"
echo "============================================================"
echo

# --- Phase 1: launch first two on cuda:0 + cuda:1 ----------------------
JOB1="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
TAG1="${JOB1%% *}"; OVL1="${JOB1#* }"
run_one "${TAG1}" "cuda:0" "${OVL1}" &
PID0=$!; GPU0_TAG="${TAG1}"

JOB2="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
TAG2="${JOB2%% *}"; OVL2="${JOB2#* }"
run_one "${TAG2}" "cuda:1" "${OVL2}" &
PID1=$!; GPU1_TAG="${TAG2}"

echo "[v3 6k] launched ${TAG1} on cuda:0 (pid=${PID0})  +  ${TAG2} on cuda:1 (pid=${PID1})"
echo "[v3 6k] queue remaining: ${QUEUE[@]:-<empty>}"
echo

# --- Phases 2+: as each job finishes, launch the next on its freed GPU --
while [[ ${#QUEUE[@]} -gt 0 ]]; do
    wait -n "${PID0}" "${PID1}"
    # Which one died?
    if ! kill -0 "${PID0}" 2>/dev/null; then
        FREED="cuda:0"; FREED_PREV="${GPU0_TAG}"
        NEXT="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
        TAG="${NEXT%% *}"; OVL="${NEXT#* }"
        echo "[$(date '+%F %T')] [v3 6k] ${FREED_PREV} done → launching ${TAG} on ${FREED}"
        run_one "${TAG}" "${FREED}" "${OVL}" &
        PID0=$!; GPU0_TAG="${TAG}"
    else
        FREED="cuda:1"; FREED_PREV="${GPU1_TAG}"
        NEXT="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}")
        TAG="${NEXT%% *}"; OVL="${NEXT#* }"
        echo "[$(date '+%F %T')] [v3 6k] ${FREED_PREV} done → launching ${TAG} on ${FREED}"
        run_one "${TAG}" "${FREED}" "${OVL}" &
        PID1=$!; GPU1_TAG="${TAG}"
    fi
    echo "[v3 6k] queue remaining: ${QUEUE[@]:-<empty>}"
done

# Wait for the final two to finish.
wait "${PID0}" 2>/dev/null
wait "${PID1}" 2>/dev/null

echo
echo "============================================================"
echo " v3 ladder finished at $(date '+%F %T')"
echo "============================================================"

for tag in v3a_b v3a_c v3b_b v3b_c; do
    log="runs/${tag}_6k.log"
    printf "  %-6s  log=%s\n" "${tag}" "${log}"
    if [ -f "${log}" ]; then
        last_epoch=$(grep -E '^=== epoch '   "${log}" | tail -1 || true)
        last_val=$(  grep -E '^  val:'       "${log}" | tail -1 || true)
        last_len=$(  grep -E '^  len:'       "${log}" | tail -1 || true)
        [ -n "${last_epoch}" ] && printf "        %s\n" "${last_epoch}"
        [ -n "${last_val}"   ] && printf "        %s\n" "${last_val}"
        [ -n "${last_len}"   ] && printf "        %s\n" "${last_len}"
    fi
    echo
done
echo "Inspect any run with:"
echo "  python -m scripts.inspect_history runs/<save_dir>/"
echo "  python -m scripts.eval_checkpoint --checkpoint runs/<save_dir>/best.pth \\"
echo "      --overlay configs/<overlay>.yaml --device cuda:0"
