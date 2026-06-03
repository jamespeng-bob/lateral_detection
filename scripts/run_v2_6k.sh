#!/usr/bin/env bash
# Re-run v2a + v2b + v2c on bobyard-server-6000 — single-GPU each, batch
# sizes set to the maximum that fits comfortably in 50 GB for each encoder:
#
#   v2a (tu-efficientnet_b3)   bs=8    ← bs=16 OOM'd (>47 GB activations)
#   v2b (tu-hgnetv2_b4)        bs=16   ← fits cleanly, verified
#   v2c (mit_b2)               bs=8    ← bs=16 would OOM (O(N^2) attention)
#
# All three are 2x the corresponding 5090 batch size (v2a/v2c were bs=4,
# v2b was bs=8). LR is held at 1e-4 across the board so the only axis of
# variation in the rerun is encoder × batch-size, not lr.
#
# Why single-GPU
# --------------
# 1. Sidesteps SyncBN entirely. The 5090 DDP attempt for v2a applied SyncBN
#    automatically and broke EfficientNet's training (dice 0.82 → 0.61).
#    Single-GPU = no DDP = no SyncBN to convert.
# 2. We have 2 GPUs and 3 jobs — single-GPU lets us run two variants in
#    parallel on the two cards. Wall-clock is ~9-10 h vs ~15 h for a
#    sequential DDP equivalent (each DDP variant would use both GPUs).
#
# Schedule (uses `wait -n` so we don't waste cycles on the slower of the
# first pair):
#
#   t=0:        v2a on cuda:0  +  v2b on cuda:1  in parallel
#   t≈first:    v2c launches on whichever GPU just freed up
#   t≈last:     summary table printed
#
# Usage (inside tmux on bobyard-server-6000):
#   tmux new -s v2_6k
#   cd ~/james/lateral_detection && source .venv/bin/activate
#   bash scripts/run_v2_6k.sh
#   # Ctrl-B then D to detach; `tmux attach -t v2_6k` to reattach.

set -uo pipefail        # no -e: continue past individual failures

cd "$(dirname "$0")/.."  # always run from the repo root
mkdir -p runs

trap_cleanup() {
    echo "[v2 6k] received signal; killing background train jobs..."
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null
    exit 130
}
trap trap_cleanup INT TERM

run_one() {
    # Args: variant_tag  device  overlay_yaml
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

echo "============================================================"
echo " v2 ladder on bobyard-server-6000   $(date '+%F %T')"
echo "   v2a (tu-efficientnet_b3)  bs=8    single-GPU"
echo "   v2b (tu-hgnetv2_b4)       bs=16   single-GPU"
echo "   v2c (mit_b2)              bs=8    single-GPU"
echo "   v2a + v2b in parallel; v2c on the first GPU to free up"
echo "============================================================"
echo

# --- Phase 1: v2a (cuda:0) + v2b (cuda:1) in parallel ----------------
run_one v2a cuda:0 configs/train_v2a_6k.yaml &
PID_A=$!
GPU_A="cuda:0"
TAG_A="v2a"

run_one v2b cuda:1 configs/train_v2b_6k.yaml &
PID_B=$!
GPU_B="cuda:1"
TAG_B="v2b"

echo "[v2 6k] launched v2a (pid=${PID_A}) on cuda:0  and  v2b (pid=${PID_B}) on cuda:1."
echo "[v2 6k] waiting for whichever finishes first..."

# wait -n returns when ANY listed pid exits. Requires bash >= 4.3 (we have 5.x).
wait -n "${PID_A}" "${PID_B}"
FIRST_RC=$?

if ! kill -0 "${PID_A}" 2>/dev/null; then
    FIRST_TAG="${TAG_A}"; FREED_GPU="${GPU_A}"
    REMAINING_PID="${PID_B}"; REMAINING_TAG="${TAG_B}"
else
    FIRST_TAG="${TAG_B}"; FREED_GPU="${GPU_B}"
    REMAINING_PID="${PID_A}"; REMAINING_TAG="${TAG_A}"
fi
echo "[v2 6k] ${FIRST_TAG} finished first (rc=${FIRST_RC}); freeing ${FREED_GPU}"

# --- Phase 2: v2c on the freed GPU -----------------------------------
run_one v2c "${FREED_GPU}" configs/train_v2c_6k.yaml &
PID_C=$!
echo "[v2 6k] launched v2c on ${FREED_GPU} (pid=${PID_C});  ${REMAINING_TAG} continues on the other GPU (pid=${REMAINING_PID})"

wait "${REMAINING_PID}"; REMAINING_RC=$?
wait "${PID_C}";         C_RC=$?

echo
echo "============================================================"
echo " v2 ladder finished at $(date '+%F %T')"
echo "============================================================"

for tag in v2a v2b v2c; do
    log="runs/${tag}_6k.log"
    printf "  %-4s  log=%s\n" "${tag}" "${log}"
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
echo "Each variant's checkpoint + history lives in runs/<variant>_*_bcedice_6k/."
echo "Compare against the 5090 ladder via scripts/eval_checkpoint.py or"
echo "scripts/inspect_history.py."
