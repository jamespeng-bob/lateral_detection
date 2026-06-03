#!/usr/bin/env bash
# Re-run v2a + v2b + v2c on bobyard-server-6000 at bs=16 single-GPU.
#
# Why this script exists
# ----------------------
# The original v2 ladder ran on the 5090 (32 GB GPUs), which forced:
#   - bs=8 single-GPU for v2b (HGNetv2-B4)        — OK
#   - bs=4 single-GPU for v2a (EfficientNet-B3)   — too small, OOMs at 8
#   - bs=4 single-GPU for v2c (MiT-B2)            — too small, OOMs at 8
# To match v2b's bs=8 we tried DDP (bs=4 per-GPU * 2 ranks = effective 8),
# which silently auto-applied SyncBN and broke v2a (dice 0.82 → 0.61).
#
# The 6000 has 51 GB GPUs → bs=16 single-GPU fits ALL three encoders. By
# going single-GPU we sidestep the SyncBN gotcha entirely AND can run two
# variants in parallel (one per GPU), so the full ladder finishes in roughly
# half the wall-clock of a sequential DDP ladder.
#
# Schedule
# --------
#   t=0:    launch v2a on cuda:0 (background)
#           launch v2b on cuda:1 (background)
#   t≈first finish:
#           launch v2c on whichever GPU just freed up
#   t=last finish:
#           print per-variant tail and a summary table
#
# Both GPUs are always busy (modulo the brief gap between the first phase's
# first finish and v2c's startup). Total wall-clock ≈ ~5-6 hours, vs ~10-15
# hours for a sequential DDP equivalent.
#
# LR is intentionally LEFT at the base 1.0e-4 — same lr the original v2b
# (bs=8) used. The only axis of variation across this rerun is encoder.
# An lr sweep belongs in v3, not here.
#
# Usage (run inside tmux on bobyard-server-6000):
#   tmux new -s v2bs16
#   cd ~/james/lateral_detection && source .venv/bin/activate
#   bash scripts/run_v2_bs16_6k.sh
#   # Ctrl-B then D to detach; `tmux attach -t v2bs16` to reattach.

set -uo pipefail        # no -e: we want to continue past individual failures

cd "$(dirname "$0")/.."  # always run from the repo root
mkdir -p runs

trap_cleanup() {
    echo "[v2 bs16 6k] received signal; killing background train jobs..."
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
    python train.py --overlay "${overlay}" --device "${device}" \
        > "${log}" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] [${tag}] exited rc=${rc}"
    return ${rc}
}

echo "============================================================"
echo " v2 bs=16 ladder on bobyard-server-6000  $(date '+%F %T')"
echo "   v2a (tu-efficientnet_b3)  bs=16  single-GPU"
echo "   v2b (tu-hgnetv2_b4)       bs=16  single-GPU"
echo "   v2c (mit_b2)              bs=16  single-GPU"
echo "   two in parallel + one on the first freed GPU"
echo "============================================================"
echo

# --- Phase 1: v2a on cuda:0 + v2b on cuda:1 in parallel --------------
run_one v2a cuda:0 configs/train_v2a_bs16.yaml &
PID_A=$!
GPU_A="cuda:0"
TAG_A="v2a"

run_one v2b cuda:1 configs/train_v2b_bs16.yaml &
PID_B=$!
GPU_B="cuda:1"
TAG_B="v2b"

echo "[v2 bs16 6k] launched v2a (pid=${PID_A}) and v2b (pid=${PID_B})."
echo "[v2 bs16 6k] waiting for whichever finishes first..."

# wait -n returns when ANY of the listed pids exits. Requires bash >= 4.3
# (we're on bash 5.x — fine). The wait returns the exit code of the first
# child; we record which one finished by checking who's still running.
wait -n "${PID_A}" "${PID_B}"
FIRST_RC=$?

if ! kill -0 "${PID_A}" 2>/dev/null; then
    FIRST_TAG="${TAG_A}"; FREED_GPU="${GPU_A}"
    REMAINING_PID="${PID_B}"; REMAINING_TAG="${TAG_B}"
else
    FIRST_TAG="${TAG_B}"; FREED_GPU="${GPU_B}"
    REMAINING_PID="${PID_A}"; REMAINING_TAG="${TAG_A}"
fi
echo "[v2 bs16 6k] ${FIRST_TAG} finished first (rc=${FIRST_RC}); freeing ${FREED_GPU}"

# --- Phase 2: launch v2c on the freed GPU ----------------------------
run_one v2c "${FREED_GPU}" configs/train_v2c_bs16.yaml &
PID_C=$!
echo "[v2 bs16 6k] launched v2c on ${FREED_GPU} (pid=${PID_C}); "\
"  ${REMAINING_TAG} still on the other GPU (pid=${REMAINING_PID})"

# Wait for both remaining jobs to finish.
wait "${REMAINING_PID}"; REMAINING_RC=$?
wait "${PID_C}";         C_RC=$?

echo
echo "============================================================"
echo " v2 bs=16 ladder finished at $(date '+%F %T')"
echo "============================================================"

# Summary
for tag in v2a v2b v2c; do
    case ${tag} in
        v2a) dir="runs/v2a_effb3_bcedice_bs16"      ;;
        v2b) dir="runs/v2b_hgnetv2b4_bcedice_bs16"  ;;
        v2c) dir="runs/v2c_mitb2_bcedice_bs16"      ;;
    esac
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
echo "Run history.json + history.png live under each runs/<tag>_bs16/ dir."
echo "Compare against the original 5090 ladder at runs/v2{a,b,c}_*_bcedice/"
echo "(now stored on the 5090 server only)."
