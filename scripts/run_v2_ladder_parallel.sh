#!/usr/bin/env bash
# Launches v2a + v2b in PARALLEL on cuda:0 + cuda:1, then v2c on whichever
# GPU finishes first. Total wall time ~6-7 hours instead of ~10 sequential.
#
# Sister of run_v2_ladder.sh (which is the safe-and-slow sequential variant).
# Both write each variant's log to runs/v2{a,b,c}.log so analysis is identical.
#
# Usage on the server (inside tmux so it survives SSH disconnect):
#   tmux new -s v2ladder
#   cd ~/james/lateral_detection && source .venv/bin/activate
#   bash scripts/run_v2_ladder_parallel.sh
#   # Ctrl-B then D to detach; tmux attach -t v2ladder to reattach
#
# REQUIRES BOTH GPUs to be free. Coordinate with anyone else using the box.

set -uo pipefail        # no -e: we continue on individual-run failure

mkdir -p runs

# Clean up children on Ctrl-C / SIGTERM so we don't leave orphan training
# jobs eating both GPUs after the user kills the script.
cleanup() {
    echo "[parallel-ladder] received signal; killing background train jobs..."
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null
    exit 130
}
trap cleanup INT TERM

echo "============================================================"
echo " v2 parallel ladder  --  $(date '+%F %T')"
echo "   v2a on cuda:0   tu-efficientnet_b3   (12.6M)"
echo "   v2b on cuda:1   tu-hgnetv2_b4        (22.5M)"
echo "   v2c launched on whichever GPU frees up first  --  mit_b2 (27.5M)"
echo "============================================================"
echo

# Stage 1: v2a on cuda:0, v2b on cuda:1, simultaneously.
python train.py --overlay configs/train_v2a.yaml --device cuda:0 \
    > runs/v2a.log 2>&1 &
PID_A=$!

python train.py --overlay configs/train_v2b.yaml --device cuda:1 \
    > runs/v2b.log 2>&1 &
PID_B=$!

t_stage1_start=$(date +%s)
echo "[$(date +%T)] v2a pid=$PID_A on cuda:0   /   v2b pid=$PID_B on cuda:1"
echo "             tail -f runs/v2a.log  (or v2b)  to watch progress"
echo

# Wait for whichever of {v2a, v2b} finishes first.
wait -n $PID_A $PID_B 2>/dev/null || true
elapsed=$(( $(date +%s) - t_stage1_start ))

# Decide which GPU is now free.
# `kill -0 PID` returns 0 if the process is still alive; non-zero if it's gone.
if kill -0 $PID_A 2>/dev/null; then
    free_device="cuda:1"
    first_done="v2b"
else
    free_device="cuda:0"
    first_done="v2a"
fi
echo "[$(date +%T)] ${first_done} finished (after ${elapsed}s); launching v2c on ${free_device}"

# Stage 2: v2c on the freed GPU.
python train.py --overlay configs/train_v2c.yaml --device "${free_device}" \
    > runs/v2c.log 2>&1 &
PID_C=$!
echo "[$(date +%T)] v2c pid=$PID_C on ${free_device}"
echo

# Wait for everything still running. Already-exited PIDs return immediately.
wait $PID_A $PID_B $PID_C 2>/dev/null || true

# Final summary: pull the last `=== epoch ===` line + last `val:` line from
# each log so the user gets an at-a-glance view without grepping themselves.
echo
echo "============================================================"
echo " v2 parallel ladder finished at $(date '+%F %T')"
echo "============================================================"
for tag in v2a v2b v2c; do
    log="runs/${tag}.log"
    if [ -f "$log" ]; then
        last_epoch=$(grep -E '^=== epoch ' "$log" | tail -1 || true)
        last_val=$(grep -E '^  val:'      "$log" | tail -1 || true)
        last_len=$(grep -E '^  len:'      "$log" | tail -1 || true)
        printf "  %-4s  %s\n" "$tag" "${last_epoch:-(no epoch lines)}"
        [ -n "$last_val" ] && printf "        %s\n" "$last_val"
        [ -n "$last_len" ] && printf "        %s\n" "$last_len"
    else
        printf "  %-4s  (no log written)\n" "$tag"
    fi
    echo
done
echo "Compare runs/{v2a,v2b,v2c}_*/history.{json,png} to pick the winner."
