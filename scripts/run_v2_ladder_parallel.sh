#!/usr/bin/env bash
# Re-run v2a + v2c via DistributedDataParallel across both GPUs so the
# effective batch matches v2b's bs=8 (per-GPU bs=4 * world_size=2 = 8).
# v2b is NOT re-run — it already trained at bs=8 single-GPU.
#
# Both v2a and v2c need both GPUs each, so they run SEQUENTIALLY (~7 h total).
# This is the right experimental control: every variant trained at the same
# effective batch size, BatchNorm stats synced across GPUs (SyncBN), so the
# only remaining axis of variation across v2a/v2b/v2c is the encoder.
#
# Usage on the server (inside tmux so it survives SSH disconnect):
#   tmux new -s v2ddp
#   cd ~/james/lateral_detection && source .venv/bin/activate
#   bash scripts/run_v2_ladder_parallel.sh
#   # Ctrl-B then D to detach; tmux attach -t v2ddp to reattach
#
# Each variant's per-GPU output streams into runs/<save_dir>/ as usual.
# tee'd full logs land at runs/{v2a,v2c}_ddp.log.

set -uo pipefail        # no -e: we continue on individual-run failure

mkdir -p runs

# Clean up child train processes on Ctrl-C / SIGTERM so we don't leave
# both GPUs pinned at 100% if the user kills the script.
cleanup() {
    echo "[v2 ddp ladder] received signal; killing background train jobs..."
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null
    exit 130
}
trap cleanup INT TERM

# Distinct master port per run so the second torchrun doesn't try to bind
# to a port still held by the first (which can happen on rapid sequential
# launches on the same host).
MASTER_PORTS=(29500 29501)

echo "============================================================"
echo " v2 DDP retry ladder  --  $(date '+%F %T')"
echo "   v2a (tu-efficientnet_b3,  per-GPU bs=4 * 2 = 8)   sequential"
echo "   v2c (mit_b2,              per-GPU bs=4 * 2 = 8)   sequential"
echo "   each variant uses both cuda:0 and cuda:1 via DDP+SyncBN"
echo "============================================================"
echo

i=0
for variant in v2a v2c; do
    port=${MASTER_PORTS[$i]}
    log="runs/${variant}_ddp.log"
    overlay="configs/train_${variant}.yaml"

    echo "==== [${variant}] starting at $(date '+%F %T')   port=${port}   log=${log} ===="
    t_start=$(date +%s)

    if torchrun \
        --nproc-per-node=2 \
        --master-port=${port} \
        train.py --overlay "${overlay}" 2>&1 | tee "${log}"; then
        status="OK"
    else
        status="FAILED"
    fi

    elapsed=$(( $(date +%s) - t_start ))
    printf "==== [%s] %s after %02dh%02dm ====\n\n" \
        "${variant}" "${status}" $((elapsed / 3600)) $(((elapsed % 3600) / 60))
    i=$((i + 1))
done

# Final at-a-glance summary.
echo
echo "============================================================"
echo " v2 DDP ladder finished at $(date '+%F %T')"
echo "============================================================"
for variant in v2a v2c; do
    log="runs/${variant}_ddp.log"
    if [ -f "$log" ]; then
        last_epoch=$(grep -E '^=== epoch ' "$log" | tail -1 || true)
        last_val=$(  grep -E '^  val:'     "$log" | tail -1 || true)
        last_len=$(  grep -E '^  len:'     "$log" | tail -1 || true)
        printf "  %-4s  %s\n" "$variant" "${last_epoch:-(no epoch lines)}"
        [ -n "$last_val" ] && printf "        %s\n" "$last_val"
        [ -n "$last_len" ] && printf "        %s\n" "$last_len"
    fi
    echo
done
echo "Compare runs/{v2a,v2c}_*/history.{json,png} against the existing"
echo "runs/v2b_hgnetv2b4_bcedice/history.* — every variant now had"
echo "effective bs=8 + SyncBN, so the only axis of variation is the encoder."
