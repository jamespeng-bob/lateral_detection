# Agent guide — lateral_detection

Persistent instructions for any AI coding agent (Cursor, Claude Code, Codex, …)
working in this repository. Read this before touching anything.

---

## Project scope

This repo is **`lateral_detection`** only. Two sister repos exist at the same
parent path (`graph_irrigation/`, `solid_lateral_deployment/`) — they are
**reference / archived** and must not be modified unless the user explicitly
asks. Reading them for context is fine.

The dataset lives **outside** this repo at `../datasets/` (sibling of the repo
root). The default `configs/base.yaml` uses this relative path and assumes the
layout is preserved.

---

## Git workflow (most important rule)

**Local Mac** is the only place that pushes:

- Edit files locally. Commit locally. `git push` from local.
- Commit messages explain *why*, not just *what* (see existing history for tone).
- Never commit credentials, model weights, run outputs, or notebook execution
  results (the `.gitignore` already covers these; verify before adding).

**Server** (`bobyard-server-5090`) is execute-only:

- The agent may SSH in and run: `git pull`, `python train.py`, `nvidia-smi`,
  `tail -f` on logs, kill jobs it spawned, quick `python -c "..."` smoke tests.
- The agent must **never** run `git commit` / `git push` / `git config` on the
  server. The Linux account is shared with a colleague — pushes from there
  would land under the wrong identity.
- If something on the server needs to be saved to the repo (e.g., a new config),
  mirror the change locally first, push from local, then `git pull` on the
  server.

For sanity-check or short server commands, the agent may SSH from its sandbox
(`ssh bobyard-server-5090 '...'`). For long-running training, prefer asking the
user to launch it in their own SSH terminal so they can `tail -f` directly.

---

## Secrets

The GCP service account JSON
(`credentials/inference-428300-7af7f5da75dc.json`) is the only sensitive file
in the tree. It is gitignored via three overlapping rules
(`inference-*.json`, `*-service-account*.json`, `credentials/`). If you ever
see it appear under `git status` as staged, stop and tell the user.

---

## Environment pins (these matter)

- **Server GPU**: 2× RTX 5090, compute capability **sm_120** (Blackwell).
- **Server CUDA driver**: 12.8.
- **PyTorch wheel**: must be **cu128**. cu126 and below lack sm_120 kernels;
  cu130 requires a CUDA 13 driver we don't have.
- **Pinned in `requirements.txt`**: `torch==2.9.1`, `torchvision==0.24.1` —
  these are the highest pair the cu128 index ships for Python 3.13.
  Do not bump unless `https://download.pytorch.org/whl/cu128/torch/` has been
  re-checked.
- **Server Python**: 3.13. **Local Python**: 3.9 (macOS system). Use
  `from __future__ import annotations` everywhere so 3.10-style union types
  don't break local imports.
- **macOS bootstrap quirk**: the venv must be created with `python3 -m venv`,
  not `python` (which doesn't exist on bare macOS).

---

## Training launch

- Single GPU:  `python train.py --overlay configs/<exp>.yaml --device cuda:0`
- DDP (multi-GPU on one host):
      `torchrun --nproc-per-node=2 --master-port=29500 train.py --overlay configs/<exp>.yaml`
- `training.batch_size` in any config is **per-GPU**. Effective global
  batch under DDP = `batch_size * world_size`. SyncBatchNorm is enabled
  automatically when `world_size > 1` so BN running stats are computed
  on the global effective batch.
- Use a different `--master-port` per concurrent torchrun (rapid
  sequential launches occasionally fail to release the port in time).

## Code conventions

- Paths: `pathlib.Path`, never `os.path.join` and never absolute paths in code.
  Relative paths resolve from CWD or the script's parent — see
  `symbols/call_symbol_localizer.py` for the canonical pattern.
- Modules in `data/`, `models/`, `training/` are imported by the trainer.
  Notebooks under `notebooks/` are inspection-only; never put logic the trainer
  needs inside a notebook.
- Notebooks must have outputs stripped on commit (`nbstripout --install` once
  per clone).
- Augmentation policy: thin lateral lines (~4 px) are sensitive. Geometric
  augmentations are restricted to horizontal flip, vertical flip, and
  rotations by multiples of 90°. Do not introduce elastic transforms,
  arbitrary rotation, perspective warp, or heavy color jitter without
  benchmarking the impact on Dice.
- Loss: foreground per tile is ~0.1–5 %. Plain BCE collapses; always pair with
  Dice. `bce_pos_weight=1.0` is the default — only bump if Dice plateaus low.

---

## Process

- For tasks larger than a one-line edit, propose a short plan first and wait
  for confirmation. Match the existing voice in commit messages and READMEs.
- Run `ast.parse` / module imports to smoke-test new Python before declaring
  it done.
- Don't fabricate file contents, command output, or training metrics. If
  something needs to be verified on the server, SSH and run it.
