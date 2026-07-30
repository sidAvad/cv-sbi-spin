# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

SPIN (Simulation-to-Patient Image-to-Image translation Network) for cardiovascular SBI. Trains paired generators G_sr (sim→real) and G_rs (real→sim) on raw observations (4 pressure waveforms + 5 scalars). At inference: x_real → G_rs → encoder → flow → posterior. Builds on the frozen v3 encoder+flow from `cv-dann-sbi`.

## Relationship to cv-dann-sbi

- Frozen v3 encoder and flow are loaded from `cv-dann-sbi` outputs.
- Same data layout, constants, and waveform format.
- Same real patient data: 802 patients at `/home/sa4604/data/real_data/onebeat_300patients/` on adamant.
- Same sim data: `/media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314` on adamant.

## Key references

- **cv-dann-sbi best model**: `exp-v3_encoder-lipschitz_dann_flow-maf5` — task=12.84, w1=0.85
- **Norm stats**: `norm_stats.json` (sim stats, shared normalization) — symlinked in this repo's root to `~/outputs/cv-dann-sbi/norm_stats.json` on adamant, the actual source of truth

## Directory layout on adamant (see global CLAUDE.md for the full convention)

- `~/projects/cv-sbi-spin/` — this repo, git clone (code only)
- `~/outputs/cv-sbi-spin/<run>/` — checkpoints, logs, `run_info.json` per run; adamant-only, not synced, not git-tracked
- `~/results/cv-sbi-spin/<run>/` — eval scripts' generated images; bidirectionally mutagen-synced with local

## Checkpoint layout

`train_spin.py` saves to `~/outputs/cv-sbi-spin/<run>/checkpoints/<start_ts>/{encoder,flow_net,G_sr,G_rs}.pt`, where `<start_ts>` matches that run's `train_log_<start_ts>.csv` and `run_info_v{version}_<start_ts>.json`. Never overwrites a prior run's checkpoints in place — each training invocation gets its own timestamped subfolder, so re-running the same `--run` name (e.g. after changing hyperparameters) preserves every prior attempt. Eval scripts resolve the latest timestamp under `checkpoints/` by default.

## Versioning and branching convention

Inherited from cv-dann-sbi: version numbers are assigned only when a run survives evaluation. Git branches (`exp/<what-you're-testing>`) for code changes; hyperparameter sweeps commit to main.

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.
- Always commit before running a full experiment.
- Git runs only on local Mac — never commit from adamant.

## Training run conventions

- **Always confirm the output run name/directory with the user before executing any training run.**
- Always commit before starting an `exp_` run so `run_info.json` captures the exact code state.
