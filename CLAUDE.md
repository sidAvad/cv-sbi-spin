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

- `~/projects/cv-sbi-spin/` — this repo, git clone (core model/training code: `dataset.py`, `models.py`, `train_spin.py`)
- `~/outputs/cv-sbi-spin/<run>/` — checkpoints, logs, `run_info.json` per run; adamant-only, not synced, not git-tracked
- `~/results/cv-sbi-spin/scripts/` — eval/plotting scripts (`eval_common.py` + `eval_*.py`); **not** git-tracked, deliberately — these are thin, disposable, iterated-on-quickly wrappers, unlike the core code in `projects/`. Bidirectionally mutagen-synced with local, so local edits appear on adamant automatically and generated images sync back down the same way.
- `~/results/cv-sbi-spin/<run>/` — eval scripts' generated images, one folder per run

## Checkpoint layout

`train_spin.py` saves to `~/outputs/cv-sbi-spin/<run>/checkpoints/<start_ts>/{encoder,flow_net,G_sr,G_rs}.pt`, where `<start_ts>` matches that run's `train_log_<start_ts>.csv` and `run_info_v{version}_<start_ts>.json`. Never overwrites a prior run's checkpoints in place — each training invocation gets its own timestamped subfolder, so re-running the same `--run` name (e.g. after changing hyperparameters) preserves every prior attempt. Eval scripts resolve the latest timestamp under `checkpoints/` by default.

## Versioning and branching convention

Inherited from cv-dann-sbi: version numbers are assigned only when a run survives evaluation. Git branches (`exp/<what-you're-testing>`) for code changes; hyperparameter sweeps commit to main.

**See `experiments.md`** for the full run↔branch↔version bookkeeping table (no results there, just which run/config corresponds to which version label). Current state: `exp-v1c_spin`'s rebalanced rerun (`--lam-cyc 2 --lam-id 1`, on `exp/wave-only-info-clamp`, merged to `main`) is retroactively **v2a** — not renamed on disk, just the documented version label. `exp/wdgrl-grl`'s `exp-v2b_spin` (WDGRL critics replacing the hinge-loss discriminator) is being tested against it as **v2b**; if it wins, it's promoted to **v3** (again, no file renaming — the run directory stays `exp-v2b_spin`, only the version label changes in docs).

**Run directories and file names are never renamed for versioning** — `exp-v1c_spin` stays `exp-v1c_spin` even though it's conceptually v2a. The run name is a historical record of when/how it was launched; the version label is a separate, retroactively-assigned judgment about whether it survived evaluation.

## Known gotchas (found and fixed this session — worth knowing if debugging something that looks similar)

- **`train_spin.py`'s `run_dir` used to be CWD-relative** (`Path("outputs")/run`) — this broke silently once code (`~/projects/cv-sbi-spin/`) and outputs (`~/outputs/cv-sbi-spin/`) were split into separate directories, writing into `~/projects/cv-sbi-spin/outputs/` instead. Fixed via an absolute `--outputs-root` default (commit `f979c79` on `main`, `9896bdf` on `exp/wdgrl-grl`). If a run's output directory is ever missing right after launch, check this hasn't regressed.
- **`eval_common.py`'s `run_posterior_inference` had an extra `np.clip`** on the final posterior mean that isn't in the original notebooks — silently pulled degenerate (zero-acceptance) patients' means back to the prior boundary instead of leaving them as-is. Fixed in both `cv-sbi-spin` and `cv-dann-sbi`'s `eval_common.py`. Verified against `cv-dann-sbi`'s actual notebook output after the fix — numbers matched almost exactly.
- **`GradientReversalLayer` is not used for WDGRL, in either repo.** Tried it in `exp-v2b_spin` attempt 1; checked `cv-dann-sbi/train_joint.py` directly and confirmed its `GradientReversalLayer` class is defined but unused there too — the actual mechanism in both repos is two separately-written loss expressions with opposite signs on the same critic-score term (mathematically identical to GRL, simpler code, no custom autograd `Function`). Don't reintroduce GRL without a specific reason — it was tried and reverted.
- **Adversarial training here needs gradient clipping.** `exp-v2b_spin` attempt 1 (no clipping) diverged catastrophically at epoch 362 (`w1_R` reached ~10¹⁶ in a single epoch). `max_norm=1.0` clipping added to all three optimizer steps (critic, generator, posterior), matching `cv-dann-sbi/train_joint.py`'s existing practice — don't remove this for a future variant without a good reason.

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.
- Always commit before running a full experiment.
- Git runs only on local Mac — never commit from adamant.

## Training run conventions

- **Always confirm the output run name/directory with the user before executing any training run.**
- Always commit before starting an `exp_` run so `run_info.json` captures the exact code state.
