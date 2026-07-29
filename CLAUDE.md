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
- **Norm stats**: `norm_stats.json` in cv-dann-sbi (sim stats, used as shared normalization)

## Versioning and branching convention

Inherited from cv-dann-sbi: version numbers are assigned only when a run survives evaluation. Git branches (`exp/<what-you're-testing>`) for code changes; hyperparameter sweeps commit to main.

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.
- Always commit before running a full experiment.
- Git runs only on local Mac — never commit from adamant.

## Training run conventions

- **Always confirm the output run name/directory with the user before executing any training run.**
- Always commit before starting an `exp_` run so `run_info.json` captures the exact code state.
