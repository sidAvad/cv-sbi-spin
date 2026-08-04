# Experiments

Bookkeeping only — which run corresponds to which branch/commit/config, and how runs map to
version numbers. No results/findings here; those live in `results/cv-sbi-spin/<run>/README.md`
(auto-generated provenance) and the images themselves.

## Run ↔ branch ↔ version map

| Run name | Branch | Key config | Version label | Status |
|---|---|---|---|---|
| `exp-v1_spin` | `main` (pre-freeze-scalars) | baseline: hinge-loss D_R/D_S, no `--freeze-scalars`, no `--clamp-info-gap` | v1 | done (2 invocations: 400 ep + `--resume` to 800 ep total) |
| `exp-v1b_spin` | `main` (pre-freeze-scalars) | `--no-info-in-G` (zero `lam_info` in generator step only) | v1b | done, 400 ep |
| `exp-v1c_spin` (original) | `exp/wave-only-info-clamp` | `--freeze-scalars --clamp-info-gap`, `--lam-cyc 10 --lam-id 5` (defaults) | v1c (superseded) | done, 400 ep — collapsed to identity ~epoch 364 |
| `exp-v1c_spin` (rerun, same run name) | `exp/wave-only-info-clamp` | same flags, rebalanced `--lam-cyc 2 --lam-id 1` | **v2a** (retroactive; not renamed on disk) | done, 400 ep — no collapse |
| `exp-v2b_spin` (attempt 1) | `exp/wdgrl-grl` | WDGRL critics (`n_critic=5`, `gp_weight=10.0`) + `GradientReversalLayer` for the generator step, `--grl-alpha 1.0 --adv-ramp 50` | v2b (attempt 1, failed) | done, 400 ep — catastrophic divergence at epoch 362 (`w1_R` → 10¹⁶), no gradient clipping |
| `exp-v2b_spin` (attempt 2, same run name) | `exp/wdgrl-grl` | same WDGRL critics, `GradientReversalLayer` **removed** — generator step reverted to direct `-D(fake).mean()` (matches `cv-dann-sbi`'s actual mechanism, confirmed by reading `train_joint.py`), `--lam-adv-max 1.0 --adv-ramp 50`, gradient clipping (`max_norm=1.0`) added to all three optimizer steps | v2b (attempt 2) | done, 400 ep — no divergence, but large epoch-to-epoch noise in `L_adv`/`loss_G`; real-patient eval showed 1.7% posterior acceptance and 90% CI coverage of 0.05–0.07 (see Tuning notes) |
| `exp-v2b_spin` (attempt 3, same run name) | `exp/wdgrl-grl` | `--lam-adv-max 0.2` (was 1.0), `--lr-gan 1e-4` (was 2e-4), critic inner loop redraws a fresh real batch via `sample_real()` each of the `n_critic` substeps (was one fixed batch reused across all 5 + the generator step) | v2b (attempt 3) | not yet run |

`exp/wave-only-info-clamp` merged into `main` (merge commit `e792b1f`) once v2a was confirmed stable —
main now carries `--freeze-scalars`/`--clamp-info-gap`, the `checkpoints/<timestamp>/` convention, and
the `--outputs-root` fix by default. `exp/wdgrl-grl` branches off that merged `main`.

If v2b (attempt 2) beats v2a, it gets promoted to **v3**; `exp-v2b_spin`'s run name/directory would not
be renamed retroactively (same policy as v2a) — the promotion is a documentation/versioning decision,
not a file move.

## Retroactive fixes (apply to already-completed runs' interpretation, not just future runs)

- **Checkpoint layout**: `checkpoints/<timestamp>/` convention added mid-project (commit `9c6df19` on
  `exp/wave-only-info-clamp`). `exp-v1c_spin`'s *original* run had saved checkpoints flat (pre-dates the
  convention) — manually moved into `checkpoints/20260727-143114/` to match the rerun's
  `checkpoints/20260728-124153/`, so "latest checkpoint" resolution works correctly across both attempts.
- **`run_dir` path bug**: `train_spin.py` used a CWD-relative `Path("outputs")/args.run`, which silently
  wrote to `~/projects/cv-sbi-spin/outputs/` once code and outputs were split into separate directories.
  Fixed on both `main` (commit `f979c79`) and `exp/wdgrl-grl` (commit `9896bdf`) with an absolute
  `--outputs-root` default. `exp-v2b_spin` attempt 1 was killed and relaunched after this fix (its first,
  mis-located attempt was deleted, not archived).
- **Eval script posterior-mean bug**: `results/cv-sbi-spin/scripts/eval_common.py`'s
  `run_posterior_inference` had an erroneous `np.clip(...)` on the final posterior mean that isn't in the
  original notebooks (they only clip in the sim-reconstruction path, not the parameter-inference path).
  Fixed; same bug existed in `cv-dann-sbi`'s `eval_common.py` and was fixed there too. Any
  `parameter_scatter.png`/`calibration.png`/reconstruction images generated before this fix are stale and
  were regenerated for `exp-v1c_spin`.
- **Eval script patient-selection consistency**: `eval_recon_sim.py` (both projects) originally selected
  example patients by simulator-success filtering (matches the original notebooks exactly), which could
  pick different patient indices per project. Changed to a deterministic, evenly-spaced rule
  (`i * len(patients)//n_examples`) in both `cv-sbi-spin` and `cv-dann-sbi` so the same patients are
  plotted in both projects' reconstruction images, enabling direct visual comparison.

## Tuning notes

- **`exp-v2b_spin` attempt 2 → attempt 3**: attempt 2 finished all 400 epochs without diverging, but
  `training_curves.png` showed large epoch-to-epoch noise in `L_adv`/`loss_G`, and real-patient eval
  (`eval_real_patient_inference.py`) showed only 1.7% posterior acceptance and 90% CI coverage of
  0.05–0.07 (ideal 0.90) — badly miscalibrated. Reconstructing `loss_G` from its logged components at
  epoch 400 (`L_adv=3.09, L_cyc=0.037, L_id=0.169, L_info=0.379, loss_G=3.72`) showed `L_adv` was ~83%
  of `loss_G` by itself despite `lam_adv_max(1.0) < lam_cyc(2.0)` — `L_adv` is an unbounded raw
  critic-score difference (no output bound on a WGAN critic), unlike the normalized L1 `cyc`/`id`
  losses (~0.03–0.4), so the lambda coefficients alone didn't reflect each term's actual contribution
  to the total loss. `cv-dann-sbi`'s WDGRL setup isn't directly comparable here (no `cyc`/`id` terms
  there at all), so its `lambda_target=0.1` wasn't copied — 0.2 was chosen from `exp-v2b_spin`'s own
  logged magnitudes instead, to bring `lam_adv·L_adv` roughly in line with the other three terms
  combined. `lr_gan` and the critic-loop fresh-batch resampling *were* brought in line with
  `cv-dann-sbi`, since those aren't loss-composition-dependent.
