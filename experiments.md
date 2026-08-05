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
| `exp-v2b_spin` (attempt 3, same run name) | `exp/wdgrl-grl` | `--lam-adv-max 0.2` (was 1.0), `--lr-gan 1e-4` (was 2e-4), critic inner loop redraws a fresh real batch via `sample_real()` each of the `n_critic` substeps (was one fixed batch reused across all 5 + the generator step) | v2b (attempt 3) | done, 400 ep — `npe_sim=12.22` (down from attempt 2's ~14.0); see Tuning notes |
| `exp-v3_spin` | `exp/latent-cycle-wdgrl` | Different architecture, not a variant of v2b: two separate encoders (`E_sim`, `E_real`) instead of raw-observation-space translation; `G_sr`/`G_rs` are small residual MLPs mapping between their 128-dim latent spaces (not UNets — no spatial/temporal structure left once encoded); two `WassersteinCritic`s (`Critic_R`, `Critic_S`), same WDGRL mechanics as `cv-dann-sbi`. Gradient routing (enforced via `.detach()` placement, not optimizer membership): `E_sim` — task loss only (`info_frac`-interpolated direct-`z_s`/round-tripped-`z_srs` NPE, replacing not just supplementing); `E_real` — adversarial only, both critics; `G_sr`/`G_rs` — adv+cyc+id+task. No reconstruction loss on `E_real` — decided mode collapse is self-defeating under WDGRL (a collapsed/low-diversity `z_r` is exactly the kind of distribution a critic can easily separate from `z_sr`'s spread, so it doesn't cheaply satisfy the adversarial objective). **First launch does not have mixup on the real batches** — was part of the original collapse-risk mitigation plan but got dropped during implementation; add before drawing conclusions from this run if collapse is suspected in eval. | v3 (unconfirmed — new architecture, not yet evaluated) | done, 400 ep (GPU 1) — see Tuning notes for `npe_direct` blowup |

`exp/wave-only-info-clamp` merged into `main` (merge commit `e792b1f`) once v2a was confirmed stable —
main now carries `--freeze-scalars`/`--clamp-info-gap`, the `checkpoints/<timestamp>/` convention, and
the `--outputs-root` fix by default. `exp/wdgrl-grl` and `exp/latent-cycle-wdgrl` both branch off that
merged `main`.

Both `exp-v2b_spin` (raw-observation-space, `exp/wdgrl-grl`) and `exp-v3_spin` (latent-space,
`exp/latent-cycle-wdgrl`) are candidates to beat v2a; whichever is confirmed by real-patient eval gets
promoted to the next open version label. `exp-v3_spin` is provisionally called v3 rather than v2c
because it's a structurally different architecture, not a hyperparameter/loss variant of v2a/v2b — the
version number is not a promise it has already won. Neither run's directory would be renamed
retroactively regardless of outcome (same policy as v2a) — promotion is a documentation/versioning
decision, not a file move.

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
- **Concurrent-run branch switching on adamant**: `git checkout` in the shared `~/projects/cv-sbi-spin/`
  clone was run while `exp-v2b_spin` attempt 3 was still training there — didn't affect the running
  process (already loaded into memory) but would have corrupted its `run_info.json`'s `git_hash`
  (re-read from `HEAD` at completion). Caught and reverted before that run finished. Fix going forward:
  a second run on a different branch uses a separate `git worktree`
  (`~/projects/cv-sbi-spin-latent-cycle-wdgrl`, symlinked `.venv`/`norm_stats.json`) rather than
  switching branches in the shared checkout while anything is training there.

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
- **`exp-v2b_spin` attempt 3, final result**: finished 400 epochs at `npe_sim=12.22` — a real
  improvement over attempt 2's converged ~14.0, and now ahead of v2a. `L_adv` is still visibly noisy
  epoch-to-epoch even with the rebalanced lambda (final few epochs: 10.13, 14.98, 5.78) — better than
  attempt 2 but not fully resolved. `w1_R`/`w1_S` (the actual W1 distance estimate, as opposed to the
  noisier `L_adv`) stayed small and centered near zero throughout, consistent with a converged,
  non-diverging critic. Real-patient eval (acceptance rate, calibration) not yet re-run for this
  attempt — attempt 2's numbers (1.7% acceptance, 90% CI coverage 0.05–0.07) should not be assumed to
  carry over given the training-curve improvement.
- **`exp-v3_spin` (latent-cycle-wdgrl), first run**: `npe_roundtrip` (the only term actually driving
  `E_sim`'s gradient once `info_frac` reaches 1.0 at epoch ~50) converged well to ~9.1 nats — better than
  either v2a or v2b's `npe_sim`. But `npe_direct` (pure `z_s`, bypassing `G_sr`/`G_rs`) blew up to
  >1000 nats by epoch 400. This tracks the design as built, not a bug: `L_task` *interpolates* from
  direct to round-trip rather than *summing* them (unlike original SPIN's `loss_posterior`, which always
  keeps a `npe_sim` term and adds `npe_srs` on top) — once `info_frac=1`, nothing in the loss anchors
  `flow.log_prob(theta, z_s)` any more, so the direct pathway is free to drift. Only matters if anything
  downstream evaluates sim data via direct `z_s` instead of round-tripping through `G_sr`/`G_rs` first —
  worth checking eval scripts route sim inputs consistently before comparing this run's numbers to
  v2a/v2b's `npe_sim`. `w1_R`/`w1_S` did not converge to ~0 by epoch 400 (ended at +0.62/+0.29,
  fairly stable but not small) — critics can still separate the domains somewhat; unclear yet whether
  that's undertraining or a genuinely harder alignment problem in latent space vs. raw-observation
  space. Comparison point: `cv-dann-sbi`'s own accepted-good encoder converges to `w1=0.85`, not to
  zero, so a nonzero residual plateau appears to be the expected steady state for this metric under
  WDGRL/gradient-penalty training rather than itself a sign of poor alignment — v3's 0.62/0.29 residual
  is on the same order as, or smaller than, that precedent. The trajectory (5.35 at epoch 1 → ~0.65 by
  epoch ~25–30 → flat through epoch 400) is a genuine plateau, not still-improving noise, and both
  `w1_R`/`w1_S` stayed one-signed throughout (unlike v2a/v2b's critics, which cross zero repeatedly) —
  worth another look if real-patient eval surfaces collapse symptoms, especially since this run has no
  mixup on the real batches (see run row above), which could inflate `w1_R` independent of true
  domain-gap difficulty by making `z_r` easier for the critic to separate. No mixup on reals for this
  run — worth adding before drawing firm conclusions if eval shows real-patient collapse symptoms.
