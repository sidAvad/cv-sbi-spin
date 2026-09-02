"""
SPIN training: sim<->real dual generators + information-preserving posterior.

v3 (exp/dual-flow-anchor): two separate encoder+flow pairs instead of one shared
one.

  E_sim  + flow_sim  : trained on genuine x_sim (npe_sim, direct) AND on the
                       sim->real->sim round trip x_srs = G_rs(G_sr(x_sim))
                       (npe_srs), combined via lam_info -- same mechanism as
                       v2c, just relabeled now that there's a second pair.
                       Round-trip gradient reaches BOTH G_sr and G_rs (via the
                       generator step's attached x_srs, unchanged from v2c).
  E_real + flow_real : NEW. Trained ONLY on (G_sr(x_sim), theta_sim) pairs --
                       exact, ground-truth-labeled (G_sr operates sample-wise,
                       so there's no matching/pseudo-labeling problem the way
                       there is for real patient data in the JDOT line). This
                       gradient reaches G_sr only, not G_rs -- a direct, one-hop
                       task anchor for the sim->real generator specifically.
                       Mirrors the same attached-in-generator-step /
                       detached-in-its-own-posterior-step pattern already used
                       for E_sim/flow_sim's round-trip term.

cyc and id both stay L1 (no adversarial critic for either, for now) at v2c's
weights. D_R/D_S (WDGRL domain-realism critics) are unchanged from v2c.

New: a high-frequency residual penalty on BOTH generators' own translation
residual (x_sr - x_s, x_rs - x_r), wave-only portion. Motivation: G_sr's direct
real-flow anchor only checks whether theta is recoverable from G_sr(x_sim) --
it can't tell a genuine physiological translation from a generator that smuggles
an arbitrary, cheaply-decodable watermark into the output. A blanket L1/L2
penalty on the residual's magnitude would fight legitimate large-scale domain
shift without particularly suppressing a smuggled code (which only needs a
few bits, contributes little to total residual magnitude, and would be
dominated by whatever large-scale shift the residual budget mostly goes to).
High-frequency content specifically is the cheap, low-visibility channel a
smuggled code would most plausibly use, so the penalty high-pass filters the
residual (fixed box-filter low-pass via avg_pool1d, kernel size --hf-kernel)
and penalizes only the remainder -- legitimate low-frequency/bulk shifts are
untouched. This is a standing structural guard, not a task-relevant term, so
its weight (--lam-hf) is constant, not ramped.

Note: this isn't the only steganography guard in the system -- --clamp-info-gap
(relu(npe_srs - npe_sim), gradient to G only when the round trip is *harder*
than a genuine sim, not just informative) already provides a related, weaker
check via the round-trip path. The two aren't redundant (clamp-info-gap can't
see the intermediate G_sr(x_sim) representation at all, only the round-tripped
endpoint after G_rs has had a chance to launder it -- see experiments.csv/the
session notes this design came out of) but both now sit in the system; noted,
not considered a problem.

v3b: exp-v3_spin trained cleanly through ~ep396 then diverged catastrophically
at ep397-399 (L_adv/L_cyc/L_id/hf_rs/w1_S all spiked to the millions), mostly
recovering by ep400. Diagnosis: npe_sim and npe_real (both independent of
G_rs) stayed completely normal throughout -- every exploding quantity passes
through G_rs regardless of input, pointing at G_rs's own weights becoming
numerically pathological, not a bad-input cascade from G_sr/E_real. Three
changes in response:

  1. Dropped hf_rs from the loss (still computed/logged, just not weighted
     in). Its original motivation -- guarding a direct task anchor against
     steganographic smuggling -- never applied to G_rs (which has no direct
     task anchor to smuggle through); it was applied there symmetrically
     rather than for a real reason, and is a plausible (unconfirmed)
     contributor to G_rs's instability. hf_sr is kept -- G_sr's anchor is
     real and the threat model still applies there.

  2. Mixup on the critic's real-batch sampling (--use-mixup, Beta(alpha,alpha)
     interpolation between random real-patient pairs) -- ported from
     cv-dann-sbi/train_joint.py's proven v3 recipe, never previously used in
     this project. Augments the small (802-patient) real-side critic data so
     D_R/D_S don't just memorize the exact patient set.

  3. (An earlier attempt at this v3b also added a scalar realignment step for
     E_real's input here -- that attempt was killed off and its outputs/
     results deleted. It was the leading suspect for why its canonical
     inference route (x_real -> G_sr -> E_real/flow_real) regressed sharply on
     Eap/Cas/Ras versus v3 despite Rap holding steady and coverage staying
     poor on both; see experiments.csv. This v3b does not include it --
     reverting isolates that variable before trying the decoupled gradient
     clipping below, rather than compounding changes the way v3 -> the earlier
     v3b attempt did.)

This v3b (rerun after the earlier attempt above): reverts the scalar
realignment described above and adds decoupled gradient clipping -- G_sr and
G_rs now get separate clip_grad_norm_ calls instead of one combined-norm call
over both. Motivation: v3's ep397-399 divergence traced specifically to
G_rs's own weights going numerically pathological while G_sr/E_sim/E_real
stayed normal throughout -- a combined clip rescales both generators'
gradients by the same factor when the combined norm exceeds the threshold, so
G_rs's exploding gradient was suppressing G_sr's own (unrelated, healthy)
gradient by the same amount rather than being isolated. Keeps the hf_rs drop
(never applied to G_rs's non-existent direct anchor to begin with) and the
D_R/D_S checkpointing fix below, both independently justified.

Also fixed: D_R/D_S were never saved in checkpoints (only encoder_sim/flow_sim/
encoder_real/flow_real/G_sr/G_rs) -- --resume would have restarted the critics
from scratch while generators resumed already-trained, giving generators a
free adversarial pass until critics relearned. Now saved/loaded like everything
else; --resume falls back to scratch-critics with a warning if an older
checkpoint predates this fix.

Schedule (per v2c):
  Phase 0  flow-warmup   (ep 1-2)    E_sim/flow_sim's flow trains, encoder frozen
  Phase 1  enc-warmup    (ep 3-12)   E_sim trains, flow_sim frozen
  Phase 2  joint         (ep 13+)    per-batch: critic loop -> generator step ->
                                      sim posterior step -> real posterior step

E_real/flow_real only train in the joint phase (their only input, G_sr(x_sim),
doesn't exist as a meaningful signal before G_sr itself starts training there).

Per-batch step order (joint phase only):
  1. Critic inner loop   - n_critic updates to D_R, D_S (WGAN-GP; G outputs detached)
  2. Generator step      - update G_sr, G_rs; L_adv + L_cyc + L_id + L_info
                            + lam_real*L_real_anchor + lam_hf*(hf_sr + hf_rs)
  3. Sim posterior step  - update E_sim, flow_sim (x_srs detached, no grad into G)
  4. Real posterior step - update E_real, flow_real (G_sr(x_s) detached, no grad into G)

All optimizer steps clip gradients to max_norm=1.0 (matching v2c/cv-dann-sbi).

Optimizers:
  opt_critic : Adam(D_R + D_S,           lr=1e-4, betas=(0.5, 0.9))
  opt_G      : Adam(G_sr + G_rs,         lr=1e-4, betas=(0.5, 0.999))
  opt_sim    : AdamW(E_sim + flow_sim,   lr=1e-4)
  opt_real   : AdamW(E_real + flow_real, lr=1e-4)

See experiments.csv for run-by-run rationale and results.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    ReducedCVDataset, RealBeatsDataset,
    load_stats, load_manifest,
    PARAM_KEYS_INFER, N_REDUCED_CHANNELS, N_SCALARS, T,
)

_WAVE_DIM = N_REDUCED_CHANNELS * T  # 804
from models import (
    LipschitzEncoder, build_flow_net,
    DualBranchGenerator, DualBranchDiscriminator,
)


# ── Losses ────────────────────────────────────────────────────────────────────

def gradient_penalty(disc_fn, x_fake, x_real, device):
    """
    1-Lipschitz gradient penalty for WGAN-GP at random interpolates between a
    (detached) fake and real batch. disc_fn already handles wave_only slicing.
    """
    eps   = torch.rand(x_fake.shape[0], 1, device=device)
    x_hat = (eps * x_fake.detach() + (1 - eps) * x_real.detach()).requires_grad_(True)
    f_hat = disc_fn(x_hat)
    grads = torch.autograd.grad(f_hat.sum(), x_hat, create_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def mixup_real(real_beats: torch.Tensor, n: int, alpha: float, device) -> torch.Tensor:
    """
    Beta(alpha,alpha)-interpolated pairs of real patients -- augments the
    critic's real-side training data so it doesn't just memorize the (small,
    802-patient) real set. Ported from cv-dann-sbi/train_joint.py's proven v3
    recipe; never previously used in this project (v3b).
    """
    idx_i = torch.randint(0, len(real_beats), (n,), device=device)
    idx_j = torch.randint(0, len(real_beats), (n,), device=device)
    lam   = torch.distributions.Beta(alpha, alpha).sample((n,)).to(device).unsqueeze(1)
    return lam * real_beats[idx_i] + (1 - lam) * real_beats[idx_j]


def high_freq_penalty(delta_wave: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    delta_wave: (B, N_REDUCED_CHANNELS, T) -- a generator's own residual
    (x_out - x_in), wave portion only.

    Fixed (non-learned) box-filter low-pass via avg_pool1d; penalize only the
    high-frequency remainder (delta - lowpass(delta)), leaving low-frequency/
    bulk shifts in delta completely unconstrained. See module docstring for
    why this targets steganographic smuggling specifically, unlike a blanket
    magnitude penalty on delta itself.
    """
    lowpass = F.avg_pool1d(delta_wave, kernel_size=kernel_size, stride=1,
                           padding=kernel_size // 2, count_include_pad=False)
    high_freq = delta_wave - lowpass
    return high_freq.abs().mean()


def loss_generator(G_sr, G_rs, D_R, D_S, E_sim, flow_sim, E_real, flow_real,
                   x_s, x_r, theta, lam_cyc, lam_id, lam_info, lam_adv, lam_real, lam_hf,
                   hf_kernel, wave_only=False, clamp_info_gap=False):
    """
    Generator step: update G_sr and G_rs.

    E_sim/flow_sim and E_real/flow_real are used as fixed functions here --
    gradients flow THROUGH them back to G_sr/G_rs (no .detach()), but their
    weights are not in opt_G so they don't update here (only via opt_sim/
    opt_real in the posterior steps). D_R/D_S are likewise fixed here.

    wave_only: generators receive/return waves only (804-dim); scalars are passed
               through unchanged and concatenated back before encoder/discriminator calls.
    clamp_info_gap: L_info = relu(npe_srs - npe_sim) instead of npe_srs -- gradient to
                    generator only when translated reals are harder than pure sims.
                    Applies to the E_sim round-trip term only (npe_srs); the new
                    E_real anchor (L_real) is unclamped -- see module docstring.

    L_G = lam_adv*L_adv + lam_cyc*L_cyc + lam_id*L_id + lam_info*L_info
        + lam_real*L_real + lam_hf*hf_sr   (hf_rs dropped in v3b, still logged)
    """
    def _gen(G, x):
        """Apply generator, routing scalars around it when wave_only."""
        if wave_only:
            waves_out = G(x[:, :_WAVE_DIM])
            return torch.cat([waves_out, x[:, _WAVE_DIM:]], dim=1)
        return G(x)

    def _disc(D, x):
        """Call discriminator on waves only when wave_only."""
        return D(x[:, :_WAVE_DIM]) if wave_only else D(x)

    # Forward passes
    x_sr  = _gen(G_sr, x_s)   # sim -> real
    x_rs  = _gen(G_rs, x_r)   # real -> sim
    x_srs = _gen(G_rs, x_sr)  # sim -> real -> sim  (round trip)
    x_rsr = _gen(G_sr, x_rs)  # real -> sim -> real (round trip)

    # WDGRL adversarial term (unchanged from v2c) -- generators push critics'
    # score on the translated (fake) side up; D_R/D_S are frozen here (not in
    # opt_G), trained separately in their own multi-step critic loop.
    L_adv = -_disc(D_R, x_sr).mean() - _disc(D_S, x_rs).mean()

    # Cycle consistency -- L1 on both round trips (unchanged, kept L1 deliberately:
    # this is G_rs's only remaining consistency check now that it doesn't get a
    # direct task anchor, and a distributional/adversarial cyc loss would only
    # guarantee aggregate shape match, not per-sample self-recovery)
    L_cyc = (x_srs - x_s).abs().mean() + (x_rsr - x_r).abs().mean()

    # Identity -- L1, no adversarial critic for now (unchanged from v2c)
    L_id = (_gen(G_rs, x_s) - x_s).abs().mean() + (_gen(G_sr, x_r) - x_r).abs().mean()

    # Round-trip information preservation (E_sim/flow_sim) -- sim->real->sim only,
    # no .detach() on E_sim/flow_sim (gradient reaches G_sr AND G_rs, since x_srs
    # is a function of both)
    if lam_info > 0:
        npe_srs = -flow_sim.log_prob(theta, condition=E_sim(x_srs)).mean()
        if clamp_info_gap:
            with torch.no_grad():
                npe_sim_ref = -flow_sim.log_prob(theta, condition=E_sim(x_s)).mean()
            L_info = torch.relu(npe_srs - npe_sim_ref)
        else:
            L_info = npe_srs
    else:
        L_info = torch.zeros(1, device=x_s.device).squeeze()

    # Direct real-side anchor (E_real/flow_real) -- gradient reaches G_sr
    # only (x_rs/G_rs never appear in this term at all)
    if lam_real > 0:
        L_real = -flow_real.log_prob(theta, condition=E_real(x_sr)).mean()
    else:
        L_real = torch.zeros(1, device=x_s.device).squeeze()

    # High-frequency residual penalty -- G_sr's own residual only (v3b dropped
    # hf_rs from the loss, see module docstring; still computed/logged for both).
    if lam_hf > 0:
        delta_sr_wave = (x_sr - x_s)[:, :_WAVE_DIM].view(-1, N_REDUCED_CHANNELS, T)
        delta_rs_wave = (x_rs - x_r)[:, :_WAVE_DIM].view(-1, N_REDUCED_CHANNELS, T)
        hf_sr = high_freq_penalty(delta_sr_wave, hf_kernel)
        hf_rs = high_freq_penalty(delta_rs_wave, hf_kernel)
    else:
        hf_sr = torch.zeros(1, device=x_s.device).squeeze()
        hf_rs = torch.zeros(1, device=x_s.device).squeeze()

    loss = (lam_adv * L_adv + lam_cyc * L_cyc + lam_id * L_id + lam_info * L_info
            + lam_real * L_real + lam_hf * hf_sr)

    with torch.no_grad():
        delta   = (x_sr - x_s).abs()
        delta_w = delta[:, :_WAVE_DIM].mean().item()
        delta_s = delta[:, _WAVE_DIM:].mean().item()

    return loss, {
        "adv":     L_adv.item(),
        "cyc":     L_cyc.item(),
        "id":      L_id.item(),
        "info":    L_info.item(),
        "real":    L_real.item(),
        "hf_sr":   hf_sr.item(),
        "hf_rs":   hf_rs.item(),
        "delta_w": delta_w,
        "delta_s": delta_s,
    }


def loss_critics(G_sr, G_rs, D_R, D_S, x_s, x_r, gp_weight, device, wave_only=False):
    """
    Critic step: update D_R and D_S as genuine WGAN-GP critics -- one call
    per n_critic inner iteration. Unchanged from v2c.

    L_critic = -(E[D(real)] - E[D(fake)]) + gp_weight * GP   (per critic, summed)
    """
    def _gen(G, x):
        if wave_only:
            return torch.cat([G(x[:, :_WAVE_DIM]), x[:, _WAVE_DIM:]], dim=1)
        return G(x)

    def _disc_R(x): return D_R(x[:, :_WAVE_DIM]) if wave_only else D_R(x)
    def _disc_S(x): return D_S(x[:, :_WAVE_DIM]) if wave_only else D_S(x)

    x_sr = _gen(G_sr, x_s).detach()
    x_rs = _gen(G_rs, x_r).detach()

    w1_R = _disc_R(x_r).mean() - _disc_R(x_sr).mean()
    w1_S = _disc_S(x_s).mean() - _disc_S(x_rs).mean()

    gp_R = gradient_penalty(_disc_R, x_sr, x_r, device)
    gp_S = gradient_penalty(_disc_S, x_rs, x_s, device)

    loss = (-w1_R + gp_weight * gp_R) + (-w1_S + gp_weight * gp_S)
    return loss, {"w1_R": w1_R.item(), "w1_S": w1_S.item(),
                 "gp_R": gp_R.item(), "gp_S": gp_S.item()}


def loss_sim_posterior(E_sim, flow_sim, x_s, theta, x_srs_detached, lam_info):
    """
    Sim posterior step: update E_sim and flow_sim.
    x_srs_detached must already be detached -- no grad flows into G from this step.

    L = -log flow_sim(theta | E_sim(x_s)) + lam_info * (-log flow_sim(theta | E_sim(x_srs_detached)))
    """
    npe_sim = -flow_sim.log_prob(theta, condition=E_sim(x_s)).mean()

    if x_srs_detached is not None and lam_info > 0:
        npe_srs = -flow_sim.log_prob(theta, condition=E_sim(x_srs_detached)).mean()
    else:
        npe_srs = torch.zeros(1, device=x_s.device).squeeze()

    loss = npe_sim + lam_info * npe_srs
    return loss, {"npe_sim": npe_sim.item(), "npe_srs": npe_srs.item()}


def loss_real_posterior(E_real, flow_real, x_sr_detached, theta):
    """
    Real posterior step: update E_real and flow_real.
    x_sr_detached (= G_sr(x_s), detached) must already be detached -- no grad
    flows into G_sr from this step. theta is theta_sim -- (G_sr(x_sim), theta_sim)
    is an exact pair, no matching/pseudo-labeling needed (unlike the JDOT line's
    real patient data, which has no known sim correspondence at all).

    L = -log flow_real(theta | E_real(x_sr_detached))
    """
    npe_real = -flow_real.log_prob(theta, condition=E_real(x_sr_detached)).mean()
    return npe_real, {"npe_real": npe_real.item()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def lambda_info_schedule(epoch, joint_start, joint_end, ramp_epochs=50):
    """Linear ramp from 0 → 1 over first ramp_epochs of joint phase."""
    joint_ep = epoch - joint_start
    return min(1.0, joint_ep / ramp_epochs)


def make_log(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = run_dir / f"train_{ts}.log"
    fh = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    return log, fh


def save_checkpoint(ckpt_dir: Path, E_sim, flow_sim, E_real, flow_real, G_sr, G_rs, D_R, D_S):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(E_sim.state_dict(),    ckpt_dir / "encoder_sim.pt")
    torch.save(flow_sim,              ckpt_dir / "flow_sim.pt")
    torch.save(E_real.state_dict(),   ckpt_dir / "encoder_real.pt")
    torch.save(flow_real,             ckpt_dir / "flow_real.pt")
    torch.save(G_sr.state_dict(),     ckpt_dir / "G_sr.pt")
    torch.save(G_rs.state_dict(),     ckpt_dir / "G_rs.pt")
    torch.save(D_R.state_dict(),      ckpt_dir / "D_R.pt")
    torch.save(D_S.state_dict(),      ckpt_dir / "D_S.pt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",          required=True)
    parser.add_argument("--version",      default="1")
    parser.add_argument("--sim-data-root", required=True)
    parser.add_argument("--real-data",    required=True)
    parser.add_argument("--n-sims",       type=int, required=True)
    parser.add_argument("--n-val-sims",   type=int, default=2000,
                        help="Held-out sims (from manifest_test.json, never trained on) used "
                             "each epoch for a validation npe_real reading -- one extra no_grad "
                             "batch/epoch, negligible cost vs the ~n_sims/batch_size training "
                             "batches already run. Set 0 to disable.")
    parser.add_argument("--calib-every",  type=int, default=10,
                        help="Compute real-patient Rap/Ras 90%% credible-interval coverage every N "
                             "joint epochs and checkpoint on it (harmonic mean of closeness-to-"
                             "nominal-0.90 for each). Rap/Ras have directly-measured ground truth "
                             "(PVR/SVR) on real patients, unlike Cas/Eap -- so unlike npe_real_val "
                             "(a sim-side proxy via G_sr(x_sim)), this measures calibration on the "
                             "actual target population. Expensive (802 patients x "
                             "--calib-n-samples flow samples each), hence not every epoch.")
    parser.add_argument("--calib-n-samples", type=int, default=300,
                        help="Posterior samples per real patient for the calibration checkpoint metric")
    parser.add_argument("--max-epochs",   type=int, default=400)
    parser.add_argument("--flow-warmup",  type=int, default=2)
    parser.add_argument("--enc-warmup",   type=int, default=10)
    parser.add_argument("--lam-cyc",      type=float, default=2.0)
    parser.add_argument("--lam-id",       type=float, default=1.0)
    parser.add_argument("--lam-info-max", type=float, default=1.0)
    parser.add_argument("--info-ramp",    type=int,   default=50,
                        help="Joint epochs over which lambda_info ramps 0→lam_info_max")
    parser.add_argument("--lam-real-max", type=float, default=1.0,
                        help="Target weight (ramped) on E_real/flow_real's direct task "
                             "anchor for G_sr -- new in v3")
    parser.add_argument("--real-ramp",    type=int,   default=50,
                        help="Joint epochs over which lam_real ramps 0→lam_real_max "
                             "(mirrors --info-ramp)")
    parser.add_argument("--lam-hf",       type=float, default=0.1,
                        help="Weight on the high-frequency residual penalty (new in v3). "
                             "Constant, not ramped -- a standing structural guard against "
                             "steganographic smuggling, not a task-relevant term that "
                             "should fade in importance over training.")
    parser.add_argument("--hf-kernel",    type=int,   default=5,
                        help="Odd kernel size for the box-filter low-pass that defines "
                             "the high-frequency remainder in the residual penalty")
    parser.add_argument("--freeze-scalars",  action="store_true",
                        help="Wave-only generators/discriminators; scalars bypass G and route directly to encoder")
    parser.add_argument("--no-sv", action="store_true",
                        help="Drop SV from the scalar tail entirely (4 scalars instead of 5) for both "
                             "E_sim and E_real -- ported from cv-dann-sbi/train_joint.py's --no-sv. SV "
                             "ground truth stays available separately (ReducedCVDataset.sv / "
                             "RealBeatsDataset.sv) for the reconstruct-and-recover-SV ablation even "
                             "though the model never sees it as input.")
    parser.add_argument("--clamp-info-gap", action="store_true",
                        help="L_info_G = relu(npe_srs - npe_sim): gradient to G only when gap > 0. "
                             "Applies to the E_sim round-trip term only, not the E_real anchor.")
    parser.add_argument("--no-info-in-G", action="store_true",
                        help="Zero lam_info in generator step; posterior still uses full ramp")
    parser.add_argument("--n-critic",     type=int,   default=5,
                        help="WDGRL: critic inner-loop updates per generator step")
    parser.add_argument("--gp-weight",    type=float, default=10.0,
                        help="WDGRL: gradient penalty weight (WGAN-GP Lipschitz constraint)")
    parser.add_argument("--lam-adv-max",  type=float, default=0.2,
                        help="WDGRL: target weight on the generator's adversarial term (ramped)")
    parser.add_argument("--adv-ramp",     type=int,   default=50,
                        help="Joint epochs over which lam_adv ramps 0→lam_adv_max")
    parser.add_argument("--use-mixup",    action="store_true",
                        help="v3b: Beta(alpha,alpha)-interpolate real patient pairs for the "
                             "critic's real-batch sampling, instead of raw draws -- ported from "
                             "cv-dann-sbi/train_joint.py's proven v3 recipe")
    parser.add_argument("--mixup-alpha",  type=float, default=0.2,
                        help="Beta distribution concentration for --use-mixup")
    parser.add_argument("--batch-size",   type=int, default=512)
    parser.add_argument("--lr-npe",       type=float, default=1e-4)
    parser.add_argument("--lr-gan",       type=float, default=1e-4)
    parser.add_argument("--latent-dim",   type=int, default=128)
    parser.add_argument("--stats-path",   default="norm_stats.json")
    parser.add_argument("--resume",       action="store_true",
                        help="Load existing checkpoints from run_dir and continue training")
    parser.add_argument("--start-epoch",  type=int, default=1,
                        help="Epoch offset for display and lambda schedule (set to prev max_epochs+1 when resuming)")
    parser.add_argument("--outputs-root", default="/home/sa4604/outputs/cv-sbi-spin",
                        help="Absolute root for run_dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.outputs_root) / args.run
    log, log_fh = make_log(run_dir)

    ts = f"{datetime.now():%Y%m%d-%H%M%S}"
    log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Run: {args.run}  v={args.version}")
    log(f"Device: {device}  max_epochs: {args.max_epochs}")

    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"

    with open(run_dir / f"run_info_v{args.version}_{ts}.json", "w") as f:
        json.dump({
            "run": args.run, "version": args.version,
            "status": "running",
            "started": ts,
            "command": " ".join(["train_spin.py"] + sys.argv[1:]),
            "git_hash": git_hash,
        }, f, indent=2)

    flow_end  = args.flow_warmup
    enc_end   = flow_end + args.enc_warmup
    joint_start = enc_end + 1
    log(f"Phase boundaries — flow_end={flow_end}  enc_end={enc_end}  joint_start={joint_start}")

    # ── Data ──────────────────────────────────────────────────────────────────
    stats    = load_stats(args.stats_path)
    sim_root = Path(args.sim_data_root)
    manifest = load_manifest(sim_root / "manifest_train.json")

    include_sv = not args.no_sv
    n_scalars  = 4 if args.no_sv else N_SCALARS
    log(f"--no-sv={args.no_sv}  (n_scalars={n_scalars} for E_sim/E_real)")

    log(f"Loading {args.n_sims} sim observations...")
    sim_ds = ReducedCVDataset(str(sim_root / "train"), manifest["index"][:args.n_sims], stats,
                              log=log, include_sv=include_sv)
    sim_dl = DataLoader(sim_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    log(f"Loading real patient beats from {args.real_data}...")
    real_ds = RealBeatsDataset(args.real_data, stats, log=log, include_sv=include_sv)
    real_sampler = RandomSampler(real_ds, replacement=True, num_samples=len(sim_ds))
    real_dl = DataLoader(real_ds, batch_size=args.batch_size, sampler=real_sampler,
                         num_workers=0, pin_memory=True, drop_last=True)

    rap_idx = PARAM_KEYS_INFER.index("Rap")
    ras_idx = PARAM_KEYS_INFER.index("Ras")
    real_gt_rap = real_ds.gt_rap  # (802,) physical units, directly measured (PVR)
    real_gt_ras = real_ds.gt_ras  # (802,) physical units, directly measured (SVR)
    rap_valid = ~np.isnan(real_gt_rap) & (real_gt_rap >= 0)
    ras_valid = ~np.isnan(real_gt_ras) & (real_gt_ras >= 0)

    val_ds = None
    if args.n_val_sims > 0:
        manifest_test = load_manifest(sim_root / "manifest_test.json")
        log(f"Loading {args.n_val_sims} held-out validation sims (manifest_test.json, "
            f"never trained on)...")
        val_ds = ReducedCVDataset(str(sim_root / "test"), manifest_test["index"][:args.n_val_sims],
                                  stats, log=log, include_sv=include_sv)
    log(f"Real beats: {len(real_ds)}  (oversampled to ~{len(sim_ds)} per epoch)")
    real_beats_gpu = real_ds.x.to(device)  # small (802, 808 or 809); kept resident for mixup_real

    log("Collecting theta stats for flow...")
    theta_all = sim_ds.theta[:min(10_000, len(sim_ds))]

    # ── Models ────────────────────────────────────────────────────────────────
    E_sim     = LipschitzEncoder(latent_dim=args.latent_dim, n_scalars=n_scalars).to(device)
    flow_sim  = build_flow_net(args.latent_dim, theta_all).to(device)
    E_real    = LipschitzEncoder(latent_dim=args.latent_dim, n_scalars=n_scalars).to(device)
    flow_real = build_flow_net(args.latent_dim, theta_all).to(device)
    G_sr    = DualBranchGenerator(wave_only=args.freeze_scalars).to(device)
    G_rs    = DualBranchGenerator(wave_only=args.freeze_scalars).to(device)
    D_R     = DualBranchDiscriminator(wave_only=args.freeze_scalars).to(device)
    D_S     = DualBranchDiscriminator(wave_only=args.freeze_scalars).to(device)

    if args.resume:
        ckpt_root = run_dir / "checkpoints"
        existing_ckpts = sorted(p for p in ckpt_root.glob("*") if p.is_dir())
        if not existing_ckpts:
            raise FileNotFoundError(f"--resume set but no checkpoint subfolders found in {ckpt_root}")
        resume_dir = existing_ckpts[-1]
        E_sim.load_state_dict(torch.load(resume_dir / "encoder_sim.pt", map_location=device, weights_only=True))
        flow_sim  = torch.load(resume_dir / "flow_sim.pt", map_location=device, weights_only=False)
        E_real.load_state_dict(torch.load(resume_dir / "encoder_real.pt", map_location=device, weights_only=True))
        flow_real = torch.load(resume_dir / "flow_real.pt", map_location=device, weights_only=False)
        G_sr.load_state_dict(torch.load(resume_dir / "G_sr.pt", map_location=device, weights_only=True))
        G_rs.load_state_dict(torch.load(resume_dir / "G_rs.pt", map_location=device, weights_only=True))
        if (resume_dir / "D_R.pt").exists():
            D_R.load_state_dict(torch.load(resume_dir / "D_R.pt", map_location=device, weights_only=True))
            D_S.load_state_dict(torch.load(resume_dir / "D_S.pt", map_location=device, weights_only=True))
            log(f"Resumed from checkpoints in {resume_dir} (including D_R/D_S)")
        else:
            log(f"Resumed from checkpoints in {resume_dir} -- D_R/D_S NOT found (checkpoint predates "
                f"critic saving), starting critics from scratch. Expect a rocky adjustment period "
                f"while they relearn against already-trained generators.")

    log(f"E_sim/E_real params: {sum(p.numel() for p in E_sim.parameters()):,} each")
    log(f"G_sr/G_rs params:  {sum(p.numel() for p in G_sr.parameters()):,} each")
    log(f"D_R/D_S params:    {sum(p.numel() for p in D_R.parameters()):,} each  (WDGRL critics)")
    log(f"WDGRL: n_critic={args.n_critic}  gp_weight={args.gp_weight}  "
        f"lam_adv_max={args.lam_adv_max} (ramped over {args.adv_ramp} joint epochs)  "
        f"grad_clip_norm=1.0")
    log(f"E_real anchor: lam_real_max={args.lam_real_max}  real_ramp={args.real_ramp}")
    log(f"High-freq residual penalty (G_sr only, v3b): lam_hf={args.lam_hf}  hf_kernel={args.hf_kernel}")
    log(f"Mixup (v3b): use_mixup={args.use_mixup}  mixup_alpha={args.mixup_alpha}")

    # ── Optimizers ────────────────────────────────────────────────────────────
    opt_G   = torch.optim.Adam(
        list(G_sr.parameters()) + list(G_rs.parameters()),
        lr=args.lr_gan, betas=(0.5, 0.999),
    )
    opt_critic = torch.optim.Adam(
        list(D_R.parameters()) + list(D_S.parameters()),
        lr=args.lr_gan, betas=(0.5, 0.9),
    )
    opt_sim = torch.optim.AdamW(
        list(E_sim.parameters()) + list(flow_sim.parameters()),
        lr=args.lr_npe,
    )
    opt_real = torch.optim.AdamW(
        list(E_real.parameters()) + list(flow_real.parameters()),
        lr=args.lr_npe,
    )

    # ── CSV logger ────────────────────────────────────────────────────────────
    if args.resume:
        existing = sorted(run_dir.glob("train_log_*.csv"))
        csv_path = existing[-1] if existing else run_dir / f"train_log_{ts}.csv"
        csv_fh   = open(csv_path, "a")
    else:
        csv_path = run_dir / f"train_log_{ts}.csv"
        csv_fh   = open(csv_path, "w")
        csv_fh.write("epoch,phase,npe_sim,npe_srs,gap,npe_real,npe_real_val,L_adv,L_cyc,L_id,L_info_G,L_real_G,"
                     "hf_sr,hf_rs,loss_G,w1_R,w1_S,gp_R,gp_S,loss_critic,delta_waves,delta_scal,"
                     "lam_info,lam_adv,lam_real,cov_rap,cov_ras,calib_score\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    end_epoch = args.start_epoch + args.max_epochs - 1
    best_calib_score = -float("inf")
    best_calib_epoch = None
    for epoch in range(1, args.max_epochs + 1):
        abs_epoch = args.start_epoch + epoch - 1

        if   abs_epoch <= flow_end:  phase = "flow-warmup"
        elif abs_epoch <= enc_end:   phase = "enc-warmup"
        else:                        phase = "joint"

        lam_info = (lambda_info_schedule(abs_epoch, joint_start, end_epoch, args.info_ramp)
                    * args.lam_info_max if phase == "joint" else 0.0)
        lam_adv = (lambda_info_schedule(abs_epoch, joint_start, end_epoch, args.adv_ramp)
                    * args.lam_adv_max if phase == "joint" else 0.0)
        lam_real = (lambda_info_schedule(abs_epoch, joint_start, end_epoch, args.real_ramp)
                    * args.lam_real_max if phase == "joint" else 0.0)

        # Freeze / unfreeze
        for p in E_sim.parameters():     p.requires_grad = (phase != "flow-warmup")
        for p in flow_sim.parameters():  p.requires_grad = (phase != "enc-warmup")
        for p in E_real.parameters():    p.requires_grad = (phase == "joint")
        for p in flow_real.parameters(): p.requires_grad = (phase == "joint")
        for p in G_sr.parameters():      p.requires_grad = (phase == "joint")
        for p in G_rs.parameters():      p.requires_grad = (phase == "joint")
        for p in D_R.parameters():       p.requires_grad = (phase == "joint")
        for p in D_S.parameters():       p.requires_grad = (phase == "joint")

        enc_npe_sum = G_loss_sum = critic_loss_sum = npe_srs_sum = npe_real_sum = 0.0
        adv_sum = cyc_sum = id_sum = info_g_sum = real_g_sum = hf_sr_sum = hf_rs_sum = 0.0
        w1_R_sum = w1_S_sum = gp_R_sum = gp_S_sum = dw_sum = ds_sum = 0.0
        n_batches = 0

        real_iter = iter(real_dl)

        def sample_real(n):
            if args.use_mixup:
                return mixup_real(real_beats_gpu, n, args.mixup_alpha, device)
            idx = torch.randint(0, len(real_ds), (n,))
            return real_ds.x[idx].to(device)

        for theta, x_s in sim_dl:
            theta = theta.to(device)
            x_s   = x_s.to(device)

            try:
                x_r = next(real_iter).to(device)
            except StopIteration:
                real_iter = iter(real_dl)
                x_r = next(real_iter).to(device)

            # ── 1. Critic inner loop (joint only) ─────────────────────────
            if phase == "joint":
                critic_loss_ep = w1_R_ep = w1_S_ep = gp_R_ep = gp_S_ep = 0.0
                critic_params = list(D_R.parameters()) + list(D_S.parameters())
                for _ in range(args.n_critic):
                    x_r_c = sample_real(x_s.shape[0])
                    opt_critic.zero_grad()
                    critic_loss, critic_info = loss_critics(
                        G_sr, G_rs, D_R, D_S, x_s, x_r_c, args.gp_weight, device,
                        wave_only=args.freeze_scalars,
                    )
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
                    opt_critic.step()
                    critic_loss_ep += critic_loss.item()
                    w1_R_ep += critic_info["w1_R"]; w1_S_ep += critic_info["w1_S"]
                    gp_R_ep += critic_info["gp_R"]; gp_S_ep += critic_info["gp_S"]
                critic_loss_sum += critic_loss_ep / args.n_critic
                w1_R_sum += w1_R_ep / args.n_critic; w1_S_sum += w1_S_ep / args.n_critic
                gp_R_sum += gp_R_ep / args.n_critic; gp_S_sum += gp_S_ep / args.n_critic

            # ── 2. Generator step (joint only) ────────────────────────────
            if phase == "joint":
                opt_G.zero_grad()
                lam_info_G = 0.0 if args.no_info_in_G else lam_info
                G_loss, G_info = loss_generator(
                    G_sr, G_rs, D_R, D_S, E_sim, flow_sim, E_real, flow_real,
                    x_s, x_r, theta, args.lam_cyc, args.lam_id, lam_info_G, lam_adv,
                    lam_real, args.lam_hf, args.hf_kernel,
                    wave_only=args.freeze_scalars,
                    clamp_info_gap=args.clamp_info_gap,
                )
                G_loss.backward()
                # v3b: decoupled per-generator clipping (was one combined-norm call over
                # both G_sr and G_rs) -- v3's ep397-399 divergence traced specifically to
                # G_rs's own weights going pathological while G_sr stayed normal; a combined
                # clip would rescale G_sr's healthy gradient by the same factor as G_rs's
                # exploding one instead of isolating the blowup to G_rs alone.
                torch.nn.utils.clip_grad_norm_(G_sr.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(G_rs.parameters(), 1.0)
                opt_G.step()
                G_loss_sum  += G_loss.item()
                adv_sum     += G_info["adv"]
                cyc_sum     += G_info["cyc"]
                id_sum      += G_info["id"]
                info_g_sum  += G_info["info"]
                real_g_sum  += G_info["real"]
                hf_sr_sum   += G_info["hf_sr"]
                hf_rs_sum   += G_info["hf_rs"]
                dw_sum      += G_info["delta_w"]
                ds_sum      += G_info["delta_s"]

            # ── 3. Sim posterior step (all phases) ────────────────────────
            opt_sim.zero_grad()

            if phase == "joint":
                with torch.no_grad():
                    if args.freeze_scalars:
                        waves_srs = G_rs(G_sr(x_s[:, :_WAVE_DIM]))
                        x_srs_detached = torch.cat([waves_srs, x_s[:, _WAVE_DIM:]], dim=1)
                    else:
                        x_srs_detached = G_rs(G_sr(x_s))
            else:
                x_srs_detached = None

            NPE_loss, NPE_info = loss_sim_posterior(
                E_sim, flow_sim, x_s, theta, x_srs_detached, lam_info,
            )
            NPE_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(E_sim.parameters()) + list(flow_sim.parameters()), 1.0)
            opt_sim.step()

            enc_npe_sum  += NPE_info.get("npe_sim", 0.0)
            npe_srs_sum  += NPE_info.get("npe_srs", 0.0)

            # ── 4. Real posterior step (joint only) ───────────────────────
            if phase == "joint":
                opt_real.zero_grad()
                with torch.no_grad():
                    if args.freeze_scalars:
                        waves_sr = G_sr(x_s[:, :_WAVE_DIM])
                        x_sr_detached = torch.cat([waves_sr, x_s[:, _WAVE_DIM:]], dim=1)
                    else:
                        x_sr_detached = G_sr(x_s)

                real_loss, real_info = loss_real_posterior(E_real, flow_real, x_sr_detached, theta)
                real_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(E_real.parameters()) + list(flow_real.parameters()), 1.0)
                opt_real.step()
                npe_real_sum += real_info["npe_real"]

            n_batches += 1

        # ── Validation npe_real (held-out sims, never trained on) ───────────
        # G_sr(x_val) -> encoder_real -> flow_real, same route as the real posterior step,
        # but on manifest_test.json data. .eval() on all three so spectral-norm's power-
        # iteration buffers (which only update `if self.training`) aren't perturbed by a
        # batch that was never actually trained on; .train() restored right after so the
        # rest of the script's behavior is unaffected.
        npe_real_val = float("nan")
        if val_ds is not None and phase == "joint":
            G_sr.eval(); E_real.eval(); flow_real.eval()
            with torch.no_grad():
                val_idx   = torch.randint(0, len(val_ds), (min(args.batch_size, len(val_ds)),))
                theta_val = val_ds.theta[val_idx].to(device)
                x_val     = val_ds.x[val_idx].to(device)
                if args.freeze_scalars:
                    waves_sr_val = G_sr(x_val[:, :_WAVE_DIM])
                    x_sr_val = torch.cat([waves_sr_val, x_val[:, _WAVE_DIM:]], dim=1)
                else:
                    x_sr_val = G_sr(x_val)
                npe_real_val = (-flow_real.log_prob(theta_val, condition=E_real(x_sr_val)).mean()).item()
            G_sr.train(); E_real.train(); flow_real.train()

        # ── Calibration checkpoint (real patients, Rap/Ras 90% CI coverage) ─────
        # Rap/Ras have directly-measured ground truth on real patients (PVR/SVR from
        # measured pressures+flow), unlike Cas/Eap (model-fit, less reliable) -- so
        # this measures calibration on the actual target population directly,
        # unlike npe_real_val (still logged above as a diagnostic, but no longer
        # drives checkpoint selection -- it's blind to posterior width/overconfidence,
        # see experiments.csv). Only every --calib-every epochs: 802 patients x
        # --calib-n-samples flow samples each is much more expensive than npe_real_val.
        cov_rap = cov_ras = calib_score = float("nan")
        if phase == "joint" and abs_epoch % args.calib_every == 0:
            G_sr.eval(); E_real.eval(); flow_real.eval()
            rap_samples_all, ras_samples_all = [], []
            with torch.no_grad():
                for i in range(0, len(real_ds), args.batch_size):
                    x_chunk = real_ds.x[i:i + args.batch_size].to(device)
                    if args.freeze_scalars:
                        waves_sr_c = G_sr(x_chunk[:, :_WAVE_DIM])
                        x_sr_chunk = torch.cat([waves_sr_c, x_chunk[:, _WAVE_DIM:]], dim=1)
                    else:
                        x_sr_chunk = G_sr(x_chunk)
                    z_chunk = E_real(x_sr_chunk)
                    for j in range(z_chunk.shape[0]):
                        s = flow_real.sample((args.calib_n_samples,),
                                             condition=z_chunk[j:j + 1]).squeeze(1).cpu().numpy()
                        rap_samples_all.append(s[:, rap_idx])
                        ras_samples_all.append(s[:, ras_idx])
            G_sr.train(); E_real.train(); flow_real.train()

            rap_samples_all = np.stack(rap_samples_all)  # (802, calib_n_samples)
            ras_samples_all = np.stack(ras_samples_all)

            def _coverage90(samples_all, gt_arr, valid):
                lo = np.percentile(samples_all, 5, axis=1)
                hi = np.percentile(samples_all, 95, axis=1)
                return float(((gt_arr[valid] >= lo[valid]) & (gt_arr[valid] <= hi[valid])).mean())

            cov_rap = _coverage90(rap_samples_all, real_gt_rap, rap_valid)
            cov_ras = _coverage90(ras_samples_all, real_gt_ras, ras_valid)
            close_rap = 1.0 - abs(cov_rap - 0.90)
            close_ras = 1.0 - abs(cov_ras - 0.90)
            calib_score = 2 * close_rap * close_ras / (close_rap + close_ras + 1e-9)

            if calib_score > best_calib_score:
                best_calib_score = calib_score
                best_calib_epoch = abs_epoch
                best_calib_ckpt_dir = run_dir / "checkpoints" / f"{ts}_best_calib"
                save_checkpoint(best_calib_ckpt_dir, E_sim, flow_sim, E_real, flow_real, G_sr, G_rs, D_R, D_S)

        # ── Epoch logging ─────────────────────────────────────────────────
        nb = max(n_batches, 1)
        npe_sim  = enc_npe_sum / n_batches
        npe_srs  = npe_srs_sum / n_batches
        npe_real = npe_real_sum / nb
        gap      = npe_srs - npe_sim
        G_loss_e      = G_loss_sum      / nb
        critic_loss_e = critic_loss_sum / nb
        adv_e    = adv_sum     / nb
        cyc_e    = cyc_sum     / nb
        id_e     = id_sum      / nb
        info_g_e = info_g_sum  / nb
        real_g_e = real_g_sum  / nb
        hf_sr_e  = hf_sr_sum   / nb
        hf_rs_e  = hf_rs_sum   / nb
        w1_R_e   = w1_R_sum    / nb
        w1_S_e   = w1_S_sum    / nb
        gp_R_e   = gp_R_sum    / nb
        gp_S_e   = gp_S_sum    / nb
        dw_e     = dw_sum      / nb
        ds_e     = ds_sum      / nb

        log(f"  ep {abs_epoch:3d}/{end_epoch}  [{phase}]"
            f"  npe_sim={npe_sim:.4f}  npe_srs={npe_srs:.4f}  gap={gap:+.4f}  npe_real={npe_real:.4f}"
            f"  npe_real_val={npe_real_val:.4f}"
            f"  L_adv={adv_e:.4f}  L_cyc={cyc_e:.4f}  L_id={id_e:.4f}  L_info={info_g_e:.4f}  L_real={real_g_e:.4f}"
            f"  hf_sr={hf_sr_e:.5f}  hf_rs={hf_rs_e:.5f}"
            f"  w1_R={w1_R_e:.4f}  w1_S={w1_S_e:.4f}  gp_R={gp_R_e:.4f}  gp_S={gp_S_e:.4f}"
            f"  |Δ|_w={dw_e:.4f}  |Δ|_s={ds_e:.4f}  λ_info={lam_info:.3f}  λ_adv={lam_adv:.3f}  λ_real={lam_real:.3f}"
            + (f"  cov_rap={cov_rap:.3f}  cov_ras={cov_ras:.3f}  calib_score={calib_score:.3f}"
               if not np.isnan(calib_score) else ""))
        csv_fh.write(f"{abs_epoch},{phase},{npe_sim:.6f},{npe_srs:.6f},{gap:.6f},{npe_real:.6f},"
                     f"{npe_real_val:.6f},"
                     f"{adv_e:.6f},{cyc_e:.6f},{id_e:.6f},{info_g_e:.6f},{real_g_e:.6f},"
                     f"{hf_sr_e:.6f},{hf_rs_e:.6f},{G_loss_e:.6f},"
                     f"{w1_R_e:.6f},{w1_S_e:.6f},{gp_R_e:.6f},{gp_S_e:.6f},{critic_loss_e:.6f},"
                     f"{dw_e:.6f},{ds_e:.6f},{lam_info:.4f},{lam_adv:.4f},{lam_real:.4f},"
                     f"{cov_rap:.6f},{cov_ras:.6f},{calib_score:.6f}\n")
        csv_fh.flush()

    # ── Save checkpoints ──────────────────────────────────────────────────────
    ckpt_dir = run_dir / "checkpoints" / ts
    save_checkpoint(ckpt_dir, E_sim, flow_sim, E_real, flow_real, G_sr, G_rs, D_R, D_S)
    log(f"Saved checkpoints to {ckpt_dir}")
    if best_calib_score > -float("inf"):
        log(f"Best calib_score={best_calib_score:.4f} at epoch {best_calib_epoch} "
            f"-> {ckpt_dir}_best_calib")

    # ── run_info (overwrite with full info on completion) ─────────────────────
    with open(run_dir / f"run_info_v{args.version}_{ts}.json", "w") as f:
        json.dump({
            "run": args.run, "version": args.version,
            "status": "done",
            "started": ts,
            "finished": f"{datetime.now():%Y%m%d-%H%M%S}",
            "command": " ".join(["train_spin.py"] + sys.argv[1:]),
            "git_hash": git_hash,
            "device": str(device),
            "schedule": {
                "flow_end": flow_end, "enc_end": enc_end,
                "joint_start": joint_start, "max_epochs": args.max_epochs,
                "info_ramp": args.info_ramp, "real_ramp": args.real_ramp,
            },
            "losses": {
                "lam_cyc": args.lam_cyc, "lam_id": args.lam_id,
                "lam_info_max": args.lam_info_max,
                "lam_real_max": args.lam_real_max,
                "lam_hf": args.lam_hf, "hf_kernel": args.hf_kernel,
                "hf_applies_to": "G_sr only (v3b)",
            },
            "flags": {
                "freeze_scalars": args.freeze_scalars,
                "clamp_info_gap": args.clamp_info_gap,
                "no_info_in_G": args.no_info_in_G,
                "use_mixup": args.use_mixup,
                "no_sv": args.no_sv,
            },
            "n_scalars": n_scalars,
            "mixup_alpha": args.mixup_alpha,
            "wdgrl": {
                "n_critic": args.n_critic,
                "gp_weight": args.gp_weight,
                "lam_adv_max": args.lam_adv_max,
                "adv_ramp": args.adv_ramp,
                "grad_clip_norm": "1.0, decoupled per-generator for G_sr/G_rs (v3b)",
            },
            "data": {
                "n_sims": args.n_sims, "n_val_sims": args.n_val_sims, "n_real_beats": len(real_ds),
                "sim_data_root": args.sim_data_root, "real_data": args.real_data,
            },
        }, f, indent=2)
    log("Done.")
    log_fh.close()
    csv_fh.close()


if __name__ == "__main__":
    main()
