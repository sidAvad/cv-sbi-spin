"""
SPIN training: sim↔real dual generators + information-preserving posterior.

v2b variant: WDGRL+GRL replaces the hinge-loss adversarial mechanism (v1/v2a).
D_R/D_S become genuine WGAN-GP critics, trained with their own multi-step inner
loop each batch (n_critic updates, gradient penalty for the Lipschitz constraint —
same scheme as cv-dann-sbi's WDGRL). The generator step no longer needs a
hand-negated adversarial loss term: GradientReversalLayer sits between each
generator's output and its (frozen, at this step) critic, so a plain minimize of
the critic's score on the reversed-gradient path trains G_sr/G_rs to fool it,
in one step, without an explicit "-D(x).mean()" formula.

Schedule (per spec §3):
  Phase 0  flow-warmup   (ep 1–2)    flow trains, encoder frozen, G/D idle
  Phase 1  enc-warmup    (ep 3–12)   encoder trains, flow frozen, G/D idle
  Phase 2  joint         (ep 13+)    per-batch: critic inner loop → generator step → posterior step

Per-batch step order (joint phase only):
  1. Critic inner loop  — n_critic updates to D_R, D_S (WGAN-GP; G_sr/G_rs outputs detached)
  2. Generator step     — update G_sr, G_rs via GRL-reversed gradient through frozen D_R/D_S
                           (encoder/flow as fixed functions for L_info, not detached)
  3. Posterior step     — update h_ω, q_ψ         (x_srs detached, no grad into G)

Optimizers:
  opt_critic : Adam(D_R + D_S,      lr=2e-4, betas=(0.5, 0.9))
  opt_G      : Adam(G_sr + G_rs,    lr=2e-4, betas=(0.5, 0.999))
  opt_NPE    : AdamW(encoder + flow, lr=1e-4)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    ReducedCVDataset, RealBeatsDataset,
    load_stats, load_manifest,
    PARAM_KEYS_INFER, N_REDUCED_CHANNELS, T,
)

_WAVE_DIM = N_REDUCED_CHANNELS * T  # 804
from models import (
    LipschitzEncoder, build_flow_net,
    DualBranchGenerator, DualBranchDiscriminator, GradientReversalLayer,
)


# ── Losses ────────────────────────────────────────────────────────────────────

def gradient_penalty(disc_fn, x_fake, x_real, device):
    """
    1-Lipschitz gradient penalty for WGAN-GP at random interpolates between a
    (detached) fake and real batch. disc_fn already handles wave_only slicing.

    create_graph=True is critical — without it the GP has no gradient through
    the critic and the Lipschitz constraint is not enforced. Ported from
    cv-dann-sbi/train_joint.py (there interpolates a flat latent z; here it
    interpolates the raw observation tensor — same formula works generically).
    """
    eps   = torch.rand(x_fake.shape[0], 1, device=device)
    x_hat = (eps * x_fake.detach() + (1 - eps) * x_real.detach()).requires_grad_(True)
    f_hat = disc_fn(x_hat)
    grads = torch.autograd.grad(f_hat.sum(), x_hat, create_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def loss_generator(G_sr, G_rs, D_R, D_S, x_s, x_r, theta,
                   encoder, flow, lam_cyc, lam_id, lam_info, grl, lam_adv=1.0,
                   wave_only=False, clamp_info_gap=False):
    """
    Generator step: update G_sr and G_rs.

    encoder + flow are used as fixed functions — gradients flow THROUGH them back to
    G_sr/G_rs (no .detach()), but their weights are not in opt_G so they don't update
    here. They update only via opt_NPE in the posterior step. D_R/D_S are likewise
    fixed here (not in opt_G) — forward value is unaffected by that; only the
    gradient reaching G_sr/G_rs matters, and grl reverses it (v2b WDGRL+GRL).

    wave_only: generators receive/return waves only (804-dim); scalars are passed through
               unchanged and concatenated back before encoder/discriminator calls.
    clamp_info_gap: L_info = relu(npe_srs - npe_sim) instead of npe_srs — gradient to
                    generator only when translated reals are harder than pure sims.

    L_G = lam_adv * L_adv + lam_cyc * L_cyc + lam_id * L_id + lam_info * L_info
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
    x_sr  = _gen(G_sr, x_s)   # sim → real
    x_rs  = _gen(G_rs, x_r)   # real → sim
    x_srs = _gen(G_rs, x_sr)  # sim → real → sim  (round trip, spec §1b + §1d)
    x_rsr = _gen(G_sr, x_rs)  # real → sim → real (round trip, spec §1b)

    # WDGRL+GRL adversarial term (v2b): grl is identity in forward (so this is
    # numerically just the critic's score on the translated samples), but reverses
    # the gradient reaching G_sr/G_rs on backward — minimizing this plain score
    # trains the generators to fool the (frozen-here) critics, no manual negation.
    L_adv = _disc(D_R, grl(x_sr)).mean() + _disc(D_S, grl(x_rs)).mean()

    # Cycle consistency — L1 on both round trips (spec §1b)
    L_cyc = (x_srs - x_s).abs().mean() + (x_rsr - x_r).abs().mean()

    # Identity — each generator should be near-identity on its target domain (spec §1c)
    L_id = (_gen(G_rs, x_s) - x_s).abs().mean() + (_gen(G_sr, x_r) - x_r).abs().mean()

    # Information preservation — sim→real→sim only, no .detach() on encoder/flow (spec §1d)
    if lam_info > 0:
        npe_srs = -flow.log_prob(theta, condition=encoder(x_srs)).mean()
        if clamp_info_gap:
            # gradient only when translated reals are harder than sims
            with torch.no_grad():
                npe_sim_ref = -flow.log_prob(theta, condition=encoder(x_s)).mean()
            L_info = torch.relu(npe_srs - npe_sim_ref)
        else:
            L_info = npe_srs
    else:
        L_info = torch.zeros(1, device=x_s.device).squeeze()

    loss = lam_adv * L_adv + lam_cyc * L_cyc + lam_id * L_id + lam_info * L_info

    with torch.no_grad():
        delta   = (x_sr - x_s).abs()
        delta_w = delta[:, :_WAVE_DIM].mean().item()
        delta_s = delta[:, _WAVE_DIM:].mean().item()

    return loss, {
        "adv":     L_adv.item(),
        "cyc":     L_cyc.item(),
        "id":      L_id.item(),
        "info":    L_info.item(),
        "delta_w": delta_w,
        "delta_s": delta_s,
    }


def loss_critics(G_sr, G_rs, D_R, D_S, x_s, x_r, gp_weight, device, wave_only=False):
    """
    Critic step (v2b): update D_R and D_S as genuine WGAN-GP critics — one call
    per n_critic inner iteration. G_sr(x_s)/G_rs(x_r) are recomputed fresh each
    call (cheap; G isn't updated during this loop, so callers may also compute
    them once outside and pass detached tensors in — kept here for a self-
    contained per-step API matching cv-dann-sbi's critic loop).

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


def loss_posterior(encoder, flow, x_s, theta, x_srs_detached, lam_info):
    """
    Posterior step: update h_ω and q_ψ.
    x_srs_detached must already be detached — no grad flows into G from this step.

    L_NPE = -log q_ψ(θ | h_ω(x_s))
          + lam_info * (-log q_ψ(θ | h_ω(x_srs_detached)))
    """
    npe_sim = -flow.log_prob(theta, condition=encoder(x_s)).mean()

    if x_srs_detached is not None and lam_info > 0:
        npe_srs = -flow.log_prob(theta, condition=encoder(x_srs_detached)).mean()
    else:
        npe_srs = torch.zeros(1, device=x_s.device).squeeze()

    loss = npe_sim + lam_info * npe_srs
    return loss, {"npe_sim": npe_sim.item(), "npe_srs": npe_srs.item()}


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",          required=True)
    parser.add_argument("--version",      default="1")
    parser.add_argument("--sim-data-root", required=True)
    parser.add_argument("--real-data",    required=True)
    parser.add_argument("--n-sims",       type=int, required=True)
    parser.add_argument("--max-epochs",   type=int, default=400)
    parser.add_argument("--flow-warmup",  type=int, default=2)
    parser.add_argument("--enc-warmup",   type=int, default=10)
    parser.add_argument("--lam-cyc",      type=float, default=10.0)
    parser.add_argument("--lam-id",       type=float, default=5.0)
    parser.add_argument("--lam-info-max", type=float, default=1.0)
    parser.add_argument("--info-ramp",    type=int,   default=50,
                        help="Joint epochs over which lambda_info ramps 0→lam_info_max")
    parser.add_argument("--freeze-scalars",  action="store_true",
                        help="Wave-only generators/discriminators; scalars bypass G and route directly to encoder")
    parser.add_argument("--clamp-info-gap", action="store_true",
                        help="L_info_G = relu(npe_srs - npe_sim): gradient to G only when gap > 0")
    parser.add_argument("--no-info-in-G", action="store_true",
                        help="Zero lam_info in generator step; posterior still uses full ramp")
    parser.add_argument("--n-critic",     type=int,   default=5,
                        help="v2b WDGRL+GRL: critic inner-loop updates per generator step")
    parser.add_argument("--gp-weight",    type=float, default=10.0,
                        help="v2b WDGRL+GRL: gradient penalty weight (WGAN-GP Lipschitz constraint)")
    parser.add_argument("--grl-alpha",    type=float, default=1.0,
                        help="v2b WDGRL+GRL: gradient reversal scale reaching G_sr/G_rs")
    parser.add_argument("--batch-size",   type=int, default=512)
    parser.add_argument("--lr-npe",       type=float, default=1e-4)
    parser.add_argument("--lr-gan",       type=float, default=2e-4)
    parser.add_argument("--latent-dim",   type=int, default=128)
    parser.add_argument("--stats-path",   default="norm_stats.json")
    parser.add_argument("--resume",       action="store_true",
                        help="Load existing checkpoints from run_dir and continue training")
    parser.add_argument("--start-epoch",  type=int, default=1,
                        help="Epoch offset for display and lambda schedule (set to prev max_epochs+1 when resuming)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path("outputs") / args.run
    log, log_fh = make_log(run_dir)

    ts = f"{datetime.now():%Y%m%d-%H%M%S}"
    log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Run: {args.run}  v={args.version}")
    log(f"Device: {device}  max_epochs: {args.max_epochs}")

    # Write run_info immediately so the command is captured even on crash
    with open(run_dir / f"run_info_v{args.version}_{ts}.json", "w") as f:
        json.dump({
            "run": args.run, "version": args.version,
            "status": "running",
            "started": ts,
            "command": " ".join(["train_spin.py"] + sys.argv[1:]),
        }, f, indent=2)

    flow_end  = args.flow_warmup
    enc_end   = flow_end + args.enc_warmup
    joint_start = enc_end + 1
    log(f"Phase boundaries — flow_end={flow_end}  enc_end={enc_end}  joint_start={joint_start}")

    # ── Data ──────────────────────────────────────────────────────────────────
    stats    = load_stats(args.stats_path)
    sim_root = Path(args.sim_data_root)
    manifest = load_manifest(sim_root / "manifest_train.json")

    log(f"Loading {args.n_sims} sim observations...")
    sim_ds = ReducedCVDataset(str(sim_root / "train"), manifest["index"][:args.n_sims], stats,
                              log=log)
    sim_dl = DataLoader(sim_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    log(f"Loading real patient beats from {args.real_data}...")
    real_ds = RealBeatsDataset(args.real_data, stats, log=log)
    # Oversample reals so each epoch sees ~as many real batches as sim batches
    real_sampler = RandomSampler(real_ds, replacement=True,
                                 num_samples=len(sim_ds))
    real_dl = DataLoader(real_ds, batch_size=args.batch_size, sampler=real_sampler,
                         num_workers=0, pin_memory=True, drop_last=True)
    log(f"Real beats: {len(real_ds)}  (oversampled to ~{len(sim_ds)} per epoch)")

    # Theta stats for flow z-scoring — data already in RAM, just slice
    log("Collecting theta stats for flow...")
    theta_all = sim_ds.theta[:min(10_000, len(sim_ds))]

    # ── Models ────────────────────────────────────────────────────────────────
    encoder = LipschitzEncoder(latent_dim=args.latent_dim).to(device)
    flow    = build_flow_net(args.latent_dim, theta_all).to(device)
    G_sr    = DualBranchGenerator(wave_only=args.freeze_scalars).to(device)
    G_rs    = DualBranchGenerator(wave_only=args.freeze_scalars).to(device)
    D_R     = DualBranchDiscriminator(wave_only=args.freeze_scalars).to(device)
    D_S     = DualBranchDiscriminator(wave_only=args.freeze_scalars).to(device)
    grl     = GradientReversalLayer(alpha=args.grl_alpha)

    if args.resume:
        ckpt_root = run_dir / "checkpoints"
        existing_ckpts = sorted(p for p in ckpt_root.glob("*") if p.is_dir())
        if not existing_ckpts:
            raise FileNotFoundError(f"--resume set but no checkpoint subfolders found in {ckpt_root}")
        resume_dir = existing_ckpts[-1]
        encoder.load_state_dict(torch.load(resume_dir / "encoder.pt", map_location=device, weights_only=True))
        flow    = torch.load(resume_dir / "flow_net.pt", map_location=device, weights_only=False)
        G_sr.load_state_dict(torch.load(resume_dir / "G_sr.pt", map_location=device, weights_only=True))
        G_rs.load_state_dict(torch.load(resume_dir / "G_rs.pt", map_location=device, weights_only=True))
        log(f"Resumed from checkpoints in {resume_dir}")

    log(f"Encoder params:    {sum(p.numel() for p in encoder.parameters()):,}")
    log(f"G_sr/G_rs params:  {sum(p.numel() for p in G_sr.parameters()):,} each")
    log(f"D_R/D_S params:    {sum(p.numel() for p in D_R.parameters()):,} each  (WDGRL critics, v2b)")
    log(f"WDGRL: n_critic={args.n_critic}  gp_weight={args.gp_weight}  grl_alpha={args.grl_alpha}")

    # ── Optimizers ────────────────────────────────────────────────────────────
    opt_G   = torch.optim.Adam(
        list(G_sr.parameters()) + list(G_rs.parameters()),
        lr=args.lr_gan, betas=(0.5, 0.999),
    )
    opt_critic = torch.optim.Adam(
        list(D_R.parameters()) + list(D_S.parameters()),
        lr=args.lr_gan, betas=(0.5, 0.9),
    )
    opt_NPE = torch.optim.AdamW(
        list(encoder.parameters()) + list(flow.parameters()),
        lr=args.lr_npe,
    )

    # ── CSV logger ────────────────────────────────────────────────────────────
    if args.resume:
        # Append to the most recent existing CSV
        existing = sorted(run_dir.glob("train_log_*.csv"))
        csv_path = existing[-1] if existing else run_dir / f"train_log_{ts}.csv"
        csv_fh   = open(csv_path, "a")
    else:
        csv_path = run_dir / f"train_log_{ts}.csv"
        csv_fh   = open(csv_path, "w")
        csv_fh.write("epoch,phase,npe_sim,npe_srs,gap,L_adv,L_cyc,L_id,L_info_G,"
                     "loss_G,w1_R,w1_S,gp_R,gp_S,loss_critic,delta_waves,delta_scal,lam_info\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    end_epoch = args.start_epoch + args.max_epochs - 1
    for epoch in range(1, args.max_epochs + 1):
        abs_epoch = args.start_epoch + epoch - 1

        if   abs_epoch <= flow_end:  phase = "flow-warmup"
        elif abs_epoch <= enc_end:   phase = "enc-warmup"
        else:                        phase = "joint"

        lam_info = (lambda_info_schedule(abs_epoch, joint_start, end_epoch, args.info_ramp)
                    * args.lam_info_max if phase == "joint" else 0.0)

        # Freeze / unfreeze
        for p in encoder.parameters(): p.requires_grad = (phase != "flow-warmup")
        for p in flow.parameters():    p.requires_grad = (phase != "enc-warmup")
        for p in G_sr.parameters():    p.requires_grad = (phase == "joint")
        for p in G_rs.parameters():    p.requires_grad = (phase == "joint")
        for p in D_R.parameters():     p.requires_grad = (phase == "joint")
        for p in D_S.parameters():     p.requires_grad = (phase == "joint")

        enc_npe_sum = G_loss_sum = critic_loss_sum = npe_srs_sum = 0.0
        adv_sum = cyc_sum = id_sum = info_g_sum = 0.0
        w1_R_sum = w1_S_sum = gp_R_sum = gp_S_sum = dw_sum = ds_sum = 0.0
        n_batches = 0

        real_iter = iter(real_dl)

        for theta, x_s in sim_dl:
            theta = theta.to(device)
            x_s   = x_s.to(device)

            try:
                x_r = next(real_iter).to(device)
            except StopIteration:
                real_iter = iter(real_dl)
                x_r = next(real_iter).to(device)

            # ── 1. Critic inner loop (joint only, v2b WDGRL+GRL) ──────────
            # n_critic updates to D_R/D_S; same (x_s, x_r) batch reused across all
            # of them (G isn't updated here, so G_sr(x_s)/G_rs(x_r) wouldn't change
            # between iterations anyway — this is a deliberate simplification vs
            # cv-dann-sbi's version, which redraws a fresh real sample each inner
            # step; here the real batch also stays fixed for the whole outer step).
            if phase == "joint":
                critic_loss_ep = w1_R_ep = w1_S_ep = gp_R_ep = gp_S_ep = 0.0
                for _ in range(args.n_critic):
                    opt_critic.zero_grad()
                    critic_loss, critic_info = loss_critics(
                        G_sr, G_rs, D_R, D_S, x_s, x_r, args.gp_weight, device,
                        wave_only=args.freeze_scalars,
                    )
                    critic_loss.backward()
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
                    G_sr, G_rs, D_R, D_S, x_s, x_r, theta,
                    encoder, flow, args.lam_cyc, args.lam_id, lam_info_G, grl,
                    wave_only=args.freeze_scalars,
                    clamp_info_gap=args.clamp_info_gap,
                )
                G_loss.backward()
                opt_G.step()
                G_loss_sum  += G_loss.item()
                adv_sum     += G_info["adv"]
                cyc_sum     += G_info["cyc"]
                id_sum      += G_info["id"]
                info_g_sum  += G_info["info"]
                dw_sum      += G_info["delta_w"]
                ds_sum      += G_info["delta_s"]

            # ── 3. Posterior step (all phases) ────────────────────────────
            opt_NPE.zero_grad()

            # Build x_srs detached (no grad into G from posterior step)
            if phase == "joint":
                with torch.no_grad():
                    if args.freeze_scalars:
                        waves_srs = G_rs(G_sr(x_s[:, :_WAVE_DIM]))
                        x_srs_detached = torch.cat([waves_srs, x_s[:, _WAVE_DIM:]], dim=1)
                    else:
                        x_srs_detached = G_rs(G_sr(x_s))
            else:
                x_srs_detached = None

            NPE_loss, NPE_info = loss_posterior(
                encoder, flow, x_s, theta, x_srs_detached, lam_info,
            )
            NPE_loss.backward()
            opt_NPE.step()

            enc_npe_sum  += NPE_info.get("npe_sim", 0.0)
            npe_srs_sum  += NPE_info.get("npe_srs", 0.0)
            n_batches    += 1

        # ── Epoch logging ─────────────────────────────────────────────────
        nb = max(n_batches, 1)
        npe_sim  = enc_npe_sum / n_batches
        npe_srs  = npe_srs_sum / n_batches
        gap      = npe_srs - npe_sim
        G_loss_e      = G_loss_sum      / nb
        critic_loss_e = critic_loss_sum / nb
        adv_e    = adv_sum     / nb
        cyc_e    = cyc_sum     / nb
        id_e     = id_sum      / nb
        info_g_e = info_g_sum  / nb
        w1_R_e   = w1_R_sum    / nb
        w1_S_e   = w1_S_sum    / nb
        gp_R_e   = gp_R_sum    / nb
        gp_S_e   = gp_S_sum    / nb
        dw_e     = dw_sum      / nb
        ds_e     = ds_sum      / nb

        log(f"  ep {abs_epoch:3d}/{end_epoch}  [{phase}]"
            f"  npe_sim={npe_sim:.4f}  npe_srs={npe_srs:.4f}  gap={gap:+.4f}"
            f"  L_adv={adv_e:.4f}  L_cyc={cyc_e:.4f}  L_id={id_e:.4f}  L_info={info_g_e:.4f}"
            f"  w1_R={w1_R_e:.4f}  w1_S={w1_S_e:.4f}  gp_R={gp_R_e:.4f}  gp_S={gp_S_e:.4f}"
            f"  |Δ|_w={dw_e:.4f}  |Δ|_s={ds_e:.4f}  λ_info={lam_info:.3f}")
        csv_fh.write(f"{abs_epoch},{phase},{npe_sim:.6f},{npe_srs:.6f},{gap:.6f},"
                     f"{adv_e:.6f},{cyc_e:.6f},{id_e:.6f},{info_g_e:.6f},{G_loss_e:.6f},"
                     f"{w1_R_e:.6f},{w1_S_e:.6f},{gp_R_e:.6f},{gp_S_e:.6f},{critic_loss_e:.6f},"
                     f"{dw_e:.6f},{ds_e:.6f},{lam_info:.4f}\n")
        csv_fh.flush()

    # ── Save checkpoints ──────────────────────────────────────────────────────
    ckpt_dir = run_dir / "checkpoints" / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), ckpt_dir / "encoder.pt")
    torch.save(flow,                  ckpt_dir / "flow_net.pt")
    torch.save(G_sr.state_dict(),     ckpt_dir / "G_sr.pt")
    torch.save(G_rs.state_dict(),     ckpt_dir / "G_rs.pt")
    log(f"Saved checkpoints to {ckpt_dir}")

    # ── run_info (overwrite with full info on completion) ─────────────────────
    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"
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
                "info_ramp": args.info_ramp,
            },
            "losses": {
                "lam_cyc": args.lam_cyc, "lam_id": args.lam_id,
                "lam_info_max": args.lam_info_max,
            },
            "flags": {
                "freeze_scalars": args.freeze_scalars,
                "clamp_info_gap": args.clamp_info_gap,
                "no_info_in_G": args.no_info_in_G,
            },
            "wdgrl": {
                "n_critic": args.n_critic,
                "gp_weight": args.gp_weight,
                "grl_alpha": args.grl_alpha,
            },
            "data": {
                "n_sims": args.n_sims, "n_real_beats": len(real_ds),
                "sim_data_root": args.sim_data_root, "real_data": args.real_data,
            },
        }, f, indent=2)
    log("Done.")
    log_fh.close()
    csv_fh.close()


if __name__ == "__main__":
    main()
