"""
SPIN training: sim↔real dual generators + information-preserving posterior.

Schedule (per spec §3):
  Phase 0  flow-warmup   (ep 1–2)    flow trains, encoder frozen, G/D idle
  Phase 1  enc-warmup    (ep 3–12)   encoder trains, flow frozen, G/D idle
  Phase 2  joint         (ep 13+)    per-batch: generator step → discriminator step → posterior step

Per-batch step order (joint phase only, §2):
  1. Generator step   — update G_sr, G_rs       (encoder/flow as fixed functions, not detached)
  2. Discriminator step — update D_R, D_S       (generator outputs detached)
  3. Posterior step   — update h_ω, q_ψ         (x_srs detached, no grad into G)

Optimizers:
  opt_G   : Adam(G_sr + G_rs,   lr=2e-4, betas=(0.5, 0.999))
  opt_D   : Adam(D_R  + D_S,    lr=2e-4, betas=(0.5, 0.999))
  opt_NPE : AdamW(encoder + flow, lr=1e-4)
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
from models import (
    LipschitzEncoder, build_flow_net,
    DualBranchGenerator, DualBranchDiscriminator,
)

_WAVE_DIM = N_REDUCED_CHANNELS * T   # 804  — waveform slice of obs vector


# ── Noise helper ──────────────────────────────────────────────────────────────

def enc_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add fresh i.i.d. Gaussian noise to encoder input. sigma=0 → no-op."""
    if sigma <= 0:
        return x
    return x + torch.randn_like(x) * sigma


# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup_real(real_x: torch.Tensor, n: int, alpha: float, device: torch.device) -> torch.Tensor:
    """On-the-fly Beta mixup of real beats. Returns n interpolated samples."""
    idx_i = torch.randint(0, len(real_x), (n,), device=device)
    idx_j = torch.randint(0, len(real_x), (n,), device=device)
    lam   = torch.distributions.Beta(alpha, alpha).sample((n,)).to(device).unsqueeze(1)
    return lam * real_x[idx_i] + (1 - lam) * real_x[idx_j]


# ── Losses ────────────────────────────────────────────────────────────────────

def loss_generator(G_sr, G_rs, D_R, D_S, x_s, x_r, theta,
                   encoder, flow, lam_cyc, lam_id, lam_info,
                   enc_noise_sigma=0.0, lam_adv=1.0):
    """
    Generator step: update G_sr and G_rs.

    encoder + flow are used as fixed functions — gradients flow THROUGH them back to
    G_sr/G_rs (no .detach()), but their weights are not in opt_G so they don't update
    here. They update only via opt_NPE in the posterior step.

    Returns total loss and dict of decomposed scalar losses + transport magnitude.
    """
    x_sr  = G_sr(x_s)
    x_rs  = G_rs(x_r)
    x_srs = G_rs(x_sr)
    x_rsr = G_sr(x_rs)

    L_adv = -D_R(x_sr).mean() - D_S(x_rs).mean()
    L_cyc = (x_srs - x_s).abs().mean() + (x_rsr - x_r).abs().mean()
    L_id  = (G_rs(x_s) - x_s).abs().mean() + (G_sr(x_r) - x_r).abs().mean()

    if lam_info > 0:
        L_info = -flow.log_prob(theta, condition=encoder(enc_noise(x_srs, enc_noise_sigma))).mean()
    else:
        L_info = torch.zeros(1, device=x_s.device).squeeze()

    loss = lam_adv * L_adv + lam_cyc * L_cyc + lam_id * L_id + lam_info * L_info

    # Transport magnitude: Δ = G_sr(x_s) − x_s, split waveform vs scalar
    with torch.no_grad():
        delta   = (x_sr - x_s).abs()
        delta_w = delta[:, :_WAVE_DIM].mean().item()
        delta_s = delta[:, _WAVE_DIM:].mean().item()

    return loss, {
        "L_adv":    L_adv.item(),
        "L_cyc":    L_cyc.item(),
        "L_id":     L_id.item(),
        "L_info_G": L_info.item(),
        "delta_w":  delta_w,
        "delta_s":  delta_s,
    }


def loss_discriminator(G_sr, G_rs, D_R, D_S, x_s, x_r):
    """
    Discriminator step: update D_R and D_S.

    Hinge loss:
      L_D_R = relu(1 - D_R(x_r)).mean() + relu(1 + D_R(x_sr.detach())).mean()
      L_D_S = relu(1 - D_S(x_s)).mean() + relu(1 + D_S(x_rs.detach())).mean()

    2.0 = confusion floor (D outputs 0 everywhere).  0.0 = perfect separation.
    Healthy alignment holds near 2.0.
    """
    x_sr = G_sr(x_s).detach()
    x_rs = G_rs(x_r).detach()

    L_D_R = (torch.relu(1 - D_R(x_r)).mean()
           + torch.relu(1 + D_R(x_sr)).mean())
    L_D_S = (torch.relu(1 - D_S(x_s)).mean()
           + torch.relu(1 + D_S(x_rs)).mean())

    loss = L_D_R + L_D_S
    return loss, {"L_D_R": L_D_R.item(), "L_D_S": L_D_S.item()}


def loss_posterior(encoder, flow, x_s, theta, x_srs_detached, lam_info,
                   enc_noise_sigma=0.0):
    """
    Posterior step: update h_ω and q_ψ.
    x_srs_detached must already be detached — no grad flows into G from this step.

    Fresh noise drawn independently for x_s and x_srs so the encoder cannot
    distinguish transported from original inputs by noise fingerprint.
    """
    npe_sim = -flow.log_prob(theta, condition=encoder(enc_noise(x_s, enc_noise_sigma))).mean()

    if x_srs_detached is not None and lam_info > 0:
        npe_srs = -flow.log_prob(
            theta, condition=encoder(enc_noise(x_srs_detached, enc_noise_sigma))
        ).mean()
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
    parser.add_argument("--run",               required=True)
    parser.add_argument("--version",           default="1")
    parser.add_argument("--sim-data-root",     required=True)
    parser.add_argument("--real-data",         required=True)
    parser.add_argument("--n-sims",            type=int, required=True)
    parser.add_argument("--max-epochs",        type=int, default=400)
    parser.add_argument("--flow-warmup",       type=int, default=2)
    parser.add_argument("--enc-warmup",        type=int, default=10)
    parser.add_argument("--lam-cyc",           type=float, default=10.0)
    parser.add_argument("--lam-id",            type=float, default=5.0)
    parser.add_argument("--lam-info-max",      type=float, default=1.0)
    parser.add_argument("--info-ramp",         type=int,   default=50)
    parser.add_argument("--batch-size",        type=int, default=512)
    parser.add_argument("--lr-npe",            type=float, default=1e-4)
    parser.add_argument("--lr-gan",            type=float, default=2e-4)
    parser.add_argument("--latent-dim",        type=int, default=128)
    parser.add_argument("--stats-path",        default="norm_stats.json")
    parser.add_argument("--enc-noise-sigma",   type=float, default=0.0,
                        help="Gaussian noise sigma on encoder inputs during training (0=off)")
    parser.add_argument("--use-mixup",         action="store_true",
                        help="Augment real pool with on-the-fly Beta mixup (Run B)")
    parser.add_argument("--mixup-alpha",       type=float, default=0.4,
                        help="Beta distribution concentration for mixup (same as v3 default)")
    parser.add_argument("--early-abort-epoch", type=int, default=100,
                        help="Epoch at which to check for divergence/steganography")
    parser.add_argument("--early-abort-gap",   type=float, default=-1.0,
                        help="Abort if gap=npe_srs-npe_sim < this value at --early-abort-epoch")
    parser.add_argument("--resume",            action="store_true")
    parser.add_argument("--start-epoch",       type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma  = args.enc_noise_sigma

    run_dir = Path("outputs") / args.run
    log, log_fh = make_log(run_dir)

    ts = f"{datetime.now():%Y%m%d-%H%M%S}"
    log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Run: {args.run}  v={args.version}")
    log(f"Device: {device}  max_epochs: {args.max_epochs}")
    log(f"enc_noise_sigma={sigma}  use_mixup={args.use_mixup}  mixup_alpha={args.mixup_alpha}")

    with open(run_dir / f"run_info_v{args.version}_{ts}.json", "w") as f:
        json.dump({
            "run": args.run, "version": args.version,
            "status": "running",
            "started": ts,
            "command": " ".join(["train_spin.py"] + sys.argv[1:]),
        }, f, indent=2)

    flow_end    = args.flow_warmup
    enc_end     = flow_end + args.enc_warmup
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
    real_sampler = RandomSampler(real_ds, replacement=True, num_samples=len(sim_ds))
    real_dl = DataLoader(real_ds, batch_size=args.batch_size, sampler=real_sampler,
                         num_workers=0, pin_memory=True, drop_last=True)
    log(f"Real beats: {len(real_ds)}  (oversampled to ~{len(sim_ds)} per epoch)")

    if args.use_mixup:
        real_x_dev = real_ds.x.to(device)
        log(f"Mixup enabled: alpha={args.mixup_alpha}  pool={len(real_x_dev)} beats")

    log("Collecting theta stats for flow...")
    theta_all = sim_ds.theta[:min(10_000, len(sim_ds))]

    # ── Models ────────────────────────────────────────────────────────────────
    encoder = LipschitzEncoder(latent_dim=args.latent_dim).to(device)
    flow    = build_flow_net(args.latent_dim, theta_all).to(device)
    G_sr    = DualBranchGenerator().to(device)
    G_rs    = DualBranchGenerator().to(device)
    D_R     = DualBranchDiscriminator().to(device)
    D_S     = DualBranchDiscriminator().to(device)

    if args.resume:
        encoder.load_state_dict(torch.load(run_dir / "encoder.pt", map_location=device, weights_only=True))
        flow    = torch.load(run_dir / "flow_net.pt", map_location=device, weights_only=False)
        G_sr.load_state_dict(torch.load(run_dir / "G_sr.pt", map_location=device, weights_only=True))
        G_rs.load_state_dict(torch.load(run_dir / "G_rs.pt", map_location=device, weights_only=True))
        log(f"Resumed from checkpoints in {run_dir}")

    log(f"Encoder params:    {sum(p.numel() for p in encoder.parameters()):,}")
    log(f"G_sr/G_rs params:  {sum(p.numel() for p in G_sr.parameters()):,} each")
    log(f"D_R/D_S params:    {sum(p.numel() for p in D_R.parameters()):,} each")

    # ── Optimizers ────────────────────────────────────────────────────────────
    opt_G   = torch.optim.Adam(
        list(G_sr.parameters()) + list(G_rs.parameters()),
        lr=args.lr_gan, betas=(0.5, 0.999),
    )
    opt_D   = torch.optim.Adam(
        list(D_R.parameters()) + list(D_S.parameters()),
        lr=args.lr_gan, betas=(0.5, 0.999),
    )
    opt_NPE = torch.optim.AdamW(
        list(encoder.parameters()) + list(flow.parameters()),
        lr=args.lr_npe,
    )

    # ── CSV logger ────────────────────────────────────────────────────────────
    CSV_HEADER = ("epoch,phase,"
                  "npe_sim,npe_srs,gap,"
                  "L_adv,L_cyc,L_id,L_info_G,loss_G,"
                  "L_D_R,L_D_S,loss_D,"
                  "delta_waves,delta_scal,"
                  "lam_info\n")
    if args.resume:
        existing = sorted(run_dir.glob("train_log_*.csv"))
        csv_path = existing[-1] if existing else run_dir / f"train_log_{ts}.csv"
        csv_fh   = open(csv_path, "a")
    else:
        csv_path = run_dir / f"train_log_{ts}.csv"
        csv_fh   = open(csv_path, "w")
        csv_fh.write(CSV_HEADER)

    # ── Training loop ─────────────────────────────────────────────────────────
    end_epoch = args.start_epoch + args.max_epochs - 1
    for epoch in range(1, args.max_epochs + 1):
        abs_epoch = args.start_epoch + epoch - 1

        if   abs_epoch <= flow_end:  phase = "flow-warmup"
        elif abs_epoch <= enc_end:   phase = "enc-warmup"
        else:                        phase = "joint"

        lam_info = (lambda_info_schedule(abs_epoch, joint_start, end_epoch, args.info_ramp)
                    * args.lam_info_max if phase == "joint" else 0.0)

        for p in encoder.parameters(): p.requires_grad = (phase != "flow-warmup")
        for p in flow.parameters():    p.requires_grad = (phase != "enc-warmup")
        for p in G_sr.parameters():    p.requires_grad = (phase == "joint")
        for p in G_rs.parameters():    p.requires_grad = (phase == "joint")
        for p in D_R.parameters():     p.requires_grad = (phase == "joint")
        for p in D_S.parameters():     p.requires_grad = (phase == "joint")

        npe_sim_sum = npe_srs_sum = 0.0
        adv_sum = cyc_sum = id_sum = info_g_sum = G_loss_sum = 0.0
        D_R_sum = D_S_sum = D_loss_sum = 0.0
        dw_sum  = ds_sum  = 0.0
        n_batches = 0

        real_iter = iter(real_dl)

        for theta, x_s in sim_dl:
            theta = theta.to(device)
            x_s   = x_s.to(device)

            if args.use_mixup and phase == "joint":
                x_r = mixup_real(real_x_dev, args.batch_size, args.mixup_alpha, device)
            else:
                try:
                    x_r = next(real_iter).to(device)
                except StopIteration:
                    real_iter = iter(real_dl)
                    x_r = next(real_iter).to(device)

            # ── 1. Generator step (joint only) ────────────────────────────
            if phase == "joint":
                opt_G.zero_grad()
                G_loss, G_info = loss_generator(
                    G_sr, G_rs, D_R, D_S, x_s, x_r, theta,
                    encoder, flow, args.lam_cyc, args.lam_id, lam_info,
                    enc_noise_sigma=sigma,
                )
                G_loss.backward()
                opt_G.step()
                G_loss_sum  += G_loss.item()
                adv_sum     += G_info["L_adv"]
                cyc_sum     += G_info["L_cyc"]
                id_sum      += G_info["L_id"]
                info_g_sum  += G_info["L_info_G"]
                dw_sum      += G_info["delta_w"]
                ds_sum      += G_info["delta_s"]

            # ── 2. Discriminator step (joint only) ────────────────────────
            if phase == "joint":
                opt_D.zero_grad()
                D_loss, D_info = loss_discriminator(G_sr, G_rs, D_R, D_S, x_s, x_r)
                D_loss.backward()
                opt_D.step()
                D_loss_sum += D_loss.item()
                D_R_sum    += D_info["L_D_R"]
                D_S_sum    += D_info["L_D_S"]

            # ── 3. Posterior step (all phases) ────────────────────────────
            opt_NPE.zero_grad()
            if phase == "joint":
                with torch.no_grad():
                    x_srs_detached = G_rs(G_sr(x_s))
            else:
                x_srs_detached = None

            NPE_loss, NPE_info = loss_posterior(
                encoder, flow, x_s, theta, x_srs_detached, lam_info,
                enc_noise_sigma=sigma,
            )
            NPE_loss.backward()
            opt_NPE.step()

            npe_sim_sum += NPE_info["npe_sim"]
            npe_srs_sum += NPE_info["npe_srs"]
            n_batches   += 1

        # ── Epoch logging ─────────────────────────────────────────────────
        nb = max(n_batches, 1)
        npe_sim  = npe_sim_sum / nb
        npe_srs  = npe_srs_sum / nb
        gap      = npe_srs - npe_sim
        G_loss_e = G_loss_sum / nb
        D_loss_e = D_loss_sum / nb
        L_adv_e  = adv_sum    / nb
        L_cyc_e  = cyc_sum    / nb
        L_id_e   = id_sum     / nb
        L_info_e = info_g_sum / nb
        L_D_R_e  = D_R_sum    / nb
        L_D_S_e  = D_S_sum    / nb
        dw_e     = dw_sum     / nb
        ds_e     = ds_sum     / nb

        log(f"  ep {abs_epoch:3d}/{end_epoch}  [{phase}]"
            f"  npe_sim={npe_sim:.4f}  npe_srs={npe_srs:.4f}  gap={gap:+.4f}"
            f"  L_adv={L_adv_e:.3f}  L_cyc={L_cyc_e:.3f}  L_id={L_id_e:.3f}  L_info={L_info_e:.3f}"
            f"  L_D_R={L_D_R_e:.3f}  L_D_S={L_D_S_e:.3f}"
            f"  |Δ|_w={dw_e:.4f}  |Δ|_s={ds_e:.4f}"
            f"  λ_info={lam_info:.3f}")

        csv_fh.write(
            f"{abs_epoch},{phase},"
            f"{npe_sim:.6f},{npe_srs:.6f},{gap:.6f},"
            f"{L_adv_e:.6f},{L_cyc_e:.6f},{L_id_e:.6f},{L_info_e:.6f},{G_loss_e:.6f},"
            f"{L_D_R_e:.6f},{L_D_S_e:.6f},{D_loss_e:.6f},"
            f"{dw_e:.6f},{ds_e:.6f},"
            f"{lam_info:.4f}\n"
        )
        csv_fh.flush()

        # ── Early abort ───────────────────────────────────────────────────
        if abs_epoch == args.early_abort_epoch and phase == "joint":
            if gap < args.early_abort_gap:
                msg = (f"\n{'='*70}\n"
                       f"EARLY ABORT: ep {abs_epoch}  gap={gap:+.4f} < threshold {args.early_abort_gap:+.1f}\n"
                       f"npe_srs ({npe_srs:.3f}) pulling far below npe_sim ({npe_sim:.3f}).\n"
                       f"Steganographic side-channel likely. Increase sigma or investigate.\n"
                       f"{'='*70}\n")
                log(msg)
                break

    # ── Save checkpoints ──────────────────────────────────────────────────────
    torch.save(encoder.state_dict(), run_dir / "encoder.pt")
    torch.save(flow,                  run_dir / "flow_net.pt")
    torch.save(G_sr.state_dict(),     run_dir / "G_sr.pt")
    torch.save(G_rs.state_dict(),     run_dir / "G_rs.pt")
    log(f"Saved checkpoints to {run_dir}")

    # ── run_info ──────────────────────────────────────────────────────────────
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
            "noise": {
                "enc_noise_sigma": sigma,
            },
            "data": {
                "n_sims": args.n_sims, "n_real_beats": len(real_ds),
                "use_mixup": args.use_mixup, "mixup_alpha": args.mixup_alpha,
                "sim_data_root": args.sim_data_root, "real_data": args.real_data,
            },
        }, f, indent=2)

    log("Done.")
    log_fh.close()
    csv_fh.close()


if __name__ == "__main__":
    main()
