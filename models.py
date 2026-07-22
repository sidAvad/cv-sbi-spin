"""
Model components for cv-sbi-spin (SPIN).

Carried over from cv-dann-sbi (unchanged):
  LipschitzReducedAutoencoderEncoder  — h_ω encoder
  build_flow_net                      — MAF5 flow q_ψ

New for SPIN:
  DualBranchGenerator    — G_sr / G_rs (1D conv waveform trunk + scalar MLP, residual output)
  DualBranchDiscriminator — D_R / D_S (spectral-norm, hinge loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from sbi.neural_nets import posterior_nn

from dataset import N_REDUCED_CHANNELS, N_SCALARS, T

LATENT_DIM = 128


# ── Spectral norm ceiling (carried over from cv-dann-sbi) ─────────────────────

class _SoftSpectralCeiling(nn.Module):
    def __init__(self, ceiling: float, n_power_iters: int = 1):
        super().__init__()
        self.ceiling       = ceiling
        self.n_power_iters = n_power_iters
        self.register_buffer('_u', None)
        self.register_buffer('_v', None)
        self.last_sigma: float = 0.0

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        W_mat = W.reshape(W.shape[0], -1)
        h, w  = W_mat.shape
        if self._u is None or self._u.shape[0] != h:
            self._u = F.normalize(W_mat.new_empty(h).normal_(), dim=0)
        if self._v is None or self._v.shape[0] != w:
            self._v = F.normalize(W_mat.new_empty(w).normal_(), dim=0)
        u, v = self._u.detach(), self._v.detach()
        with torch.no_grad():
            for _ in range(self.n_power_iters):
                v = F.normalize(W_mat.t() @ u, dim=0, eps=1e-12)
                u = F.normalize(W_mat @ v,     dim=0, eps=1e-12)
        sigma = (u @ (W_mat @ v)).abs()
        if self.training:
            self._u.copy_(u)
            self._v.copy_(v)
        scale = (self.ceiling / sigma.clamp(min=1e-12)).clamp(max=1.0)
        self.last_sigma = (sigma * scale).item()
        return W * scale

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        return W


def _sn_ceiling(module: nn.Module, ceiling: float) -> nn.Module:
    P.register_parametrization(module, 'weight', _SoftSpectralCeiling(ceiling))
    return module


# ── Encoder h_ω (carried over from cv-dann-sbi) ───────────────────────────────

class LipschitzEncoder(nn.Module):
    CONV_LAYERS = [
        (N_REDUCED_CHANNELS, 64,  7, 1),
        (64,                 128, 5, 2),
        (128,                256, 5, 2),
        (256,                256, 3, 1),
    ]

    def __init__(self, latent_dim: int = LATENT_DIM, sn_ceiling: float = 2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.sn_ceiling = sn_ceiling
        self.wave_len   = N_REDUCED_CHANNELS * T
        feat_dim = self.CONV_LAYERS[-1][1]

        layers = []
        for in_ch, out_ch, k, s in self.CONV_LAYERS:
            layers += [
                _sn_ceiling(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, stride=s), sn_ceiling),
                nn.SiLU(),
            ]
        self.cnn          = nn.Sequential(*layers)
        self.scalar_projs = nn.ModuleList(
            [_sn_ceiling(nn.Linear(1, feat_dim), sn_ceiling) for _ in range(N_SCALARS)]
        )
        self.attn_pool = _sn_ceiling(nn.Linear(feat_dim, 1), sn_ceiling)
        self.proj      = _sn_ceiling(nn.Linear(feat_dim, latent_dim), sn_ceiling)

    @property
    def output_dim(self):
        return self.latent_dim

    def describe(self):
        return {
            "type": "LipschitzEncoder",
            "input_waveforms": f"({N_REDUCED_CHANNELS}, {T})",
            "input_scalars": "MAP, SBP, DBP, SV, HR_z",
            "latent_dim": self.latent_dim,
            "sn_ceiling": self.sn_ceiling,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waves   = x[:, :self.wave_len].view(-1, N_REDUCED_CHANNELS, T)
        scalars = x[:, self.wave_len:]

        h = self.cnn(waves).transpose(1, 2)                               # (B, T', 256)
        scalar_tokens = torch.stack(
            [proj(scalars[:, i:i+1]) for i, proj in enumerate(self.scalar_projs)],
            dim=1,
        )                                                                  # (B, 5, 256)
        h = torch.cat([scalar_tokens, h], dim=1)
        w = self.attn_pool(h).softmax(dim=1)
        return self.proj((w * h).sum(dim=1))


# ── Flow q_ψ ─────────────────────────────────────────────────────────────────

def build_flow_net(latent_dim: int, theta_stats: torch.Tensor,
                   hidden_features: int = 128, num_transforms: int = 5) -> nn.Module:
    """MAF5 flow via sbi's posterior_nn with identity embedding."""
    build_fn = posterior_nn(
        model="maf",
        embedding_net=nn.Identity(),
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        z_score_theta="independent",
        z_score_x="none",
    )
    z_dummy = torch.zeros(len(theta_stats), latent_dim)
    return build_fn(theta_stats.cpu(), z_dummy)


# ── Generators G_sr / G_rs ────────────────────────────────────────────────────

class DualBranchGenerator(nn.Module):
    """
    Dual-branch generator for sim↔real transport.

    Waveform branch: 1D conv encoder-decoder with skip connections over (4, 201).
    Scalar branch:   small MLP over the 5 scalars.
    Fused at bottleneck; residual output (x_out = x_in + Δ) for identity init.

    Use one instance for G_sr and one for G_rs — same architecture, separate weights.
    """
    pass  # TODO


# ── Discriminators D_R / D_S ──────────────────────────────────────────────────

class DualBranchDiscriminator(nn.Module):
    """
    Dual-branch discriminator with spectral normalization.
    Outputs unbounded scalar per sample for hinge loss.

    Use one instance for D_R (real domain) and one for D_S (sim domain).
    """
    pass  # TODO
