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

    Waveform branch: 3-level UNet 1D conv encoder-decoder with skip connections.
    Scalar branch:   small MLP (5 → scalar_hidden → scalar_hidden).
    Fused at bottleneck via broadcasted concat + pointwise conv.
    Residual output: x_out = x_in + Δ, with zero-init output heads (identity at init).

    Use one instance for G_sr (sim→real) and one for G_rs (real→sim).

    Shape trace (wave branch, default channels):
      enc1: (B, 4, 201)  → (B, 32, 201)   [no stride]
      enc2: (B, 32, 201) → (B, 64, 100)   [stride-2]
      enc3: (B, 64, 100) → (B, 128, 50)   [stride-2]
      enc4: (B, 128, 50) → (B, 256, 25)   [stride-2, bottleneck]
      up3+mix3: up (B,256,25)→(B,128,50), cat enc3 skip → mix → (B,128,50)
      up2+mix2: up (B,128,50)→(B,64,100), cat enc2 skip → mix → (B,64,100)
      up1+mix1: up (B,64,100)→(B,32,201), cat enc1 skip → mix → (B,32,201)  [output_padding=1]
    """

    def __init__(self, wave_ch: int = 32, scalar_hidden: int = 32, bottleneck_ch: int = 256):
        super().__init__()
        self.wave_len = N_REDUCED_CHANNELS * T  # 804

        # Waveform encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(N_REDUCED_CHANNELS, wave_ch, 7, padding=3), nn.SiLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(wave_ch,    wave_ch*2, 4, stride=2, padding=1), nn.SiLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(wave_ch*2,  wave_ch*4, 4, stride=2, padding=1), nn.SiLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(wave_ch*4, bottleneck_ch, 4, stride=2, padding=1), nn.SiLU()
        )

        # Scalar branch
        self.scalar_enc = nn.Sequential(
            nn.Linear(N_SCALARS, scalar_hidden), nn.SiLU(),
            nn.Linear(scalar_hidden, scalar_hidden), nn.SiLU(),
        )

        # Bottleneck fusion: broadcast scalar features over spatial dim, concat, fuse
        self.bottleneck = nn.Conv1d(bottleneck_ch + scalar_hidden, bottleneck_ch, 1)

        # Waveform decoder: upsample first, then concat skip, then mix
        self.up3  = nn.ConvTranspose1d(bottleneck_ch, wave_ch*4, 4, stride=2, padding=1)
        self.mix3 = nn.Sequential(nn.Conv1d(wave_ch*4*2, wave_ch*4, 3, padding=1), nn.SiLU())

        self.up2  = nn.ConvTranspose1d(wave_ch*4, wave_ch*2, 4, stride=2, padding=1)
        self.mix2 = nn.Sequential(nn.Conv1d(wave_ch*2*2, wave_ch*2, 3, padding=1), nn.SiLU())

        self.up1  = nn.ConvTranspose1d(wave_ch*2, wave_ch, 4, stride=2, padding=1, output_padding=1)
        self.mix1 = nn.Sequential(nn.Conv1d(wave_ch*2, wave_ch, 3, padding=1), nn.SiLU())

        # Output heads — zero-init so generator starts as identity
        self.wave_head   = nn.Conv1d(wave_ch, N_REDUCED_CHANNELS, 1)
        self.scalar_head = nn.Linear(scalar_hidden, N_SCALARS)
        nn.init.zeros_(self.wave_head.weight);   nn.init.zeros_(self.wave_head.bias)
        nn.init.zeros_(self.scalar_head.weight); nn.init.zeros_(self.scalar_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waves   = x[:, :self.wave_len].view(-1, N_REDUCED_CHANNELS, T)
        scalars = x[:, self.wave_len:]                                   # (B, 5)

        # Encode
        h1 = self.enc1(waves)   # (B, 32, 201)
        h2 = self.enc2(h1)      # (B, 64, 100)
        h3 = self.enc3(h2)      # (B, 128,  50)
        h4 = self.enc4(h3)      # (B, 256,  25)

        # Scalar features broadcast to bottleneck spatial dim
        s        = self.scalar_enc(scalars)                               # (B, 32)
        s_spatial = s.unsqueeze(-1).expand(-1, -1, h4.shape[-1])         # (B, 32, 25)

        # Fuse at bottleneck
        h = self.bottleneck(torch.cat([h4, s_spatial], dim=1))           # (B, 256, 25)

        # Decode: upsample → cat skip → mix
        h = self.mix3(torch.cat([F.silu(self.up3(h)), h3], dim=1))   # (B, 128, 50)
        h = self.mix2(torch.cat([F.silu(self.up2(h)), h2], dim=1))   # (B,  64, 100)
        h = self.mix1(torch.cat([F.silu(self.up1(h)), h1], dim=1))   # (B,  32, 201)

        # Residual deltas
        delta_waves   = self.wave_head(h)            # (B, 4, 201)
        delta_scalars = self.scalar_head(s)          # (B, 5)

        out_waves   = waves   + delta_waves          # (B, 4, 201)
        out_scalars = scalars + delta_scalars        # (B, 5)

        return torch.cat([out_waves.reshape(-1, self.wave_len), out_scalars], dim=1)  # (B, 809)


# ── Discriminators D_R / D_S ──────────────────────────────────────────────────

_sn = nn.utils.spectral_norm  # standard PyTorch SN for discriminators


class DualBranchDiscriminator(nn.Module):
    """
    Dual-branch discriminator with spectral normalization and hinge loss.
    Outputs unbounded scalar per sample — no sigmoid.

    Kept deliberately small: 802 reals overfits a large discriminator quickly.

    Wave branch:   SN-Conv ×3 (stride-2 each) → global avg pool → (B, 128)
    Scalar branch: SN-Linear → LeakyReLU       → (B, 32)
    Fusion:        cat → SN-Linear ×2          → (B,)

    Use one instance for D_R (real domain) and one for D_S (sim domain).
    """

    def __init__(self, wave_ch: int = 32):
        super().__init__()
        self.wave_len = N_REDUCED_CHANNELS * T  # 804

        self.wave_branch = nn.Sequential(
            _sn(nn.Conv1d(N_REDUCED_CHANNELS, wave_ch,    4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            _sn(nn.Conv1d(wave_ch,            wave_ch*2,  4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            _sn(nn.Conv1d(wave_ch*2,          wave_ch*4,  4, stride=2, padding=1)), nn.LeakyReLU(0.2),
        )  # → (B, 128, 25)

        self.scalar_branch = nn.Sequential(
            _sn(nn.Linear(N_SCALARS, wave_ch)), nn.LeakyReLU(0.2),
        )  # → (B, 32)

        fused_dim = wave_ch*4 + wave_ch  # 128 + 32
        self.head = nn.Sequential(
            _sn(nn.Linear(fused_dim, wave_ch*2)), nn.LeakyReLU(0.2),
            _sn(nn.Linear(wave_ch*2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waves   = x[:, :self.wave_len].view(-1, N_REDUCED_CHANNELS, T)
        scalars = x[:, self.wave_len:]

        h_w = self.wave_branch(waves).mean(dim=-1)   # global avg pool → (B, 128)
        h_s = self.scalar_branch(scalars)             # (B, 32)
        return self.head(torch.cat([h_w, h_s], dim=1)).squeeze(-1)  # (B,)
