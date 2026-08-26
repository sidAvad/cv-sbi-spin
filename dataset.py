"""
Dataset utilities for cv-sbi-spin (SPIN).

Sim data: ReducedCVDataset — (theta_infer, x) pairs, legacy normalization.
Real data: RealBeatsDataset — unlabeled x tensors from per-patient H5 files.

Both datasets preload all data into RAM at init.
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PARAM_KEYS = [
    "AVD", "Bla", "Blv", "Bra", "Brv",
    "Cas", "Cvp", "Cvs", "Eap",
    "Eedref_la", "Eedref_lv", "Eedref_ra", "Eedref_rv",
    "Emax_LA", "Emax_LV", "Emax_RA", "Emax_RV",
    "HR", "Rap", "Ras", "Tmax", "Tmax_a",
    "Vs", "τ", "τ_a",
]

_HR_IDX        = PARAM_KEYS.index("HR")
PARAM_KEYS_INFER = [k for k in PARAM_KEYS if k != "HR"]  # 24-dim

WAVE_KEYS_REDUCED = ["Prv", "Pra", "Pvp", "Pap"]
N_REDUCED_CHANNELS = len(WAVE_KEYS_REDUCED)   # 4
N_SCALARS          = 5                         # MAP, SBP, DBP, SV, HR
T                  = 201
OBS_DIM            = N_REDUCED_CHANNELS * T + N_SCALARS  # 809


class ReducedCVDataset(Dataset):
    """
    Returns (theta_infer, x) where:
      theta_infer : (24,)  — all params except HR
      x           : (809,) or (808,) if include_sv=False — 4 z-scored waveforms
                    (4*201) + scalars: MAP, SBP, DBP (via z-scored Pas), [SV (via Vlv
                    std) if include_sv], HR_z

    include_sv=False drops SV from x entirely (shrinks the scalar tail from 5 to 4) --
    for the --no-sv ablation, ported from cv-dann-sbi/train_joint.py's --no-sv. SV is
    still computed and stored separately in self.sv (z-scored, same formula) regardless
    of include_sv, so held-out ground truth remains available for the ablation's
    reconstruct-and-recover-SV comparison even when it's excluded from x.

    All data preloaded into RAM at init. self.theta, self.x, self.sv are accessible
    directly for theta stats collection etc.
    """

    def __init__(self, data_dir, index_entries, stats, log=print, include_sv: bool = True):
        w = stats["waves"]
        p = stats["parameters"]
        self.include_sv = include_sv
        obs_dim = OBS_DIM if include_sv else OBS_DIM - 1

        wave_mean = np.array([w[k]["mean"] for k in WAVE_KEYS_REDUCED], dtype=np.float32)[:, None]
        wave_std  = np.array([w[k]["std"]  for k in WAVE_KEYS_REDUCED], dtype=np.float32)[:, None] + 1e-8

        pas_mean = w["Pas"]["mean"];  pas_std = w["Pas"]["std"] + 1e-8
        vlv_std  = w["Vlv"]["std"] + 1e-8
        hr_mean  = p["HR"]["mean"];   hr_std  = p["HR"]["std"] + 1e-8

        n = len(index_entries)
        theta_buf = np.empty((n, len(PARAM_KEYS_INFER)), dtype=np.float32)
        x_buf     = np.empty((n, obs_dim),                dtype=np.float32)
        sv_buf    = np.empty((n,),                        dtype=np.float32)

        handles = {}
        log_interval = max(1, n // 20)
        for i, entry in enumerate(index_entries):
            path = os.path.join(data_dir, entry["file"])
            if path not in handles:
                handles[path] = h5py.File(path, "r")
            g = handles[path][entry["group"]]

            theta_raw = np.array([float(g[f"parameters/{k}"][()]) for k in PARAM_KEYS],
                                 dtype=np.float32)
            hr_raw = theta_raw[_HR_IDX]
            theta_buf[i] = np.concatenate([theta_raw[:_HR_IDX], theta_raw[_HR_IDX + 1:]])

            waves = np.stack([g[f"waves/{k}"][:] for k in WAVE_KEYS_REDUCED]).astype(np.float32)
            waves = (waves - wave_mean) / wave_std  # (4, 201)

            pas   = g["waves/Pas"][:].astype(np.float32)
            pas_z = (pas - pas_mean) / pas_std
            vlv   = g["waves/Vlv"][:].astype(np.float32)
            sv    = (vlv.max() - vlv.min()) / vlv_std
            hr_z  = (hr_raw - hr_mean) / hr_std
            sv_buf[i] = sv

            x_buf[i, :N_REDUCED_CHANNELS * T] = waves.ravel()
            scalars = [pas_z.mean(), pas_z.max(), pas_z.min(), sv, hr_z] if include_sv \
                     else [pas_z.mean(), pas_z.max(), pas_z.min(), hr_z]
            x_buf[i, N_REDUCED_CHANNELS * T:] = scalars

            if (i + 1) % log_interval == 0 or i == n - 1:
                log(f"  sims {i + 1}/{n} ({100*(i+1)//n}%)")

        for fh in handles.values():
            fh.close()

        self.theta = torch.from_numpy(theta_buf)
        self.x     = torch.from_numpy(x_buf)
        self.sv    = torch.from_numpy(sv_buf)

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]


class RealBeatsDataset(Dataset):
    """
    Unlabeled real patient beats. Returns x tensors only (no theta).

    Each .h5 file = one patient; takes the first beat_* group.
    Same legacy normalization as ReducedCVDataset — shared sim-fitted affine map.
    All 802 beats preloaded into RAM at init (~2.5 MB).

    include_sv=False drops SV from x (shrinks scalar tail 5->4), matching
    ReducedCVDataset's --no-sv ablation support. self.sv (z-scored) is always stored
    separately regardless of include_sv, self.file (patient filenames, no extension)
    for identifying which patient a given row is when cross-referencing ground truth
    SV later.
    """

    def __init__(self, data_dir, stats, log=print, include_sv: bool = True):
        w = stats["waves"]
        p = stats["parameters"]
        self.include_sv = include_sv
        obs_dim = OBS_DIM if include_sv else OBS_DIM - 1

        wave_mean = np.array([w[k]["mean"] for k in WAVE_KEYS_REDUCED], dtype=np.float32)[:, None]
        wave_std  = np.array([w[k]["std"]  for k in WAVE_KEYS_REDUCED], dtype=np.float32)[:, None] + 1e-8

        pas_mean = w["Pas"]["mean"];  pas_std = w["Pas"]["std"] + 1e-8
        vlv_std  = w["Vlv"]["std"] + 1e-8
        hr_mean  = p["HR"]["mean"];   hr_std  = p["HR"]["std"] + 1e-8

        import pathlib
        files = sorted(str(fp) for fp in pathlib.Path(data_dir).glob("*.h5"))
        n = len(files)
        x_buf  = np.empty((n, obs_dim), dtype=np.float32)
        sv_buf = np.empty((n,),         dtype=np.float32)

        for i, fpath in enumerate(files):
            with h5py.File(fpath, "r") as f:
                beat_keys = sorted(k for k in f.keys() if k.startswith("beat_"))
                g = f[beat_keys[0]]

                waves = np.stack([g[f"waves/{k}"][:] for k in WAVE_KEYS_REDUCED]).astype(np.float32)
                waves = (waves - wave_mean) / wave_std

                map_z = (float(g["summaries/map"][()]) - pas_mean) / pas_std
                sbp_z = (float(g["summaries/sbp"][()]) - pas_mean) / pas_std
                dbp_z = (float(g["summaries/dbp"][()]) - pas_mean) / pas_std
                sv_z  = float(g["summaries/sv"][()]) / vlv_std
                hr_z  = (float(g["parameters/HR"][()]) - hr_mean) / hr_std
                sv_buf[i] = sv_z

                x_buf[i, :N_REDUCED_CHANNELS * T] = waves.ravel()
                scalars = [map_z, sbp_z, dbp_z, sv_z, hr_z] if include_sv \
                         else [map_z, sbp_z, dbp_z, hr_z]
                x_buf[i, N_REDUCED_CHANNELS * T:] = scalars

        log(f"  loaded {n} real beats")
        self.x    = torch.from_numpy(x_buf)
        self.sv   = torch.from_numpy(sv_buf)
        self.file = [pathlib.Path(fp).stem for fp in files]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


def load_stats(stats_path="norm_stats.json"):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"{stats_path} not found.")
    with open(stats_path) as f:
        return json.load(f)


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)
