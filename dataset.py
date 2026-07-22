"""
Dataset utilities for cv-sbi-spin (SPIN).

Sim data: ReducedCVDataset — (theta_infer, x) pairs, legacy normalization.
Real data: RealBeatsDataset — unlabeled x tensors from per-patient H5 files.
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
      x           : (809,) — 4 z-scored waveforms (4*201) + 5 scalars
                             scalars: MAP, SBP, DBP (via z-scored Pas), SV (via Vlv std), HR_z

    Legacy normalization only: shared sim-fitted affine map for both sims and reals.
    """

    def __init__(self, data_dir, index_entries, stats):
        self.data_dir = data_dir
        self.index    = index_entries
        self._handles = {}

        w = stats["waves"]
        p = stats["parameters"]

        self.wave_mean = torch.tensor(
            [w[k]["mean"] for k in WAVE_KEYS_REDUCED], dtype=torch.float32
        ).unsqueeze(1)
        self.wave_std = torch.tensor(
            [w[k]["std"] for k in WAVE_KEYS_REDUCED], dtype=torch.float32
        ).unsqueeze(1)

        self._pas_mean = w["Pas"]["mean"];  self._pas_std = w["Pas"]["std"] + 1e-8
        self._vlv_std  = w["Vlv"]["std"] + 1e-8
        self._hr_mean  = p["HR"]["mean"];   self._hr_std  = p["HR"]["std"] + 1e-8

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        path  = os.path.join(self.data_dir, entry["file"])
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        g = self._handles[path][entry["group"]]

        theta   = torch.tensor(
            [float(g[f"parameters/{k}"][()]) for k in PARAM_KEYS], dtype=torch.float32
        )
        hr_raw      = theta[_HR_IDX].item()
        theta_infer = torch.cat([theta[:_HR_IDX], theta[_HR_IDX + 1:]])  # (24,)

        waves = torch.from_numpy(
            np.stack([g[f"waves/{k}"][:] for k in WAVE_KEYS_REDUCED]).astype(np.float32)
        )
        waves = (waves - self.wave_mean) / (self.wave_std + 1e-8)  # (4, 201)

        pas_z = torch.from_numpy(g["waves/Pas"][:].astype(np.float32))
        pas_z = (pas_z - self._pas_mean) / self._pas_std

        vlv_z = torch.from_numpy(g["waves/Vlv"][:].astype(np.float32))
        sv    = (vlv_z.max() - vlv_z.min()) / self._vlv_std

        hr_z  = torch.tensor((hr_raw - self._hr_mean) / self._hr_std, dtype=torch.float32)

        scalars = torch.stack([pas_z.mean(), pas_z.max(), pas_z.min(), sv, hr_z])  # (5,)
        x = torch.cat([waves.reshape(-1), scalars])  # (809,)

        return theta_infer, x

    def close(self):
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()


class RealBeatsDataset(Dataset):
    """
    Unlabeled real patient beats. Returns x tensors only (no theta).

    Each .h5 file = one patient; takes the first beat_* group.
    Same legacy normalization as ReducedCVDataset — shared sim-fitted affine map.
    """

    def __init__(self, data_dir, stats):
        w = stats["waves"]
        p = stats["parameters"]

        self.wave_mean = torch.tensor(
            [w[k]["mean"] for k in WAVE_KEYS_REDUCED], dtype=torch.float32
        ).unsqueeze(1)
        self.wave_std = torch.tensor(
            [w[k]["std"] for k in WAVE_KEYS_REDUCED], dtype=torch.float32
        ).unsqueeze(1)

        self._pas_mean = w["Pas"]["mean"];  self._pas_std = w["Pas"]["std"] + 1e-8
        self._vlv_std  = w["Vlv"]["std"] + 1e-8
        self._hr_mean  = p["HR"]["mean"];   self._hr_std  = p["HR"]["std"] + 1e-8

        self.files = sorted(str(p) for p in __import__("pathlib").Path(data_dir).glob("*.h5"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], "r") as f:
            beat_keys = sorted(k for k in f.keys() if k.startswith("beat_"))
            g = f[beat_keys[0]]

            waves = torch.from_numpy(
                np.stack([g[f"waves/{k}"][:].astype(np.float32) for k in WAVE_KEYS_REDUCED])
            )
            waves = (waves - self.wave_mean) / (self.wave_std + 1e-8)

            # Real files have pre-computed scalar summaries — apply same sim-fitted normalization
            # so same physical value → same normalized point as in ReducedCVDataset.
            map_z = torch.tensor(
                (float(g["summaries/map"][()]) - self._pas_mean) / self._pas_std,
                dtype=torch.float32,
            )
            sbp_z = torch.tensor(
                (float(g["summaries/sbp"][()]) - self._pas_mean) / self._pas_std,
                dtype=torch.float32,
            )
            dbp_z = torch.tensor(
                (float(g["summaries/dbp"][()]) - self._pas_mean) / self._pas_std,
                dtype=torch.float32,
            )
            sv_z  = torch.tensor(
                float(g["summaries/sv"][()]) / self._vlv_std,
                dtype=torch.float32,
            )
            hr_z  = torch.tensor(
                (float(g["parameters/HR"][()]) - self._hr_mean) / self._hr_std,
                dtype=torch.float32,
            )

            scalars = torch.stack([map_z, sbp_z, dbp_z, sv_z, hr_z])
            return torch.cat([waves.reshape(-1), scalars])  # (809,)


def load_stats(stats_path="norm_stats.json"):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"{stats_path} not found.")
    with open(stats_path) as f:
        return json.load(f)


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)
