#!/usr/bin/env python3
"""
timeshap_smile_pipeline.py
=========================================================
One-file Temporal explainability for GRFNet-MultiScale with:
  - Temporal SHAP (from frozen NPZ LOSO dataset + checkpoints)
  - Temporal SMILE (from aligned CSV trials + checkpoints)
  - Optional vGRF overlay plots (with safe scaling controls)
  - Event-based temporal explainability (input/attribution/output landmark alignment + perturbation)
  - OPTIONAL: TimeSHAP-style event-wise KernelSHAP (stance “events” as features)

This "FULL" version restores the per-activity figures that worked in your older script:
  ✅ temporal_smile_by_activity.png (activity grid)
  ✅ temporal_smile_phase_bar.png   (phase bar summary)

And keeps your newer TimeSHAP plotting consistency:
  ✅ TimeSHAP E-bins → upsample to T_norm=100 (repeat or interp)
  ✅ Same heatmap style, same phase markers, same Top-K overlays
  ✅ BEFORE vs AFTER comparison figure

USAGE (examples)
----------------
# SMILE (+ per-activity, + overlays, + event analysis, + TimeSHAP)
python3 temporal_xai_onefile_eventfixed_TIMESHAPCONSISTENT_FULL.py smile \
  --data_dir /path/Dataset_Aligned \
  --ckpt_glob "/path/checkpoints/**/GRFNet_MultiScale.pt" \
  --outdir /path/temporal_xai \
  --device cuda \
  --n_samples 50 \
  --window_size 15 \
  --stride 10 \
  --imu_zscore fold \
  --overlay_scale none \
  --event_analysis \
  --event_landmark peak_vgrf \
  --event_topk 4 \
  --event_win_pct 6 \
  --timeshap \
  --timeshap_E 20 \
  --timeshap_K 300 \
  --timeshap_target peak_vgrf \
  --timeshap_max_samples 20 \
  --timeshap_upsample repeat

# SHAP
python3 temporal_xai_onefile_eventfixed_TIMESHAPCONSISTENT_FULL.py shap \
  --npz /path/frozen_dataset_loso_539.npz \
  --ckpt_glob "/path/checkpoints/**/GRFNet_MultiScale.pt" \
  --outdir /path/temporal_xai \
  --device cuda \
  --sample_n 50 \
  --bg_n 100

# BOTH
python3 temporal_xai_onefile_eventfixed_TIMESHAPCONSISTENT_FULL.py both \
  --npz /path/frozen_dataset_loso_539.npz \
  --data_dir /path/Dataset_Aligned \
  --ckpt_glob "/path/checkpoints/**/GRFNet_MultiScale.pt" \
  --outdir /path/temporal_xai \
  --device cuda
"""

import os
import re
import glob
import argparse
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib as mpl

from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, kendalltau
from scipy.special import comb

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PADDING_VALUE = -9999.0
EPS = 1e-9

CHANNEL_LABELS = [
    "waist_accX", "waist_accY", "waist_accZ",
    "waist_gyroX", "waist_gyroY", "waist_gyroZ",
    "wrist_accX",  "wrist_accY",  "wrist_accZ",
    "wrist_gyroX", "wrist_gyroY", "wrist_gyroZ",
]

PHASES = {
    "Loading":    (0.00, 0.40),
    "Mid-stance": (0.40, 0.80),
    "Push-off":   (0.80, 1.00),
}
PHASE_COLORS = {
    "Loading":    "#d4e6f1",
    "Mid-stance": "#d5f5e3",
    "Push-off":   "#fdebd0",
}

SENSOR_COLORS = ["#1f77b4"] * 6 + ["#ff7f0e"] * 6
ACTIVITY_ORDER = ["walking", "jogging", "running", "step_down", "heel_drop"]

FORCE_COL_CANDS = ["Force_Z", "force_z_N", "force_z", "Force_Vertical", "ForceZ", "Fz"]
MASS_COL_CANDS  = ["mass_kg", "Mass_kg", "body_mass_kg", "BodyMass_kg", "weight_kg", "Weight_kg"]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE STYLE
# ─────────────────────────────────────────────────────────────────────────────
def _apply_plot_style(fig_style: str, font_scale: float = 1.0):
    base = 8.0 * float(font_scale)
    mpl.rcParams.update({
        "font.size": base,
        "axes.titlesize": base + 2,
        "axes.labelsize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "figure.titlesize": base + 3,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

def _figsize_heatmap(fig_style: str, n_channels: int = 12):
    # IEEE single column ~3.5 in, double ~7.2 in
    if fig_style == "singlecol":
        w = 3.5
        h = max(2.6, 0.22 * n_channels)
        return (w, h)
    return (7.2, max(3.2, 0.20 * n_channels))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def _zero_pad(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[x == PADDING_VALUE] = 0.0
    return x

class MultiScaleGRFNet(nn.Module):
    def __init__(self, input_dim=12, num_filters=64, num_blocks=4,
                 dropout=0.15, use_global_context=True,
                 use_dilations=True, use_residual=True):
        super().__init__()
        self.use_global_context = use_global_context
        self.use_dilations      = use_dilations
        self.use_residual       = use_residual

        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, 1),
            nn.BatchNorm1d(num_filters)
        )

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            d = (2 ** i) if use_dilations else 1
            self.blocks.append(nn.Sequential(
                nn.Conv1d(num_filters, num_filters, 3, padding=d, dilation=d),
                nn.BatchNorm1d(num_filters), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(num_filters, num_filters, 3, padding=d, dilation=d),
                nn.BatchNorm1d(num_filters),
            ))

        if use_global_context:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            head_in = num_filters * 2
        else:
            self.global_pool = None
            head_in = num_filters

        self.output_head = nn.Sequential(
            nn.Linear(head_in, num_filters), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(num_filters, 1)
        )

    def forward(self, x):
        # x: (B,T,C)
        h = self.proj(_zero_pad(x).transpose(1, 2))  # (B,F,T)
        for blk in self.blocks:
            out = blk(h)
            h = F.relu(out + h) if self.use_residual else F.relu(out)

        h_t = h.transpose(1, 2)  # (B,T,F)

        if self.use_global_context:
            g = self.global_pool(h).squeeze(-1).unsqueeze(1).expand(-1, h.size(2), -1)
            h_t = torch.cat([h_t, g], dim=-1)

        return self.output_head(h_t).squeeze(-1)  # (B,T)

class ScalarWrapper(nn.Module):
    """Reduces (B,T) → (B,1) via masked mean for SHAP compatibility."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        valid   = ~(x == PADDING_VALUE).all(dim=-1)   # (B,T)
        valid_f = valid.float()
        y = self.base(x)                              # (B,T)
        if y.dim() == 3:
            y = y.squeeze(-1)
        y = y * valid_f
        denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return y.sum(dim=1, keepdim=True) / denom     # (B,1)

def load_ckpt(path, device):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        return ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADING + STANCE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def _infer_activity(stem: str) -> str:
    s = stem.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", s) if t]

    def has_exact(*opts):
        return any(t in opts for t in tokens)

    def has_prefix(*opts):
        return any(any(t.startswith(o) for o in opts) for t in tokens)

    if has_prefix("heel"):
        return "heel_drop"

    # step-down patterns (safer than plain drop)
    if has_exact("stepdown", "step-down", "step") or has_prefix("stepdown", "step"):
        return "step_down"
    if has_prefix("dropdown", "stepdrop") or ("drop" in tokens and "step" in tokens):
        return "step_down"
    if has_prefix("drop") and ("heel" not in tokens):
        return "step_down"

    if has_prefix("walk"):
        return "walking"
    if has_prefix("jog"):
        return "jogging"
    if has_prefix("run"):
        return "running"
    return "unknown"

def _extract_pid(stem: str) -> str:
    m = re.search(r"(?:^|_)(P)(\d+)(?:_|$)", stem.upper())
    if m:
        return f"P{int(m.group(2)):02d}"
    for p in stem.split("_"):
        pu = p.upper()
        if pu.startswith("P") and pu[1:].isdigit():
            return f"P{int(pu[1:]):02d}"
    return "P00"

def _try_get_mass_kg(df: pd.DataFrame):
    for c in MASS_COL_CANDS:
        if c in df.columns:
            v = df[c].dropna().values
            if len(v):
                try:
                    return float(v[0])
                except Exception:
                    pass
    return None

def _extract_stance(X, y, thr, min_dur, min_peak, fs):
    mask = y > thr
    if not mask.any():
        return None, None

    idx = np.where(mask)[0]
    gaps = np.diff(idx)

    # pick longest contiguous segment (allow small gaps)
    if (gaps > 5).any():
        segs, s = [], 0
        for gi in np.where(gaps > 5)[0]:
            seg = idx[s:gi+1]
            if len(seg):
                segs.append(seg)
            s = gi + 1
        if s < len(idx):
            segs.append(idx[s:])
        if not segs:
            return None, None
        seg = max(segs, key=len)
        onset, offset = seg[0], seg[-1]
    else:
        onset, offset = idx[0], idx[-1]

    if (offset - onset + 1) < int(min_dur * fs):
        return None, None
    if np.max(y[onset:offset+1]) < min_peak:
        return None, None

    return X[onset:offset+1], y[onset:offset+1]

def load_csv_trials(data_dir, thr=80.0, fs=100.0):
    schemas = [
        ["Waist_AccX","Waist_AccY","Waist_AccZ","Waist_GyroX","Waist_GyroY","Waist_GyroZ",
         "Wrist_AccX","Wrist_AccY","Wrist_AccZ","Wrist_GyroX","Wrist_GyroY","Wrist_GyroZ"],
        ["waist_accX","waist_accY","waist_accZ","waist_gyroX","waist_gyroY","waist_gyroZ",
         "wrist_accX","wrist_accY","wrist_accZ","wrist_gyroX","wrist_gyroY","wrist_gyroZ"],
    ]

    X_all, y_all, pids, acts, masses = [], [], [], [], []
    imu_cols = None
    fcol = None

    for fp in tqdm(sorted(Path(data_dir).glob("*.csv")), desc="CSV"):
        if "alignment_log" in fp.name.lower():
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        if fcol is None:
            for c in FORCE_COL_CANDS:
                if c in df.columns:
                    fcol = c
                    break
        if fcol is None or fcol not in df.columns:
            continue

        if imu_cols is None:
            for s in schemas:
                if all(c in df.columns for c in s):
                    imu_cols = s
                    break
        if imu_cols is None or not all(c in df.columns for c in imu_cols):
            continue

        act = _infer_activity(fp.stem)
        pid = _extract_pid(fp.stem)
        mkg = _try_get_mass_kg(df)

        peak_thr, dur_thr = (250.0, 0.08) if act == "heel_drop" else (800.0, 0.15)

        Xs, ys = _extract_stance(
            df[imu_cols].values.astype(np.float32),
            df[fcol].values.astype(np.float32),
            thr, dur_thr, peak_thr, fs
        )
        if Xs is None:
            continue

        X_all.append(Xs)
        y_all.append(ys)
        pids.append(pid)
        acts.append(act)
        masses.append(mkg)

    print(f"[INFO] Loaded {len(X_all)} trials | {dict(Counter(acts))}")
    return X_all, y_all, pids, acts, masses


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _normalise_tc(arr_tc: np.ndarray, T_norm: int) -> np.ndarray:
    """Linear interpolation: (T_i,C)->(T_norm,C)."""
    arr_tc = np.asarray(arr_tc)
    T_i = arr_tc.shape[0]
    C = arr_tc.shape[1]
    if T_i == T_norm:
        return arr_tc.astype(np.float32)

    src = np.linspace(0, 1, T_i)
    dst = np.linspace(0, 1, T_norm)
    out = np.zeros((T_norm, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(dst, src, arr_tc[:, c])
    return out

def _corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def _zscore(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    mu = np.asarray(mu).reshape(1, -1)
    sd = np.asarray(sd).reshape(1, -1)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd

def _rank_vector(importances: np.ndarray) -> np.ndarray:
    order = np.argsort(-importances)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(importances) + 1)
    return ranks

def _compute_fold_imu_stats(X_all, tr_idx):
    vals = np.concatenate([X_all[i] for i in tr_idx], axis=0)  # (sumT, C)
    mu = np.nanmean(vals, axis=0)
    sd = np.nanstd(vals, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)

def _global_energy_scale(samples_by_act: dict, lo=2.0, hi=98.0) -> tuple:
    all_vals = np.concatenate([
        s["xai_energy"] for act_s in samples_by_act.values() for s in act_s
    ]).astype(np.float32)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(all_vals, lo))
    vmax = float(np.percentile(all_vals, hi))
    if vmax - vmin < 1e-9:
        vmin, vmax = 0.0, float(all_vals.max() + 1e-6)
    return vmin, vmax


# ─────────────────────────────────────────────────────────────────────────────
# EVENT-BASED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _pct_from_idx(i: int, T: int) -> float:
    if T <= 1:
        return 0.0
    return 100.0 * float(i) / float(T - 1)

def _peak_idx(x: np.ndarray, mode="abs") -> int:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or (not np.isfinite(x).any()):
        return 0
    if mode == "abs":
        return int(np.nanargmax(np.abs(x)))
    if mode in ("value", "pos"):
        return int(np.nanargmax(x))
    raise ValueError(f"Unknown mode: {mode}")

def _output_landmark_idx(y: np.ndarray, which: str):
    y = np.asarray(y).astype(float)
    if y.size == 0:
        return 0
    if which == "peak_vgrf":
        return int(np.nanargmax(y))
    if which == "peak_rfd":
        dy = np.gradient(y)
        return int(np.nanargmax(dy))
    return int(np.nanargmax(y))

def _event_value_at_landmark(y: np.ndarray, which: str):
    y = np.asarray(y).astype(float)
    if y.size == 0:
        return 0, np.nan
    idx = _output_landmark_idx(y, which)
    return idx, float(y[idx])

@torch.no_grad()
def _perturb_window_effect(model, X_in_tc, baseline_c, c_idx, center_idx, win, device, landmark: str):
    X = torch.tensor(X_in_tc, dtype=torch.float32).unsqueeze(0).to(device)  # (1,T,C)
    y0 = model(X).squeeze(0).detach().cpu().numpy()

    Xp = X.clone()
    a = max(0, int(center_idx) - int(win))
    b = min(X_in_tc.shape[0], int(center_idx) + int(win) + 1)
    Xp[0, a:b, int(c_idx)] = float(baseline_c)
    y1 = model(Xp).squeeze(0).detach().cpu().numpy()

    rmse = float(np.sqrt(np.mean((y0 - y1) ** 2))) if y0.size else np.nan
    peak0 = float(np.nanmax(y0)) if y0.size else np.nan
    peak1 = float(np.nanmax(y1)) if y1.size else np.nan
    dpeak = peak0 - peak1

    idx0, _ = _event_value_at_landmark(y0, landmark)
    devent = float(y0[idx0] - y1[idx0]) if (y0.size and y1.size) else np.nan
    return rmse, dpeak, devent


# ─────────────────────────────────────────────────────────────────────────────
# TIME-SHAP STYLE EVENT-WISE KERNELSHAP (stance events as "features")
# ─────────────────────────────────────────────────────────────────────────────
def _event_blocks(T_norm=100, E=20):
    edges = np.linspace(0, T_norm, E + 1).round().astype(int)
    blocks = [(int(edges[i]), int(edges[i + 1])) for i in range(E)]
    blocks = [(a, b if b > a else min(a + 1, T_norm)) for a, b in blocks]
    return blocks

def _kernel_weight(m, E):
    if m <= 0 or m >= E:
        return 1e6
    denom = comb(E, m) * m * (E - m)
    if denom <= 0:
        return 1e6
    return float((E - 1) / denom)

def _build_baseline_tc(bv_c, T_norm=100, baseline_mode="const", baseline_tc=None):
    if baseline_mode == "tc":
        if baseline_tc is None:
            raise ValueError("baseline_mode='tc' requires baseline_tc")
        return np.asarray(baseline_tc, dtype=np.float32)
    bv = np.asarray(bv_c, dtype=np.float32).reshape(1, -1)
    return np.repeat(bv, repeats=T_norm, axis=0)

@torch.no_grad()
def _predict_from_norm_tc(model, X_norm_tc, device):
    Xb = torch.tensor(X_norm_tc, dtype=torch.float32).unsqueeze(0).to(device)
    y = model(Xb).squeeze(0).detach().cpu().numpy()
    return np.asarray(y, dtype=np.float32)

def _target_scalar(y_pred_norm, target="peak_vgrf", phase=None, t_idx=None):
    y = np.asarray(y_pred_norm, dtype=float)
    if y.size == 0:
        return np.nan
    if target == "peak_vgrf":
        return float(np.nanmax(y))
    if target == "phase_mean":
        if phase is None:
            raise ValueError("phase_mean requires phase=(a,b) indices")
        a, b = phase
        a = int(np.clip(a, 0, len(y)))
        b = int(np.clip(b, 0, len(y)))
        if b <= a:
            return float(np.nanmean(y))
        return float(np.nanmean(y[a:b]))
    if target == "value_at":
        if t_idx is None:
            raise ValueError("value_at requires t_idx")
        i = int(np.clip(int(t_idx), 0, len(y) - 1))
        return float(y[i])
    raise ValueError(f"Unknown target: {target}")

def _sample_coalitions(E, K, rng):
    Z = [np.zeros(E, dtype=int), np.ones(E, dtype=int)]
    for _ in range(max(0, K - 2)):
        m = int(rng.integers(1, E))
        idx = rng.choice(E, size=m, replace=False)
        z = np.zeros(E, dtype=int)
        z[idx] = 1
        Z.append(z)
    return np.stack(Z, axis=0)

def _weighted_linear_solve(Z, y, w):
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)

    K, E = Z.shape
    Xmat = np.concatenate([np.ones((K, 1)), Z], axis=1)

    sw = np.sqrt(np.clip(w, 1e-12, np.inf))
    Xw = Xmat * sw[:, None]
    yw = y * sw

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    phi0 = float(beta[0])
    phi = beta[1:].astype(np.float32)
    return phi, phi0

@torch.no_grad()
def timeshap_eventwise_kernelshap(
    model,
    X_in_tc,               # (T_i,C) input in SAME SPACE as model input
    bv_c,                  # (C,) baseline values in SAME SPACE
    device,
    T_norm=100,
    E=20,
    K=300,
    seed=0,
    baseline_mode="const",
    baseline_tc=None,
    target="peak_vgrf",
    phase_name=None,       # used only if target=="phase_mean"
):
    rng = np.random.default_rng(seed)

    X_norm_tc = _normalise_tc(X_in_tc, T_norm)
    baseline_norm_tc = _build_baseline_tc(bv_c, T_norm, baseline_mode, baseline_tc)

    blocks = _event_blocks(T_norm=T_norm, E=E)
    Z = _sample_coalitions(E=E, K=K, rng=rng)
    m = Z.sum(axis=1).astype(int)
    w = np.array([_kernel_weight(mi, E) for mi in m], dtype=float)

    if target == "phase_mean":
        if phase_name is None:
            raise ValueError("phase_mean target requires phase_name")
        t0, t1 = PHASES[phase_name]
        a = int(t0 * T_norm)
        b = int(t1 * T_norm)
        phase_idx = (a, b)
    else:
        phase_idx = None

    C = X_norm_tc.shape[1]
    phi_events = np.zeros((E, C), dtype=np.float32)
    phi0 = np.zeros((C,), dtype=np.float32)

    # per-channel KernelSHAP on events (mask only channel c inside stance bins)
    for c in range(C):
        y_vals = []
        for z in Z:
            Xp = X_norm_tc.copy()
            for e, (a, b) in enumerate(blocks):
                if int(z[e]) == 0:
                    Xp[a:b, c] = baseline_norm_tc[a:b, c]
            y_pred_norm = _predict_from_norm_tc(model, Xp, device=device)
            if target == "peak_vgrf":
                y_vals.append(_target_scalar(y_pred_norm, target="peak_vgrf"))
            elif target == "phase_mean":
                y_vals.append(_target_scalar(y_pred_norm, target="phase_mean", phase=phase_idx))
            elif target == "value_at":
                t_idx = int(np.nanargmax(y_pred_norm)) if y_pred_norm.size else 0
                y_vals.append(_target_scalar(y_pred_norm, target="value_at", t_idx=t_idx))
            else:
                raise ValueError(target)

        y_vals = np.asarray(y_vals, dtype=float)
        phic, phi0c = _weighted_linear_solve(Z, y_vals, w)
        phi_events[:, c] = phic
        phi0[c] = float(phi0c)

    meta = {
        "T_norm": int(T_norm),
        "E": int(E),
        "K": int(K),
        "target": str(target),
        "phase_name": str(phase_name) if phase_name is not None else "",
    }
    return phi_events, phi0, meta


# ─────────────────────────────────────────────────────────────────────────────
# SHAP (padding-safe + fold-safe z-score) + plots
# ─────────────────────────────────────────────────────────────────────────────
def _shap_normalise_shape(sv):
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv
    if sv.ndim == 4:
        if sv.shape[-1] == 1:
            return sv.squeeze(-1)
        if sv.shape[1] == 1:
            return sv.squeeze(1)
    raise RuntimeError(f"Unexpected SHAP shape: {sv.shape}")

def _valid_time_mask_from_X(X_btc: torch.Tensor) -> torch.Tensor:
    return ~(X_btc == PADDING_VALUE).all(dim=-1)

def _masked_mean_std(X_btc: torch.Tensor, valid_bt: torch.Tensor):
    X = X_btc.cpu().numpy()
    V = valid_bt.cpu().numpy()
    Xv = X[V]  # (n_valid, C)
    mu = np.nanmean(Xv, axis=0).astype(np.float32)
    sd = np.nanstd(Xv, axis=0).astype(np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd

def _apply_zscore_with_padding(X_btc: torch.Tensor, mu: np.ndarray, sd: np.ndarray) -> torch.Tensor:
    Xn = X_btc.clone()
    valid = _valid_time_mask_from_X(Xn)  # (B,T)

    mu_t = torch.tensor(mu, dtype=Xn.dtype, device=Xn.device).view(1, 1, -1)
    sd_t = torch.tensor(sd, dtype=Xn.dtype, device=Xn.device).view(1, 1, -1)

    Xn[valid] = (Xn[valid] - mu_t.expand_as(Xn)[valid]) / sd_t.expand_as(Xn)[valid]
    return Xn

def _compute_shap(model, bg, sm, device):
    try:
        import shap as shap_lib
    except ImportError as e:
        raise ImportError("pip install shap") from e

    model.eval()
    wrapped = ScalarWrapper(model).to(device).eval()

    bg = bg.to(device)
    sm = sm.to(device)

    _shap_warn_filter = ("ignore", "The NumPy global RNG was seeded", FutureWarning)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(*_shap_warn_filter)
            exp = shap_lib.DeepExplainer(wrapped, bg)
            sv = exp.shap_values(sm, check_additivity=False)
    except Exception as e:
        print(f"  DeepExplainer → GradientExplainer ({e})")
        with warnings.catch_warnings():
            warnings.filterwarnings(*_shap_warn_filter)
            exp = shap_lib.GradientExplainer(wrapped, bg)
            sv = exp.shap_values(sm)

    if isinstance(sv, list):
        sv = sv[0]
    return _shap_normalise_shape(np.asarray(sv))  # (B,T,C)

def temporal_shap_loso(
    npz_path, ckpt_glob, outdir, device="cpu",
    bg_n=100, sample_n=50, T_norm=100, seed=42,
    imu_zscore="fold",
    fig_style="doublecol",
    heat_gamma=0.55,
    heat_vmax_pct=99.0,
    heat_vmin_pct=0.0,
    vis_norm="global",
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    d = np.load(npz_path, allow_pickle=True)
    X = torch.from_numpy(d["X"]).float()        # (N,T,C) with padding
    pids = d["pids"].tolist()
    feat = d["feature_names"].tolist() if "feature_names" in d else CHANNEL_LABELS

    valid_mask = _valid_time_mask_from_X(X)

    ckpts = sorted(glob.glob(ckpt_glob, recursive=True))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints: {ckpt_glob}")

    pid_to_ckpt = {}
    for c in ckpts:
        m = re.search(r"test[_\\-]?(P\\d{2})", c, re.IGNORECASE)
        if m:
            pid_to_ckpt[m.group(1).upper()] = c

    uniq = sorted(set([str(p).upper() for p in pids]))
    folds = [
        (p,
         np.array([i for i, q in enumerate(pids) if str(q).upper() != p]),
         np.array([i for i, q in enumerate(pids) if str(q).upper() == p]))
        for p in uniq
    ]

    fold_temporal, fold_ranks = [], []

    for fi, (test_pid, tr_idx, te_idx) in enumerate(folds, 1):
        ckpt_path = pid_to_ckpt.get(test_pid, ckpts[min(fi-1, len(ckpts)-1)])
        print(f"\nFold {fi:02d} | TEST={test_pid} | CKPT={Path(ckpt_path).name}")

        model = MultiScaleGRFNet(input_dim=X.shape[-1])
        model.load_state_dict(load_ckpt(ckpt_path, "cpu"))
        model = model.to(device).eval()

        rng = np.random.default_rng(seed + fi)
        bg_idx = rng.choice(len(tr_idx), size=min(bg_n, len(tr_idx)), replace=False)
        sm_idx = rng.choice(len(te_idx), size=min(sample_n, len(te_idx)), replace=False)

        bg = X[tr_idx[bg_idx]].clone()
        sm = X[te_idx[sm_idx]].clone()

        if imu_zscore == "fold":
            mu, sd = _masked_mean_std(X[tr_idx], valid_mask[tr_idx])
            bg = _apply_zscore_with_padding(bg, mu, sd)
            sm = _apply_zscore_with_padding(sm, mu, sd)
        elif imu_zscore == "global":
            mu, sd = _masked_mean_std(X, valid_mask)
            bg = _apply_zscore_with_padding(bg, mu, sd)
            sm = _apply_zscore_with_padding(sm, mu, sd)

        shap_btc = _compute_shap(model, bg, sm, device)
        abs_btc = np.abs(shap_btc)

        B, _, C = abs_btc.shape
        fold_tc_list = []

        sm_valid = valid_mask[te_idx[sm_idx]].cpu().numpy()

        for b in range(B):
            valid_len = int(sm_valid[b].sum())
            if valid_len < 2:
                continue
            tc_raw = abs_btc[b, :valid_len, :]
            tc_norm = _normalise_tc(tc_raw, T_norm)
            fold_tc_list.append(tc_norm)

        fold_mean_tc = (
            np.stack(fold_tc_list).mean(axis=0)
            if fold_tc_list else np.zeros((T_norm, C), np.float32)
        )
        fold_temporal.append(fold_mean_tc)

        imp_vals = abs_btc[sm_valid]
        imp_c = imp_vals.mean(axis=0) if imp_vals.size else np.zeros(C)
        fold_ranks.append(_rank_vector(imp_c))

        _plot_single_heatmap(
            fold_mean_tc,
            title=f"Temporal SHAP | Fold {fi:02d} | TEST={test_pid}",
            out_path=Path(outdir) / f"temporal_shap_fold_{fi:02d}_{test_pid}.png",
            T_norm=T_norm,
            fig_style=fig_style
        )

    agg_mean = np.stack(fold_temporal).mean(axis=0)
    agg_std  = np.stack(fold_temporal).std(axis=0)

    _plot_temporal_heatmap_full(
        imp_mean=agg_mean,
        imp_std=agg_std,
        title="Temporal SHAP — Aggregate (all LOSO folds)",
        out_path=Path(outdir) / "temporal_shap_aggregate.png",
        T_norm=T_norm,
        overlay_topk=4,
        overlay_band=True,
        fig_style=fig_style,
        heat_gamma=heat_gamma,
        heat_vmax_pct=heat_vmax_pct,
        heat_vmin_pct=heat_vmin_pct,
        vis_norm=vis_norm,
    )

    _save_phase_csv(agg_mean, T_norm, Path(outdir) / "shap_phase_summary.csv")
    _save_tc_csv(agg_mean, T_norm, Path(outdir) / "shap_temporal_agg.csv")
    _save_stability(fold_ranks, feat, Path(outdir) / "shap_stability_summary.txt")

    print(f"\n✅ Temporal SHAP done → {outdir}")


# ─────────────────────────────────────────────────────────────────────────────
# SMILE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _temporal_smile_one(model, X_in, baseline_vals, window_size, stride, device):
    """
    Temporal SMILE for ONE sample (T_i, C) in SAME SPACE as model input.
    Returns (T_i, C) importance map.
    """
    T_i, C = X_in.shape
    X_b = torch.tensor(X_in, dtype=torch.float32).unsqueeze(0).to(device)  # (1,T_i,C)
    bv = baseline_vals.to(device)

    pred_orig = model(X_b)  # (1, T_i)

    importance_tc = np.zeros((T_i, C), dtype=np.float32)
    counts = np.zeros(T_i, dtype=np.float32)

    n_wins = max(1, int(np.floor((T_i - window_size) / stride)) + 1)
    for w in range(n_wins):
        t0 = w * stride
        t1 = min(t0 + window_size, T_i)
        counts[t0:t1] += 1.0

        for c in range(C):
            Xp = X_b.clone()
            Xp[0, t0:t1, c] = bv[c].item()
            pred_p = model(Xp)
            diff = (pred_orig - pred_p).squeeze(0)  # (T_i,)
            impact = diff[t0:t1].abs().mean().item()
            importance_tc[t0:t1, c] += impact

    importance_tc /= np.maximum(counts[:, None], 1.0)
    return importance_tc

def _overlay_scale_pred(y_true_N, y_pred_model, mode, bw_kg=None, mass_kg=None):
    """Scale prediction ONLY for plotting, to compare with y_true in Newtons."""
    if mode == "none":
        return y_pred_model
    if mode == "auto":
        pt = float(np.nanmax(y_true_N))
        pp = float(np.nanmax(y_pred_model))
        if pp < 1e-6:
            return y_pred_model
        return y_pred_model * (pt / pp)
    if mode == "bw":
        m = mass_kg if (mass_kg is not None and np.isfinite(mass_kg)) else bw_kg
        if m is None or not np.isfinite(m):
            return _overlay_scale_pred(y_true_N, y_pred_model, "auto", bw_kg=bw_kg, mass_kg=mass_kg)
        BW = float(m) * 9.81
        return y_pred_model * BW
    return y_pred_model


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SUMMARIES
# ─────────────────────────────────────────────────────────────────────────────
def _save_stability(fold_ranks, feat_names, out_path):
    fold_ranks = np.array(fold_ranks)  # (F,C)
    F_, C = fold_ranks.shape

    rhos, taus = [], []
    for i in range(F_):
        for j in range(i + 1, F_):
            rhos.append(spearmanr(fold_ranks[i], fold_ranks[j]).correlation)
            taus.append(kendalltau(fold_ranks[i], fold_ranks[j]).correlation)

    top1 = np.bincount(np.argmin(fold_ranks, axis=1), minlength=C) / F_
    top3_freq = np.zeros(C)
    for f in range(F_):
        for c in np.argsort(fold_ranks[f])[:3]:
            top3_freq[c] += 1
    top3_freq /= F_

    with open(out_path, "w") as f:
        f.write(f"Folds: {F_}\nFold-pairs: {len(rhos)}\n")
        f.write(f"Median Spearman rho: {np.nanmedian(rhos):.4f}\n")
        f.write(f"Mean Kendall tau:    {np.nanmean(taus):.4f}\n\n")
        f.write("Channel | MeanRank | Top1% | Top3%\n")
        mean_ranks = fold_ranks.mean(axis=0)
        order = np.argsort(mean_ranks)
        for c in order:
            f.write(f"{feat_names[c]:<20} {mean_ranks[c]:5.1f}   "
                    f"{top1[c]*100:5.1f}%   {top3_freq[c]*100:5.1f}%\n")
    print(f"[OK] {Path(out_path).name}")

def _save_phase_csv(imp_mean: np.ndarray, T_norm: int, out_path: Path):
    rows = []
    for c, name in enumerate(CHANNEL_LABELS):
        row = {"channel": name}
        for phase, (t0, t1) in PHASES.items():
            t0i, t1i = int(t0 * T_norm), int(t1 * T_norm)
            row[phase] = float(imp_mean[t0i:t1i, c].mean())
        row["overall"] = float(imp_mean[:, c].mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] {out_path.name}")

def _save_tc_csv(imp_mean: np.ndarray, T_norm: int, out_path: Path):
    pct = np.linspace(0, 100, T_norm)
    df = pd.DataFrame(imp_mean, columns=CHANNEL_LABELS)
    df.insert(0, "stance_pct", pct)
    df.to_csv(out_path, index=False)
    print(f"[OK] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING (heatmaps)
# ─────────────────────────────────────────────────────────────────────────────
def _add_phases(ax, T_norm, y_text=None, fontsize=8):
    for phase, (t0, t1) in PHASES.items():
        ax.axvline(x=t0 * (T_norm - 1), color="gray", lw=0.9, ls="--", alpha=0.6)
        if y_text is not None:
            ax.text((t0 + t1) / 2 * (T_norm - 1), y_text, phase,
                    ha="center", fontsize=fontsize,
                    color="dimgray", style="italic",
                    bbox=dict(facecolor="white", alpha=0.35, pad=1, edgecolor="none"))

def _plot_single_heatmap(imp_tc, title, out_path, T_norm=100, sigma=1.2, fig_style="doublecol"):
    C = imp_tc.shape[1]
    smooth = np.stack([gaussian_filter1d(imp_tc[:, c], sigma) for c in range(C)], axis=1)

    fig_w, fig_h = _figsize_heatmap(fig_style, n_channels=C)
    fig, ax = plt.subplots(figsize=(fig_w, max(fig_h, 3.0)), constrained_layout=True)

    vmax = float(smooth.max()) if float(smooth.max()) > 0 else 1e-6
    im = ax.imshow(smooth.T, aspect="auto", cmap="YlOrRd", origin="lower",
                   vmin=0, vmax=vmax, interpolation="bilinear")

    _add_phases(ax, T_norm, y_text=C - 0.6, fontsize=8 if fig_style != "singlecol" else 7)

    ax.set_yticks(range(C))
    ax.set_yticklabels(CHANNEL_LABELS, fontsize=8 if fig_style != "singlecol" else 7)
    for tl, col in zip(ax.get_yticklabels(), SENSOR_COLORS):
        tl.set_color(col)

    xticks = np.arange(0, T_norm + 1, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{p}%" for p in xticks], fontsize=8 if fig_style != "singlecol" else 7)
    ax.set_xlabel("Stance phase (%)")
    ax.set_title(title, fontsize=10, fontweight="bold")

    if fig_style == "singlecol":
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.12, pad=0.12)
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cbar.set_label("Attribution magnitude")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _plot_temporal_heatmap_full(
    imp_mean, imp_std, title, out_path, T_norm=100, sigma=1.5,
    overlay_topk=4, overlay_band=True,
    fig_style="doublecol",
    heat_gamma=0.55,
    heat_vmax_pct=99.0,
    heat_vmin_pct=0.0,
    vis_norm="global",
):
    C = imp_mean.shape[1]

    smooth_mean = np.stack([gaussian_filter1d(imp_mean[:, c], sigma) for c in range(C)], axis=1)
    smooth_std  = np.stack([gaussian_filter1d(imp_std[:, c],  sigma) for c in range(C)], axis=1)

    fig, ax = plt.subplots(figsize=_figsize_heatmap(fig_style, n_channels=C), constrained_layout=True)

    vals = smooth_mean[np.isfinite(smooth_mean)]
    if vals.size == 0:
        vmin, vmax = 0.0, 1e-6
    else:
        vmin = float(np.percentile(vals, heat_vmin_pct))
        vmax = float(np.percentile(vals, heat_vmax_pct))
        if vmax - vmin < 1e-9:
            vmin, vmax = 0.0, float(vals.max() + 1e-6)

    disp = smooth_mean.copy()
    if vis_norm == "per_phase":
        for _, (t0, t1) in PHASES.items():
            a = int(t0 * T_norm); b = int(t1 * T_norm)
            seg = disp[a:b, :]
            segv = seg[np.isfinite(seg)]
            if segv.size > 0:
                seg_max = float(np.percentile(segv, heat_vmax_pct))
                if seg_max > 1e-9:
                    disp[a:b, :] = seg / seg_max
        vmin, vmax = 0.0, 1.0

    norm = mcolors.PowerNorm(gamma=float(heat_gamma), vmin=vmin, vmax=vmax)

    im = ax.imshow(
        disp.T,
        aspect="auto",
        cmap="YlOrRd",
        origin="lower",
        norm=norm,
        interpolation="bilinear"
    )

    _add_phases(ax, T_norm, y_text=C - 0.4, fontsize=9 if fig_style != "singlecol" else 7.5)

    ax.set_yticks(range(C))
    ax.set_yticklabels(CHANNEL_LABELS, fontsize=10 if fig_style != "singlecol" else 7)
    for tl, col in zip(ax.get_yticklabels(), SENSOR_COLORS):
        tl.set_color(col)

    xticks = np.linspace(0, T_norm - 1, 11)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(p)}%" for p in np.linspace(0, 100, 11)],
                       fontsize=8 if fig_style != "singlecol" else 7)

    ax.set_xlabel("Stance phase (%)", fontsize=12 if fig_style != "singlecol" else 9)
    ax.set_title(title, fontsize=13 if fig_style != "singlecol" else 10, fontweight="bold", pad=10)

    if fig_style == "singlecol":
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.12, pad=0.12)
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cbar.set_label("Attribution magnitude", fontsize=10 if fig_style != "singlecol" else 8)

    phase_patches = [Patch(facecolor=PHASE_COLORS[p], alpha=0.7, label=p) for p in PHASES]
    leg_phase = ax.legend(
        handles=phase_patches,
        loc="upper right",
        fontsize=9 if fig_style != "singlecol" else 7,
        title="Phase",
        title_fontsize=9 if fig_style != "singlecol" else 7
    )

    # Overlay top-k curves
    topk = np.argsort(imp_mean.mean(axis=0))[::-1][:max(1, int(overlay_topk))]
    x = np.linspace(0, T_norm - 1, T_norm)
    color_cycle = plt.cm.tab10.colors

    curve_labels, curve_colors = [], []
    for i, c in enumerate(topk):
        mu = gaussian_filter1d(imp_mean[:, c], sigma)
        sd = gaussian_filter1d(imp_std[:, c], sigma)

        denom = float(mu.max()) if float(mu.max()) > 1e-9 else 1e-9
        y_mu = (mu / denom) * (C - 1)

        col = color_cycle[i % len(color_cycle)]
        curve_labels.append(CHANNEL_LABELS[c])
        curve_colors.append(col)

        ax.plot(x, y_mu, lw=3.0 if fig_style != "singlecol" else 2.4,
                color="black", alpha=0.28, zorder=19)
        ax.plot(x, y_mu, lw=2.1 if fig_style != "singlecol" else 1.8,
                color=col, alpha=0.95, zorder=20)

        if overlay_band:
            y_lo = np.clip(((mu - sd) / denom) * (C - 1), 0, C - 1)
            y_hi = np.clip(((mu + sd) / denom) * (C - 1), 0, C - 1)
            ax.fill_between(x, y_lo, y_hi, color=col, alpha=0.10, zorder=18)

    ax.add_artist(leg_phase)

    if len(curve_labels) > 0:
        topk_handles = [Patch(facecolor=curve_colors[i], label=curve_labels[i])
                        for i in range(len(curve_labels))]
        if fig_style == "doublecol":
            ax.legend(
                handles=topk_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                framealpha=0.95,
                title=f"Top-{len(topk_handles)}",
                fontsize=9,
                title_fontsize=9,
            )
        else:
            ax.legend(
                handles=topk_handles,
                loc="upper left",
                framealpha=0.9,
                title=f"Top-{len(topk_handles)}",
                fontsize=7,
                title_fontsize=7,
            )
            ax.add_artist(leg_phase)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# PER-ACTIVITY + PHASE SUMMARY (restored from older version)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_activity_grid(fold_temporal_by_act: dict, T_norm: int, out_path: Path,
                        fig_style="doublecol", heat_gamma=0.55, heat_vmax_pct=99.0):
    """
    fold_temporal_by_act: act -> list of (T_norm,C) arrays from all folds/samples
    Produces a grid with one heatmap per activity (mean over provided arrays).
    """
    acts = [a for a in ACTIVITY_ORDER if a in fold_temporal_by_act and len(fold_temporal_by_act[a]) > 0]
    if not acts:
        print("[WARN] _plot_activity_grid: no data, skipping.")
        return

    C = 12
    n = len(acts)
    ncols = 2 if (fig_style == "doublecol" and n > 1) else 1
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(14, 4.2 * nrows)) if fig_style == "doublecol" else plt.figure(figsize=(7.2, 4.0 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.22, hspace=0.28)

    # global vmax
    mats = []
    for a in acts:
        mat = np.stack(fold_temporal_by_act[a]).mean(axis=0)  # (T,C)
        mats.append(mat)
    allv = np.concatenate([m.ravel() for m in mats])
    vmax = float(np.nanpercentile(allv[np.isfinite(allv)], heat_vmax_pct)) if np.isfinite(allv).any() else 1e-6
    vmax = max(vmax, 1e-6)
    norm = mcolors.PowerNorm(gamma=float(heat_gamma), vmin=0.0, vmax=vmax)

    last_im = None
    for i, act in enumerate(acts):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])

        mean_tc = np.stack(fold_temporal_by_act[act]).mean(axis=0)  # (T,C)
        smooth = np.stack([gaussian_filter1d(mean_tc[:, k], 1.2) for k in range(C)], axis=1)

        last_im = ax.imshow(smooth.T, aspect="auto", cmap="YlOrRd", origin="lower", norm=norm, interpolation="bilinear")
        _add_phases(ax, T_norm, y_text=C - 0.4, fontsize=8)
        ax.set_title(act.replace("_", "-").title(), fontweight="bold")

        ax.set_yticks(range(C))
        ax.set_yticklabels(CHANNEL_LABELS, fontsize=8)
        for tl, col in zip(ax.get_yticklabels(), SENSOR_COLORS):
            tl.set_color(col)

        xticks = np.linspace(0, T_norm - 1, 11)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(p)}%" for p in np.linspace(0, 100, 11)], fontsize=8)
        ax.set_xlabel("Stance phase (%)")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=fig.axes, fraction=0.02, pad=0.01)
        cbar.set_label("Attribution magnitude")

    fig.suptitle("Temporal SMILE — By Activity (aggregate across folds)", fontweight="bold", y=0.995)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")

def _plot_phase_bar(imp_mean: np.ndarray, T_norm: int, out_path: Path):
    """
    Bar plot: per-channel mean attribution per phase (Loading/Mid/Push) + overall.
    """
    C = imp_mean.shape[1]
    phase_vals = {}
    for phase, (t0, t1) in PHASES.items():
        a = int(t0 * T_norm)
        b = int(t1 * T_norm)
        phase_vals[phase] = imp_mean[a:b, :].mean(axis=0)

    overall = imp_mean.mean(axis=0)
    df = pd.DataFrame({
        "channel": CHANNEL_LABELS,
        "Loading": phase_vals["Loading"],
        "Mid-stance": phase_vals["Mid-stance"],
        "Push-off": phase_vals["Push-off"],
        "Overall": overall
    })

    # sort by overall desc
    df = df.sort_values("Overall", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(14, 5.5))
    ax = fig.add_subplot(111)

    x = np.arange(C)
    w = 0.18
    ax.bar(x - 1.5*w, df["Loading"].values, width=w, label="Loading")
    ax.bar(x - 0.5*w, df["Mid-stance"].values, width=w, label="Mid-stance")
    ax.bar(x + 0.5*w, df["Push-off"].values, width=w, label="Push-off")
    ax.bar(x + 1.5*w, df["Overall"].values, width=w, label="Overall")

    ax.set_xticks(x)
    ax.set_xticklabels(df["channel"].values, rotation=45, ha="right")
    ax.set_ylabel("Mean attribution")
    ax.set_title("Temporal SMILE — Phase summary (mean attribution per channel)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=9)

    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW: TimeSHAP plotting consistent with SHAP/SMILE
# ─────────────────────────────────────────────────────────────────────────────
def _upsample_timeshap_repeat(phi_events_EC: np.ndarray, T_norm: int) -> np.ndarray:
    """
    Step-like upsampling: repeat each event bin to fill T_norm timesteps.
    If T_norm not divisible by E, last bin is extended/truncated safely.
    """
    phi = np.asarray(phi_events_EC, dtype=np.float32)
    E, C = phi.shape
    if E <= 0:
        return np.zeros((T_norm, C), dtype=np.float32)

    reps = T_norm // E
    if reps <= 0:
        return _upsample_timeshap_interp(phi, T_norm)

    up = np.repeat(phi, reps, axis=0)  # (E*reps, C)
    if up.shape[0] < T_norm:
        pad = np.repeat(phi[-1:, :], T_norm - up.shape[0], axis=0)
        up = np.vstack([up, pad])
    if up.shape[0] > T_norm:
        up = up[:T_norm, :]
    return up.astype(np.float32)

def _upsample_timeshap_interp(phi_events_EC: np.ndarray, T_norm: int) -> np.ndarray:
    """
    Smooth upsampling: interpolate between event-bin centers.
    """
    phi = np.asarray(phi_events_EC, dtype=np.float32)
    E, C = phi.shape
    if E <= 1:
        return np.repeat(phi[:1, :], T_norm, axis=0).astype(np.float32)

    x_src = np.linspace(0, 1, E)
    x_dst = np.linspace(0, 1, T_norm)
    out = np.zeros((T_norm, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(x_dst, x_src, phi[:, c])
    return out.astype(np.float32)

def plot_timeshap_temporal_heatmap(
    phi_events_EC: np.ndarray,                 # (E,C)
    title: str,
    out_path,
    T_norm=100,
    upsample_method="repeat",                  # "repeat" or "interp"
    fig_style="doublecol",
    heat_gamma=0.55,
    heat_vmax_pct=99.0,
    heat_vmin_pct=0.0,
    vis_norm="global",
    overlay_topk=4,
    overlay_band=False,
    sigma=1.5,
):
    """
    TimeSHAP -> consistent heatmap in (T_norm,C) like SHAP/SMILE.
    Uses mean/std interface of _plot_temporal_heatmap_full by setting std=0.
    """
    phi = np.asarray(phi_events_EC, dtype=np.float32)

    if upsample_method == "repeat":
        tc = _upsample_timeshap_repeat(phi, T_norm=T_norm)
    elif upsample_method == "interp":
        tc = _upsample_timeshap_interp(phi, T_norm=T_norm)
    else:
        raise ValueError("upsample_method must be 'repeat' or 'interp'")

    imp_mean = np.abs(tc).astype(np.float32)
    imp_std  = np.zeros_like(imp_mean, dtype=np.float32)

    _plot_temporal_heatmap_full(
        imp_mean=imp_mean,
        imp_std=imp_std,
        title=title,
        out_path=Path(out_path),
        T_norm=T_norm,
        sigma=sigma,
        overlay_topk=overlay_topk,
        overlay_band=overlay_band,
        fig_style=fig_style,
        heat_gamma=heat_gamma,
        heat_vmax_pct=heat_vmax_pct,
        heat_vmin_pct=heat_vmin_pct,
        vis_norm=vis_norm,
    )

def plot_timeshap_format_comparison(
    phi_events_EC: np.ndarray,
    out_path,
    T_norm=100,
    fig_style="doublecol",
    heat_vmax_pct=99.0,
    heat_gamma=0.55,
):
    """
    Side-by-side: BEFORE (E bins) vs AFTER (T_norm timesteps).
    Left: original eventwise heatmap (E x C)
    Right: upsampled (T_norm x C) with phase markers (simple imshow)
    """
    phi = np.asarray(phi_events_EC, dtype=np.float32)
    E, C = phi.shape
    after = _upsample_timeshap_repeat(phi, T_norm=T_norm)
    before = np.abs(phi)
    after_abs = np.abs(after)

    vmax_b = float(np.nanpercentile(before, heat_vmax_pct)) if np.isfinite(before).any() else 1e-6
    vmax_a = float(np.nanpercentile(after_abs, heat_vmax_pct)) if np.isfinite(after_abs).any() else 1e-6
    vmax = max(vmax_b, vmax_a, 1e-6)

    norm = mcolors.PowerNorm(gamma=float(heat_gamma), vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(14, 5.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.35], wspace=0.18)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(before.T, aspect="auto", cmap="YlOrRd", origin="lower", norm=norm, interpolation="nearest")
    ax0.set_title(f"BEFORE: TimeSHAP event bins (E={E})", fontweight="bold")
    ax0.set_xlabel("Event bin")
    ax0.set_yticks(range(C))
    ax0.set_yticklabels(CHANNEL_LABELS, fontsize=8)
    for tl, col in zip(ax0.get_yticklabels(), SENSOR_COLORS):
        tl.set_color(col)

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(after_abs.T, aspect="auto", cmap="YlOrRd", origin="lower", norm=norm, interpolation="bilinear")
    _add_phases(ax1, T_norm, y_text=C - 0.4, fontsize=8)
    ax1.set_title(f"AFTER: Upsampled to T={T_norm} (1% stance)", fontweight="bold")
    ax1.set_xlabel("Stance phase (%)")
    ax1.set_yticks(range(C))
    ax1.set_yticklabels(CHANNEL_LABELS, fontsize=8)
    for tl, col in zip(ax1.get_yticklabels(), SENSOR_COLORS):
        tl.set_color(col)
    xticks = np.linspace(0, T_norm - 1, 11)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{int(p)}%" for p in np.linspace(0, 100, 11)], fontsize=8)

    cbar = fig.colorbar(im1, ax=[ax0, ax1], fraction=0.03, pad=0.02)
    cbar.set_label("Attribution magnitude (|phi|)")

    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL eventwise heatmap (kept for “before”)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_eventwise_shap_heatmap(mean_abs_EC: np.ndarray, title: str, out_path: Path, fig_style="doublecol"):
    mat = np.asarray(mean_abs_EC, dtype=float)
    E_, C_ = mat.shape
    fig, ax = plt.subplots(figsize=_figsize_heatmap(fig_style, n_channels=C_), constrained_layout=True)

    vmax = float(np.nanpercentile(mat, 99.0)) if np.isfinite(mat).any() else 1e-6
    im = ax.imshow(mat.T, aspect="auto", cmap="YlOrRd", origin="lower",
                   vmin=0.0, vmax=max(vmax, 1e-6), interpolation="bilinear")

    ax.set_yticks(range(C_))
    ax.set_yticklabels(CHANNEL_LABELS, fontsize=10 if fig_style != "singlecol" else 7)
    for tl, col in zip(ax.get_yticklabels(), SENSOR_COLORS):
        tl.set_color(col)

    ax.set_xticks(np.arange(0, E_, max(1, E_ // 10)))
    ax.set_xticklabels([str(int(i)) for i in ax.get_xticks()], fontsize=8 if fig_style != "singlecol" else 7)
    ax.set_xlabel("Event bin (stance partition)", fontsize=11 if fig_style != "singlecol" else 9)
    ax.set_title(title, fontsize=12 if fig_style != "singlecol" else 10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cbar.set_label("mean |phi|", fontsize=10 if fig_style != "singlecol" else 8)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# GRF OVERLAY PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def _phase_shading(ax) -> None:
    for phase, (t0, t1) in PHASES.items():
        ax.axvspan(t0 * 100, t1 * 100, alpha=0.08, color=PHASE_COLORS[phase], zorder=0)

def _phase_labels(ax) -> None:
    ylim = ax.get_ylim()
    span = max(ylim[1] - ylim[0], 1e-3)
    for phase, (t0, t1) in PHASES.items():
        ax.text((t0 + t1) / 2 * 100,
                ylim[0] + 0.95 * span,
                phase,
                ha="center", va="top", fontsize=7.5,
                color="dimgray", style="italic",
                bbox=dict(facecolor="white", alpha=0.30, pad=1.2, edgecolor="none"),
                zorder=10)

def _xtick_pct(ax, step: int = 10) -> None:
    ticks = np.arange(0, 101, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks], fontsize=8)

def _mean_r(samples: list) -> str:
    rs = [s["r"] for s in samples if not np.isnan(s["r"])]
    return f"{np.mean(rs):.3f}" if rs else "N/A"

def _activity_arrays(samples: list, sigma: float):
    y_true_mat = np.stack([s["y_true_norm"] for s in samples])
    y_pred_mat = np.stack([s["y_pred_norm"] for s in samples])
    xai_stack = np.stack([s["xai_energy"] for s in samples])
    att_mean = gaussian_filter1d(xai_stack.mean(axis=0).astype(float), sigma)
    return y_true_mat, y_pred_mat, att_mean

def _colored_fill_under_curve(ax, pct, y_curve, att, gmin, gmax,
                              cmap_name="YlOrRd", alpha=0.72, gamma=0.6):
    cmap = mpl.colormaps.get_cmap(cmap_name)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=gmin, vmax=gmax)
    for i in range(len(pct) - 1):
        ax.fill_between(
            pct[i:i+2], 0, y_curve[i:i+2],
            color=cmap(norm(float(att[i]))),
            alpha=alpha, linewidth=0, zorder=2
        )
    return cm.ScalarMappable(cmap=cmap, norm=norm)

def _plot_xai_on_grf_by_activity(samples_by_act, T_norm, out_path, sigma=1.2, max_individual=6):
    acts = [a for a in ACTIVITY_ORDER if a in samples_by_act and len(samples_by_act[a]) > 0]
    if not acts:
        print("[WARN] _plot_xai_on_grf_by_activity: no data, skipping.")
        return

    pct = np.linspace(0, 100, T_norm)
    gmin, gmax = _global_energy_scale(samples_by_act)

    fig, axes = plt.subplots(len(acts), 1, figsize=(14, 3.6 * len(acts)), sharex=True)
    if len(acts) == 1:
        axes = [axes]

    for ax, act in zip(axes, acts):
        samples = samples_by_act[act]
        y_true_mat, y_pred_mat, att_mean = _activity_arrays(samples, sigma=sigma)

        N_act = len(samples)
        y_true_mu, y_true_sd = y_true_mat.mean(axis=0), y_true_mat.std(axis=0)
        y_pred_mu, y_pred_sd = y_pred_mat.mean(axis=0), y_pred_mat.std(axis=0)

        _phase_shading(ax)

        for i in range(min(max_individual, N_act)):
            ax.plot(pct, y_true_mat[i], color="black", lw=0.5, alpha=0.15, zorder=1)
            ax.plot(pct, y_pred_mat[i], color="firebrick", lw=0.5, alpha=0.15, ls="--", zorder=1)

        _colored_fill_under_curve(ax, pct, y_pred_mu, att_mean, gmin=gmin, gmax=gmax)

        ax.plot(pct, y_true_mu, color="black", lw=2.5, label=f"Force plate mean (n={N_act})", zorder=7)
        ax.fill_between(pct, np.maximum(y_true_mu - y_true_sd, 0), y_true_mu + y_true_sd,
                        alpha=0.10, color="black", zorder=5)

        ax.plot(pct, y_pred_mu, color="darkred", lw=2.2, ls="--", label="Model prediction mean", zorder=6)
        ax.fill_between(pct, np.maximum(y_pred_mu - y_pred_sd, 0), y_pred_mu + y_pred_sd,
                        alpha=0.10, color="darkred", zorder=5)

        ax.set_xlim(0, 100)
        ax.set_ylabel("vGRF (N)", fontsize=10)
        ax.set_title(f"{act.replace('_','-').title()}   (n = {N_act},  mean r = {_mean_r(samples)})",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, axis="y", alpha=0.22)
        ax.autoscale(enable=True, axis="y", tight=False)
        _phase_labels(ax)

    axes[-1].set_xlabel("Stance phase (%)", fontsize=11)
    _xtick_pct(axes[-1])

    sm_global = cm.ScalarMappable(
        cmap=mpl.colormaps.get_cmap("YlOrRd"),
        norm=mcolors.Normalize(vmin=gmin, vmax=gmax),
    )
    sm_global.set_array([])
    cbar = fig.colorbar(sm_global, ax=axes, orientation="vertical", fraction=0.018, pad=0.01)
    cbar.set_label("Attribution energy (raw)\n(sum across channels, global scale)", fontsize=9)

    fig.suptitle("Activity-specific vGRF Prediction with Temporal SMILE Attribution Overlay",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.subplots_adjust(top=0.93, hspace=0.35)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {Path(out_path).name}")

def _plot_xai_on_grf_aggregate(samples_by_act, T_norm, out_path, sigma=1.5):
    all_samples = [s for act_s in samples_by_act.values() for s in act_s]
    if not all_samples:
        print("[WARN] _plot_xai_on_grf_aggregate: no data, skipping.")
        return

    pct = np.linspace(0, 100, T_norm)
    N = len(all_samples)
    gmin, gmax = _global_energy_scale(samples_by_act)

    y_true_mat, y_pred_mat, att_mean = _activity_arrays(all_samples, sigma=sigma)
    y_true_mu, y_true_sd = y_true_mat.mean(axis=0), y_true_mat.std(axis=0)
    y_pred_mu, y_pred_sd = y_pred_mat.mean(axis=0), y_pred_mat.std(axis=0)

    r_wave = _corr(y_true_mu, y_pred_mu)
    r_txt = f"{r_wave:.3f}" if not np.isnan(r_wave) else "N/A"

    act_counts = {a: len(samples_by_act[a]) for a in ACTIVITY_ORDER if a in samples_by_act}
    counts_str = "  |  ".join(f"{a}: {c}" for a, c in act_counts.items())

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.8, 1.0], hspace=0.38)

    ax_a = fig.add_subplot(gs[0])
    _phase_shading(ax_a)

    sm = _colored_fill_under_curve(ax_a, pct, y_pred_mu, att_mean, gmin=gmin, gmax=gmax)

    ax_a.plot(pct, y_true_mu, color="black", lw=2.5, label=f"Force plate mean (N={N})", zorder=7)
    ax_a.fill_between(pct, np.maximum(y_true_mu - y_true_sd, 0), y_true_mu + y_true_sd,
                      alpha=0.10, color="black", zorder=5)

    ax_a.plot(pct, y_pred_mu, color="darkred", lw=2.2, ls="--", label="Model prediction mean", zorder=6)
    ax_a.fill_between(pct, np.maximum(y_pred_mu - y_pred_sd, 0), y_pred_mu + y_pred_sd,
                      alpha=0.10, color="darkred", zorder=5)

    ax_a.set_xlim(0, 100)
    ax_a.set_ylabel("vGRF (N)", fontsize=11)
    ax_a.set_title(
        f"vGRF Prediction with Temporal SMILE Attribution — Aggregate\n"
        f"N = {N} samples  |  Pearson r (mean waveforms) = {r_txt}\n{counts_str}",
        fontsize=11, fontweight="bold", pad=8)
    ax_a.legend(loc="upper right", fontsize=9, ncol=2)
    ax_a.grid(True, axis="y", alpha=0.25)
    ax_a.autoscale(enable=True, axis="y", tight=False)
    _phase_labels(ax_a)

    cbar = plt.colorbar(sm, ax=ax_a, orientation="vertical", fraction=0.025, pad=0.01)
    cbar.set_label("Attribution energy (raw)\n(sum across channels)", fontsize=8)

    ax_b = fig.add_subplot(gs[1])
    _phase_shading(ax_b)
    ax_b.fill_between(pct, 0, att_mean, alpha=0.35, zorder=3)
    ax_b.plot(pct, att_mean, lw=1.8, zorder=4)
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel("Stance phase (%)", fontsize=11)
    ax_b.set_ylabel("Attribution energy\n(raw)", fontsize=9)
    ax_b.set_title("Mean attribution energy across all samples (sum across channels)", fontsize=9)
    ax_b.grid(True, axis="y", alpha=0.25)

    _xtick_pct(ax_b)
    _xtick_pct(ax_a)

    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# SMILE main
# ─────────────────────────────────────────────────────────────────────────────
def temporal_smile_loso(
    data_dir, ckpt_glob, outdir, device="cpu",
    n_samples=95, window_size=15, stride=10,
    baseline="mean", T_norm=100, seed=42,
    make_grf_overlays=True,
    imu_zscore="none",
    overlay_scale="none",
    bw_kg=None,
    fig_style="doublecol",
    heat_gamma=0.55,
    heat_vmax_pct=99.0,
    heat_vmin_pct=0.0,
    vis_norm="global",
    # event controls
    event_analysis=False,
    event_topk=4,
    event_win_pct=6.0,
    event_landmark="peak_vgrf",
    # TimeSHAP controls
    timeshap=False,
    timeshap_E=20,
    timeshap_K=300,
    timeshap_target="peak_vgrf",
    timeshap_phase="Loading",
    timeshap_max_samples=20,
    timeshap_upsample="repeat",
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    X_all, y_all, pids_all, acts_all, masses_all = load_csv_trials(data_dir)
    if not X_all:
        raise RuntimeError("No data loaded from CSV dir")

    ckpts = sorted(glob.glob(ckpt_glob, recursive=True))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints: {ckpt_glob}")

    pid_to_ckpt = {}
    for c in ckpts:
        m = re.search(r"test[_\\-]?(P\\d{2})", c, re.IGNORECASE)
        if m:
            pid_to_ckpt[m.group(1).upper()] = c

    uniq = sorted(set(pids_all))
    folds = [
        (p,
         [i for i, q in enumerate(pids_all) if q != p],
         [i for i, q in enumerate(pids_all) if q == p])
        for p in uniq
    ]

    fold_temporal = []
    fold_temporal_by_act = defaultdict(list)
    fold_ranks = []
    samples_by_act = defaultdict(list)

    event_rows = []
    timeshap_rows = []
    timeshap_agg = []

    rng = np.random.default_rng(seed)
    timeshap_done = 0

    for fi, (test_pid, tr_idx, te_idx) in enumerate(folds, 1):
        ckpt_path = pid_to_ckpt.get(test_pid, ckpts[min(fi - 1, len(ckpts) - 1)])
        print(f"\nFold {fi:02d} | TEST={test_pid} | CKPT={Path(ckpt_path).name}")

        model = MultiScaleGRFNet()
        model.load_state_dict(load_ckpt(ckpt_path, "cpu"))
        model = model.to(device).eval()

        mu_fold, sd_fold = None, None
        if imu_zscore in ("fold", "global"):
            if imu_zscore == "fold":
                mu_fold, sd_fold = _compute_fold_imu_stats(X_all, tr_idx)
            else:
                all_vals = np.concatenate(X_all, axis=0)
                mu_fold = np.nanmean(all_vals, axis=0).astype(np.float32)
                sd_fold = np.nanstd(all_vals, axis=0).astype(np.float32)
                sd_fold = np.where(sd_fold < 1e-8, 1.0, sd_fold).astype(np.float32)

        # SMILE baseline in SAME SPACE as model input
        if baseline == "mean":
            if mu_fold is not None:
                bv = torch.zeros(12, dtype=torch.float32)  # mean in z-space
            else:
                tr_vals_raw = np.concatenate([X_all[i] for i in tr_idx], axis=0)
                bv = torch.tensor(np.nanmean(tr_vals_raw, axis=0), dtype=torch.float32)
        else:
            bv = torch.zeros(12, dtype=torch.float32)

        # balance test samples by activity within fold
        te_by_act = defaultdict(list)
        for i in te_idx:
            te_by_act[acts_all[i]].append(i)

        per_act = max(1, n_samples // max(len(te_by_act), 1))
        chosen = []
        for act, idxs in te_by_act.items():
            idxs_sh = list(idxs)
            rng.shuffle(idxs_sh)
            chosen.extend(idxs_sh[:per_act])
        chosen = chosen[:n_samples]

        fold_tc_list = []
        fold_rank_imp = np.zeros(12, dtype=np.float64)

        for idx in tqdm(chosen, desc=f"  Fold {fi:02d} SMILE"):
            X_raw  = X_all[idx].copy()
            y_true = y_all[idx].copy()
            act    = acts_all[idx]
            mass_kg = masses_all[idx]

            X_in = _zscore(X_raw, mu_fold, sd_fold) if (mu_fold is not None) else X_raw

            # Temporal SMILE
            tc = _temporal_smile_one(model, X_in, bv, window_size, stride, device)  # (T_i,12)
            tc_norm = _normalise_tc(tc, T_norm)

            fold_tc_list.append(tc_norm)
            fold_temporal_by_act[act].append(tc_norm)
            fold_rank_imp += tc_norm.mean(axis=0)

            # TimeSHAP-style KernelSHAP (limited samples)
            if timeshap and (timeshap_done < int(timeshap_max_samples)):
                try:
                    phi_events, phi0, meta = timeshap_eventwise_kernelshap(
                        model=model,
                        X_in_tc=X_in,
                        bv_c=bv.detach().cpu().numpy(),
                        device=device,
                        T_norm=T_norm,
                        E=timeshap_E,
                        K=timeshap_K,
                        seed=seed + fi * 1000 + idx,
                        target=timeshap_target,
                        phase_name=timeshap_phase if timeshap_target == "phase_mean" else None,
                    )
                    timeshap_agg.append(phi_events)
                    for e in range(phi_events.shape[0]):
                        for c in range(phi_events.shape[1]):
                            timeshap_rows.append({
                                "fold_test_pid": test_pid,
                                "activity": act,
                                "sample_index": int(idx),
                                "event_bin": int(e),
                                "channel": CHANNEL_LABELS[c],
                                "phi": float(phi_events[e, c]),
                                "target": meta["target"],
                                "E": meta["E"],
                                "K": meta["K"],
                            })
                    timeshap_done += 1
                except Exception as e:
                    print(f"[WARN] TimeSHAP failed on sample {idx}: {e}")

            # Event-based (optional)
            if event_analysis:
                X_b_native = torch.tensor(X_in, dtype=torch.float32).unsqueeze(0).to(device)
                y_pred_native = model(X_b_native).squeeze(0).detach().cpu().numpy()

                y_pred_norm = _normalise_tc(y_pred_native[:, None], T_norm).squeeze(1)
                y_true_norm = _normalise_tc(y_true[:, None], T_norm).squeeze(1)

                out_idx_pred = _output_landmark_idx(y_pred_norm, which=event_landmark)
                out_pct_pred = _pct_from_idx(out_idx_pred, T_norm)

                out_idx_true = _output_landmark_idx(y_true_norm, which=event_landmark)
                out_pct_true = _pct_from_idx(out_idx_true, T_norm)

                imp_c = tc_norm.mean(axis=0)
                topk = np.argsort(imp_c)[::-1][:max(1, int(event_topk))]

                win_native = int(round((float(event_win_pct) / 100.0) * (X_in.shape[0] - 1)))
                win_native = int(np.clip(win_native, 3, 25))

                for c in topk:
                    xin_norm = _normalise_tc(X_in[:, c][:, None], T_norm).squeeze(1)
                    xin_sm = gaussian_filter1d(xin_norm.astype(float), 1.5)
                    in_idx = _peak_idx(xin_sm, mode="abs")
                    in_pct = _pct_from_idx(in_idx, T_norm)

                    att_sm = gaussian_filter1d(tc_norm[:, c].astype(float), 1.2)
                    att_idx = int(np.nanargmax(att_sm))
                    att_pct = _pct_from_idx(att_idx, T_norm)

                    in_idx_native = int(round((in_pct / 100.0) * (X_in.shape[0] - 1)))

                    base_c = float(bv[c].item())
                    rmse, dpeak, devent = _perturb_window_effect(
                        model=model,
                        X_in_tc=X_in,
                        baseline_c=base_c,
                        c_idx=int(c),
                        center_idx=in_idx_native,
                        win=win_native,
                        device=device,
                        landmark=event_landmark,
                    )

                    event_rows.append({
                        "fold_test_pid": test_pid,
                        "activity": act,
                        "channel": CHANNEL_LABELS[c],
                        "in_peak_pct": float(in_pct),
                        "att_peak_pct": float(att_pct),
                        "out_event_pct_pred": float(out_pct_pred),
                        "out_event_pct_true": float(out_pct_true),
                        "lag_in_to_out_pred_pct": float(out_pct_pred - in_pct),
                        "lag_in_to_out_true_pct": float(out_pct_true - in_pct),
                        "lag_att_to_out_pred_pct": float(out_pct_pred - att_pct),
                        "lag_att_to_out_true_pct": float(out_pct_true - att_pct),
                        "mean_attr": float(imp_c[c]),
                        "perturb_win_pct": float(event_win_pct),
                        "perturb_rmse_native_units": float(rmse),
                        "perturb_dpeak_native_units": float(dpeak),
                        "perturb_devent_native_units": float(devent),
                        "landmark": event_landmark,
                    })

            # GRF overlays (optional)
# GRF overlays (optional)
            if make_grf_overlays:
                X_b = torch.tensor(X_in, dtype=torch.float32).unsqueeze(0).to(device)
                y_pred = model(X_b).squeeze(0).detach().cpu().numpy()

                y_true_N = _normalise_tc(y_true[:, None], T_norm).squeeze(1)
                y_pred_m = _normalise_tc(y_pred[:, None], T_norm).squeeze(1)

                # ✅ FIX: Always scale for visualization (use auto if none specified)
                y_pred_plot = _overlay_scale_pred(
                    y_true_N=y_true_N,
                    y_pred_model=y_pred_m,
                    mode=overlay_scale if overlay_scale != "none" else "auto",  # ✅ fallback to auto
                    bw_kg=bw_kg,
                    mass_kg=mass_kg
                )

                # ✅ FIX: Correlation AFTER scaling (apples-to-apples)
                r_val = _corr(y_true_N, y_pred_plot)

                xai_energy = tc_norm.sum(axis=1)

                samples_by_act[act].append({
                    "y_true_norm": y_true_N.astype(np.float32),
                    "y_pred_norm": y_pred_plot.astype(np.float32),  # ✅ stores scaled version
                    "xai_energy":  xai_energy.astype(np.float32),
                    "r": float(r_val) if r_val is not None else np.nan,
                })

        fold_mean_tc = (
            np.stack(fold_tc_list).mean(axis=0)
            if fold_tc_list else np.zeros((T_norm, 12), np.float32)
        )
        fold_temporal.append(fold_mean_tc)
        fold_ranks.append(_rank_vector(fold_rank_imp / max(len(fold_tc_list), 1)))

        fold_out = Path(outdir) / f"temporal_smile_fold_{fi:02d}_{test_pid}.png"
        _plot_single_heatmap(
            fold_mean_tc,
            title=f"Temporal SMILE | Fold {fi:02d} | TEST={test_pid}",
            out_path=fold_out,
            T_norm=T_norm,
            fig_style=fig_style
        )

    # Aggregate SMILE
    agg_mean = np.stack(fold_temporal).mean(axis=0)
    agg_std  = np.stack(fold_temporal).std(axis=0)

    _plot_temporal_heatmap_full(
        imp_mean=agg_mean,
        imp_std=agg_std,
        title="Temporal SMILE — Aggregate (all LOSO folds)",
        out_path=Path(outdir) / "temporal_smile_aggregate.png",
        T_norm=T_norm,
        overlay_topk=4,
        overlay_band=False,
        fig_style=fig_style,
        heat_gamma=heat_gamma,
        heat_vmax_pct=heat_vmax_pct,
        heat_vmin_pct=heat_vmin_pct,
        vis_norm=vis_norm,
    )

    # ✅ restored per-activity outputs
    _plot_activity_grid(
        fold_temporal_by_act=fold_temporal_by_act,
        T_norm=T_norm,
        out_path=Path(outdir) / "temporal_smile_by_activity.png",
        fig_style=fig_style,
        heat_gamma=heat_gamma,
        heat_vmax_pct=heat_vmax_pct,
    )
    _plot_phase_bar(
        imp_mean=agg_mean,
        T_norm=T_norm,
        out_path=Path(outdir) / "temporal_smile_phase_bar.png",
    )

    _save_phase_csv(agg_mean, T_norm, Path(outdir) / "smile_phase_summary.csv")
    _save_tc_csv(agg_mean, T_norm, Path(outdir) / "smile_temporal_agg.csv")
    _save_stability(fold_ranks, CHANNEL_LABELS, Path(outdir) / "smile_stability_summary.txt")

    if make_grf_overlays and samples_by_act:
        _plot_xai_on_grf_by_activity(samples_by_act, T_norm, Path(outdir) / "grf_overlay_by_activity.png")
        _plot_xai_on_grf_aggregate(samples_by_act, T_norm, Path(outdir) / "grf_overlay_aggregate.png")

    # Event outputs
    if event_analysis and event_rows:
        ev = pd.DataFrame(event_rows)
        ev_path = Path(outdir) / "event_alignment_smile.csv"
        ev.to_csv(ev_path, index=False)
        print(f"[OK] {ev_path.name}")

        summ = ev.groupby("channel").agg({
            "mean_attr": "mean",
            "lag_in_to_out_pred_pct": ["median", "mean"],
            "lag_in_to_out_true_pct": ["median", "mean"],
            "lag_att_to_out_pred_pct": ["median", "mean"],
            "lag_att_to_out_true_pct": ["median", "mean"],
            "perturb_devent_native_units": ["median", "mean"],
            "perturb_dpeak_native_units": ["median", "mean"],
            "perturb_rmse_native_units": ["median", "mean"],
        })
        summ_path = Path(outdir) / "event_alignment_smile_summary.csv"
        summ.to_csv(summ_path)
        print(f"[OK] {summ_path.name}")

    # TimeSHAP outputs (UPDATED plotting)
    if timeshap and timeshap_rows:
        ts_df = pd.DataFrame(timeshap_rows)
        ts_path = Path(outdir) / "timeshap_eventwise_long.csv"
        ts_df.to_csv(ts_path, index=False)
        print(f"[OK] {ts_path.name}")

        mats = np.stack(timeshap_agg, axis=0)              # (S,E,C)
        mean_abs_EC = np.mean(np.abs(mats), axis=0)        # (E,C)

        ts_agg_path = Path(outdir) / "timeshap_eventwise_mean_abs.csv"
        pd.DataFrame(mean_abs_EC, columns=CHANNEL_LABELS).assign(event_bin=np.arange(mean_abs_EC.shape[0])).to_csv(
            ts_agg_path, index=False
        )
        print(f"[OK] {ts_agg_path.name}")

        # BEFORE (event-bin heatmap)
        _plot_eventwise_shap_heatmap(
            mean_abs_EC,
            title=f"TimeSHAP Eventwise (mean |phi|, E={timeshap_E}) — BEFORE",
            out_path=Path(outdir) / "timeshap_eventwise_heatmap_BEFORE.png",
            fig_style=fig_style,
        )

        # AFTER (consistent with SHAP/SMILE)
        plot_timeshap_temporal_heatmap(
            phi_events_EC=mean_abs_EC,
            title=f"TimeSHAP (upsampled to T={T_norm}) — CONSISTENT with SHAP/SMILE",
            out_path=Path(outdir) / "timeshap_temporal_heatmap.png",
            T_norm=T_norm,
            upsample_method=timeshap_upsample,
            fig_style=fig_style,
            heat_gamma=heat_gamma,
            heat_vmax_pct=heat_vmax_pct,
            heat_vmin_pct=heat_vmin_pct,
            vis_norm=vis_norm,
            overlay_topk=4,
            overlay_band=False,
        )

        # Comparison figure
        plot_timeshap_format_comparison(
            phi_events_EC=mean_abs_EC,
            out_path=Path(outdir) / "timeshap_format_comparison.png",
            T_norm=T_norm,
            fig_style=fig_style,
            heat_vmax_pct=heat_vmax_pct,
            heat_gamma=heat_gamma,
        )

    print(f"\n✅ Temporal SMILE done → {outdir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Temporal XAI (SHAP + SMILE) for GRFNet-MultiScale (+ overlays + event analysis + optional TimeSHAP)"
    )
    ap.add_argument("mode", choices=["shap", "smile", "both"])

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--T_norm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_glob", required=True)

    # SHAP
    ap.add_argument("--npz", default=None)
    ap.add_argument("--bg_n", type=int, default=100)
    ap.add_argument("--sample_n", type=int, default=50)

    # SMILE
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--n_samples", type=int, default=95)
    ap.add_argument("--window_size", type=int, default=15)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--baseline", default="mean", choices=["zero", "mean"])
    ap.add_argument("--no_grf_overlays", action="store_true")

    # Critical controls
    ap.add_argument("--imu_zscore", default="none", choices=["none", "fold", "global"],
                    help="Apply z-score to IMU before inference/SMILE/SHAP. Use if training used z-score.")
    ap.add_argument("--overlay_scale", default="none", choices=["none", "auto", "bw"],
                    help="Plot-only scaling for overlay. Prefer none or bw if model outputs N/BW.")
    ap.add_argument("--bw_kg", type=float, default=None,
                    help="Body mass (kg) for --overlay_scale bw if mass not in CSV.")

    # Figure styling
    ap.add_argument("--fig_style", default="doublecol", choices=["singlecol", "doublecol"],
                    help="Figure layout preset. singlecol≈IEEE single-column width.")
    ap.add_argument("--font_scale", type=float, default=1.0,
                    help="Global multiplier for all font sizes.")
    ap.add_argument("--heat_gamma", type=float, default=0.55,
                    help="PowerNorm gamma for heatmaps (<1 boosts mid/low values).")
    ap.add_argument("--heat_vmax_pct", type=float, default=99.0,
                    help="Upper percentile used as vmax for heatmaps (stabilises contrast).")
    ap.add_argument("--heat_vmin_pct", type=float, default=0.0,
                    help="Lower percentile used as vmin for heatmaps.")
    ap.add_argument("--vis_norm", default="global", choices=["global", "per_phase"],
                    help="VISUAL-ONLY normalisation. per_phase boosts mid/push contrast but changes color meaning.")

    # Event-based temporal explainability
    ap.add_argument("--event_analysis", action="store_true",
                    help="Compute event-based alignment + perturbation analysis.")
    ap.add_argument("--event_topk", type=int, default=4,
                    help="Top-K channels (by mean attribution) used for event analysis (per sample).")
    ap.add_argument("--event_win_pct", type=float, default=6.0,
                    help="Window size (% stance) around input peak for perturbation (native-length mapped).")
    ap.add_argument("--event_landmark", default="peak_vgrf",
                    choices=["peak_vgrf", "peak_rfd"],
                    help="Which output landmark to align against (computed on pred and truth).")

    # TimeSHAP-style event-wise KernelSHAP
    ap.add_argument("--timeshap", action="store_true",
                    help="Run TimeSHAP-style event-wise KernelSHAP (events over stance bins).")
    ap.add_argument("--timeshap_E", type=int, default=20,
                    help="Number of stance events (features). Recommended 20.")
    ap.add_argument("--timeshap_K", type=int, default=300,
                    help="Number of coalition samples per explained sample. Start ~300.")
    ap.add_argument("--timeshap_target", default="peak_vgrf",
                    choices=["peak_vgrf", "phase_mean", "value_at"],
                    help="Scalar target used for KernelSHAP.")
    ap.add_argument("--timeshap_phase", default="Loading",
                    choices=list(PHASES.keys()),
                    help="Phase used only if --timeshap_target phase_mean.")
    ap.add_argument("--timeshap_max_samples", type=int, default=20,
                    help="Max number of samples to run TimeSHAP on (runtime control).")
    ap.add_argument("--timeshap_upsample", default="repeat", choices=["repeat", "interp"],
                    help="How to upsample TimeSHAP E-bins to T_norm for consistent plotting.")

    args = ap.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    _apply_plot_style(args.fig_style, args.font_scale)

    if args.mode in ("shap", "both"):
        if not args.npz:
            ap.error("--npz required for shap/both mode")
        temporal_shap_loso(
            npz_path=args.npz,
            ckpt_glob=args.ckpt_glob,
            outdir=os.path.join(args.outdir, "shap"),
            device=device,
            bg_n=args.bg_n,
            sample_n=args.sample_n,
            T_norm=args.T_norm,
            seed=args.seed,
            imu_zscore=args.imu_zscore,
            fig_style=args.fig_style,
            heat_gamma=args.heat_gamma,
            heat_vmax_pct=args.heat_vmax_pct,
            heat_vmin_pct=args.heat_vmin_pct,
            vis_norm=args.vis_norm,
        )

    if args.mode in ("smile", "both"):
        if not args.data_dir:
            ap.error("--data_dir required for smile/both mode")
        temporal_smile_loso(
            data_dir=args.data_dir,
            ckpt_glob=args.ckpt_glob,
            outdir=os.path.join(args.outdir, "smile"),
            device=device,
            n_samples=args.n_samples,
            window_size=args.window_size,
            stride=args.stride,
            baseline=args.baseline,
            T_norm=args.T_norm,
            seed=args.seed,
            make_grf_overlays=(not args.no_grf_overlays),
            imu_zscore=args.imu_zscore,
            overlay_scale=args.overlay_scale,
            bw_kg=args.bw_kg,
            fig_style=args.fig_style,
            heat_gamma=args.heat_gamma,
            heat_vmax_pct=args.heat_vmax_pct,
            heat_vmin_pct=args.heat_vmin_pct,
            vis_norm=args.vis_norm,
            event_analysis=args.event_analysis,
            event_topk=args.event_topk,
            event_win_pct=args.event_win_pct,
            event_landmark=args.event_landmark,
            timeshap=args.timeshap,
            timeshap_E=args.timeshap_E,
            timeshap_K=args.timeshap_K,
            timeshap_target=args.timeshap_target,
            timeshap_phase=args.timeshap_phase,
            timeshap_max_samples=args.timeshap_max_samples,
            timeshap_upsample=args.timeshap_upsample,
        )

if __name__ == "__main__":
    main()
