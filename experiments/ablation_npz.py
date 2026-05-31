#!/usr/bin/env python3
"""
ablation_loso_full.py  (FULL CLEAN SCRIPT + NPZ FREEZE OPTION)
==============================================================

What this script does:
- Loads aligned CSV trials (force + IMU)
- Extracts stance windows using force-threshold rules
- Builds variable-length trial lists (X_all, y_all, pids, acts)
- OPTIONAL: exports a frozen padded dataset NPZ:
      X:    (N,T,C) float32   padded with PADDING_VALUE
      y:    (N,T)   float32   padded with PADDING_VALUE
      mask: (N,T)   bool      True where y != PADDING_VALUE
      pids: (N,)    object
      acts: (N,)    object
  and exits (no training).
- Otherwise runs 10-fold LOSO by participant:
    * Baselines: Linear (RidgeCV), XGBoost (if available) on original scale
    * Deep models: trained on normalized targets (train stats only),
                   evaluated on original scale
    * Includes GRFNet-MultiScale ablations for your table:
        1) Full
        2) w/o global context
        3) w/o dilations (all dilation=1)
        4) w/o residual connections
        5) MSE-only (no corr term)

Usage examples
--------------
# (A) Only freeze dataset NPZ (fast, no training):
python3 ablation_loso_full.py \
  --aligned_dir /home/805478/Dataset_Aligned_FINAL_forcheck_2Jan_v36_y \
  --output_dir /home/805478/Ablation_Results \
  --config both \
  --export_npz

# (B) Full LOSO training:
python3 ablation_loso_full.py \
  --aligned_dir /home/805478/Dataset_Aligned_FINAL_forcheck_2Jan_v36_y \
  --output_dir /home/805478/Ablation_Results \
  --config both
"""

import argparse
import re
import math
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import RidgeCV

# Optional XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    xgb = None
    HAS_XGB = False


# =============================================================================
# CONFIG
# =============================================================================
RANDOM_STATE  = 42
PADDING_VALUE = -9999.0


# =============================================================================
# UTILS
# =============================================================================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))

def fisher_z_mean_std(r_vals):
    r = np.asarray(r_vals, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan, np.nan
    r = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r)
    z_mean = z.mean()
    z_std  = z.std(ddof=1) if len(z) > 1 else 0.0
    r_mean = np.tanh(z_mean)
    r_lo   = np.tanh(z_mean - z_std)
    r_hi   = np.tanh(z_mean + z_std)
    r_std_equiv = (r_hi - r_lo) / 2
    return float(r_mean), float(r_std_equiv)

def _zero_pad(x: torch.Tensor, pad_value=PADDING_VALUE) -> torch.Tensor:
    x = x.clone()
    x[x == pad_value] = 0.0
    return x

def safe_save_counter(counter_obj, out_csv: Path, key="reason", val="count"):
    rows = [{key: k, val: int(v)} for k, v in counter_obj.items()]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[key, val])
    df = df.sort_values(by=val, ascending=False)
    df.to_csv(out_csv, index=False)


# =============================================================================
# FREEZE DATASET TO NPZ (for reuse in XAI scripts)
# =============================================================================
def freeze_to_npz(X_all, y_all, pids, acts, out_npz: str, feature_names=None):
    """
    Convert variable-length trial lists into padded arrays:
      X:    (N,T,C) float32
      y:    (N,T)   float32
      mask: (N,T)   bool  (True where y != PADDING_VALUE)
      pids/acts: object arrays

    Also stores:
      feature_names: (C,) object array  (IMPORTANT for correct SHAP labeling)
    """
    N = len(X_all)
    assert N == len(y_all) == len(pids) == len(acts)
    C = X_all[0].shape[1]
    T = max(x.shape[0] for x in X_all)

    X = np.full((N, T, C), PADDING_VALUE, dtype=np.float32)
    y = np.full((N, T),    PADDING_VALUE, dtype=np.float32)

    for i, (xi, yi) in enumerate(zip(X_all, y_all)):
        ti = min(len(xi), T)
        X[i, :ti, :] = xi[:ti]
        y[i, :ti]    = yi[:ti]

    mask = (y != PADDING_VALUE)

    if feature_names is None:
        feature_names = [f"ch_{i:02d}" for i in range(C)]
    else:
        if len(feature_names) != C:
            raise ValueError(f"feature_names length ({len(feature_names)}) != channels ({C})")

    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        mask=mask,
        pids=np.array(pids, dtype=object),
        acts=np.array(acts, dtype=object),
        padding_value=np.array([PADDING_VALUE], dtype=np.float32),
        feature_names=np.array(feature_names, dtype=object),
    )
    print(f"[INFO] ✅ Saved NPZ: {out_npz}")
    print(f"[INFO] Shapes: X={X.shape}, y={y.shape}, mask={mask.shape}, C={C}, T={T}")
    print(f"[INFO] feature_names saved: {feature_names}")



# =============================================================================
# TARGET NORMALIZATION (TRAIN STATS ONLY)
# =============================================================================
class TargetNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.std  = 1.0

    def fit(self, y_list):
        y_all = np.concatenate([y.reshape(-1) for y in y_list]).astype(np.float64)
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size == 0:
            self.mean, self.std = 0.0, 1.0
            return
        self.mean = float(y_all.mean())
        self.std  = float(y_all.std())
        if self.std < 1e-8:
            self.std = 1.0

    def transform(self, y_list):
        return [(y - self.mean) / self.std for y in y_list]

    def inverse_transform(self, y_norm):
        return y_norm * self.std + self.mean


# =============================================================================
# ACTIVITY + PID HELPERS
# =============================================================================
def infer_activity(stem: str) -> str:
    s = stem.lower()
    tokens = re.split(r"[^a-z0-9]+", s)

    def has_prefix(pref: str) -> bool:
        return any(t.startswith(pref) for t in tokens if t)

    if has_prefix("heel"):
        return "heel"
    if has_prefix("drop"):
        return "drop"
    if has_prefix("walk"):
        return "walking"
    if has_prefix("jog"):
        return "jogging"
    if has_prefix("run"):
        return "running"

    return "unknown"

def extract_pid(stem: str) -> str:
    m = re.search(r"(?:^|_)(P)(\d+)(?:_|$)", stem.upper())
    if m:
        return f"P{int(m.group(2)):02d}"
    for part in stem.split("_"):
        pu = part.upper()
        if pu.startswith("P") and pu[1:].isdigit():
            return f"P{int(pu[1:]):02d}"
    print(f"[WARN] Could not extract participant ID from: {stem}")
    return "P00"


# =============================================================================
# STANCE EXTRACTION
# =============================================================================
def extract_stance(X, y, contact_thr, min_dur, min_peak, fs, stance_reasons: Counter):
    mask = y > contact_thr
    if not mask.any():
        stance_reasons["no_contact"] += 1
        return None, None

    idx = np.where(mask)[0]
    gaps = np.diff(idx)

    # handle fragmented contact; keep longest segment
    if (gaps > 5).any():
        segments = []
        start = 0
        for gi in np.where(gaps > 5)[0]:
            seg = idx[start:gi + 1]
            if len(seg) > 0:
                segments.append(seg)
            start = gi + 1
        if start < len(idx):
            segments.append(idx[start:])
        if not segments:
            stance_reasons["no_segments"] += 1
            return None, None
        seg = max(segments, key=len)
        onset, offset = seg[0], seg[-1]
    else:
        onset, offset = idx[0], idx[-1]

    if (offset - onset + 1) < int(min_dur * fs):
        stance_reasons["too_short"] += 1
        return None, None

    if float(np.max(y[onset:offset + 1])) < min_peak:
        stance_reasons["low_peak"] += 1
        return None, None

    return X[onset:offset + 1], y[onset:offset + 1]


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(aligned_dir, args):
    files = sorted(Path(aligned_dir).glob("*.csv"))

    imu_schemas = {
        "both": [
            ["Waist_AccX","Waist_AccY","Waist_AccZ","Waist_GyroX","Waist_GyroY","Waist_GyroZ",
             "Wrist_AccX","Wrist_AccY","Wrist_AccZ","Wrist_GyroX","Wrist_GyroY","Wrist_GyroZ"],
            ["waist_accX","waist_accY","waist_accZ","waist_gyroX","waist_gyroY","waist_gyroZ",
             "wrist_accX","wrist_accY","wrist_accZ","wrist_gyroX","wrist_gyroY","wrist_gyroZ"],
        ],
        "waist": [
            ["Waist_AccX","Waist_AccY","Waist_AccZ","Waist_GyroX","Waist_GyroY","Waist_GyroZ"],
            ["waist_accX","waist_accY","waist_accZ","waist_gyroX","waist_gyroY","waist_gyroZ"],
        ],
        "wrist": [
            ["Wrist_AccX","Wrist_AccY","Wrist_AccZ","Wrist_GyroX","Wrist_GyroY","Wrist_GyroZ"],
            ["wrist_accX","wrist_accY","wrist_accZ","wrist_gyroX","wrist_gyroY","wrist_gyroZ"],
        ],
    }[args.config]

    force_candidates = ["Force_Z", "force_z_N", "force_z", "Force_Vertical", "ForceZ", "Fz", "forceZ"]

    X_all, y_all, pids, acts = [], [], [], []
    imu_cols_used = None
    force_col_used = None

    act_counts = Counter()
    skipped = Counter()
    stance_reasons = Counter()

    print(f"\n[INFO] Loading IMU config: {args.config}")
    print("[INFO] Stance rules: heel=(250N,0.08s), drop=(800N,0.15s), locomotion=(800N,0.15s)")

    for fp in tqdm(files, desc="Loading"):
        if fp.name.lower() == "alignment_log.csv" or "alignment_log" in fp.name.lower():
            skipped["alignment_log"] += 1
            continue

        try:
            df = pd.read_csv(fp)
        except Exception:
            skipped["csv_error"] += 1
            continue

        if force_col_used is None:
            for cand in force_candidates:
                if cand in df.columns:
                    force_col_used = cand
                    print(f"[INFO] Force column: {force_col_used}")
                    break
        if force_col_used is None or force_col_used not in df.columns:
            skipped["no_force"] += 1
            continue

        if imu_cols_used is None:
            for schema in imu_schemas:
                if all(c in df.columns for c in schema):
                    imu_cols_used = schema
                    print(f"[INFO] IMU schema detected: {len(schema)} channels")
                    break
        if imu_cols_used is None:
            skipped["no_imu_schema_detected"] += 1
            continue
        if not all(c in df.columns for c in imu_cols_used):
            skipped["imu_missing"] += 1
            continue

        X = df[imu_cols_used].values.astype(np.float32)
        y = df[force_col_used].values.astype(np.float32)

        act = infer_activity(fp.stem)
        pid = extract_pid(fp.stem)
        act_counts[act] += 1

        if act == "heel":
            peak_thr, dur_thr = 250.0, 0.08
        elif act == "drop":
            peak_thr, dur_thr = 800.0, 0.15
        else:
            peak_thr, dur_thr = 800.0, 0.15

        X_st, y_st = extract_stance(
            X, y,
            contact_thr=args.contact_threshold,
            min_dur=dur_thr,
            min_peak=peak_thr,
            fs=args.fs,
            stance_reasons=stance_reasons
        )

        if X_st is None:
            skipped["stance_failed"] += 1
            continue

        if act not in args.allowed_acts:
            skipped["activity_filtered"] += 1
            continue

        X_all.append(X_st)
        y_all.append(y_st)
        pids.append(pid)
        acts.append(act)

    print(f"\n[INFO] Loaded {len(X_all)} trials")
    print(f"[INFO] Activities: {dict(Counter(acts))}")
    print(f"[INFO] Participants: {dict(Counter(pids))}")
    print(f"[INFO] Skipped: {dict(skipped)}")
    print(f"[INFO] Stance rejections: {dict(stance_reasons)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_save_counter(act_counts, out_dir / "activity_raw_counts.csv", key="activity", val="count")
    safe_save_counter(skipped, out_dir / "skipped_reasons.csv", key="reason", val="count")
    safe_save_counter(stance_reasons, out_dir / "stance_rejection_reasons.csv", key="reason", val="count")

    if len(X_all) == 0:
        raise RuntimeError("No trials loaded after filtering.")

    return X_all, y_all, pids, acts, imu_cols_used



# =============================================================================
# DATASET
# =============================================================================
class StanceDataset(Dataset):
    def __init__(self, X_list, y_list, max_len):
        if len(X_list) == 0:
            raise ValueError("StanceDataset received empty X_list.")
        self.X_list = X_list
        self.y_list = y_list
        self.max_len = int(max_len)
        self.n_channels = X_list[0].shape[1]

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        X, y = self.X_list[idx], self.y_list[idx]

        if len(X) > self.max_len:
            X = X[:self.max_len]
            y = y[:self.max_len]

        T = len(X)
        X_pad = np.full((self.max_len, self.n_channels), PADDING_VALUE, dtype=np.float32)
        y_pad = np.full((self.max_len,), PADDING_VALUE, dtype=np.float32)
        X_pad[:T] = X
        y_pad[:T] = y
        return torch.from_numpy(X_pad), torch.from_numpy(y_pad)


# =============================================================================
# MODELS
# =============================================================================
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        h, _ = self.lstm(_zero_pad(x))
        return self.fc(h).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ImprovedTransformer(nn.Module):
    def __init__(self, input_dim=12, d_model=128, nhead=4, num_layers=2,
                 dropout=0.2, max_len=20000):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        pad_mask = (x == PADDING_VALUE).all(dim=-1)
        x = x.clone()
        x[pad_mask.unsqueeze(-1).expand_as(x)] = 0.0

        h = self.proj(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        h = self.tr(h, src_key_padding_mask=pad_mask)
        return self.fc(h).squeeze(-1)


class SimpleTCN(nn.Module):
    def __init__(self, input_dim=12, num_filters=64, num_blocks=4, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        ch = input_dim
        for i in range(num_blocks):
            dil = 2 ** i
            layers += [
                nn.Conv1d(ch, num_filters, kernel_size,
                          padding=(kernel_size - 1) * dil // 2, dilation=dil),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            ch = num_filters
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        h = self.net(_zero_pad(x).transpose(1, 2)).transpose(1, 2)
        return self.fc(h).squeeze(-1)


class SimpleGRFNet(nn.Module):
    def __init__(self, input_dim=12, num_filters=64, num_blocks=3, dropout=0.15):
        super().__init__()
        self.proj = nn.Conv1d(input_dim, num_filters, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm1d(num_filters),
            ) for _ in range(num_blocks)
        ])
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        h = self.proj(_zero_pad(x).transpose(1, 2))
        for blk in self.blocks:
            h = F.relu(blk(h) + h)
        return self.fc(h.transpose(1, 2)).squeeze(-1)


class ImprovedGRFNet(nn.Module):
    def __init__(self, input_dim=12, num_filters=64, num_blocks=3, dropout=0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, 1),
            nn.BatchNorm1d(num_filters)
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm1d(num_filters),
            ) for _ in range(num_blocks)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(num_filters // 2, 1)
        )

    def forward(self, x):
        h = self.proj(_zero_pad(x).transpose(1, 2))
        for blk in self.blocks:
            h = F.relu(blk(h) + h)
        return self.output_head(h.transpose(1, 2)).squeeze(-1)


class MultiScaleGRFNet(nn.Module):
    """
    GRFNet-MultiScale with ablation toggles:
      - use_global_context: remove global pooling concat (w/o global context)
      - use_dilations: set all dilations=1 (w/o dilations)
      - use_residual: remove residual add (w/o residual connections)
    """
    def __init__(self,
                 input_dim=12,
                 num_filters=64,
                 num_blocks=4,
                 dropout=0.15,
                 use_global_context=True,
                 use_dilations=True,
                 use_residual=True):
        super().__init__()
        self.use_global_context = bool(use_global_context)
        self.use_dilations = bool(use_dilations)
        self.use_residual = bool(use_residual)

        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, 1),
            nn.BatchNorm1d(num_filters)
        )

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = (2 ** i) if self.use_dilations else 1
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(num_filters, num_filters, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(num_filters, num_filters, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(num_filters),
                )
            )

        if self.use_global_context:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            head_in = num_filters * 2
        else:
            self.global_pool = None
            head_in = num_filters

        self.output_head = nn.Sequential(
            nn.Linear(head_in, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, 1)
        )

    def forward(self, x):
        x_pad = _zero_pad(x)
        h = self.proj(x_pad.transpose(1, 2))  # (B,C,T)

        for blk in self.blocks:
            out = blk(h)
            if self.use_residual:
                h = F.relu(out + h)
            else:
                h = F.relu(out)

        h_t = h.transpose(1, 2)  # (B,T,C)

        if self.use_global_context:
            g = self.global_pool(h).squeeze(-1)          # (B,C)
            g = g.unsqueeze(1).expand(-1, h.size(2), -1) # (B,T,C)
            h_t = torch.cat([h_t, g], dim=-1)            # (B,T,2C)

        return self.output_head(h_t).squeeze(-1)


class HybridGRFNet(nn.Module):
    def __init__(self, input_dim=12, num_filters=64, num_blocks=4, dropout=0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, 1),
            nn.BatchNorm1d(num_filters)
        )

        self.ms_blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.ms_blocks.append(
                nn.Sequential(
                    nn.Conv1d(num_filters, num_filters, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.temporal_attn = nn.Sequential(
            nn.Conv1d(num_filters, num_filters // 4, 1),
            nn.ReLU(),
            nn.Conv1d(num_filters // 4, 1, 1),
            nn.Sigmoid()
        )

        self.output_head = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, 1)
        )

    def forward(self, x):
        x_pad = _zero_pad(x)
        h = self.proj(x_pad.transpose(1, 2))
        for blk in self.ms_blocks:
            h = blk(h) + h
        attn = self.temporal_attn(h)
        h = h * attn
        return self.output_head(h.transpose(1, 2)).squeeze(-1)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
def masked_mse(pred, target):
    mask = (target != PADDING_VALUE)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[mask], target[mask])

def combined_loss(pred, target, alpha=0.85):
    """
    Your original style: corr term scaled by detached mse.
    """
    mask = (target != PADDING_VALUE)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    pred_masked = pred[mask]
    target_masked = target[mask]
    mse_loss = F.mse_loss(pred_masked, target_masked)

    pred_centered = pred_masked - pred_masked.mean()
    target_centered = target_masked - target_masked.mean()

    correlation = (pred_centered * target_centered).sum() / (
        torch.sqrt((pred_centered ** 2).sum()) *
        torch.sqrt((target_centered ** 2).sum()) + 1e-8
    )
    corr_loss = 1.0 - correlation
    total_loss = alpha * mse_loss + (1 - alpha) * corr_loss * mse_loss.detach()
    return total_loss


# =============================================================================
# TRAINING / EVAL
# =============================================================================
@torch.no_grad()
def eval_epoch(model, loader, device, normalizer=None):
    model.eval()
    preds, targs = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        mask = (y != PADDING_VALUE)

        pred_np = pred[mask].detach().cpu().numpy()
        targ_np = y[mask].detach().cpu().numpy()

        if normalizer is not None:
            pred_np = normalizer.inverse_transform(pred_np)
            targ_np = normalizer.inverse_transform(targ_np)

        preds.append(pred_np)
        targs.append(targ_np)

    preds = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    targs = np.concatenate(targs) if targs else np.array([], dtype=np.float32)
    return pearson_r(preds, targs), rmse(preds, targs)

def train_epoch(model, loader, opt, device, loss_fn):
    model.train()
    losses = []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else np.nan

def train_model(model, train_loader, val_loader, device, epochs, lr, patience,
                use_improved_loss=False, use_corr_loss=True, normalizer=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(1, patience // 2), verbose=False
    )

    if use_improved_loss and use_corr_loss:
        loss_fn = combined_loss
    else:
        loss_fn = masked_mse

    best_metric = -1e9
    best_state = None
    bad = 0

    for _epoch in range(epochs):
        train_epoch(model, train_loader, opt, device, loss_fn)
        r_val, rmse_val = eval_epoch(model, val_loader, device, normalizer)

        metric = r_val - 1e-6 * (rmse_val if np.isfinite(rmse_val) else 0.0)
        scheduler.step(r_val)

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =============================================================================
# BASELINES
# =============================================================================
def stack_timesteps(X_list, y_list):
    Xs = np.vstack([x for x in X_list]).astype(np.float32)
    ys = np.hstack([y for y in y_list]).astype(np.float32)
    return Xs, ys

def train_linear(X_list, y_list):
    X, y = stack_timesteps(X_list, y_list)
    model = RidgeCV(alphas=np.logspace(-3, 6, 20))
    model.fit(X, y)
    return model

def eval_linear(model, X_list, y_list):
    X, y = stack_timesteps(X_list, y_list)
    pred = model.predict(X).astype(np.float32)
    return pearson_r(pred, y), rmse(pred, y)

def train_xgb(X_list, y_list, seed):
    if not HAS_XGB:
        return None
    X, y = stack_timesteps(X_list, y_list)
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=8,
        tree_method="hist",
    )
    model.fit(X, y)
    return model

def eval_xgb(model, X_list, y_list):
    if model is None:
        return np.nan, np.nan
    X, y = stack_timesteps(X_list, y_list)
    pred = model.predict(X).astype(np.float32)
    return pearson_r(pred, y), rmse(pred, y)


# =============================================================================
# LOSO HELPERS
# =============================================================================
def choose_val_pid(unique_pids, test_pid, pids_train, preferred_next_pid):
    if preferred_next_pid != test_pid and any(p == preferred_next_pid for p in pids_train):
        return preferred_next_pid
    for pid in unique_pids:
        if pid != test_pid and any(p == pid for p in pids_train):
            return pid
    return None


# =============================================================================
# LOSO MAIN
# =============================================================================
def run_loso(X_all, y_all, pids, args):
    unique_pids = sorted(set(pids))
    results_rows = []

    print(f"\n[INFO] LOSO folds: {len(unique_pids)}")
    print("[INFO] Baselines: trained/evaluated on original scale")
    print("[INFO] Deep models: trained on normalized targets, evaluated on original scale")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, test_pid in enumerate(unique_pids, 1):
        print("\n" + "=" * 90)
        print(f"FOLD {fold_idx}/{len(unique_pids)} | TEST = {test_pid}")
        print("=" * 90)

        set_seed(args.seed + fold_idx)

        test_idx  = [i for i, p in enumerate(pids) if p == test_pid]
        train_idx = [i for i, p in enumerate(pids) if p != test_pid]

        next_pid = unique_pids[fold_idx % len(unique_pids)]
        val_pid = choose_val_pid(unique_pids, test_pid, [pids[i] for i in train_idx], next_pid)

        if val_pid is None:
            rng = np.random.default_rng(args.seed + fold_idx)
            rng.shuffle(train_idx)
            n_val = max(1, int(0.1 * len(train_idx)))
            val_idx = train_idx[:n_val]
            train_idx2 = train_idx[n_val:]
        else:
            val_idx = [i for i in train_idx if pids[i] == val_pid]
            train_idx2 = [i for i in train_idx if pids[i] != val_pid]
            if len(val_idx) == 0:
                # fallback
                for alt in unique_pids:
                    if alt != test_pid:
                        tmp = [i for i in train_idx if pids[i] == alt]
                        if len(tmp) > 0:
                            val_pid = alt
                            val_idx = tmp
                            train_idx2 = [i for i in train_idx if pids[i] != val_pid]
                            break

        X_train = [X_all[i] for i in train_idx2]
        y_train = [y_all[i] for i in train_idx2]
        X_val   = [X_all[i] for i in val_idx]
        y_val   = [y_all[i] for i in val_idx]
        X_test  = [X_all[i] for i in test_idx]
        y_test  = [y_all[i] for i in test_idx]

        # Normalizer fitted ONLY on training targets
        normalizer = TargetNormalizer()
        normalizer.fit(y_train)

        y_train_norm = normalizer.transform(y_train)
        y_val_norm   = normalizer.transform(y_val)
        y_test_norm  = normalizer.transform(y_test)

        print(f"[INFO] Train={len(X_train)} | Val({val_pid})={len(X_val)} | Test={len(X_test)}")
        print(f"[INFO] Target norm: mean={normalizer.mean:.2f}N, std={normalizer.std:.2f}N")

        # Baselines
        print("\n[Linear]")
        lin = train_linear(X_train, y_train)
        r_lin, e_lin = eval_linear(lin, X_test, y_test)
        print(f"  r={r_lin:.4f}, RMSE={e_lin:.2f}")

        print("\n[XGBoost]")
        xgbm = train_xgb(X_train, y_train, args.seed + fold_idx)
        r_xgb, e_xgb = eval_xgb(xgbm, X_test, y_test)
        print(f"  r={r_xgb:.4f}, RMSE={e_xgb:.2f}")

        # Deep datasets/loaders
        max_len = max(len(x) for x in (X_train + X_val + X_test))
        input_dim = X_train[0].shape[1]

        # Guard: no val
        if len(X_val) == 0:
            n_samples = max(1, min(len(X_train), args.batch_size))
            X_val = X_train[:n_samples]
            y_val_norm = y_train_norm[:n_samples]

        train_ds = StanceDataset(X_train, y_train_norm, max_len)
        val_ds   = StanceDataset(X_val,   y_val_norm,   max_len)
        test_ds  = StanceDataset(X_test,  y_test_norm,  max_len)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False)

        # Models (include MultiScale ablations)
        models_to_test = [
            ("LSTM",              lambda **kw: SimpleLSTM(**kw),              False, True),
            ("Transformer",       lambda **kw: ImprovedTransformer(**kw),     False, True),
            ("TCN",               lambda **kw: SimpleTCN(**kw),               False, True),
            ("GRFNet-Original",   lambda **kw: SimpleGRFNet(**kw),            False, True),
            ("GRFNet-Improved",   lambda **kw: ImprovedGRFNet(**kw),          True,  True),

            # ====== ABLATIONS YOU NEED ======
            ("GRFNet-MultiScale",                 lambda **kw: MultiScaleGRFNet(**kw, use_global_context=True,  use_dilations=True,  use_residual=True),  True,  True),
            ("GRFNet-MultiScale (w/o global)",    lambda **kw: MultiScaleGRFNet(**kw, use_global_context=False, use_dilations=True,  use_residual=True),  True,  True),
            ("GRFNet-MultiScale (w/o dilations)", lambda **kw: MultiScaleGRFNet(**kw, use_global_context=True,  use_dilations=False, use_residual=True),  True,  True),
            ("GRFNet-MultiScale (w/o residual)",  lambda **kw: MultiScaleGRFNet(**kw, use_global_context=True,  use_dilations=True,  use_residual=False), True,  True),
            ("GRFNet-MultiScale (MSE only)",      lambda **kw: MultiScaleGRFNet(**kw, use_global_context=True,  use_dilations=True,  use_residual=True),  True,  False),

            ("GRFNet-Hybrid",     lambda **kw: HybridGRFNet(**kw),            True,  True),
        ]

        for name, factory, use_loss, use_corr_loss in models_to_test:
            print(f"\n[{name}]")
            model = factory(input_dim=input_dim).to(args.device)

            model = train_model(
                model, train_loader, val_loader, args.device,
                epochs=args.epochs, lr=args.lr, patience=args.patience,
                use_improved_loss=use_loss, use_corr_loss=use_corr_loss,
                normalizer=normalizer
            )

            r, e = eval_epoch(model, test_loader, args.device, normalizer)
            print(f"  r={r:.4f}, RMSE={e:.2f}")

            results_rows.append({
                "fold": fold_idx, "test_pid": test_pid, "val_pid": val_pid,
                "model": name, "r": r, "rmse": e,
                "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
            })

        # Add baselines to results table
        for name, r_val, e_val in [("Linear", r_lin, e_lin), ("XGBoost", r_xgb, e_xgb)]:
            results_rows.append({
                "fold": fold_idx, "test_pid": test_pid, "val_pid": val_pid,
                "model": name, "r": r_val, "rmse": e_val,
                "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
            })

        # Save incrementally
        pd.DataFrame(results_rows).to_csv(out_dir / "loso_results_long.csv", index=False)

    return pd.DataFrame(results_rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", choices=["both", "waist", "wrist"], default="both")

    # NPZ export mode
    parser.add_argument("--export_npz", action="store_true",
                        help="Freeze dataset to NPZ (X,y,mask,pids,acts) and exit.")
    parser.add_argument("--npz_path", default=None,
                        help="Where to save NPZ (default: output_dir/frozen_dataset_loso_539.npz)")

    # training params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--contact_threshold", type=float, default=80.0)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--allowed_acts", nargs="+",
                        default=["walking", "jogging", "running", "heel", "drop"])

    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load trials
    X_all, y_all, pids, acts, imu_cols_used = load_data(args.aligned_dir, args)
    print(f"\n[INFO] Final dataset: {len(X_all)} trials | channels={X_all[0].shape[1]}")

    # 2) Optional: freeze NPZ and exit
    if args.export_npz:
        out_npz = args.npz_path
        if out_npz is None:
            out_npz = str(out_dir / "frozen_dataset_loso_539.npz")
        # Normalize feature names to the exact order used in X
        # (This is critical for SHAP correctness)
        feature_names = [c.strip() for c in imu_cols_used]
        freeze_to_npz(X_all, y_all, pids, acts, out_npz, feature_names=feature_names)

        print("[INFO] Exiting after NPZ export (no training).")
        return

    # 3) Train/eval LOSO
    df = run_loso(X_all, y_all, pids, args)
    df.to_csv(out_dir / "loso_results_long.csv", index=False)

    # 4) Summary
    summary_rows = []
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        r_mean, r_std = fisher_z_mean_std(sub["r"].values)
        summary_rows.append({
            "model": model,
            "r_mean_fisher": r_mean,
            "r_std_fisher": r_std,
            "rmse_mean": float(np.nanmean(sub["rmse"].values)),
            "rmse_std": float(np.nanstd(sub["rmse"].values, ddof=1)),
            "n_folds": int(len(sub)),
        })

    summary = pd.DataFrame(summary_rows).sort_values("r_mean_fisher", ascending=False)
    summary.to_csv(out_dir / "loso_summary.csv", index=False)

    print(f"\n✅ Done! Results saved to: {out_dir}")
    print("\n" + "="*90)
    print("FINAL SUMMARY (Fisher-z mean ± equiv std)")
    print("="*90)
    print(summary.to_string(index=False))

    print("\n" + "="*90)
    print("MULTISCALE ABLATIONS ONLY (copy into LaTeX table)")
    print("="*90)
    ab = summary[summary["model"].str.contains("GRFNet-MultiScale", regex=False)]
    if not ab.empty:
        print(ab.to_string(index=False))


if __name__ == "__main__":
    main()
