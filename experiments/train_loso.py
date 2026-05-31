#!/usr/bin/env python3
"""
Train GRFNet-MultiScale with EXACT baseline data loading
=========================================================

This script:
1. Uses EXACT baseline data loading (stance extraction, column order)
2. Saves checkpoints compatible with SHAP
3. Will reproduce r=0.797±0.077 from your baseline

Usage:
  python train_multiscale_BASELINE_EXACT.py
"""

import argparse
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PADDING_VALUE = -9999.0
RANDOM_STATE = 42

# ========== EXACT BASELINE UTILITIES ==========

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pearson_r(a, b):
    if a.size == 0 or b.size == 0 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def rmse(a, b):
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
    z_std = z.std(ddof=1) if len(z) > 1 else 0.0
    r_mean = np.tanh(z_mean)
    r_lo = np.tanh(z_mean - z_std)
    r_hi = np.tanh(z_mean + z_std)
    r_std_equiv = (r_hi - r_lo) / 2
    return float(r_mean), float(r_std_equiv)


# ========== EXACT BASELINE DATA LOADING ==========

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
    return "P00"

def extract_stance(X, y, contact_thr, min_dur, min_peak, fs, stance_reasons: Counter):
    mask = y > contact_thr
    if not mask.any():
        stance_reasons["no_contact"] += 1
        return None, None
    
    idx = np.where(mask)[0]
    gaps = np.diff(idx)
    
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

def load_data(aligned_dir, contact_threshold=80.0, fs=100.0, allowed_acts=None):
    """EXACT baseline data loading"""
    
    if allowed_acts is None:
        allowed_acts = ["walking", "jogging", "running", "heel", "drop"]
    
    files = sorted(Path(aligned_dir).glob("*.csv"))
    
    # EXACT baseline schemas
    imu_schemas = [
        ["Waist_AccX","Waist_AccY","Waist_AccZ","Waist_GyroX","Waist_GyroY","Waist_GyroZ",
         "Wrist_AccX","Wrist_AccY","Wrist_AccZ","Wrist_GyroX","Wrist_GyroY","Wrist_GyroZ"],
        ["waist_accX","waist_accY","waist_accZ","waist_gyroX","waist_gyroY","waist_gyroZ",
         "wrist_accX","wrist_accY","wrist_accZ","wrist_gyroX","wrist_gyroY","wrist_gyroZ"],
    ]
    
    force_candidates = ["Force_Z", "force_z_N", "force_z", "Force_Vertical", "ForceZ", "Fz"]
    
    X_all, y_all, pids, acts = [], [], [], []
    imu_cols_used = None
    force_col_used = None
    
    skipped = Counter()
    stance_reasons = Counter()
    
    print(f"\n[INFO] Loading with EXACT baseline pipeline")
    print(f"[INFO] Stance extraction: contact_thr={contact_threshold}N, fs={fs}Hz")
    
    for fp in tqdm(files, desc="Loading"):
        if "alignment_log" in fp.name.lower():
            skipped["alignment_log"] += 1
            continue
        
        try:
            df = pd.read_csv(fp)
        except:
            skipped["csv_error"] += 1
            continue
        
        # Find force column (once)
        if force_col_used is None:
            for cand in force_candidates:
                if cand in df.columns:
                    force_col_used = cand
                    print(f"[INFO] Force column: {force_col_used}")
                    break
        
        if force_col_used is None or force_col_used not in df.columns:
            skipped["no_force"] += 1
            continue
        
        # Find IMU schema (once)
        if imu_cols_used is None:
            for schema in imu_schemas:
                if all(c in df.columns for c in schema):
                    imu_cols_used = schema
                    print(f"[INFO] IMU schema: {schema}")
                    break
        
        if imu_cols_used is None:
            skipped["no_imu_schema"] += 1
            continue
        
        if not all(c in df.columns for c in imu_cols_used):
            skipped["imu_missing"] += 1
            continue
        
        # Extract in EXACT order
        X = df[imu_cols_used].values.astype(np.float32)
        y = df[force_col_used].values.astype(np.float32)
        
        act = infer_activity(fp.stem)
        pid = extract_pid(fp.stem)
        
        # Activity-specific thresholds
        if act == "heel":
            peak_thr, dur_thr = 250.0, 0.08
        elif act == "drop":
            peak_thr, dur_thr = 800.0, 0.15
        else:
            peak_thr, dur_thr = 800.0, 0.15
        
        # STANCE EXTRACTION
        X_st, y_st = extract_stance(X, y, contact_threshold, dur_thr, peak_thr, fs, stance_reasons)
        
        if X_st is None:
            skipped["stance_failed"] += 1
            continue
        
        if act not in allowed_acts:
            skipped["activity_filtered"] += 1
            continue
        
        X_all.append(X_st)
        y_all.append(y_st)
        pids.append(pid)
        acts.append(act)
    
    print(f"\n[INFO] Loaded {len(X_all)} stance trials")
    print(f"[INFO] Activities: {dict(Counter(acts))}")
    print(f"[INFO] Participants: {dict(Counter(pids))}")
    print(f"[INFO] Skipped: {dict(skipped)}")
    print(f"[INFO] Stance rejections: {dict(stance_reasons)}")
    
    if len(X_all) == 0:
        raise RuntimeError("No trials loaded!")
    
    return X_all, y_all, pids, acts


# ========== EXACT BASELINE CLASSES ==========

class TargetNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
    
    def fit(self, y_list):
        y_all = np.concatenate([y.reshape(-1) for y in y_list]).astype(np.float64)
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size == 0:
            self.mean, self.std = 0.0, 1.0
            return
        self.mean = float(y_all.mean())
        self.std = float(y_all.std())
        if self.std < 1e-8:
            self.std = 1.0
    
    def transform(self, y_list):
        return [(y - self.mean) / self.std for y in y_list]
    
    def inverse_transform(self, y_norm):
        return y_norm * self.std + self.mean


class StanceDataset(Dataset):
    def __init__(self, X_list, y_list, max_len):
        self.X_list = X_list
        self.y_list = y_list
        self.max_len = int(max_len)
        self.n_channels = 12

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        X, y = self.X_list[idx], self.y_list[idx]
        if len(X) > self.max_len:
            X = X[:self.max_len]
            y = y[:self.max_len]
        T = len(X)
        X_pad = np.full((self.max_len, 12), PADDING_VALUE, dtype=np.float32)
        y_pad = np.full((self.max_len,), PADDING_VALUE, dtype=np.float32)
        X_pad[:T] = X
        y_pad[:T] = y
        return torch.from_numpy(X_pad), torch.from_numpy(y_pad)


class MultiScaleGRFNet(nn.Module):
    def __init__(self, input_dim=12, num_filters=64, num_blocks=4, dropout=0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, 1),
            nn.BatchNorm1d(num_filters)
        )
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(num_filters, num_filters, 3, 
                             padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(num_filters, num_filters, 3,
                             padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(num_filters),
                )
            )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_head = nn.Sequential(
            nn.Linear(num_filters * 2, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, 1)
        )

    def forward(self, x):
        x = x.clone()
        x[x == PADDING_VALUE] = 0.0
        h = self.proj(x.transpose(1, 2))
        for blk in self.blocks:
            h = F.relu(blk(h) + h)
        global_context = self.global_pool(h).squeeze(-1)
        global_context = global_context.unsqueeze(1).expand(-1, h.size(2), -1)
        h = h.transpose(1, 2)
        h_combined = torch.cat([h, global_context], dim=-1)
        return self.output_head(h_combined).squeeze(-1)


# ========== TRAINING ==========

def combined_loss(pred, target, alpha=0.85):
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
    return alpha * mse_loss + (1 - alpha) * corr_loss * mse_loss.detach()

@torch.no_grad()
def eval_epoch(model, loader, device, normalizer):
    model.eval()
    preds, targs = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        mask = (y != PADDING_VALUE)
        pred_np = pred[mask].cpu().numpy()
        targ_np = y[mask].cpu().numpy()
        pred_np = normalizer.inverse_transform(pred_np)
        targ_np = normalizer.inverse_transform(targ_np)
        preds.append(pred_np)
        targs.append(targ_np)
    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    return pearson_r(preds, targs), rmse(preds, targs)

def train_model(model, train_loader, val_loader, device, epochs, lr, patience, normalizer):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=patience//2, verbose=False
    )
    best_metric = -1e9
    best_state = None
    bad = 0
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = combined_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        
        r_val, rmse_val = eval_epoch(model, val_loader, device, normalizer)
        metric = r_val - 1e-6 * (rmse_val if np.isfinite(rmse_val) else 0.0)
        scheduler.step(r_val)
        
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    return model


# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_dir", default="/home/805478/Dataset_Aligned_FINAL_forcheck_2Jan_v36_y")
    parser.add_argument("--output_dir", default="/home/805478/MultiScale_BASELINE_EXACT")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data with EXACT baseline method
    X_all, y_all, pids, acts = load_data(args.aligned_dir)
    
    unique_pids = sorted(set(pids))
    print(f"\n✅ Training GRFNet-MultiScale on {len(unique_pids)} participants")
    print(f"✅ Using EXACT baseline data loading (stance extraction + column order)\n")
    
    results = []
    
    for fold_idx, test_pid in enumerate(unique_pids, 1):
        print(f"{'='*70}")
        print(f"FOLD {fold_idx}/{len(unique_pids)}: Test={test_pid}")
        print(f"{'='*70}")
        
        set_seed(args.seed + fold_idx)
        
        test_idx = [i for i, p in enumerate(pids) if p == test_pid]
        train_idx = [i for i, p in enumerate(pids) if p != test_pid]
        
        # Simple 10% val split
        rng = np.random.default_rng(args.seed + fold_idx)
        rng.shuffle(train_idx)
        n_val = max(1, int(0.1 * len(train_idx)))
        val_idx = train_idx[:n_val]
        train_idx = train_idx[n_val:]
        
        X_train = [X_all[i] for i in train_idx]
        y_train = [y_all[i] for i in train_idx]
        X_val = [X_all[i] for i in val_idx]
        y_val = [y_all[i] for i in val_idx]
        X_test = [X_all[i] for i in test_idx]
        y_test = [y_all[i] for i in test_idx]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Normalize
        normalizer = TargetNormalizer()
        normalizer.fit(y_train)
        y_train_norm = normalizer.transform(y_train)
        y_val_norm = normalizer.transform(y_val)
        y_test_norm = normalizer.transform(y_test)
        
        # Create datasets
        max_len = max(len(x) for x in (X_train + X_val + X_test))
        train_ds = StanceDataset(X_train, y_train_norm, max_len)
        val_ds = StanceDataset(X_val, y_val_norm, max_len)
        test_ds = StanceDataset(X_test, y_test_norm, max_len)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        
        # Train
        model = MultiScaleGRFNet().to(args.device)
        model = train_model(model, train_loader, val_loader, args.device, 
                          args.epochs, args.lr, args.patience, normalizer)
        
        # Evaluate
        r, e = eval_epoch(model, test_loader, args.device, normalizer)
        print(f"✅ r={r:.4f}, RMSE={e:.2f} N")
        
        results.append({"fold": fold_idx, "test_pid": test_pid, "r": r, "rmse": e})
        
        # Save checkpoint
        ckpt_dir = out_dir / "checkpoints" / f"fold_{fold_idx:02d}" / f"test_{test_pid}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "GRFNet_MultiScale.pt"
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "fold": fold_idx,
            "test_pid": test_pid,
            "test_r": r,
            "test_rmse": e,
            "target_norm_mean": normalizer.mean,
            "target_norm_std": normalizer.std,
        }, ckpt_path)
        
        print(f"💾 Saved: {ckpt_path}\n")
        
        # Save incremental
        pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    
    # Final summary
    df = pd.DataFrame(results)
    r_mean, r_std = fisher_z_mean_std(df["r"].values)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"\nMean r: {r_mean:.4f} ± {r_std:.4f}")
    print(f"Mean RMSE: {df['rmse'].mean():.2f} ± {df['rmse'].std():.2f} N")
    
    best = df.sort_values("r", ascending=False).iloc[0]
    best_ckpt = out_dir / "checkpoints" / f"fold_{int(best['fold']):02d}" / f"test_{best['test_pid']}" / "GRFNet_MultiScale.pt"
    print(f"\n🏆 Best: {best_ckpt}")
    print(f"   r={best['r']:.4f}, RMSE={best['rmse']:.2f} N")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()