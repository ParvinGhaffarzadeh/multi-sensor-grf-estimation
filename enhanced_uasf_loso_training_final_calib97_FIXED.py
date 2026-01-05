#!/usr/bin/env python3
"""
ENHANCED UASF-GRFNet (REAL DATA) — LOSO CV with Uncertainty & Biomechanical Priors
==================================================================================

REAL DATA ONLY:
- Loads your aligned CSVs
- Extracts stance from force threshold
- Pads sequences to max length (padding is zeros + mask, NOT synthetic data)
- Trains EnhancedUASF-GRFNet with heteroscedastic NLL + biomech priors + (optional) activity CE

Key fixes:
1) Robust activity parsing for v3.7 filenames (e.g. P01_Walk_ID1_walk0001_f_8_aligned.csv)
2) No forced 'unknown' class unless it actually exists
3) Safer class weights:
   - weight=0 for classes absent in the TRAIN fold
   - sqrt-inverse frequency (less extreme than pure inverse)
4) Optional merge_heel_to_walk (recommended when heel count is tiny)

Author: Parvin Ghaffarzadeh
"""

from __future__ import annotations

import os
import re
import random
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy.optimize import minimize_scalar

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from enhanced_uasf_model_final import (
    EnhancedUASF_GRFNet,
    BiomechTargets,
    BiomechanicalPriors,
    combined_training_loss,
    masked_mse,
    validate_robustness,
)

warnings.filterwarnings("ignore")


# =============================================================================
# A) Participant mass mapping (kg) for BW normalization
# =============================================================================
PID_TO_MASS_KG: Dict[str, float] = {
    "P01": 95.8,
    "P02": 74.5,
    "P03": 67.8,
    "P04": 68.6,
    "P05": 74.0,
    "P06": 68.2,
    "P07": 85.2,
    "P08": 68.8,
    "P09": 60.2,
    "P10": 76.4,
}
G_CONST = 9.81


# =============================================================================
# 0) Reproducibility
# =============================================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 1) Stance extraction
# =============================================================================
def extract_stance_phase(
    waist: np.ndarray,
    wrist: np.ndarray,
    force_N: np.ndarray,
    contact_threshold: float = 100.0,
    min_stance_length: int = 30,
    max_stance_length: int = 500,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    trials_w, trials_r, trials_f = [], [], []
    contact = force_N > contact_threshold

    in_stance = False
    stance_start = 0
    for i in range(len(contact)):
        if contact[i] and not in_stance:
            stance_start = i
            in_stance = True
        elif (not contact[i]) and in_stance:
            stance_end = i
            L = stance_end - stance_start
            if min_stance_length <= L <= max_stance_length:
                trials_w.append(waist[stance_start:stance_end])
                trials_r.append(wrist[stance_start:stance_end])
                trials_f.append(force_N[stance_start:stance_end])
            in_stance = False

    return trials_w, trials_r, trials_f


# =============================================================================
# 2) Robust activity parsing for v3.7
# =============================================================================
ACT_SET = {"walk", "jog", "run", "heel", "drop"}

def parse_pid_act_v37(stem: str, merge_heel_to_walk: bool = True) -> Tuple[str, str]:
    """
    Examples:
      P01_Walk_ID1_walk0001_f_8_aligned
      P02_drop0003_f_2_aligned
      P03_Jog_ID7_jog0002_f_5_aligned

    Returns: (pid, act)
    """
    parts = stem.split("_")
    pid = "unknown"
    act = "unknown"

    # PID: first token like P01
    if parts and re.match(r"^P\d+$", parts[0], re.IGNORECASE):
        pid = parts[0].upper()

    # Prefer 2nd token if it is exactly Walk/Jog/Run/Heel/Drop
    if len(parts) >= 2:
        p1 = parts[1].lower()
        if p1 in ACT_SET:
            act = p1

    # Otherwise search any token that starts with walk/jog/run/heel/drop
    if act == "unknown":
        for p in parts:
            m = re.match(r"^(walk|jog|run|heel|drop)", p, re.IGNORECASE)
            if m:
                act = m.group(1).lower()
                break

    if merge_heel_to_walk and act == "heel":
        act = "walk"

    return pid, act


# =============================================================================
# 3) Data loader
# =============================================================================
def load_aligned_trials_real(
    aligned_dir: str,
    extract_stance: bool = True,
    participant_ids: Optional[List[str]] = None,
    contact_threshold: float = 100.0,
    min_stance_length: int = 30,
    max_stance_length: int = 500,
    merge_heel_to_walk: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str], List[str]]:

    aligned_dir = Path(aligned_dir)
    csv_files = sorted(list(aligned_dir.glob("*.csv")))

    if participant_ids is not None:
        csv_files = [f for f in csv_files if any(pid in f.stem for pid in participant_ids)]

    waist_cols = ["waist_accX", "waist_accY", "waist_accZ", "waist_gyroX", "waist_gyroY", "waist_gyroZ"]
    wrist_cols = ["wrist_accX", "wrist_accY", "wrist_accZ", "wrist_gyroX", "wrist_gyroY", "wrist_gyroZ"]

    all_waist, all_wrist, all_force, all_pids, all_acts = [], [], [], [], []

    print(f"Loading {len(csv_files)} CSVs from: {aligned_dir}")

    for f in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        pid, act = parse_pid_act_v37(f.stem, merge_heel_to_walk=merge_heel_to_walk)

        # columns check
        if not all(c in df.columns for c in waist_cols + wrist_cols):
            continue

        # force column
        if "force_z_N" in df.columns:
            force_N = df["force_z_N"].values.astype(np.float32)
        elif "vGRF" in df.columns:
            force_N = df["vGRF"].values.astype(np.float32)
        else:
            continue

        waist = df[waist_cols].values.astype(np.float32)
        wrist = df[wrist_cols].values.astype(np.float32)

        if extract_stance:
            tw, tr, tf = extract_stance_phase(
                waist, wrist, force_N,
                contact_threshold=contact_threshold,
                min_stance_length=min_stance_length,
                max_stance_length=max_stance_length,
            )
            for w_seg, r_seg, f_seg in zip(tw, tr, tf):
                all_waist.append(w_seg)
                all_wrist.append(r_seg)
                all_force.append(f_seg)
                all_pids.append(pid)
                all_acts.append(act)
        else:
            all_waist.append(waist)
            all_wrist.append(wrist)
            all_force.append(force_N)
            all_pids.append(pid)
            all_acts.append(act)

    print(f"\n✅ Loaded {len(all_waist)} stance trials")
    if len(all_force) > 0:
        concat_force = np.concatenate(all_force, axis=0)
        print(f"   Force stats (N): mean={concat_force.mean():.1f}, std={concat_force.std():.1f}, max={concat_force.max():.1f}")

    if len(all_acts) > 0:
        vc = pd.Series(all_acts).value_counts().to_dict()
        print(f"   Activity counts: {vc}")

    return all_waist, all_wrist, all_force, all_pids, all_acts


# =============================================================================
# 4) Dataset
# =============================================================================
class EnhancedUASFDataset(Dataset):
    def __init__(
        self,
        waist_list: List[np.ndarray],
        wrist_list: List[np.ndarray],
        force_list: List[np.ndarray],
        pids: List[str],
        acts: List[str],
        act2idx: Dict[str, int],
        max_length: int,
        normalize: bool = True,
        force_scale: str = "BW",
        pid_to_mass_kg: Optional[Dict[str, float]] = None,
    ):
        assert len(waist_list) == len(wrist_list) == len(force_list) == len(pids) == len(acts)
        self.waist_list = waist_list
        self.wrist_list = wrist_list
        self.force_list = force_list
        self.pids = pids
        self.acts = acts
        self.act2idx = act2idx
        self.max_length = int(max_length)
        self.normalize = normalize
        self.force_scale = force_scale
        self.pid_to_mass_kg = pid_to_mass_kg or {}

        print(f"Dataset: {len(self.waist_list)} trials, max_length={self.max_length}, force_scale={self.force_scale}")

    def __len__(self) -> int:
        return len(self.waist_list)

    def __getitem__(self, idx: int):
        waist = self.waist_list[idx].astype(np.float32)
        wrist = self.wrist_list[idx].astype(np.float32)
        force = self.force_list[idx].astype(np.float32)

        # Per-trial z-score normalization
        if self.normalize:
            waist = (waist - waist.mean(axis=0)) / (waist.std(axis=0) + 1e-8)
            wrist = (wrist - wrist.mean(axis=0)) / (wrist.std(axis=0) + 1e-8)

        # Force scaling
        scale = self.force_scale.lower()
        pid = self.pids[idx]
        if scale == "kn":
            force = force / 1000.0
        elif scale == "n":
            pass
        elif scale == "bw":
            mass = float(self.pid_to_mass_kg.get(pid, 1.0))
            if not np.isfinite(mass) or mass <= 0:
                mass = 1.0
            force = force / (mass * G_CONST)
        else:
            raise ValueError("force_scale must be 'kN', 'N', or 'BW'")

        T = int(force.shape[0])

        waist_t = torch.tensor(waist, dtype=torch.float32)
        wrist_t = torch.tensor(wrist, dtype=torch.float32)
        force_t = torch.tensor(force, dtype=torch.float32)

        # Padding (zeros) + mask = not synthetic signal (model ignores padding via mask)
        if T < self.max_length:
            pad_len = self.max_length - T
            waist_t = F.pad(waist_t, (0, 0, 0, pad_len))
            wrist_t = F.pad(wrist_t, (0, 0, 0, pad_len))
            force_t = F.pad(force_t, (0, pad_len))

        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:T] = True

        act = self.acts[idx]
        act_idx = int(self.act2idx.get(act, 0))  # 0 is safe fallback

        return waist_t, wrist_t, force_t, mask, act_idx, pid


# =============================================================================
# 5) Metrics
# =============================================================================
def compute_batch_r(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> List[float]:
    pred_cpu = pred.detach().cpu()
    target_cpu = target.detach().cpu()
    mask_cpu = mask.detach().cpu()
    rs = []
    for i in range(pred_cpu.size(0)):
        p = pred_cpu[i, mask_cpu[i]].numpy()
        t = target_cpu[i, mask_cpu[i]].numpy()
        if len(p) > 10:
            r, _ = stats.pearsonr(p, t)
            if not np.isnan(r):
                rs.append(float(r))
    return rs


def interval_metrics_95(
    pred: torch.Tensor,
    total_u: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    z = 1.96
    m = mask.bool()
    low = pred - z * total_u
    high = pred + z * total_u

    inside = ((target >= low) & (target <= high) & m).sum().item()
    total = m.sum().item()

    width = (high - low)
    mpiw = width[m].mean().item() if total > 0 else 0.0
    picp = inside / max(total, 1)
    return float(picp), float(mpiw)


def compute_picp_mpiw_from_tensors(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    z_score: float = 1.96,
) -> Tuple[float, float]:
    """Compute PICP and MPIW from tensors.

    This helper exists because some calibration blocks want quick interval
    metrics without going through eval_epoch().

    Notes:
        - All tensors should be shape (B, T) (or broadcastable).
        - `masks` should be 0/1 or bool with same shape.
        - `uncertainties` is assumed to be 1-sigma (std). Intervals are
          [pred ± z_score * std].
    """
    # Ensure boolean mask
    m = masks.bool()
    if m.sum().item() == 0:
        return 0.0, 0.0

    low = predictions - z_score * uncertainties
    high = predictions + z_score * uncertainties
    inside = ((targets >= low) & (targets <= high) & m).sum().item()
    total = m.sum().item()

    width = (high - low)
    mpiw = width[m].mean().item()
    picp = inside / max(total, 1)
    return float(picp), float(mpiw)




# =============================================================================
# 5b) Temperature scaling for uncertainty calibration (post-hoc)
# =============================================================================
def calibrate_temperature(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_picp: float = 0.95,
    z_score: float = 1.96,
    bounds: Tuple[float, float] = (0.1, 3.0),
) -> float:
    """Find scalar temperature T to scale uncertainties so PICP matches target_picp."""

    def compute_picp(temp: float) -> float:
        scaled = uncertainties * temp
        low = predictions - z_score * scaled
        high = predictions + z_score * scaled
        m = masks.bool()
        covered = ((targets >= low) & (targets <= high) & m).sum().item()
        total = m.sum().item()
        return covered / max(total, 1)

    def objective(temp: float) -> float:
        return abs(compute_picp(temp) - target_picp)

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    T = float(res.x)
    return T


# =============================================================================
# Robust Temperature Calibration (Clipped T + Optional Monotonic Search)
# =============================================================================

def calibrate_temperature_monotonic(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_picp: float = 0.95,
    z_score: float = 1.96,
    bounds: Tuple[float, float] = (0.1, 3.0),
    tol: float = 0.01,
    max_iter: int = 50,
    min_samples: int = 50,
) -> Tuple[float, str]:
    """
    Monotonic binary search temperature calibration.

    PICP(T) is (almost always) monotonic increasing in T, so binary search is
    often more stable than minimize_scalar when data are small/noisy.

    Returns:
        (T, status) where status ∈ {'calibrated', 'bounds_hit', 'insufficient_data'}
    """

    def compute_picp(temp: float) -> float:
        scaled = uncertainties * temp
        low = predictions - z_score * scaled
        high = predictions + z_score * scaled
        m = masks.bool()
        covered = ((targets >= low) & (targets <= high) & m).sum().item()
        total = m.sum().item()
        return covered / max(total, 1)

    n_valid = int(masks.sum().item())
    if n_valid < min_samples:
        return 1.0, "insufficient_data"

    T_low, T_high = float(bounds[0]), float(bounds[1])
    picp_low = compute_picp(T_low)
    picp_high = compute_picp(T_high)

    # If target not achievable within bounds, clip to boundary (preserve direction).
    if target_picp <= picp_low:
        return T_low, "bounds_hit"
    if target_picp >= picp_high:
        return T_high, "bounds_hit"

    for _ in range(max_iter):
        T_mid = 0.5 * (T_low + T_high)
        picp_mid = compute_picp(T_mid)
        if abs(picp_mid - target_picp) < tol:
            return T_mid, "calibrated"
        if picp_mid < target_picp:
            T_low = T_mid
        else:
            T_high = T_mid
        if abs(T_high - T_low) < 1e-4:
            break

    return 0.5 * (T_low + T_high), "calibrated"


def calibrate_temperature_robust(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_picp: float = 0.95,
    z_score: float = 1.96,
    bounds: Tuple[float, float] = (0.1, 3.0),
    min_samples: int = 50,
    use_monotonic: bool = True,
) -> Tuple[float, str]:
    """
    Robust temperature calibration with safeguards.

    Returns:
        (T, status) where status ∈ {'calibrated', 'bounds_hit', 'insufficient_data'}

    Notes:
        - If bounds are hit, we *keep* the clipped T (do NOT revert to 1.0),
          because it preserves the information that the validation set requires
          wider/narrower intervals than allowed by bounds.
    """

    n_valid = int(masks.sum().item())
    if n_valid < min_samples:
        return 1.0, "insufficient_data"

    if use_monotonic:
        return calibrate_temperature_monotonic(
            predictions, uncertainties, targets, masks,
            target_picp=target_picp,
            z_score=z_score,
            bounds=bounds,
            tol=0.01,
            max_iter=50,
            min_samples=min_samples,
        )

    # Fallback: scipy bounded scalar optimisation
    def compute_picp(temp: float) -> float:
        scaled = uncertainties * temp
        low = predictions - z_score * scaled
        high = predictions + z_score * scaled
        m = masks.bool()
        covered = ((targets >= low) & (targets <= high) & m).sum().item()
        total = m.sum().item()
        return covered / max(total, 1)

    def objective(temp: float) -> float:
        return abs(compute_picp(temp) - target_picp)

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    T = float(res.x)
    if abs(T - bounds[0]) < 0.01 or abs(T - bounds[1]) < 0.01:
        return T, "bounds_hit"
    return T, "calibrated"


def calibrate_with_pooling(
    model: "EnhancedUASF_GRFNet",
    val_loaders_list: List[DataLoader],
    device: str,
    target_picp: float = 0.95,
    min_samples: int = 50,
    use_monotonic: bool = True,
) -> Tuple[float, str, int]:
    """
    Optional helper: pool multiple validation loaders to stabilise calibration
    when single-fold validation is small.

    Returns:
        (T, status, n_valid_samples)
    """
    model.eval()
    preds, uncs, tgts, masks_list = [], [], [], []
    with torch.no_grad():
        for loader in val_loaders_list:
            for waist, wrist, target, mask, act_idx, _ in loader:
                waist = waist.to(device)
                wrist = wrist.to(device)
                target = target.to(device)
                mask = mask.to(device)
                out = model(waist, wrist, return_all=True)
                preds.append(out["prediction"].cpu())
                uncs.append(out["total_uncertainty"].cpu())
                tgts.append(target.cpu())
                masks_list.append(mask.cpu())

    preds = torch.cat(preds, dim=0)
    uncs = torch.cat(uncs, dim=0)
    tgts = torch.cat(tgts, dim=0)
    masks_cat = torch.cat(masks_list, dim=0)

    T, status = calibrate_temperature_robust(
        preds, uncs, tgts, masks_cat,
        target_picp=target_picp,
        min_samples=min_samples,
        use_monotonic=use_monotonic,
    )
    return T, status, int(masks_cat.sum().item())


@torch.no_grad()
def collect_for_calibration(
    model: EnhancedUASF_GRFNet,
    loader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect (pred, unc, target, mask) on CPU for temperature calibration."""
    model.eval()
    preds, uncs, tgts, masks = [], [], [], []
    for waist, wrist, target, mask, act_idx, _ in loader:
        waist = waist.to(device)
        wrist = wrist.to(device)
        target = target.to(device)
        mask = mask.to(device)
        out = model(waist, wrist, return_all=True)
        preds.append(out["prediction"].detach().cpu())
        uncs.append(out["total_uncertainty"].detach().cpu())
        tgts.append(target.detach().cpu())
        masks.append(mask.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(uncs, dim=0), torch.cat(tgts, dim=0), torch.cat(masks, dim=0)



# =============================================================================
# 6) Train / Eval Loops
# =============================================================================
def train_epoch(
    model: EnhancedUASF_GRFNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    priors: BiomechanicalPriors,
    class_weights: Optional[torch.Tensor],
    lambda_activity: float,
    beta_uncertainty: float,
) -> Dict[str, float]:

    model.train()
    totals = {"total": 0.0, "nll": 0.0, "mse": 0.0, "priors": 0.0, "act": 0.0}

    for waist, wrist, target, mask, act_idx, _ in tqdm(loader, desc="Training", leave=False):
        waist = waist.to(device, non_blocking=True)
        wrist = wrist.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        act_idx = act_idx.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(waist, wrist, return_all=True)

        loss, loss_dict = combined_training_loss(
            model_outputs=out,
            target=target,
            activity_labels=act_idx,
            mask=mask,
            priors=priors,
            class_weights=class_weights,
            lambda_activity=lambda_activity,
            beta_uncertainty=beta_uncertainty,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        totals["total"] += float(loss_dict["total"].item())
        totals["nll"] += float(loss_dict["nll"].item())
        totals["mse"] += float(loss_dict["mse"].item())
        totals["priors"] += float(loss_dict["priors_total"].item())
        totals["act"] += float(loss_dict["activity"].item())

    n = max(len(loader), 1)
    for k in totals:
        totals[k] /= n
    return totals


@torch.no_grad()
def eval_epoch(
    model: EnhancedUASF_GRFNet,
    loader: DataLoader,
    device: str,
    compute_weights: bool = True,
    compute_interval: bool = True,
    temperature: float = 1.0,
) -> Dict[str, float]:

    model.eval()
    mse_sum = 0.0
    rs: List[float] = []

    w_list = []
    correct = 0
    total_act = 0

    picp_list = []
    mpiw_list = []

    for waist, wrist, target, mask, act_idx, _ in loader:
        waist = waist.to(device)
        wrist = wrist.to(device)
        target = target.to(device)
        mask = mask.to(device)
        act_idx = act_idx.to(device)

        out = model(waist, wrist, return_all=True)
        pred = out["prediction"]

        mse = masked_mse(pred, target, mask)
        mse_sum += float(mse.item())
        rs.extend(compute_batch_r(pred, target, mask))

        if compute_weights:
            w_list.append(out["fusion_weights"].detach().cpu().numpy())

        probs = out["activity_probs"]
        yhat = torch.argmax(probs, dim=-1)
        correct += int((yhat == act_idx).sum().item())
        total_act += int(act_idx.numel())

        if compute_interval:
            picp, mpiw = interval_metrics_95(pred, out["total_uncertainty"] * float(temperature), target, mask)
            picp_list.append(picp)
            mpiw_list.append(mpiw)

    mean_mse = mse_sum / max(len(loader), 1)
    mean_r = float(np.mean(rs)) if len(rs) else 0.0

    w_mean = None
    if compute_weights and len(w_list):
        W = np.concatenate(w_list, axis=0)
        w_mean = W.mean(axis=0).tolist()

    act_acc = correct / max(total_act, 1)
    picp95 = float(np.mean(picp_list)) if len(picp_list) else 0.0
    mpiw95 = float(np.mean(mpiw_list)) if len(mpiw_list) else 0.0

    return {
        "mse": float(mean_mse),
        "r": float(mean_r),
        "w_waist": float(w_mean[0]) if w_mean is not None else float("nan"),
        "w_wrist": float(w_mean[1]) if w_mean is not None else float("nan"),
        "act_acc": float(act_acc),
        "PICP95": float(picp95),
        "MPIW95": float(mpiw95),
    }


# =============================================================================
# 7) LOSO Utilities
# =============================================================================
def create_loso_splits(unique_pids: List[str]) -> List[Tuple[str, str, List[str]]]:
    splits = []
    for i, test_pid in enumerate(unique_pids):
        val_pid = unique_pids[(i + 1) % len(unique_pids)]
        train_pids = [p for p in unique_pids if p not in [test_pid, val_pid]]
        splits.append((test_pid, val_pid, train_pids))
    return splits


def compute_class_weights(
    act_indices: List[int],
    n_classes: int,
    smoothing: float = 0.0,
    max_weight: float = 5.0,
    min_weight: float = 0.2,
) -> torch.Tensor:
    """
    Safer weights:
      - unseen classes in TRAIN -> weight=0 (ignored)
      - sqrt-inverse frequency (less extreme than inverse)
      - normalize present classes to mean 1
      - clamp to [min_weight, max_weight]
    """
    counts = np.bincount(np.array(act_indices, dtype=np.int64), minlength=n_classes).astype(np.float32)

    w = np.zeros_like(counts, dtype=np.float32)
    present = counts > 0

    w[present] = 1.0 / np.sqrt(counts[present])
    w[present] = w[present] / (w[present].mean() + 1e-8)

    if smoothing > 0:
        w[present] = (1.0 - smoothing) * w[present] + smoothing * np.ones_like(w[present])

    w[present] = np.clip(w[present], min_weight, max_weight)
    # unseen stay 0
    return torch.tensor(w, dtype=torch.float32)


# =============================================================================
# 8) Training Configuration
# =============================================================================
@dataclass
class TrainConfig:
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 5e-4
    weight_decay: float = 1e-2
    max_epochs: int = 150
    patience: int = 30
    print_every: int = 5

    n_filters: int = 128
    dropout: float = 0.15

    # strongly suggest small activity weight (your main goal is GRF)
    lambda_activity: float = 0.01
    beta_uncertainty: float = 0.2

    # post-hoc calibration for prediction intervals (does NOT change predictions)
    use_temperature_scaling: bool = False
    target_picp: float = 0.95

    lambda_impulse: float = 0.1
    lambda_temporal: float = 0.05
    lambda_magnitude: float = 0.05

    target_mean_bw: float = 1.0
    peak_cap_bw: float = 3.0

    target_mean_kn: float = 0.70
    peak_cap_kn: float = 4.50

    class_weight_clip: float = 5.0
    class_weight_min: float = 0.2
    class_weight_smoothing: float = 0.0


# =============================================================================
# 9) LOSO runner
# =============================================================================
def run_loso(
    data_dir: str,
    save_dir: str,
    device: str,
    seed: int,
    cfg: TrainConfig,
    extract_stance: bool = True,
    force_scale: str = "BW",
    merge_heel_to_walk: bool = True,
) -> pd.DataFrame:

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENHANCED UASF-GRFNet (REAL DATA) — LOSO CV")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data:   {data_dir}")
    print(f"Save:   {save_dir}")
    print(f"Stance: {'ON' if extract_stance else 'OFF'} | Force: {force_scale}")
    print()

    waist_all, wrist_all, force_all, pids_all, acts_all = load_aligned_trials_real(
        data_dir,
        extract_stance=extract_stance,
        participant_ids=None,
        merge_heel_to_walk=merge_heel_to_walk,
    )

    unique_pids = sorted(list(set(pids_all)))
    print(f"✅ Participants ({len(unique_pids)}): {unique_pids}")

    # build activity set WITHOUT forcing unknown unless it exists
    act_counts = Counter(acts_all)
    unique_acts = sorted(act_counts.keys())

    # If somehow unknown is present with 0 count (rare), remove it
    if "unknown" in act_counts and act_counts["unknown"] == 0:
        unique_acts = [a for a in unique_acts if a != "unknown"]

    act2idx = {a: i for i, a in enumerate(unique_acts)}
    idx2act = {i: a for a, i in act2idx.items()}
    n_acts = len(unique_acts)
    print(f"✅ Activities ({n_acts}): {unique_acts}")

    global_max_len = max([w.shape[0] for w in waist_all]) if len(waist_all) else 0
    print(f"✅ Global max_length: {global_max_len}")
    print()

    splits = create_loso_splits(unique_pids)
    results_rows = []

    for fold_i, (test_pid, val_pid, train_pids) in enumerate(splits, start=1):
        print("\n" + "=" * 80)
        print(f"FOLD {fold_i:02d}/{len(splits)} | test={test_pid} | val={val_pid} | train={len(train_pids)} pids")
        print("=" * 80)

        train_idx = [i for i, pid in enumerate(pids_all) if pid in train_pids]
        val_idx = [i for i, pid in enumerate(pids_all) if pid == val_pid]
        test_idx = [i for i, pid in enumerate(pids_all) if pid == test_pid]


        # ---- Per-fold activity distribution diagnostics (paper-friendly) ----
        def _count_acts(idxs: List[int]) -> Dict[str, int]:
            return dict(Counter([acts_all[i] for i in idxs]))

        train_act_counts = _count_acts(train_idx)
        val_act_counts = _count_acts(val_idx)
        test_act_counts = _count_acts(test_idx)

        print(f"\n📊 Fold {fold_i:02d} Activity Distribution:")
        print(f"   Train: {train_act_counts}")
        print(f"   Val:   {val_act_counts}")
        print(f"   Test:  {test_act_counts}")

        missing_in_train = set(unique_acts) - set(train_act_counts.keys())
        if missing_in_train:
            print(f"   ⚠️  Missing in train: {sorted(list(missing_in_train))}")

        train_ds = EnhancedUASFDataset(
            [waist_all[i] for i in train_idx],
            [wrist_all[i] for i in train_idx],
            [force_all[i] for i in train_idx],
            [pids_all[i] for i in train_idx],
            [acts_all[i] for i in train_idx],
            act2idx=act2idx,
            max_length=global_max_len,
            normalize=True,
            force_scale=force_scale,
            pid_to_mass_kg=PID_TO_MASS_KG,
        )
        val_ds = EnhancedUASFDataset(
            [waist_all[i] for i in val_idx],
            [wrist_all[i] for i in val_idx],
            [force_all[i] for i in val_idx],
            [pids_all[i] for i in val_idx],
            [acts_all[i] for i in val_idx],
            act2idx=act2idx,
            max_length=global_max_len,
            normalize=True,
            force_scale=force_scale,
            pid_to_mass_kg=PID_TO_MASS_KG,
        )
        test_ds = EnhancedUASFDataset(
            [waist_all[i] for i in test_idx],
            [wrist_all[i] for i in test_idx],
            [force_all[i] for i in test_idx],
            [pids_all[i] for i in test_idx],
            [acts_all[i] for i in test_idx],
            act2idx=act2idx,
            max_length=global_max_len,
            normalize=True,
            force_scale=force_scale,
            pid_to_mass_kg=PID_TO_MASS_KG,
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, pin_memory=True)

        # Class weights from TRAIN only
        train_act_idx = [act2idx[acts_all[i]] for i in train_idx]
        cw = compute_class_weights(
            train_act_idx,
            n_classes=n_acts,
            smoothing=cfg.class_weight_smoothing,
            max_weight=cfg.class_weight_clip,
            min_weight=cfg.class_weight_min,
        )
        cw_map = {idx2act[i]: float(cw[i].item()) for i in range(n_acts)}
        print(f"Activity CE weights: {cw_map}")
        cw = cw.to(device)

        seed_everything(seed)
        model = EnhancedUASF_GRFNet(n_filters=cfg.n_filters, dropout=cfg.dropout, n_activities=n_acts).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n✅ Model params: {total_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        fs = force_scale.lower()
        if fs == "bw":
            biomech_targets = BiomechTargets(target_mean=cfg.target_mean_bw, peak_cap=cfg.peak_cap_bw)
        elif fs == "kn":
            biomech_targets = BiomechTargets(target_mean=cfg.target_mean_kn, peak_cap=cfg.peak_cap_kn)
        elif fs == "n":
            biomech_targets = BiomechTargets(target_mean=cfg.target_mean_kn * 1000.0, peak_cap=cfg.peak_cap_kn * 1000.0)
        else:
            raise ValueError("force_scale must be BW, kN, or N")

        priors = BiomechanicalPriors(
            lambda_impulse=cfg.lambda_impulse,
            lambda_temporal=cfg.lambda_temporal,
            lambda_magnitude=cfg.lambda_magnitude,
            targets=biomech_targets,
        ).to(device)

        best_val_r = -1.0
        best_path = str(Path(save_dir) / f"fold{fold_i:02d}_test{test_pid}_val{val_pid}.pt")
        patience_ctr = 0

        for epoch in range(1, cfg.max_epochs + 1):
            tr = train_epoch(
                model, train_loader, optimizer, device,
                priors=priors,
                class_weights=cw,
                lambda_activity=cfg.lambda_activity,
                beta_uncertainty=cfg.beta_uncertainty,
            )
            va = eval_epoch(model, val_loader, device, compute_weights=False, compute_interval=False)
            scheduler.step(va["mse"])

            if va["r"] > best_val_r:
                best_val_r = va["r"]
                patience_ctr = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_ctr += 1

            if epoch == 1 or epoch % cfg.print_every == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train: total={tr['total']:.4f} nll={tr['nll']:.4f} mse={tr['mse']:.4f} priors={tr['priors']:.4f} act={tr['act']:.4f} | "
                    f"val: mse={va['mse']:.4f} r={va['r']:.4f}"
                )

            if patience_ctr >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (best val r={best_val_r:.4f})")
                break

        print(f"✅ Fold best val r: {best_val_r:.4f}")

        
        model.load_state_dict(torch.load(best_path, map_location=device))
        te = eval_epoch(model, test_loader, device, compute_weights=True, compute_interval=True, temperature=1.0)


        # Optional: calibrate uncertainty on validation set, then report calibrated intervals on test
        te_picp_cal = te["PICP95"]
        te_mpiw_cal = te["MPIW95"]
        temp_T = 1.0
        cal_status = "not_used"

        if getattr(cfg, "use_temperature_scaling", False):
            try:
                vp, vu, vt, vm = collect_for_calibration(model, val_loader, device)
                n_val_samples = int(vm.sum().item())
                print(f"   📊 Validation samples: {n_val_samples}")

                target_picp = float(getattr(cfg, "target_picp", 0.97))

                temp_T, cal_status = calibrate_temperature_robust(
                    vp, vu, vt, vm,
                    target_picp=target_picp,
                    bounds=(0.1, 3.0),
                    min_samples=50,
                    use_monotonic=True,
                )

                # Post-check: ensure validation coverage is at least target_picp (conservative)
                if cal_status != "insufficient_data":
                    val_picp, val_mpiw = compute_picp_mpiw_from_tensors(vp, vu * temp_T, vt, vm, z_score=1.96)
                    if val_picp + 1e-6 < target_picp:
                        T_lo, T_hi = float(temp_T), 3.0
                        picp_hi, _ = compute_picp_mpiw_from_tensors(vp, vu * T_hi, vt, vm, z_score=1.96)
                        if picp_hi + 1e-6 < target_picp:
                            temp_T = T_hi
                            cal_status = "bounds_hit"
                        else:
                            for _ in range(30):
                                T_mid = 0.5 * (T_lo + T_hi)
                                picp_mid, _ = compute_picp_mpiw_from_tensors(vp, vu * T_mid, vt, vm, z_score=1.96)
                                if picp_mid + 1e-6 < target_picp:
                                    T_lo = T_mid
                                else:
                                    T_hi = T_mid
                            temp_T = T_hi
                            cal_status = f"{cal_status}_inflated"
                        val_picp, val_mpiw = compute_picp_mpiw_from_tensors(vp, vu * temp_T, vt, vm, z_score=1.96)

                    print(
                        f"   ✅ Val calibration check: target={target_picp:.3f} | "
                        f"Val PICP95={val_picp:.3f} | Val MPIW95={val_mpiw:.3f}"
                    )

                if cal_status == "insufficient_data":
                    # Not enough validation points → do not scale
                    temp_T = 1.0
                    te_picp_cal = te["PICP95"]
                    te_mpiw_cal = te["MPIW95"]
                    print(f"   ⚠️  Insufficient validation data, skipping calibration (T=1.0)")
                else:
                    # IMPORTANT: apply temperature even when bounds are hit (clipped T preserves direction)
                    te_cal = eval_epoch(
                        model, test_loader, device,
                        compute_weights=False,
                        compute_interval=True,
                        temperature=temp_T,
                    )
                    te_picp_cal = te_cal["PICP95"]
                    te_mpiw_cal = te_cal["MPIW95"]

                    if cal_status == "bounds_hit":
                        print(f"   🌡️  Temp scaling: T={temp_T:.3f} ({cal_status}) ⚠️ | "
                              f"Test PICP95={te_picp_cal:.3f} | Test MPIW95={te_mpiw_cal:.3f}")
                    else:
                        print(f"   🌡️  Temp scaling: T={temp_T:.3f} ({cal_status}) | "
                              f"Test PICP95={te_picp_cal:.3f} | Test MPIW95={te_mpiw_cal:.3f}")

            except Exception as e:
                print(f"   ⚠️  Temperature scaling failed: {e}")
                temp_T = 1.0
                te_picp_cal = te["PICP95"]
                te_mpiw_cal = te["MPIW95"]
                cal_status = "error"

        print(
            f"🧪 TEST | mse={te['mse']:.4f} r={te['r']:.4f} | "
            f"weights=[{te['w_waist']:.3f},{te['w_wrist']:.3f}] | "
            f"act_acc={te['act_acc']:.3f} | PICP95={te['PICP95']:.3f} | MPIW95={te['MPIW95']:.3f} "
            f"(PID={test_pid})"
        )
        print(f"💾 Saved fold model: {best_path}")

        # Robustness
        try:
            batch = next(iter(test_loader))
            waist_b, wrist_b = batch[0].to(device), batch[1].to(device)
            rob = validate_robustness(model, waist_b, wrist_b, confidence_level=0.95)
            print(f"🛡️ Robustness scenarios: {list(rob.keys())}")
        except Exception as e:
            print(f"⚠️ Robustness validation skipped: {e}")

        results_rows.append({
            "fold": fold_i,
            "test_pid": test_pid,
            "val_pid": val_pid,
            "test_r": te["r"],
            "test_mse": te["mse"],
            "w_waist": te["w_waist"],
            "w_wrist": te["w_wrist"],
            "act_acc": te["act_acc"],
            "PICP95": te["PICP95"],
            "MPIW95": te["MPIW95"],
            "temp_T": temp_T,
            "PICP95_cal": te_picp_cal,
            "MPIW95_cal": te_mpiw_cal,
            "cal_status": cal_status,
            "model_path": Path(best_path).as_posix(),
            "n_acts": n_acts,
        })

        pd.DataFrame(results_rows).to_csv(Path(save_dir) / "loso_partial_results.csv", index=False)

    df = pd.DataFrame(results_rows)
    df.to_csv(Path(save_dir) / "loso_final_results.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL LOSO SUMMARY")
    print("=" * 80)
    show_cols = ["test_pid", "val_pid", "test_r", "test_mse", "PICP95", "MPIW95", "PICP95_cal", "MPIW95_cal", "act_acc"]
    print(df[show_cols].to_string(index=False))
    print("-" * 80)
    print(f"Mean test r:   {df['test_r'].mean():.4f} ± {df['test_r'].std(ddof=0):.4f}")
    print(f"Mean test MSE: {df['test_mse'].mean():.4f} ± {df['test_mse'].std(ddof=0):.4f}")
    print(f"Mean PICP95:   {df['PICP95'].mean():.4f} ± {df['PICP95'].std(ddof=0):.4f}")
    print(f"Mean MPIW95:   {df['MPIW95'].mean():.4f} ± {df['MPIW95'].std(ddof=0):.4f}")
    if 'PICP95_cal' in df.columns and df['PICP95_cal'].notna().any():
        print(f"Mean PICP95_cal: {df['PICP95_cal'].mean():.4f}")
    if 'MPIW95_cal' in df.columns and df['MPIW95_cal'].notna().any():
        print(f"Mean MPIW95_cal: {df['MPIW95_cal'].mean():.4f}")
    if 'cal_status' in df.columns:
        counts = df['cal_status'].value_counts(dropna=False).to_dict()
        print(f"Calibration status counts: {counts}")
    print(f"Mean act_acc:  {df['act_acc'].mean():.4f} ± {df['act_acc'].std(ddof=0):.4f}")
    print(f"Saved: {Path(save_dir) / 'loso_final_results.csv'}")
    print("=" * 80)

    return df


# =============================================================================
# 10) CLI / Main
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Enhanced UASF-GRFNet (real data) — LOSO CV")
    p.add_argument("--data_dir", type=str, required=True, help="Folder of aligned CSVs")
    p.add_argument("--save_dir", type=str, default="results/enhanced_uasf")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--extract_stance", action="store_true", help="Enable stance extraction")
    p.add_argument("--no_extract_stance", action="store_true", help="Disable stance extraction")

    p.add_argument("--force_scale", type=str, default="BW", choices=["BW", "kN", "N"])

    p.add_argument("--merge_heel_to_walk", action="store_true", help="Merge heel into walk (recommended)")
    p.add_argument("--no_merge_heel_to_walk", action="store_true", help="Keep heel as a separate class")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=4)

    # optionally expose activity weight
    p.add_argument("--lambda_activity", type=float, default=0.01)

    p.add_argument("--beta_uncertainty", type=float, default=0.2)
    p.add_argument("--calibrate_temperature", action="store_true",
                   help="Post-hoc temperature scaling on VAL to target PICP (does not change predictions)")
    p.add_argument("--target_picp", type=float, default=0.95)

    return p


def main():
    args, _ = build_argparser().parse_known_args()
    seed_everything(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    extract_stance = True
    if args.no_extract_stance:
        extract_stance = False
    elif args.extract_stance:
        extract_stance = True

    merge_heel_to_walk = True
    if args.no_merge_heel_to_walk:
        merge_heel_to_walk = False
    elif args.merge_heel_to_walk:
        merge_heel_to_walk = True

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        lambda_activity=args.lambda_activity,
        beta_uncertainty=args.beta_uncertainty,
        use_temperature_scaling=bool(args.calibrate_temperature),
        target_picp=float(args.target_picp),
    )

    run_loso(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        device=device,
        seed=args.seed,
        cfg=cfg,
        extract_stance=extract_stance,
        force_scale=args.force_scale,
        merge_heel_to_walk=merge_heel_to_walk,
    )


if __name__ == "__main__":
    main()
