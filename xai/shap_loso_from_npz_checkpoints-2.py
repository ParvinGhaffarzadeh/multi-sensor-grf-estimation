#!/usr/bin/env python3
"""
shap_loso_beeswarm_stability_masked.py
=====================================

LOSO SHAP analysis for a padded (N,T,C) NPZ dataset.

Fixes vs. common pitfalls:
- Never feed -9999 padding into SHAP: we zero it before explaining.
- Scalar target is a MASKED mean over valid timesteps (ignores padding).
- Beeswarm + importance exclude padded timesteps using NPZ "mask".
- Random (seeded) background + sample selection per fold.
- Robust SHAP output shape normalization.
"""

import os
import re
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau


# =============================================================================
# CONFIG
# =============================================================================
PADDING_VALUE = -9999.0


# =============================================================================
# MODEL (must match training)
# =============================================================================
def _zero_pad(x: torch.Tensor, pad_value=PADDING_VALUE) -> torch.Tensor:
    x = x.clone()
    x[x == pad_value] = 0.0
    return x


class MultiScaleGRFNet(nn.Module):
    def __init__(
        self,
        input_dim=12,
        num_filters=64,
        num_blocks=4,
        dropout=0.15,
        use_global_context=True,
        use_dilations=True,
        use_residual=True,
    ):
        super().__init__()
        self.use_global_context = bool(use_global_context)
        self.use_dilations = bool(use_dilations)
        self.use_residual = bool(use_residual)

        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, 1),
            nn.BatchNorm1d(num_filters),
        )

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = (2 ** i) if self.use_dilations else 1
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        num_filters,
                        num_filters,
                        3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(
                        num_filters,
                        num_filters,
                        3,
                        padding=dilation,
                        dilation=dilation,
                    ),
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
            nn.Linear(num_filters, 1),
        )

    def forward(self, x):
        # x: (B,T,C)
        x_pad = _zero_pad(x)                         # -9999 -> 0
        h = self.proj(x_pad.transpose(1, 2))         # (B,C,T)

        for blk in self.blocks:
            out = blk(h)
            if self.use_residual:
                h = F.relu(out + h)
            else:
                h = F.relu(out)

        h_t = h.transpose(1, 2)                      # (B,T,C)

        if self.use_global_context:
            g = self.global_pool(h).squeeze(-1)              # (B,C)
            g = g.unsqueeze(1).expand(-1, h.size(2), -1)     # (B,T,C)
            h_t = torch.cat([h_t, g], dim=-1)                # (B,T,2C)

        return self.output_head(h_t).squeeze(-1)     # (B,T)


# =============================================================================
# SHAP WRAPPER (masked scalar target)
# =============================================================================
class ScalarWrapper(nn.Module):
    """
    Wrap a sequence model y=(B,T) into a scalar output (B,1)
    using a MASKED mean/sum over valid (non-padding) timesteps,
    so SHAP attributions are not biased by padding length.
    """

    def __init__(self, base_model: nn.Module, reduce="mean", pad_value=PADDING_VALUE):
        super().__init__()
        assert reduce in ("mean", "sum")
        self.base = base_model
        self.reduce = reduce
        self.pad_value = pad_value

    def forward(self, x):
        # valid timestep if NOT all channels are padding
        valid = ~(x == self.pad_value).all(dim=-1)  # (B,T)
        valid_f = valid.float()

        y = self.base(x)                            # (B,T) or (B,T,1)
        if y.dim() == 3:
            y = y.squeeze(-1)

        y = y * valid_f                             # zero-out padded timesteps

        if self.reduce == "sum":
            return y.sum(dim=1, keepdim=True)       # (B,1)

        denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return y.sum(dim=1, keepdim=True) / denom   # (B,1)


# =============================================================================
# HELPERS
# =============================================================================
def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_shap_shape(shap_values: np.ndarray) -> np.ndarray:
    """
    Make SHAP output consistent for sequence inputs: (B, T, C)
    Common returns:
      - (B, T, C)        ok
      - (B, T, C, 1)     -> squeeze last dim
      - (B, 1, T, C)     -> squeeze dim=1
    """
    if shap_values.ndim == 3:
        return shap_values

    if shap_values.ndim == 4:
        if shap_values.shape[-1] == 1:
            return np.squeeze(shap_values, axis=-1)
        if shap_values.shape[1] == 1:
            return np.squeeze(shap_values, axis=1)

    raise RuntimeError(f"Unexpected SHAP shape: {shap_values.shape}")


def beeswarm_plot(shap_flat, data_flat, outpath, feature_names, max_display=12):
    ensure_dir(os.path.dirname(outpath))
    shap.summary_plot(
        shap_flat,
        data_flat,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def rank_vector(importances: np.ndarray):
    order = np.argsort(-importances)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(importances) + 1)
    return ranks


def parse_fold_pid_from_path(path: str):
    fold = None
    pid = None
    m1 = re.search(r"fold[_\-]?(\d{1,2})", path, re.IGNORECASE)
    if m1:
        fold = int(m1.group(1))
    m2 = re.search(r"test[_\-]?(P\d{2})", path, re.IGNORECASE)
    if m2:
        pid = m2.group(1).upper()
    return fold, pid


def build_loso_indices(pids_list):
    uniq = sorted(set(pids_list))
    folds = []
    for pid in uniq:
        test_idx = np.array([i for i, p in enumerate(pids_list) if p == pid], dtype=int)
        train_idx = np.array([i for i, p in enumerate(pids_list) if p != pid], dtype=int)
        folds.append((pid, train_idx, test_idx))
    return folds


def compute_shap(model, background, samples, device, try_deep=True):
    """
    Returns:
      shap_values: (B,T,C)
      samples_np:  (B,T,C) (the samples actually explained)
    """
    model.eval()

    wrapped = ScalarWrapper(model, reduce="mean").to(device)
    wrapped.eval()

    # IMPORTANT: zero-pad BEFORE SHAP sees inputs
    background = _zero_pad(background).to(device)
    samples = _zero_pad(samples).to(device)

    if try_deep:
        try:
            explainer = shap.DeepExplainer(wrapped, background)
            shap_values = explainer.shap_values(samples, check_additivity=False)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = _normalize_shap_shape(to_numpy(shap_values))
            return shap_values, to_numpy(samples)
        except Exception as e:
            print("DeepExplainer failed → GradientExplainer. Reason:", str(e))

    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = explainer.shap_values(samples)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = _normalize_shap_shape(to_numpy(shap_values))
    return shap_values, to_numpy(samples)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to frozen_dataset_loso_539.npz")
    ap.add_argument("--ckpt_glob", required=True, help="Quoted glob to checkpoints, e.g. '/path/**/GRFNet_MultiScale.pt'")
    ap.add_argument("--outdir", required=True, help="Output directory for beeswarms + stability")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bg_n", type=int, default=100)
    ap.add_argument("--sample_n", type=int, default=50)
    ap.add_argument("--max_display", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    d = np.load(args.npz, allow_pickle=True)
    if "X" not in d or "pids" not in d or "mask" not in d:
        raise SystemExit("NPZ must contain keys: X, pids, mask (from your freeze script).")

    X = torch.from_numpy(d["X"]).float()           # (N,T,C)
    mask = torch.from_numpy(d["mask"]).bool()      # (N,T) True=valid
    pids = d["pids"].tolist()

    if "feature_names" in d:
        feature_names = d["feature_names"].tolist()
    else:
        # Fallback if missing
        C = X.shape[-1]
        feature_names = [f"ch_{i:02d}" for i in range(C)]
        print("[WARN] NPZ missing feature_names. Using generic channel names.")

    device = torch.device(args.device)
    ensure_dir(args.outdir)

    # Collect checkpoints
    ckpts = sorted(glob.glob(args.ckpt_glob, recursive=True))
    if len(ckpts) == 0:
        raise SystemExit(f"No checkpoints matched ckpt_glob: {args.ckpt_glob}")

    # Map pid->ckpt (prefer those with test_Pxx in path)
    pid_to_ckpt = {}
    for c in ckpts:
        _, pid = parse_fold_pid_from_path(c)
        if pid is not None and pid not in pid_to_ckpt:
            pid_to_ckpt[pid] = c

    folds = build_loso_indices(pids)

    all_shap_flat, all_data_flat = [], []
    fold_ranks = []
    mapping_lines = []

    for fold_i, (test_pid, train_idx, test_idx) in enumerate(folds, start=1):
        if test_pid not in pid_to_ckpt:
            ckpt_path = ckpts[fold_i - 1] if fold_i - 1 < len(ckpts) else ckpts[-1]
            warn = f"[WARN] No checkpoint tagged with {test_pid}. Using: {ckpt_path}"
            print(warn)
            mapping_lines.append(warn)
        else:
            ckpt_path = pid_to_ckpt[test_pid]
            mapping_lines.append(f"[OK] {test_pid} -> {ckpt_path}")

        print("\n" + "-" * 72)
        print(f"Fold {fold_i:02d} | TEST={test_pid} | CKPT={ckpt_path}")
        print("-" * 72)

        # Load model
        model = MultiScaleGRFNet(
            input_dim=X.shape[-1],
            use_global_context=True,
            use_dilations=True,
            use_residual=True,
        )
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        model = model.to(device)

        X_train = X[train_idx]
        X_test = X[test_idx]
        mask_test_full = mask[test_idx]  # (N_test, T)

        bg_n = min(args.bg_n, len(X_train))
        sm_n = min(args.sample_n, len(X_test))

        # Seeded random selection (important!)
        rng = np.random.default_rng(args.seed + fold_i)
        bg_idx = rng.choice(len(X_train), size=bg_n, replace=False) if bg_n > 0 else np.array([], dtype=int)
        sm_idx = rng.choice(len(X_test), size=sm_n, replace=False) if sm_n > 0 else np.array([], dtype=int)

        bg = X_train[bg_idx]
        sm = X_test[sm_idx]
        mask_sm = mask_test_full[sm_idx].cpu().numpy()  # (B,T) aligned with sm

        # Compute SHAP
        shap_btc, data_btc = compute_shap(model, bg, sm, device=device, try_deep=True)  # (B,T,C), (B,T,C)

        # Filter to valid timesteps only (exclude padding)
        valid_rows = np.where(mask_sm.reshape(-1))[0]
        shap_flat = shap_btc.reshape(-1, shap_btc.shape[-1])[valid_rows]
        data_flat = data_btc.reshape(-1, data_btc.shape[-1])[valid_rows]

        # Per-fold beeswarm
        out_png = os.path.join(args.outdir, f"beeswarm_fold_{fold_i:02d}_test_{test_pid}.png")
        beeswarm_plot(
            shap_flat,
            data_flat,
            outpath=out_png,
            feature_names=feature_names,
            max_display=args.max_display,
        )
        print(f"[OK] Saved {os.path.basename(out_png)}")

        all_shap_flat.append(shap_flat)
        all_data_flat.append(data_flat)

        # Per-fold importance, masked
        # abs(shap) over valid timesteps only
        abs_shap = np.abs(shap_btc)  # (B,T,C)
        imp = abs_shap[mask_sm].mean(axis=0)  # (C,)
        fold_ranks.append(rank_vector(imp))

    # Save mapping audit
    with open(os.path.join(args.outdir, "checkpoint_mapping.txt"), "w") as f:
        f.write("\n".join(mapping_lines) + "\n")

    # Aggregate beeswarm (all folds)
    agg_shap = np.vstack(all_shap_flat)
    agg_data = np.vstack(all_data_flat)
    agg_png = os.path.join(args.outdir, "beeswarm_AGGREGATE_ALL_FOLDS.png")
    beeswarm_plot(
        agg_shap,
        agg_data,
        outpath=agg_png,
        feature_names=feature_names,
        max_display=args.max_display,
    )
    print("[OK] Saved beeswarm_AGGREGATE_ALL_FOLDS.png")

    # Stability summary
    fold_ranks = np.array(fold_ranks)  # (F,C)
    Ff, C = fold_ranks.shape

    rhos, taus = [], []
    for i in range(Ff):
        for j in range(i + 1, Ff):
            rhos.append(spearmanr(fold_ranks[i], fold_ranks[j]).correlation)
            taus.append(kendalltau(fold_ranks[i], fold_ranks[j]).correlation)

    rhos = np.array(rhos, dtype=float)
    taus = np.array(taus, dtype=float)

    top1_idx = np.argmin(fold_ranks, axis=1)
    top1_freq = np.bincount(top1_idx, minlength=C) / Ff

    top3_freq = np.zeros(C, dtype=float)
    for f in range(Ff):
        top3 = np.argsort(fold_ranks[f])[:3]
        top3_freq[top3] += 1
    top3_freq /= Ff

    out_txt = os.path.join(args.outdir, "stability_summary.txt")
    with open(out_txt, "w") as f:
        f.write(f"Folds: {Ff}\n")
        f.write(f"Fold pairs: {len(rhos)}\n")
        f.write(f"Median Spearman rho: {np.nanmedian(rhos):.6f}\n")
        f.write(f"Mean Kendall tau: {np.nanmean(taus):.6f}\n\n")

        f.write("Top-1 frequency:\n")
        for name, val in zip(feature_names, top1_freq):
            f.write(f"{name}: {val * 100:.1f}%\n")

        f.write("\nTop-3 frequency:\n")
        for name, val in zip(feature_names, top3_freq):
            f.write(f"{name}: {val * 100:.1f}%\n")

    print(f"[OK] Saved {out_txt}")


if __name__ == "__main__":
    main()
