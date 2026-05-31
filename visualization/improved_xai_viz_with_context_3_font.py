#!/usr/bin/env python3
"""
improved_xai_viz_with_context_2.py
==================================
4x3 grid of subplots — one per IMU channel.
Each subplot:
  - 1-pixel-tall attribution heatmap (YlOrRd)
    stretched to fill the axes height
  - Phase background shading (Loading / Mid-stance / Push-off)
  - Black mean IMU signal overlaid (from --data_dir)
  - Per-channel colour scale (each channel uses its own vmax)

USAGE (per-activity):
  python3 improved_xai_viz_with_context_2.py \
    --smile_csv    /path/smile_temporal_agg.csv \
    --timeshap_csv /path/timeshap_eventwise_mean_abs.csv \
    --data_dir     /path/Dataset_Aligned \
    --outdir       /path/xai_visualizations

Per-activity (add --activity):
  python3 improved_xai_viz_with_context_2.py \
    --smile_csv  /path/smile_temporal_agg.csv \
    --data_dir   /path/Dataset_Aligned \
    --outdir     /path/xai_visualizations \
    --activity   walking
"""

import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# ── paper-quality font settings ───────────────────────────────────────────────
matplotlib.rcParams.update({
    # Use a clean serif font matching most IEEE/Elsevier papers.
    # Falls back to DejaVu Sans if Times is unavailable.
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          9,          # base size — all relative sizes scale from this
    "axes.titlesize":     10,         # subplot channel titles
    "axes.labelsize":     9,          # x/y axis labels
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.titlesize":   11,         # suptitle
    # Crisp rendering
    "pdf.fonttype":       42,         # embed fonts as TrueType in PDF
    "ps.fonttype":        42,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.minor.width":  0.4,
    "ytick.minor.width":  0.4,
    "lines.linewidth":    1.5,
    "axes.spines.top":    False,      # cleaner look — remove top/right spines
    "axes.spines.right":  False,
})

# ── constants ─────────────────────────────────────────────────────────────────
CHANNEL_LABELS = [
    "waist_accX","waist_accY","waist_accZ",
    "waist_gyroX","waist_gyroY","waist_gyroZ",
    "wrist_accX","wrist_accY","wrist_accZ",
    "wrist_gyroX","wrist_gyroY","wrist_gyroZ",
]
N_CH = 12
SENSOR_COLORS = ["#1a6fad"] * 6 + ["#d4622a"] * 6

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

ACTIVITY_ORDER = ["walking","jogging","running","heel_drop","step_down"]
ACTIVITY_DISPLAY = {
    "walking":"Walking","jogging":"Jogging","running":"Running",
    "heel_drop":"Heel drop","step_down":"Step drop",
}
STANCE_THR = {
    "walking":50.0,"jogging":80.0,"running":80.0,
    "heel_drop":40.0,"step_down":40.0,"unknown":50.0,
}
FORCE_COLS = ["force_z_N","Force_Z","force_z","ForceZ","Fz"]
IMU_SCHEMAS = [
    ["waist_accX","waist_accY","waist_accZ","waist_gyroX","waist_gyroY","waist_gyroZ",
     "wrist_accX","wrist_accY","wrist_accZ","wrist_gyroX","wrist_gyroY","wrist_gyroZ"],
    ["Waist_AccX","Waist_AccY","Waist_AccZ","Waist_GyroX","Waist_GyroY","Waist_GyroZ",
     "Wrist_AccX","Wrist_AccY","Wrist_AccZ","Wrist_GyroX","Wrist_GyroY","Wrist_GyroZ"],
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _infer_activity(stem):
    s = stem.lower()
    t = re.split(r"[^a-z0-9]+", s)
    if any(x.startswith("heel") for x in t): return "heel_drop"
    if any(x in ("stepdown","step") for x in t): return "step_down"
    if any(x.startswith("drop") for x in t): return "step_down"
    if any(x.startswith("walk") for x in t): return "walking"
    if any(x.startswith("jog")  for x in t): return "jogging"
    if any(x.startswith("run")  for x in t): return "running"
    return "unknown"

def _normalise(arr, T):
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 1: arr = arr[:,None]
    Ti, C = arr.shape
    if Ti == T: return arr.squeeze() if C==1 else arr
    src = np.linspace(0,1,Ti); dst = np.linspace(0,1,T)
    out = np.zeros((T,C),np.float32)
    for c in range(C):
        out[:,c] = np.interp(dst, src, arr[:,c])
    return out.squeeze() if C==1 else out

def _load_attr(csv_path, T_norm=100):
    df  = pd.read_csv(csv_path)
    try:
        mat = df[CHANNEL_LABELS].values.astype(np.float32)
    except KeyError:
        found = [c for c in CHANNEL_LABELS if c in df.columns]
        mat   = df[found].values.astype(np.float32)
    return np.abs(_normalise(mat, T_norm))   # (T_norm, 12)

def _load_signals(data_dir, T_norm=100, activity_filter=None):
    """Returns dict: activity -> list of (T_norm,12) IMU arrays."""
    records = defaultdict(list)
    csvs = [p for p in sorted(Path(data_dir).glob("*.csv"))
            if "alignment_log" not in p.name.lower()]
    for fp in tqdm(csvs, desc="Loading"):
        try: df = pd.read_csv(fp)
        except: continue
        fc = next((c for c in FORCE_COLS if c in df.columns), None)
        ic = next((s for s in IMU_SCHEMAS if all(c in df.columns for c in s)), None)
        if fc is None or ic is None: continue
        act = _infer_activity(fp.stem)
        if activity_filter and act != activity_filter: continue
        fz  = df[fc].values.astype(np.float32)
        imu = df[ic].values.astype(np.float32)
        thr = STANCE_THR.get(act, 50.0)
        mask = fz > thr
        if not mask.any(): continue
        idx  = np.where(mask)[0]
        gaps = np.diff(idx)
        segs, s = [], 0
        for gi in np.where(gaps > 5)[0]:
            segs.append(idx[s:gi+1]); s = gi+1
        segs.append(idx[s:])
        segs = [sg for sg in segs if len(sg) >= 5]
        if not segs: continue
        best = max(segs, key=lambda sg: fz[sg].max())
        on  = max(0, int(best[0])-2)
        off = min(len(fz)-1, int(best[-1])+2)
        records[act].append(_normalise(imu[on:off+1], T_norm))
    return dict(records)

# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_grid(attr_sm, mean_signals, act_label, n_trials,
               out_path, T_norm, sigma,
               method_name="SMILE"):           # FIX: added method_name param
    """
    4×3 grid — one subplot per IMU channel.
    Background: per-channel YlOrRd attribution heatmap.
    Line: mean IMU signal (black, white halo).

    Parameters
    ----------
    method_name : str
        "SMILE" or "TimeSHAP" — used in the figure subtitle so the
        TimeSHAP figures no longer incorrectly say "SMILE attribution".
    """
    pct = np.linspace(0, 100, T_norm)
    fig, axes = plt.subplots(4, 3, figsize=(18, 13), facecolor="white")

    # FIX: subtitle now reflects the actual method being plotted
    fig.suptitle(
        f"{method_name} Attribution + IMU Signal — {act_label}  "
        f"(n = {n_trials} trials · all participants)\n"
        f"Background = {method_name} attribution (per-channel scale) · "
        f"Black line = mean IMU signal",
        fontsize=13, fontweight="bold", y=1.002,
    )

    for ci, ax in enumerate(axes.flat):
        attr_ch = gaussian_filter1d(attr_sm[:, ci].astype(float), sigma)

        # ── per-channel colour scale ───────────────────────────────────────
        ch_vmax = float(np.percentile(attr_ch[np.isfinite(attr_ch)], 95))
        ch_vmax = max(ch_vmax, 1e-6)
        ch_norm = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=ch_vmax)

        bg = np.vstack([attr_ch, attr_ch])   # (2, T_norm)

        im = ax.imshow(
            bg, aspect="auto",
            cmap="YlOrRd", norm=ch_norm,
            origin="lower",
            extent=[0, 100, -1.5, 1.5],
            interpolation="bilinear", zorder=0,
        )

        # ── phase shading + boundaries ─────────────────────────────────────
        for ph, (t0, t1) in PHASES.items():
            ax.axvspan(t0*100, t1*100,
                       color=PHASE_COLORS[ph], alpha=0.18, zorder=1)
            if t0 > 0:
                ax.axvline(t0*100, color="#555555", lw=0.9,
                           ls="--", alpha=0.55, zorder=2)

        # phase labels (top row only)
        row_i = ci // 3
        if row_i == 0:
            for ph, (t0, t1) in PHASES.items():
                ax.text((t0+t1)/2*100, 1.42,
                        ph.replace("Mid-stance","Mid"),
                        ha="center", va="top",
                        fontsize=8, color="#333333",
                        style="italic", fontweight="bold", zorder=5,
                        bbox=dict(facecolor="white", alpha=0.6,
                                  pad=1.5, edgecolor="none"))

        # ── mean signal ────────────────────────────────────────────────────
        if mean_signals is not None:
            sig = gaussian_filter1d(mean_signals[:, ci].astype(float), sigma)
            # z-score → clip to ±3 → map to [-1, 1]
            mu, sd = sig.mean(), sig.std()
            sd = max(sd, 1e-6)
            sig_z = np.clip((sig - mu) / sd, -3, 3) / 3.0
            ax.plot(pct, sig_z, color="white", lw=3.2,
                    alpha=0.65, zorder=3, solid_capstyle="round")
            ax.plot(pct, sig_z, color="black", lw=1.7,
                    alpha=0.95, zorder=4, solid_capstyle="round")
            # y=0 reference
            ax.axhline(0, color="white", lw=0.5, ls=":",
                       alpha=0.4, zorder=1)

        # ── titles + axes ──────────────────────────────────────────────────
        sensor_col = "#d4622a" if CHANNEL_LABELS[ci].startswith("wrist") \
                     else "#1a6fad"
        ax.set_title(CHANNEL_LABELS[ci], fontsize=10,
                     fontweight="bold", color=sensor_col, pad=4)
        ax.set_xlim(0, 100)
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([r"$-1\sigma$", "0", r"$+1\sigma$"], fontsize=8)

        col_i = ci % 3
        if row_i == 3:
            ax.set_xlabel("Stance phase (%)", fontsize=9)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.set_xticklabels(
                [f"{x}%" for x in np.arange(0,101,20)], fontsize=8)
        else:
            ax.set_xticks([])

        ax.grid(True, axis="x", alpha=0.15, lw=0.5, ls="--")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#cccccc")

        # per-subplot colorbar
        cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.01)
        cb.set_label("Attribution", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.3)

    # ── global legend ──────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=PHASE_COLORS[p], alpha=0.5, label=p) for p in PHASES
    ] + [
        Line2D([0],[0], color="black", lw=2, label="Mean IMU signal (z-score)"),
        Line2D([0],[0], color="#d4622a", lw=2, label="Wrist channels"),
        Line2D([0],[0], color="#1a6fad", lw=2, label="Waist channels"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(
        left=0.06, right=0.97,
        top=0.95, bottom=0.06,
        hspace=0.44, wspace=0.32,
    )
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smile_csv",    required=True)
    ap.add_argument("--timeshap_csv", default=None)
    ap.add_argument("--data_dir",     default=None)
    ap.add_argument("--outdir",       required=True)
    ap.add_argument("--activity",     default=None,
                    choices=ACTIVITY_ORDER + [None],
                    help="Single activity (omit for all 5)")
    ap.add_argument("--T_norm", type=int,   default=100)
    ap.add_argument("--sigma",  type=float, default=1.5)
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load SMILE
    smile_tc = _load_attr(args.smile_csv, T_norm=args.T_norm)
    print(f"[INFO] SMILE: {smile_tc.shape}  vmax={smile_tc.max():.4f}")

    # load TimeSHAP (optional)
    ts_tc = None
    if args.timeshap_csv:
        ts_tc = _load_attr(args.timeshap_csv, T_norm=args.T_norm)
        print(f"[INFO] TimeSHAP: {ts_tc.shape}  vmax={ts_tc.max():.4f}")

    # load signals
    signals_by_act = {}
    if args.data_dir:
        signals_by_act = _load_signals(
            args.data_dir, T_norm=args.T_norm,
            activity_filter=args.activity)

    # determine which activities to plot
    acts = [args.activity] if args.activity else ACTIVITY_ORDER

    for act in acts:
        recs = signals_by_act.get(act, [])
        n_t  = len(recs)
        lbl  = ACTIVITY_DISPLAY.get(act, act)

        # mean signal across trials
        if recs:
            mean_imu = np.stack(recs).mean(0)  # (T_norm,12)
        else:
            mean_imu = None
            print(f"[WARN] No signal data for {act} — heatmap only")

        # ── SMILE figure ──────────────────────────────────────────────────
        _plot_grid(
            attr_sm      = smile_tc,
            mean_signals = mean_imu,
            act_label    = lbl,
            n_trials     = n_t,
            out_path     = out_dir / f"smile_{act}.png",
            T_norm       = args.T_norm,
            sigma        = args.sigma,
            method_name  = "SMILE",        # FIX: explicit method label
        )

        # ── TimeSHAP figure (optional) ────────────────────────────────────
        if ts_tc is not None:
            _plot_grid(
                attr_sm      = ts_tc,           # FIX: was accidentally ts_tc anyway,
                mean_signals = mean_imu,        #      but now method_name is correct
                act_label    = lbl,             # FIX: removed redundant "(TimeSHAP)"
                n_trials     = n_t,             #      from act_label — it's in the
                out_path     = out_dir / f"timeshap_{act}.png",  # subtitle now
                T_norm       = args.T_norm,
                sigma        = args.sigma,
                method_name  = "TimeSHAP",     # FIX: correct method label
            )

    print(f"\n✅ All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
