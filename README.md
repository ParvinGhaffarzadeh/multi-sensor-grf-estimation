# Edge-First Ground Reaction Force Estimation with Consumer Smartwatches

[![IEEE Internet Computing](https://img.shields.io/badge/IEEE%20Internet%20Computing-2026-blue)](https://www.computer.org/csdl/magazine/ic)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://zenodo.org/records/17376717)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

This repository contains the official implementation for:

> **Edge-First Ground Reaction Force Estimation with Consumer Smartwatches**  
> Parvin Ghaffarzadeh, Debarati Chakraborty, Koorosh Aslansefat, Ali Dostan, Yiannis Papadopoulos  
> *IEEE Internet Computing*, 2026  
> Dataset DOI: [10.5281/zenodo.17376717](https://doi.org/10.5281/zenodo.17376717) (CC-BY-4.0)

The system estimates vertical ground reaction force (vGRF) from consumer Apple Watch IMU data using a compact temporal convolutional network (GRFNet-MultiScale), with inference running locally on an iPhone — no cloud connectivity required.

---

## Key Contributions

- **End-to-end edge pipeline**: dual Apple Watch Series 6 → iPhone-side preprocessing, storage, and inference via WatchConnectivity
- **GRFNet-MultiScale**: compact 1D dilated residual TCN (d∈{1,2,4,8}, k=3, global context branch); ~1.2M parameters (dual-sensor), ~605K (wrist-only)
- **Strict LOSO evaluation**: 10-fold leave-one-subject-out on 539 stance windows from 10 participants
- **Reproducible temporal XAI**: Temporal SMILE and TimeSHAP both identify early-stance wrist acceleration as the dominant signal; ranking stability confirmed across all 45 pairwise fold comparisons (median Spearman ρ=0.755, mean Kendall τ=0.589)

---

## Results Summary

### Predictive Performance (10-fold LOSO)

| Configuration | Pearson r | RMSE |
|---|---|---|
| Dual-sensor (wrist + waist) | 0.798 | 257 N |
| Wrist-only | 0.658 | — |
| Waist-only | — | — |

Wrist-only retains **82.5%** of dual-sensor correlation, providing a practical smartwatch-only fallback.

### Activity-Stratified Performance

| Activity | r (LOSO) |
|---|---|
| Jogging | 0.867 |
| Running | 0.829 |
| Walking | 0.560 |
| Step-down (20 cm) | 0.442 |
| Heel drop | 0.301 |

The validated deployment region is **cyclic locomotion** (walking, jogging, running). Heel-drop-like impulsive tasks are not currently supported.

### Derived Metrics

| Metric | r |
|---|---|
| Impulse | 0.983 |
| Effective contact duration | 0.982 |
| Peak vGRF | 0.741 |
| Loading rate | 0.588 |

The system is best suited for **longitudinal load monitoring** rather than exact peak-force assessment.

---

## Installation

### Option 1: Conda (Recommended)
```bash
git clone https://github.com/ParvinGhaffarzadeh/multi-sensor-grf-estimation.git
cd multi-sensor-grf-estimation

conda create -n grfnet python=3.11
conda activate grfnet

pip install -r requirements.txt
```

### Option 2: Virtual Environment
```bash
git clone https://github.com/ParvinGhaffarzadeh/multi-sensor-grf-estimation.git
cd multi-sensor-grf-estimation

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Verify
```python
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "from model.grfnet_multiscale import MultiScaleGRFNet; print('Model imported OK')"
```

---

## Quick Start

### Load Pretrained Model
```python
import torch
from model.grfnet_multiscale import MultiScaleGRFNet

# Dual-sensor model (~1.2M parameters)
model = MultiScaleGRFNet(input_dim=12)
model.load_state_dict(torch.load('pretrained/GRFNet_MultiScale.pt'))
model.eval()
print("Model loaded")
```

### Predict vGRF
```python
import torch

# Input: (batch, 198 timesteps, 12 channels) — wrist + waist IMU at 100 Hz
x = torch.randn(1, 198, 12)

with torch.no_grad():
    vgrf_pred = model(x)  # Output: (1, 198, 1) — predicted vGRF waveform

print(f"Prediction shape: {vgrf_pred.shape}")
```

### Run XAI Pipeline (Temporal SMILE + TimeSHAP)
```bash
# Full SMILE + TimeSHAP pipeline across all 10 LOSO folds
python xai/timeshap_smile_pipeline.py smile \
    --data_dir /path/to/Dataset_Aligned_FINAL \
    --ckpt_glob "pretrained/checkpoints/**/GRFNet_MultiScale.pt" \
    --outdir results/xai/ \
    --device cuda \
    --n_samples 50 \
    --imu_zscore fold \
    --timeshap \
    --timeshap_E 20 \
    --timeshap_K 300 \
    --timeshap_target peak_vgrf \
    --timeshap_max_samples 20

# SHAP beeswarm per fold
python xai/shap_beeswarm_loso.py \
    --npz /path/to/frozen_dataset_loso_539.npz \
    --ckpt_glob "pretrained/checkpoints/**/*.pt" \
    --outdir results/xai/beeswarm/ \
    --device cuda \
    --bg_n 100 \
    --sample_n 50

# Per-activity XAI heatmaps (SMILE vs TimeSHAP)
python visualization/plot_xai_heatmaps.py \
    --smile_csv results/xai/smile_temporal_agg.csv \
    --timeshap_csv results/xai/timeshap_eventwise_mean_abs.csv \
    --data_dir /path/to/Dataset_Aligned_FINAL \
    --outdir results/xai/figures/
```

Output:
- `results/xai/temporal_smile_aggregate.png` — aggregate SMILE heatmap (Fig. 3 in paper)
- `results/xai/temporal_smile_by_activity.png` — per-activity SMILE heatmaps
- `results/xai/timeshap_eventwise_heatmap.png` — TimeSHAP eventwise heatmap
- `results/xai/smile_phase_summary.csv` — loading/mid-stance/push-off attribution %
- `results/xai/smile_stability_summary.txt` — fold stability (Spearman ρ, Kendall τ)
- `results/xai/timeshap_eventwise_mean_abs.csv` — TimeSHAP mean |φ| per channel
- `results/xai/beeswarm/beeswarm_fold_*.png` — per-fold SHAP beeswarm plots (Fig. 3a)
- `results/xai/figures/smile_walking.png` etc. — per-activity paired heatmaps (Fig. 3b–k)

---

## Reproducing Paper Results

### 1. Download Dataset
```bash
wget https://zenodo.org/records/17376717/files/Dataset_Aligned.zip
unzip Dataset_Aligned.zip
```

Dataset: 10 participants, 539 quality-screened trials (from 598 recorded), 5 activities, Apple Watch Series 6 (wrist + waist, 100 Hz) + AMTI OR6-7 force plate (1000 Hz, downsampled to 100 Hz). See companion dataset paper [Ghaffarzadeh et al., *Scientific Data*, 2026].

### 2. LOSO Training
```bash
python experiments/train_loso.py \
    --aligned_dir /path/to/Dataset_Aligned_FINAL \
    --outdir results/checkpoints/
```

Expected output: `results/checkpoints/fold_XX/test_PXX/GRFNet_MultiScale.pt` for all 10 folds.  
Reproduces: r=0.797±0.077 (published result).

### 3. Ablation Study
```bash
python experiments/ablation.py \
    --aligned_dir /path/to/Dataset_Aligned_FINAL \
    --outdir results/ablation/ \
    --config both
```

Reproduces: multi-scale dilation vs. single-scale (d=1), Transformer comparison
(r=0.581, 2.1M params, ~42ms).

### 4. Clinical / Derived Metrics
```bash
python experiments/clinical_metrics.py \
    --aligned_dir /path/to/Dataset_Aligned_FINAL \
    --ckpt_dir results/checkpoints/ \
    --outdir results/metrics/
```

Expected outputs (match paper):
- Impulse r=0.983, Contact duration r=0.982, Peak vGRF r=0.741, Loading rate r=0.588

### 5. Freeze Dataset NPZ
```bash
python experiments/ablation_npz.py \
    --aligned_dir /path/to/Dataset_Aligned_FINAL \
    --freeze_npz results/frozen_dataset_loso_539.npz
```

Produces `frozen_dataset_loso_539.npz` — required by `shap_beeswarm_loso.py`.  
Shape: (539, 198, 12), padded with PADDING_VALUE=-9999.0.

### 6. XAI Analysis
```bash
# Temporal SMILE + TimeSHAP (produces Fig. 3b–k)
python xai/timeshap_smile_pipeline.py smile \
    --data_dir /path/to/Dataset_Aligned_FINAL \
    --ckpt_glob "results/checkpoints/**/GRFNet_MultiScale.pt" \
    --outdir results/xai/ \
    --device cuda \
    --n_samples 50 \
    --imu_zscore fold \
    --timeshap \
    --timeshap_E 20 \
    --timeshap_K 300 \
    --timeshap_target peak_vgrf \
    --timeshap_max_samples 20

# SHAP beeswarm (Fig. 3a)
python xai/shap_beeswarm_loso.py \
    --npz results/frozen_dataset_loso_539.npz \
    --ckpt_glob "results/checkpoints/**/*.pt" \
    --outdir results/xai/beeswarm/ \
    --device cuda \
    --bg_n 100 \
    --sample_n 50

# Per-activity paired heatmaps (Fig. 3b–k)
python visualization/plot_xai_heatmaps.py \
    --smile_csv results/xai/smile_temporal_agg.csv \
    --timeshap_csv results/xai/timeshap_eventwise_mean_abs.csv \
    --data_dir /path/to/Dataset_Aligned_FINAL \
    --outdir results/xai/figures/
```

---

## Repository Structure

```
multi-sensor-grf-estimation/
├── README.md
├── LICENSE
├── requirements.txt
│
├── model/
│   ├── __init__.py
│   └── grfnet_multiscale.py             # GRFNet-MultiScale architecture
│                                        # class MultiScaleGRFNet
│                                        # d∈{1,2,4,8}, k=3, ~1.2M params
│                                        # (from predict_grf.py)
│
├── data_processing/
│   ├── __init__.py
│   └── add_mass_to_csvs.py              # Adds mass_kg column by participant ID
│
├── experiments/
│   ├── train_loso.py                    # 10-fold LOSO training loop
│   │                                    # Reproduces r=0.797±0.077
│   │                                    # (from train_multiscale_BASELINE_EXACT.py)
│   ├── ablation.py                      # Dilation ablation + Transformer comparison
│   │                                    # (from full_baseline_LOSO_allAct_11-savebest.py)
│   ├── ablation_npz.py                  # Freezes dataset to NPZ + LOSO baselines
│   │                                    # (from ablation_loso_full_with_npz.py)
│   └── clinical_metrics.py             # Derived metrics: impulse, contact duration,
│                                        # peak vGRF, loading rate
│                                        # (from clinical_metrics_stance_only2.py)
│
├── xai/
│   ├── timeshap_smile_pipeline.py       # Main XAI pipeline — Temporal SMILE +
│   │                                    # TimeSHAP across all 10 LOSO folds
│   │                                    # Produces Fig. 3b–k in paper
│   │                                    # (from TimeSHAP-line3Final-2.py)
│   └── shap_beeswarm_loso.py            # SHAP beeswarm per fold + aggregate
│                                        # Produces Fig. 3a in paper
│                                        # (from shap_loso_from_npz_checkpoints.py)
│
├── visualization/
│   └── plot_xai_heatmaps.py             # Per-activity SMILE vs TimeSHAP heatmaps
│                                        # Produces smile_walking.png etc.
│                                        # (from improved_xai_viz_with_context_3_font.py)
│
├── pretrained/
│   ├── GRFNet_MultiScale.pt             # Dual-sensor pretrained weights
│   └── GRFNet_MultiScale_wrist.pt       # Wrist-only pretrained weights
│
└── sample_data/
    ├── sample_walking.csv
    └── data_format.md
```

---

## Data Format

Each aligned CSV file contains synchronized IMU and force plate data at 100 Hz:

| Column | Description | Unit |
|---|---|---|
| `waist_accX/Y/Z` | Waist Apple Watch acceleration | m/s² |
| `waist_gyroX/Y/Z` | Waist Apple Watch angular velocity | rad/s |
| `wrist_accX/Y/Z` | Wrist Apple Watch acceleration | m/s² |
| `wrist_gyroX/Y/Z` | Wrist Apple Watch angular velocity | rad/s |
| `force_z_N` | Vertical GRF from AMTI OR6-7 | N |
| `mass_kg` | Participant body mass | kg |

Input tensor shape to model: **(198 × 12)** — 1.98 s window, 12 inertial channels.

For full specification see [sample_data/data_format.md](sample_data/data_format.md).

---

## Citation

If you use this code, please cite both the paper and the dataset:

```bibtex
@article{ghaffarzadeh2026edge,
  title={Edge-First Ground Reaction Force Estimation with Consumer Smartwatches},
  author={Ghaffarzadeh, Parvin and Chakraborty, Debarati and Aslansefat, Koorosh
          and Dostan, Ali and Papadopoulos, Yiannis},
  journal={IEEE Internet Computing},
  year={2026},
  volume={XX},
  number={X},
  pages={1--9},
  publisher={IEEE}
}

@article{ghaffarzadeh2026dataset,
  title={A Multi-Modal Dataset for Ground Reaction Force Estimation Using
         Consumer Wearable Sensors},
  author={Ghaffarzadeh, Parvin and Chakraborty, Debarati and Aslansefat, Koorosh
          and Dostan, Ali and Papadopoulos, Yiannis},
  journal={Scientific Data},
  year={2026},
  doi={10.1038/s41597-026-07183-6},
  note={Dataset: \url{https://doi.org/10.5281/zenodo.17376717}}
}
```

---

## Contact

**Parvin Ghaffarzadeh** (corresponding author)
- Institution: University of Hull, Department of Computer Science (DAIM)
- Email: p.ghaffarzadeh@hull.ac.uk
- GitHub: [@ParvinGhaffarzadeh](https://github.com/ParvinGhaffarzadeh)

Supervisory team: Prof. Yiannis Papadopoulos (lead), Dr. Debarati Chakraborty,
Dr. Koorosh Aslansefat (University of Hull); Dr. Ali Dostan (Nottingham Trent University).

---

## Contributing and Security

Please read [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) before opening issues or pull requests. Use [SUPPORT.md](SUPPORT.md) for guidance on where to ask questions or report reproducible problems.

---

## Acknowledgments

This research was funded by the EPSRC National Edge AI Hub for Real Data
(EP/Y028813/1) and conducted at the University of Hull with ethical approval
(FHS-24-25.036). GPU compute provided by the University of Hull HPC cluster (ViperOOD).

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Paper status:** Published, IEEE Internet Computing, 2026  
**Repository status:** Active  
**Last updated:** June 2026
