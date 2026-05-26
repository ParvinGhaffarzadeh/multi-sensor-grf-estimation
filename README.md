# Edge-First Ground Reaction Force Estimation with Consumer Smartwatches

[![IEEE Internet Computing](https://img.shields.io/badge/IEEE%20Internet%20Computing-2026-blue)](https://www.computer.org/csdl/magazine/ic)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://zenodo.org/records/17376717)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
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
git clone https://github.com/parvinghaffarzadeh/grfnet-multiscale.git
cd grfnet-multiscale

conda create -n grfnet python=3.11
conda activate grfnet

pip install -r requirements.txt
```

### Option 2: Virtual Environment
```bash
git clone https://github.com/parvinghaffarzadeh/grfnet-multiscale.git
cd grfnet-multiscale

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Verify
```python
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "from model.grfnet_multiscale import GRFNetMultiScale; print('Model imported OK')"
```

---

## Quick Start

### Load Pretrained Model
```python
import torch
from model.grfnet_multiscale import GRFNetMultiScale

# Dual-sensor model (~1.2M parameters)
model = GRFNetMultiScale(in_channels=12)
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
python xai/run_smile_timeshap_complete.py \
    --data-dir /path/to/Dataset_Aligned_FINAL \
    --ckpt pretrained/GRFNet_MultiScale.pt \
    --outdir results/xai/
```

Output:
- `results/xai/aggregates/aggregate_smile_two_panel_*.png` — per-activity heatmaps
- `results/xai/aggregates/aggregate_timeshap_two_panel_*.png` — TimeSHAP panels
- `results/xai/single_trials/` — per-trial signal + attribution figures

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
    --data-dir data/Dataset_Aligned_FINAL \
    --outdir results/checkpoints/ \
    --epochs 100 \
    --batch-size 32 \
    --seed 42
```

### 3. Evaluate
```bash
python experiments/evaluate_loso.py \
    --data-dir data/Dataset_Aligned_FINAL \
    --ckpt-dir results/checkpoints/ \
    --outdir results/metrics/
```

Expected outputs:
- `results/metrics/loso_summary.csv` — per-participant r and RMSE (Table in paper)
- `results/metrics/activity_stratified.csv` — per-activity performance
- `results/metrics/derived_metrics.csv` — impulse, contact duration, peak vGRF, loading rate

### 4. Ablation Study
```bash
python experiments/ablation.py \
    --data-dir data/Dataset_Aligned_FINAL \
    --outdir results/ablation/
```

Reproduces: multi-scale dilation vs. single-scale (d=1), Transformer comparison (r=0.581, 2.1M params, ~42ms).

### 5. Generate Figures
```bash
python visualization/generate_all_figures.py \
    --results-dir results/ \
    --outdir figures/ \
    --dpi 300
```

---

## Repository Structure

```
grfnet-multiscale/
├── README.md
├── LICENSE
├── requirements.txt
│
├── model/
│   ├── __init__.py
│   └── grfnet_multiscale.py        # GRFNet-MultiScale architecture
│                                   # d∈{1,2,4,8}, k=3, ~1.2M params
│
├── data_processing/
│   ├── __init__.py
│   ├── preprocessing.py            # Alignment, low-pass filter (IMU: 10 Hz, FP: 20 Hz)
│   ├── stance_extraction.py        # Fixed 198-sample windows, z-score normalization
│   └── data_loader.py              # PyTorch Dataset/DataLoader
│
├── experiments/
│   ├── train_loso.py               # 10-fold LOSO training loop
│   ├── evaluate_loso.py            # Evaluation: r, RMSE, derived metrics
│   ├── ablation.py                 # Dilation ablation + Transformer comparison
│   └── config.yaml                 # Hyperparameters
│
├── xai/
│   ├── run_smile_timeshap_complete.py     # Main XAI pipeline (all folds)
│   ├── enhanced_stance_smile_from_outputs.py   # Activity SMILE heatmaps
│   ├── enhanced_stance_smile_all_folds_2.py    # Per-fold composites
│   └── plot_aggregate_two_panel.py        # Two-panel aggregate figures
│
├── visualization/
│   ├── generate_all_figures.py
│   └── plot_utils.py
│
├── pretrained/
│   ├── GRFNet_MultiScale.pt        # Dual-sensor pretrained weights
│   └── GRFNet_MultiScale_wrist.pt  # Wrist-only pretrained weights
│
├── sample_data/
│   ├── sample_walking.csv
│   └── data_format.md
│
└── notebooks/
    ├── 01_dataset_exploration.ipynb
    ├── 02_training_demo.ipynb
    ├── 03_evaluation_analysis.ipynb
    └── 04_xai_demo.ipynb
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
- GitHub: [@parvinghaffarzadeh](https://github.com/parvinghaffarzadeh)

Supervisory team: Prof. Yiannis Papadopoulos (lead), Dr. Debarati Chakraborty,
Dr. Koorosh Aslansefat (University of Hull); Dr. Ali Dostan (Nottingham Trent University).

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
**Last updated:** May 2026
