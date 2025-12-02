# Activity-Specific Ground Reaction Force Prediction Using Wearable IMU Sensors

[![IEEE TBME](https://img.shields.io/badge/IEEE%20TBME-Under%20Review-blue)](https://www.embs.org/tbme/)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://zenodo.org/records/17376717)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📋 Overview

This repository contains the official implementation of our IEEE TBME paper on **activity-specific ground reaction force (vGRF) estimation** using consumer wearable IMU sensors (Apple Watch) and deep learning.

**Key Contribution:** We demonstrate that sensor placement effectiveness is **highly activity-dependent** - wrist sensors excel in natural arm-swing activities (walking, running) but fail in constrained movements (heel drops with hands crossed on chest), providing critical evidence-based guidance for clinical wearable deployment.

### Research Highlights
- 🏃 **5 Diverse Activities**: Walking, jogging, running, drop landings, heel drops
- 📊 **585 Validated Trials** from 10 healthy participants
- 📍 **Dual Sensor Comparison**: Wrist vs. waist placement across activities
- 🧠 **GRFNet**: Novel 1D CNN with 194,497 parameters (2.0× more efficient than InceptionTime)
- 🔍 **Validated Explainability**: Gradient-based, Integrated Gradients, and SmoothGrad saliency analysis
- 💊 **Clinical Relevance**: Evidence-based sensor placement for osteoporosis monitoring and injury prevention

---

## 🚀 Key Findings

### Activity-Specific Sensor Performance

| Activity | Wrist (r) | Waist (r) | **Recommended** |
|----------|-----------|-----------|-----------------|
| **Walking** | 0.634 ± 0.296 | 0.521 ± 0.265 | **Wrist ✓** |
| **Jogging** | 0.876 ± 0.166 | **0.933 ± 0.064** | **Waist ✓** |
| **Running** | **0.822 ± 0.238** | 0.766 ± 0.334 | **Wrist ✓** |
| **Drop Landings** | **0.614 ± 0.251** | 0.482 ± 0.256 | **Wrist ✓** |
| **Heel Drops** | ⚠️ 0.186 | ⚠️ 0.119 | ❌ Both Poor |

**Critical Finding:** Wrist sensors **fail catastrophically** in heel drops (r=0.186, RMSE=239.8 N) when arms are constrained. This demonstrates the **absolute necessity of task-specific sensor selection** for clinical applications.

### Model Efficiency
- **GRFNet**: 194,497 parameters → **Deployable on consumer wearables**
- **InceptionTime** (best baseline): 388,512 parameters
- **Parameter reduction**: 2.0× with competitive accuracy (9% difference in r)
- **Computational cost**: <0.01 s per stance prediction (real-time capable)

### Best Overall Sensor Recommendation
**Wrist placement outperforms waist in 4 of 5 activities**, making it the practical choice for continuous free-living monitoring. Acceptability of wrist-worn devices (smartwatch) exceeds waist-worn sensors for long-term compliance.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended) or CPU
- ~2 GB disk space for models and data

### Option 1: Conda (Recommended)
```bash
# Clone repository
git clone https://github.com/parvinghaffarzadeh/multi-sensor-grf-estimation.git
cd multi-sensor-grf-estimation

# Create conda environment
conda create -n vgrf python=3.11
conda activate vgrf

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Virtual Environment
```bash
# Clone repository
git clone https://github.com/parvinghaffarzadeh/multi-sensor-grf-estimation.git
cd multi-sensor-grf-estimation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation
```python
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "from models.grfnet import GRFNet; print('GRFNet imported successfully')"
python -c "import pandas; import numpy; print('Dependencies ready')"
```

---

## 🎯 Quick Start

### 1. Load Pretrained Models
```python
import torch
from models.grfnet import GRFNet

# Load wrist sensor model (recommended for free-living monitoring)
model_wrist = GRFNet(in_channels=6, hidden_channels=96)
model_wrist.load_state_dict(torch.load('pretrained_models/WRIST_GRFNet.pth'))
model_wrist.eval()

# Load waist sensor model (optimal for rhythmic activities like jogging)
model_waist = GRFNet(in_channels=6, hidden_channels=96)
model_waist.load_state_dict(torch.load('pretrained_models/WAIST_GRFNet.pth'))
model_waist.eval()

print("✓ Models loaded successfully")
```

### 2. Predict vGRF from IMU Data
```python
import pandas as pd
import torch
from data_processing.preprocessing import preprocess_imu

# Load aligned CSV data (wrist and waist IMU + force plate)
df = pd.read_csv('sample_data/sample_walking.csv')

# Prepare data (shape: batch_size, timesteps, 6 channels)
imu_wrist = preprocess_imu(df, sensor='wrist')
imu_waist = preprocess_imu(df, sensor='waist')

# Predict vGRF for both sensors
with torch.no_grad():
    vgrf_wrist = model_wrist(imu_wrist)    # Wrist prediction
    vgrf_waist = model_waist(imu_waist)    # Waist prediction

print(f"Wrist prediction shape: {vgrf_wrist.shape}")
print(f"Waist prediction shape: {vgrf_waist.shape}")
```

### 3. Visualize Predictions
```python
from visualization.plot_utils import plot_grf_comparison

# Compare predictions with ground truth
plot_grf_comparison(
    ground_truth=df['force_z_N'].values,
    predicted_wrist=vgrf_wrist.numpy(),
    predicted_waist=vgrf_waist.numpy(),
    title="Walking Trial - Wrist vs. Waist Comparison",
    save_path='results/walking_comparison.png'
)
```

### 4. Batch Prediction on Multiple Trials
```python
from data_processing.data_loader import GRFDataset, DataLoader

# Load dataset
dataset = GRFDataset(
    root_dir='path/to/Dataset_Aligned',
    sensor='wrist',
    activity='walking'
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Predict all trials
all_predictions = []
with torch.no_grad():
    for batch in dataloader:
        predictions = model_wrist(batch)
        all_predictions.append(predictions.cpu().numpy())

print(f"Total predictions: {len(all_predictions)} batches")
```

---

## 📊 Reproducing Paper Results

### Dataset Access
The multi-modal dataset is available on Zenodo:
```bash
# Download dataset
wget https://zenodo.org/records/17376717/files/Dataset_Aligned.zip
unzip Dataset_Aligned.zip
```

Dataset includes:
- **10 participants** (6 male, 4 female)
- **585 synchronized trials** across 5 activities
- **Apple Watch Series 6** at wrist and waist
- **AMTI BP400600 force plate** (1000 Hz reference)
- **740 wrist + 1,263 waist stance segments**

For dataset details, see the companion paper: [Ghaffarzadeh et al., "A multi-modal dataset for ground reaction force estimation using consumer wearable sensors" (under revision)]

### Train Models from Scratch
```bash
# Download and prepare dataset
cd data/
wget https://zenodo.org/records/17376717/files/Dataset_Aligned.zip
unzip Dataset_Aligned.zip

# Train wrist sensor model
python ../experiments/train_wrist.py \
    --data_dir Dataset_Aligned \
    --epochs 150 \
    --batch_size 64 \
    --lr 0.001 \
    --seed 42

# Train waist sensor model
python ../experiments/train_waist.py \
    --data_dir Dataset_Aligned \
    --epochs 150 \
    --batch_size 64 \
    --lr 0.001 \
    --seed 42
```

### Run Full Multi-Sensor Comparison
```bash
python experiments/multi_sensor_comparison.py \
    --aligned_dir data/Dataset_Aligned \
    --output_dir results/ \
    --n_seeds 5
```

**Expected Output:**
- `results/performance_by_activity.csv` - Table 1 from paper
- `results/performance_matrix.csv` - Table 2 (activity × sensor)
- `results/baseline_comparison.csv` - Supplementary baseline analysis
- `results/figures/` - All manuscript figures

### Run Explainability Analysis
```bash
python explainability/explainability_metrics.py \
    --model_path pretrained_models/WAIST_GRFNet.pth \
    --data_dir data/Dataset_Aligned \
    --sensor waist \
    --methods gradient integrated_gradients smoothgrad \
    --n_stances 100
```

**Output:**
- `explainability_results/waist/metrics_table.csv` - Table 5 from paper
- `explainability_results/waist/saliency_maps.png` - Figure 8
- `explainability_results/waist/faithfulness_analysis.pdf` - Detailed analysis

### Generate All Manuscript Figures
```bash
# Generate 8 publication-quality figures (300 DPI)
python visualization/generate_all_figures.py \
    --output_dir figures/ \
    --dpi 300
```

**Generated Figures:**
1. `fig1_dataset_overview.png` - Stance distribution (Figure 1)
2. `fig2_grfnet_architecture.png` - Network architecture (Figure 2)
3. `fig8_training_curves.png` - Training progress (Figure 3)
4. `fig9_stance_extraction.png` - Stance detection (Figure 4)
5. `fig7_waveform_examples.png` - Prediction examples (Figure 5)
6. `fig11_performance_matrix.png` - Performance heatmap (Figure 6)
7. `fig12_baseline_comparison.png` - Model comparison (Figure 7)
8. `fig10_saliency_maps.png` - Explainability analysis (Figure 8)

---

## 📁 Repository Structure
```
multi-sensor-grf-estimation/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│   ├── torch>=2.0
│   ├── pandas>=1.5
│   ├── numpy>=1.23
│   ├── scikit-learn>=1.2
│   ├── matplotlib>=3.7
│   └── scipy>=1.10
│
├── models/
│   ├── __init__.py
│   ├── grfnet.py                     # GRFNet architecture
│   ├── baselines.py                  # InceptionTime, TCN, LSTM, Transformer
│   └── model_card.md                 # Model documentation
│
├── data_processing/
│   ├── __init__.py
│   ├── stance_extraction.py          # Automatic stance detection (80 N threshold)
│   ├── preprocessing.py              # Synchronization & normalization
│   ├── data_loader.py                # PyTorch Dataset/DataLoader
│   └── validation.py                 # Data quality checks
│
├── experiments/
│   ├── __init__.py
│   ├── train_wrist.py                # Wrist sensor training pipeline
│   ├── train_waist.py                # Waist sensor training pipeline
│   ├── multi_sensor_comparison.py    # Complete comparison framework
│   ├── baseline_comparison.py        # Train all 6 baselines
│   ├── config.yaml                   # Hyperparameters & settings
│   └── utils.py                      # Training utilities
│
├── explainability/
│   ├── __init__.py
│   ├── saliency_methods.py           # Gradient, IG, SmoothGrad implementations
│   ├── explainability_metrics.py     # Faithfulness, robustness, sparsity
│   └── visualization.py              # Saliency visualization
│
├── visualization/
│   ├── __init__.py
│   ├── generate_all_figures.py       # Generate all 8 manuscript figures
│   ├── plot_utils.py                 # Plotting helper functions
│   └── style.py                      # Publication-quality formatting
│
├── pretrained_models/
│   ├── WRIST_GRFNet.pth             # Wrist-mounted sensor model
│   ├── WAIST_GRFNet.pth             # Waist-mounted sensor model
│   ├── model_card.md                 # Model card with performance metrics
│   └── README.md                     # Model usage documentation
│
├── sample_data/
│   ├── sample_walking.csv            # Example: Walking trial
│   ├── sample_jogging.csv            # Example: Jogging trial
│   ├── sample_drop_landing.csv       # Example: Drop landing
│   └── data_format.md                # Complete data specification
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   # Statistics & distributions
│   ├── 02_model_training_demo.ipynb  # End-to-end training walkthrough
│   ├── 03_evaluation_analysis.ipynb  # Performance visualizations
│   ├── 04_explainability_demo.ipynb  # Saliency maps & interpretation
│   └── 05_clinical_guidelines.ipynb  # Sensor placement recommendations
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py                 # GRFNet unit tests
│   ├── test_preprocessing.py         # Data preprocessing tests
│   └── test_data_loader.py           # DataLoader tests
│
└── docs/
    ├── INSTALLATION.md               # Detailed installation guide
    ├── DATASET.md                    # Dataset documentation
    ├── MODELS.md                     # Model architecture details
    ├── RESULTS.md                    # Complete results tables
    └── TROUBLESHOOTING.md            # Common issues & solutions
```

---

## 📝 Data Format

### Input Data: Synchronized CSV Files
Each file contains real-time IMU + force plate data at 100 Hz synchronization:

| Column | Sensor | Description | Unit |
|--------|--------|-------------|------|
| `wrist_accX/Y/Z` | Apple Watch | Acceleration | m/s² |
| `wrist_gyroX/Y/Z` | Apple Watch | Angular velocity | rad/s |
| `waist_accX/Y/Z` | Apple Watch | Acceleration | m/s² |
| `waist_gyroX/Y/Z` | Apple Watch | Angular velocity | rad/s |
| `force_z_N` | AMTI Force Plate | Vertical GRF | N |

**Example CSV:**
```csv
time,wrist_accX,wrist_accY,wrist_accZ,wrist_gyroX,wrist_gyroY,wrist_gyroZ,waist_accX,waist_accY,waist_accZ,waist_gyroX,waist_gyroY,waist_gyroZ,force_z_N
0.00,0.123,0.456,9.789,0.012,0.034,0.056,0.234,0.567,9.890,0.023,0.045,0.067,1234.5
0.01,0.125,0.458,9.791,0.013,0.035,0.057,0.235,0.568,9.891,0.024,0.046,0.068,1245.3
...
```

**Dataset Statistics:**
- **Sampling rate**: 100 Hz (synchronized)
- **Trials per activity**: ~117 per activity
- **Stance segments**: 740 (wrist) + 1,263 (waist)
- **Stance duration**: 0.6–1.4 seconds (60–140 samples)
- **Force plate range**: 0–2,000 N

For complete specifications, see [sample_data/data_format.md](sample_data/data_format.md).

---

## 📄 Citation

If you use this code, pretrained models, or findings in your research, please cite our paper:

```bibtex
@article{ghaffarzadeh2024activity,
  title={Activity-Specific Ground Reaction Force Prediction: Wrist vs. Waist IMU Sensor Comparison Using Convolutional Neural Networks and Explainable AI},
  author={Ghaffarzadeh, Parvin and Chakraborty, Debarati and Aslansefat, Koorosh and Papadopoulos, Yiannis and Dostan, Ali},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2024},
  volume={XX},
  number={XX},
  pages={XXX--XXX},
  doi={10.1109/TBME.2024.XXXXXX},
  publisher={IEEE}
}
```

Also cite the dataset paper:
```bibtex
@article{ghaffarzadeh2024dataset,
  title={A Multi-Modal Dataset for Ground Reaction Force Estimation Using Consumer Wearable Sensors},
  author={Ghaffarzadeh, Parvin and Chakraborty, Debarati and Aslansefat, Koorosh and Dostan, Ali and Papadopoulos, Yiannis},
  journal={Scientific Data},
  year={2024},
  volume={XX},
  number={XX},
  pages={XXX},
  doi={10.5281/zenodo.17376717}
}
```

---

## 📖 Model Documentation

### GRFNet Architecture
- **Input**: 6-channel IMU (3 acceleration + 3 gyroscope)
- **Output**: 1-channel vGRF time series (same length as input)
- **Layers**: 4 × 1D Conv + BatchNorm + ReLU + Dropout
- **Parameters**: 194,497 (2× fewer than InceptionTime)
- **Inference time**: <0.01 s per stance (real-time capable)

See [pretrained_models/model_card.md](pretrained_models/model_card.md) for detailed specs.

---

## 📧 Contact & Support

**Parvin Ghaffarzadeh** (Author & Maintainer)
- 🏫 **Institution**: University of Hull, Department of Artificial Intelligence and Modelling (DAIM)
- 📧 **Email**: p.ghaffarzadeh@hull.ac.uk
- 🔗 **GitHub**: [@parvinghaffarzadeh](https://github.com/parvinghaffarzadeh)
- 📍 **Location**: Kingston upon Hull, UK

**Supervision Team**:
- **Primary Supervisor**: Dr. Debarati Chakraborty (University of Hull)
- **Co-Supervisors**: Dr. Koorosh Aslansefat, Prof. Yiannis Papadopoulos (University of Hull)
- **External Supervisor**: Dr. Ali Dostan (Nottingham Trent University)

**For Questions About:**
- 📋 Dataset access → Open issue or contact p.ghaffarzadeh@hull.ac.uk
- 🧠 Model architecture → See [docs/MODELS.md](docs/MODELS.md)
- 🔧 Installation issues → See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- 📊 Results reproduction → See [docs/RESULTS.md](docs/RESULTS.md)

---

## 🙏 Acknowledgments

- **Dataset Collection**: University of Hull Biomechanics Laboratory
- **Funding**: University of Hull Department of Artificial Intelligence and Modelling
- **Equipment**: AMTI BP400600 force plate, Apple Watch Series 6
- **Framework**: Built with PyTorch, NumPy, SciPy, Matplotlib
- **Computing**: GPU compute resources provided by [Institution/Service]

---

## 📚 Related Work

**Companion Paper (Dataset):**
- Ghaffarzadeh et al. (2024). "A Multi-Modal Dataset for Ground Reaction Force Estimation Using Consumer Wearable Sensors." *Scientific Data* (under review)
  - Dataset DOI: https://zenodo.org/records/17376717

**References:**
- Chen et al. (2021). "Deep learning for sensor-based human activity recognition"
- Goodfellow et al. (2016). "Deep Learning" (MIT Press)
- See full bibliography in paper or [docs/REFERENCES.md](docs/REFERENCES.md)

---

## ⚖️ License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

**You are free to:**
- ✓ Use this code commercially
- ✓ Modify and distribute
- ✓ Use for private purposes

**You must:**
- ⚠️ Include license and copyright notice
- ⚠️ State changes made to the code

---

## 🤝 Contributing

We welcome contributions to improve this project!

**How to contribute:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

**Contribution Guidelines:**
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Reference related issues

---

## ⭐ Citation Count & Impact

If you use this repository, please:
- ⭐ Star this repository to help others discover it
- 🔗 Link to this GitHub in your work
- 📝 Cite using the BibTeX provided above

---

**Last Updated:** December 2024
**Repository Status:** Active Development
**Paper Status:** Under Review (IEEE TBME)

---

## 🚀 Quick Links

- [📖 Full Documentation](docs/)
- [📊 Dataset on Zenodo](https://zenodo.org/records/17376717)
- [📋 Reproduce Results](docs/RESULTS.md)
- [🔧 Installation Guide](docs/INSTALLATION.md)
- [❓ FAQ & Troubleshooting](docs/TROUBLESHOOTING.md)
- [📝 Paper (Preprint)](https://arxiv.org/abs/XXXX.XXXXX) *(Coming soon)*
