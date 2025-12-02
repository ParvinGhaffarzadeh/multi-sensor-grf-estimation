# Activity-Specific vGRF Prediction: Wrist and Waist Sensor Comparison

[![Paper](https://img.shields.io/badge/IEEE%20TBME-Paper-blue)](link-to-paper)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📋 Overview

This repository contains the official implementation of our IEEE TBME paper on **activity-specific ground reaction force (vGRF) estimation** using consumer wearable IMU sensors and deep learning.

**Key Contribution:** We demonstrate that sensor placement effectiveness is **highly activity-dependent** - wrist sensors excel in natural locomotion but fail in constrained movements (e.g., hands-on-chest tasks), providing critical guidance for clinical wearable deployment.

### Research Highlights
- 🏃 **5 Activities Evaluated**: Walking, jogging, running, drop landings, heel drops
- 📊 **1,263 stance phases** from real-world data
- 🧠 **GRFNet**: Novel 1D CNN with 194K parameters (2× more efficient than InceptionTime)
- 🔍 **Explainability Analysis**: Gradient-based saliency with RexQual metrics
- 📍 **Clinical Guidelines**: Evidence-based sensor placement recommendations

---

## 🚀 Key Findings

### Performance by Activity

| Activity | Wrist Sensor (r) | Waist Sensor (r) | Best Choice |
|----------|------------------|------------------|-------------|
| Walking  | 0.634 ± 0.296    | 0.521 ± 0.265    | Wrist ✓     |
| Jogging  | 0.876 ± 0.166    | **0.933 ± 0.064**| Waist ✓     |
| Running  | 0.822 ± 0.238    | 0.766 ± 0.334    | Wrist ✓     |
| Drop Landings | 0.614 ± 0.251 | 0.482 ± 0.256   | Wrist ✓     |
| Heel Drops | ⚠️ 0.186 ± 0.108 | 0.119 ± 0.203  | Neither (both poor) |

**Critical Finding:** Wrist sensors **fail catastrophically** in heel drops (r=0.186) due to arm immobilization (hands crossed on chest). This demonstrates the importance of task-specific sensor selection.

### Model Efficiency
- **GRFNet**: 194,497 parameters
- **InceptionTime** (baseline): 388,512 parameters
- **Efficiency gain**: 2.0× parameter reduction with competitive performance

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU

### Option 1: Conda (Recommended)
```bash
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
```

---

## 🎯 Quick Start

### 1. Load Pretrained Model
```python
import torch
from models.grfnet import GRFNet

# Load wrist sensor model
model = GRFNet(in_channels=6, hidden_channels=96)
model.load_state_dict(torch.load('pretrained_models/WRIST_GRFNet.pth'))
model.eval()

# Your IMU data: shape (batch_size, timesteps, 6)
# Channels: [accX, accY, accZ, gyroX, gyroY, gyroZ]
predictions = model(imu_data)  # Returns: (batch_size, timesteps)
```

### 2. Predict from CSV File
```python
import pandas as pd
from data_processing.preprocessing import preprocess_imu

# Load aligned CSV data
df = pd.read_csv('sample_data/sample_walking.csv')
imu_data = preprocess_imu(df, sensor='wrist')

# Predict GRF
with torch.no_grad():
    vgrf_predicted = model(imu_data)

print(f"Predicted vGRF shape: {vgrf_predicted.shape}")
```

### 3. Visualize Predictions
```python
from visualization.plot_utils import plot_grf_comparison

plot_grf_comparison(
    ground_truth=df['force_z_N'].values,
    predicted=vgrf_predicted.numpy(),
    title="Walking Trial - Wrist Sensor"
)
```

---

## 📊 Reproducing Paper Results

### Train Models from Scratch
```bash
# Train wrist sensor model
python experiments/train_wrist.py \
    --data_dir /path/to/Dataset_Aligned \
    --epochs 150 \
    --batch_size 64 \
    --lr 0.001

# Train waist sensor model
python experiments/train_waist.py \
    --data_dir /path/to/Dataset_Aligned \
    --epochs 150 \
    --batch_size 64 \
    --lr 0.001
```

### Run Multi-Sensor Comparison
```bash
python experiments/multi_sensor_comparison.py \
    --aligned_dir /path/to/Dataset_Aligned \
    --output_dir results/
```

**Expected Output:**
- Performance metrics by activity
- Statistical comparisons
- Visualization figures saved to `results/figures/`

### Generate Explainability Analysis
```bash
python explainability/explainability_metrics.py \
    --model_path pretrained_models/WAIST_GRFNet.pth \
    --data_dir /path/to/Dataset_Aligned \
    --sensor waist \
    --n_stances 100
```

**Output:**
- `explainability_results/waist/explainability_metrics.csv`
- `explainability_results/waist/explainability_table.tex`

### Generate All Manuscript Figures
```bash
python visualization/generate_figures.py
```

**Generated Figures:**
1. `fig1_dataset_overview.png` - Stance count distribution
2. `fig2_grfnet_architecture.png` - Network architecture
3. `fig7_waveform_examples.png` - Prediction examples
4. `fig8_training_curves.png` - Training progress
5. `fig9_stance_extraction.png` - Stance detection demo
6. `fig10_saliency_maps.png` - Explainability analysis
7. `fig11_performance_matrix.png` - Performance heatmap
8. `fig12_baseline_comparison.png` - Model comparison

---

## 📁 Repository Structure
```
multi-sensor-grf-estimation/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (optional)
│
├── models/
│   ├── grfnet.py                     # GRFNet architecture
│   ├── baselines.py                  # InceptionTime, TCN, etc.
│   └── __init__.py
│
├── data_processing/
│   ├── stance_extraction.py          # Automatic stance detection
│   ├── preprocessing.py              # Data alignment & normalization
│   ├── data_loader.py                # PyTorch Dataset/DataLoader
│   └── __init__.py
│
├── experiments/
│   ├── train_wrist.py                # Wrist sensor training
│   ├── train_waist.py                # Waist sensor training
│   ├── multi_sensor_comparison.py    # Full comparison pipeline
│   └── config.yaml                   # Hyperparameter configuration
│
├── explainability/
│   ├── saliency_methods.py           # Gradient, IG, SmoothGrad
│   ├── explainability_metrics.py     # RexQual metrics calculator
│   └── __init__.py
│
├── visualization/
│   ├── generate_figures.py           # Generate all manuscript figures
│   ├── plot_utils.py                 # Plotting helper functions
│   └── __init__.py
│
├── pretrained_models/
│   ├── WRIST_GRFNet.pth             # Pretrained wrist model
│   ├── WAIST_GRFNet.pth             # Pretrained waist model
│   └── model_card.md                 # Model documentation
│
├── sample_data/
│   ├── sample_walking.csv            # Example walking trial
│   ├── sample_jogging.csv            # Example jogging trial
│   └── data_format.md                # Data format specification
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Dataset statistics
│   ├── 02_model_training_demo.ipynb  # Training walkthrough
│   ├── 03_evaluation_analysis.ipynb  # Results visualization
│   └── 04_explainability_demo.ipynb  # Saliency analysis demo
│
└── tests/
    ├── test_model.py                 # Unit tests for GRFNet
    └── test_preprocessing.py         # Unit tests for data processing
```

---

## 📝 Data Format

### Input: Aligned CSV Files
Each CSV file should contain synchronized IMU and force plate data at 100 Hz:

| Column | Description | Unit |
|--------|-------------|------|
| `wrist_accX/Y/Z` | Wrist acceleration | m/s² |
| `wrist_gyroX/Y/Z` | Wrist angular velocity | rad/s |
| `waist_accX/Y/Z` | Waist acceleration | m/s² |
| `waist_gyroX/Y/Z` | Waist angular velocity | rad/s |
| `force_z_N` | Vertical ground reaction force | N |

**Example:**
```csv
wrist_accX,wrist_accY,wrist_accZ,wrist_gyroX,wrist_gyroY,wrist_gyroZ,waist_accX,waist_accY,waist_accZ,waist_gyroX,waist_gyroY,waist_gyroZ,force_z_N
0.123,0.456,9.789,0.012,0.034,0.056,0.234,0.567,9.890,0.023,0.045,0.067,1234.5
...
```

See `sample_data/data_format.md` for detailed specifications.

---

## 🎓 Citation

If you use this code or findings in your research, please cite:
```bibtex
@article{ghaffarzadeh2024vgrf,
  title={Activity-Specific Vertical Ground Reaction Force Prediction: Wrist and Waist Sensor Comparison},
  author={Ghaffarzadeh, Parvin and [Co-authors]},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2024},
  volume={XX},
  number={XX},
  pages={XXX-XXX},
  doi={10.1109/TBME.2024.XXXXXX}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

**Parvin Ghaffarzadeh**
- Email: [your-email@university.edu]
- GitHub: [@parvinghaffarzadeh](https://github.com/parvinghaffarzadeh)
- Institution: [Your Institution]

For questions about the paper or code, please open an issue or contact via email.

---

## 🙏 Acknowledgments

- Dataset collected with [mention equipment/lab]
- Computing resources provided by [mention institution]
- Built with PyTorch, NumPy, and Matplotlib

---

## 📚 Related Publications

[List any related papers or preprints here]

---

**Last Updated:** December 2024
