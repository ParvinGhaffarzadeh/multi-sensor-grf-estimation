
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
