import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PADDING_VALUE = -9999.0

class MultiScaleGRFNet(nn.Module):
    def __init__(self, input_dim=12, num_filters=64, num_blocks=4, dropout=0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, 1),
            nn.BatchNorm1d(num_filters),
        )
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
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
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_head = nn.Sequential(
            nn.Linear(num_filters * 2, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, 1),
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="best_model_meta.json")
    ap.add_argument("--weights", required=True, help="weights-only .pt")
    ap.add_argument("--input_csv", required=True, help="aligned CSV to predict on")
    ap.add_argument("--out_csv", default="predictions.csv")
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)

    imu_cols = meta["imu_cols_used"]
    mean = float(meta["normalizer_mean"])
    std = float(meta["normalizer_std"])
    input_dim = int(meta.get("input_dim", len(imu_cols)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiScaleGRFNet(input_dim=input_dim).to(device)
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    df = pd.read_csv(args.input_csv)
    missing = [c for c in imu_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing IMU columns in input_csv: {missing}")

    X = df[imu_cols].values.astype(np.float32)  # (T, C)
    X_t = torch.from_numpy(X).unsqueeze(0).to(device)  # (1, T, C)

    with torch.no_grad():
        pred_norm = model(X_t).squeeze(0).cpu().numpy()
    pred_N = pred_norm * std + mean

    out = pd.DataFrame({"pred_force_z_N": pred_N})
    out.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)

if __name__ == "__main__":
    main()
