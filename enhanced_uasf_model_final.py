#!/usr/bin/env python3
"""
Enhanced UASF-GRFNet with Biomechanical Priors and Uncertainty Quantification
============================================================================
Model + losses + robustness utilities ONLY (no data loading).

REAL DATA ONLY: No synthetic padding, no synthetic augmentation in this file.

Author: Parvin Ghaffarzadeh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def z_score_from_confidence(confidence_level: float = 0.95) -> float:
    """Z score for common confidence levels."""
    return {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(float(confidence_level), 1.96)


# -----------------------------------------------------------------------------
# PART 1: ENCODER WITH EPISTEMIC UNCERTAINTY
# -----------------------------------------------------------------------------

class EnhancedUncertaintySensorEncoder(nn.Module):
    """
    Sensor encoder with epistemic uncertainty estimation (per-trial scalar).

    Inputs:  x (B, T, C)
    Outputs: features (B, F, T), epistemic_unc (B, 1) positive
    """

    def __init__(self, n_channels: int = 6, n_filters: int = 128, dropout: float = 0.15):
        super().__init__()

        self.conv1 = nn.Conv1d(n_channels, n_filters, kernel_size=19, padding=9)
        self.gn1 = nn.GroupNorm(8, n_filters)

        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=11, padding=5)
        self.gn2 = nn.GroupNorm(8, n_filters)

        self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size=11, padding=20, dilation=4)
        self.gn3 = nn.GroupNorm(8, n_filters)

        self.uncertainty_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus(),  # positive
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = self.dropout(self.relu(self.gn1(self.conv1(x))))

        residual = x
        x = self.dropout(self.relu(self.gn2(self.conv2(x) + residual)))

        residual = x
        x = self.dropout(self.relu(self.gn3(self.conv3(x) + residual)))

        features = x
        epistemic_unc = self.uncertainty_head(features)  # (B, 1)
        return features, epistemic_unc


# -----------------------------------------------------------------------------
# PART 2: ACTIVITY-AWARE ADAPTIVE FUSION
# -----------------------------------------------------------------------------

class ActivityAwareFusion(nn.Module):
    """
    Fusion weights conditioned on:
      1) encoder epistemic uncertainties
      2) predicted activity probabilities

    Inputs:
      waist_feat, wrist_feat: (B, F, T)
      waist_unc, wrist_unc:   (B, 1)

    Outputs:
      fused:         (B, F, T)
      weights:       (B, 2) [waist, wrist] sum=1
      activity_probs:(B, n_activities)
    """

    def __init__(self, n_filters: int = 128, n_activities: int = 5, dropout: float = 0.3):
        super().__init__()
        self.n_activities = int(n_activities)

        self.activity_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.n_activities),
        )

        self.weight_generator = nn.Sequential(
            nn.Linear(2 + self.n_activities, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1),
        )

        self.value_waist = nn.Conv1d(n_filters, n_filters, kernel_size=1)
        self.value_wrist = nn.Conv1d(n_filters, n_filters, kernel_size=1)

    def forward(
        self,
        waist_feat: torch.Tensor,
        wrist_feat: torch.Tensor,
        waist_unc: torch.Tensor,
        wrist_unc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        combined_feat = torch.cat([waist_feat, wrist_feat], dim=1)  # (B, 2F, T)
        activity_logits = self.activity_classifier(combined_feat)
        activity_probs = torch.softmax(activity_logits, dim=-1)

        fusion_input = torch.cat([waist_unc, wrist_unc, activity_probs], dim=1)  # (B, 2+nA)
        weights = self.weight_generator(fusion_input)  # (B,2)

        v_waist = self.value_waist(waist_feat)
        v_wrist = self.value_wrist(wrist_feat)

        ww = weights[:, 0:1].unsqueeze(-1)  # (B,1,1)
        wr = weights[:, 1:2].unsqueeze(-1)

        fused = ww * v_waist + wr * v_wrist
        return fused, weights, activity_probs


# -----------------------------------------------------------------------------
# PART 3: DECODER WITH ALEATORIC UNCERTAINTY
# -----------------------------------------------------------------------------

class EnhancedDecoder(nn.Module):
    """Predicts GRF waveform (B,T) and aleatoric sigma(t) (B,T) positive."""

    def __init__(self, n_filters: int = 128, dropout: float = 0.15):
        super().__init__()

        self.prediction_path = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=11, padding=5),
            nn.GroupNorm(8, n_filters),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(n_filters, n_filters // 2, kernel_size=9, padding=4),
            nn.GroupNorm(8, n_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(n_filters // 2, 1, kernel_size=7, padding=3),
        )

        self.uncertainty_path = nn.Sequential(
            nn.Conv1d(n_filters, n_filters // 2, kernel_size=11, padding=5),
            nn.GroupNorm(8, n_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(n_filters // 2, 1, kernel_size=7, padding=3),
            nn.Softplus(),  # sigma>0
        )

    def forward(self, fused_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.prediction_path(fused_features).squeeze(1)
        ale = self.uncertainty_path(fused_features).squeeze(1)
        return pred, ale


# -----------------------------------------------------------------------------
# PART 4: COMPLETE MODEL
# -----------------------------------------------------------------------------

class EnhancedUASF_GRFNet(nn.Module):
    """
    forward(return_all=True) outputs:
      prediction:        (B, T)
      aleatoric_unc:     (B, T)
      epistemic_unc:     (B, 1)
      total_uncertainty: (B, T)
      fusion_weights:    (B, 2)
      activity_probs:    (B, n_activities)
    """

    def __init__(self, n_filters: int = 128, dropout: float = 0.15, n_activities: int = 5):
        super().__init__()
        self.n_activities = int(n_activities)

        self.waist_encoder = EnhancedUncertaintySensorEncoder(6, n_filters, dropout)
        self.wrist_encoder = EnhancedUncertaintySensorEncoder(6, n_filters, dropout)

        self.fusion = ActivityAwareFusion(n_filters, self.n_activities)
        self.decoder = EnhancedDecoder(n_filters, dropout)

    def forward(
        self,
        waist: torch.Tensor,
        wrist: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:

        waist_feat, waist_epi = self.waist_encoder(waist)

        if wrist is None:
            fused_feat = waist_feat
            weights = torch.tensor([[1.0, 0.0]], device=waist.device, dtype=waist.dtype).repeat(waist.size(0), 1)
            activity_probs = torch.zeros(waist.size(0), self.n_activities, device=waist.device, dtype=waist.dtype)
            wrist_epi = torch.zeros_like(waist_epi)
        else:
            wrist_feat, wrist_epi = self.wrist_encoder(wrist)
            fused_feat, weights, activity_probs = self.fusion(waist_feat, wrist_feat, waist_epi, wrist_epi)

        pred, ale = self.decoder(fused_feat)

        if not return_all:
            return {"prediction": pred}

        combined_epi = weights[:, 0:1] * waist_epi + weights[:, 1:2] * wrist_epi  # (B,1)
        epi_t = combined_epi.expand(-1, pred.size(1))                               # (B,T)
        total_u = torch.sqrt(epi_t**2 + ale**2 + 1e-8)

        return {
            "prediction": pred,
            "aleatoric_unc": ale,
            "epistemic_unc": combined_epi,
            "total_uncertainty": total_u,
            "fusion_weights": weights,
            "activity_probs": activity_probs,
        }

    @torch.no_grad()
    def predict_with_confidence(
        self,
        waist: torch.Tensor,
        wrist: Optional[torch.Tensor] = None,
        confidence_level: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        out = self.forward(waist, wrist, return_all=True)
        z = z_score_from_confidence(confidence_level)

        pred = out["prediction"]
        sigma = out["total_uncertainty"]

        return {
            **out,
            "lower_bound": pred - z * sigma,
            "upper_bound": pred + z * sigma,
        }


# -----------------------------------------------------------------------------
# PART 5: LOSSES
# -----------------------------------------------------------------------------

@dataclass
class BiomechTargets:
    """Targets in same scale as training force (BW/kN/N)."""
    target_mean: float = 1.0
    peak_cap: float = 3.0


class BiomechanicalPriors(nn.Module):
    """
    Light priors. Inputs:
      pred: (B,T)
      mask: (B,T) bool
    """

    def __init__(
        self,
        lambda_impulse: float = 0.1,
        lambda_temporal: float = 0.05,
        lambda_magnitude: float = 0.05,
        targets: BiomechTargets = BiomechTargets(),
        peak_center: float = 0.4,
        peak_width: float = 0.2,
    ):
        super().__init__()
        self.lambda_impulse = float(lambda_impulse)
        self.lambda_temporal = float(lambda_temporal)
        self.lambda_magnitude = float(lambda_magnitude)
        self.targets = targets
        self.peak_center = float(peak_center)
        self.peak_width = float(peak_width)

    def negative_impulse_penalty(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        impulse = (pred * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return F.relu(-impulse).mean()

    def temporal_consistency_penalty(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        penalties = []
        B, _ = pred.shape
        for i in range(B):
            valid = pred[i][mask[i]]
            if valid.numel() < 10:
                continue
            peak_idx = torch.argmax(valid)
            loc = peak_idx.float() / valid.numel()  # 0..1
            deviation = (loc - self.peak_center) / self.peak_width
            penalties.append(F.relu(torch.abs(deviation) - 1.0))
        if not penalties:
            return torch.tensor(0.0, device=pred.device)
        return torch.stack(penalties).mean()

    def magnitude_consistency_penalty(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        mean_force = (pred * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        mean_penalty = torch.abs(mean_force - self.targets.target_mean).mean()

        peak_force = (pred * m).max(dim=1)[0]
        peak_penalty = F.relu(peak_force - self.targets.peak_cap).mean()

        return mean_penalty + peak_penalty

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        imp = self.negative_impulse_penalty(pred, mask)
        tmp = self.temporal_consistency_penalty(pred, mask)
        mag = self.magnitude_consistency_penalty(pred, mask)

        priors_total = (
            self.lambda_impulse * imp
            + self.lambda_temporal * tmp
            + self.lambda_magnitude * mag
        )

        return {
            "priors_total": priors_total,
            "impulse_penalty": imp,
            "temporal_penalty": tmp,
            "magnitude_penalty": mag,
        }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    return (((pred - target) ** 2) * m).sum() / m.sum().clamp(min=1.0)


def heteroscedastic_nll(
    pred: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    m = mask.float()
    var = sigma**2 + 1e-6
    per_t = (pred - target) ** 2 / (2.0 * var) + beta * torch.log(var)
    return (per_t * m).sum() / m.sum().clamp(min=1.0)


def activity_classification_loss(
    activity_probs: torch.Tensor,
    activity_labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logp = torch.log(activity_probs + 1e-8)
    return F.nll_loss(logp, activity_labels, weight=class_weights)


def combined_training_loss(
    model_outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    activity_labels: torch.Tensor,
    mask: torch.Tensor,
    priors: BiomechanicalPriors,
    class_weights: Optional[torch.Tensor] = None,
    lambda_activity: float = 0.01,
    beta_uncertainty: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    pred = model_outputs["prediction"]
    ale = model_outputs["aleatoric_unc"]
    act_probs = model_outputs["activity_probs"]

    nll = heteroscedastic_nll(pred, ale, target, mask, beta=beta_uncertainty)
    prior_terms = priors(pred, mask)
    act = activity_classification_loss(act_probs, activity_labels, class_weights=class_weights)

    total = nll + prior_terms["priors_total"] + lambda_activity * act

    logs = {
        "total": total,
        "nll": nll,
        "mse": masked_mse(pred, target, mask),
        "priors_total": prior_terms["priors_total"],
        "impulse_penalty": prior_terms["impulse_penalty"],
        "temporal_penalty": prior_terms["temporal_penalty"],
        "magnitude_penalty": prior_terms["magnitude_penalty"],
        "activity": act,
    }
    return total, logs


# -----------------------------------------------------------------------------
# PART 6: ROBUSTNESS HELPERS
# -----------------------------------------------------------------------------

@torch.no_grad()
def predict_single_sensor(
    model: EnhancedUASF_GRFNet,
    sensor: torch.Tensor,
    which: str = "waist",
    confidence_level: float = 0.95,
) -> Dict[str, torch.Tensor]:
    model.eval()
    z = z_score_from_confidence(confidence_level)

    which = which.lower().strip()
    if which == "waist":
        feat, epi = model.waist_encoder(sensor)
        weights = torch.tensor([[1.0, 0.0]], device=sensor.device, dtype=sensor.dtype).repeat(sensor.size(0), 1)
    elif which == "wrist":
        feat, epi = model.wrist_encoder(sensor)
        weights = torch.tensor([[0.0, 1.0]], device=sensor.device, dtype=sensor.dtype).repeat(sensor.size(0), 1)
    else:
        raise ValueError("which must be 'waist' or 'wrist'")

    pred, ale = model.decoder(feat)
    epi_t = epi.expand(-1, pred.size(1))
    total_u = torch.sqrt(epi_t**2 + ale**2 + 1e-8)

    return {
        "prediction": pred,
        "aleatoric_unc": ale,
        "epistemic_unc": epi,
        "total_uncertainty": total_u,
        "lower_bound": pred - z * total_u,
        "upper_bound": pred + z * total_u,
        "fusion_weights": weights,
        "activity_probs": torch.zeros(sensor.size(0), model.n_activities, device=sensor.device, dtype=sensor.dtype),
    }


@torch.no_grad()
def validate_robustness(
    model: EnhancedUASF_GRFNet,
    waist: torch.Tensor,
    wrist: torch.Tensor,
    confidence_level: float = 0.95,
    corrupt_ratio: float = 0.3,
    corrupt_scale: float = 5.0,
) -> Dict[str, Dict[str, torch.Tensor]]:

    model.eval()
    out_both = model.predict_with_confidence(waist, wrist, confidence_level=confidence_level)
    out_waist = predict_single_sensor(model, waist, which="waist", confidence_level=confidence_level)
    out_wrist = predict_single_sensor(model, wrist, which="wrist", confidence_level=confidence_level)

    wrist_corrupted = wrist.clone()
    T = wrist.shape[1]
    corrupt_mask = torch.rand(T, device=wrist.device) < float(corrupt_ratio)
    wrist_corrupted[:, corrupt_mask, :] = torch.randn_like(wrist_corrupted[:, corrupt_mask, :]) * float(corrupt_scale)

    out_corrupted = model.predict_with_confidence(waist, wrist_corrupted, confidence_level=confidence_level)

    return {
        "both_sensors": out_both,
        "waist_only": out_waist,
        "wrist_only": out_wrist,
        "corrupted_wrist": out_corrupted,
    }


__all__ = [
    "EnhancedUncertaintySensorEncoder",
    "ActivityAwareFusion",
    "EnhancedDecoder",
    "EnhancedUASF_GRFNet",
    "BiomechTargets",
    "BiomechanicalPriors",
    "masked_mse",
    "heteroscedastic_nll",
    "activity_classification_loss",
    "combined_training_loss",
    "predict_single_sensor",
    "validate_robustness",
    "z_score_from_confidence",
]
