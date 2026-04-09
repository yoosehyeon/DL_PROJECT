"""HybridPdM - 모델 해석.

Captum의 IntegratedGradients를 활용하여 예측에 가장 기여한 입력 채널을 산출한다.
Top-K 피처 중요도를 다음 task별로 지원:

  · CNN 분류    : 입력 (B, C, L). 채널·시점별 |attr| 평균 → Top-K
  · LSTM 회귀   : 입력 (B, F, L). feature 축으로 집계 → Top-K
  · AE 이상탐지 : reconstruction error를 target으로 하는 wrapper로 attribution

설계 원칙:
  - captum이 설치되지 않은 환경에서도 import 자체는 가능하도록 lazy import.
  - 모델 forward가 (B,) scalar를 반환하지 않는 경우 wrapper로 감싼다.
  - 결과는 numpy로 반환 + Top-K 인덱스/이름 함께 제공.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

import config


# ---------------------------------------------------------------------------
# Captum 지연 로딩
# ---------------------------------------------------------------------------

def _get_captum():
    """Captum을 lazy import. 설치 안 됐을 때 친절한 에러."""
    try:
        from captum.attr import IntegratedGradients
        return IntegratedGradients
    except ImportError as e:
        raise RuntimeError(
            "captum is required for explain.py. Install with `pip install captum`."
        ) from e


# ---------------------------------------------------------------------------
# Forward wrapper (target scalar 보장)
# ---------------------------------------------------------------------------

class _BinaryClsWrapper(nn.Module):
    """이진 분류 모델 출력 (B,1)/(B,) → 확률 (B,)로 변환."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        logit = self.model(x).view(-1)
        return torch.sigmoid(logit)


class _MultiClsWrapper(nn.Module):
    """다중 분류 모델 (B,C) → 지정 클래스 확률 (B,)."""
    def __init__(self, model: nn.Module, target_class: int):
        super().__init__()
        self.model = model
        self.target_class = target_class

    def forward(self, x):
        logits = self.model(x)
        return torch.softmax(logits, dim=1)[:, self.target_class]


class _RegressionWrapper(nn.Module):
    """회귀 모델 출력 (B,) 그대로 사용."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).view(-1)


class _AEWrapper(nn.Module):
    """AE 재구성 오차 (B,)를 target으로. 큰 오차일수록 '이상'."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        recon = self.model(x)
        return ((recon - x) ** 2).mean(dim=1)


# ---------------------------------------------------------------------------
# 핵심 attribution 함수
# ---------------------------------------------------------------------------

def _ig_attribute(
    wrapped: nn.Module,
    X: np.ndarray,
    device: torch.device,
    n_steps: int = 32,
    baseline_value: float = 0.0,
) -> np.ndarray:
    """IntegratedGradients로 attribution 계산. 입력과 동일 shape의 numpy 반환."""
    IG = _get_captum()
    wrapped.to(device).eval()
    x_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    baseline = torch.full_like(x_t, baseline_value)
    ig = IG(wrapped)
    attr = ig.attribute(x_t, baselines=baseline, n_steps=n_steps)
    return attr.detach().cpu().numpy()


def _aggregate_to_features(attr: np.ndarray, mode: str) -> np.ndarray:
    """sample 차원 평균 → feature별 중요도 (1D).

    mode:
      - "BCL_channel"  : (B,C,L) → 채널 축 |attr| 평균   → (C,)
      - "BCL_seq"      : (B,C,L) → 시퀀스 축 |attr| 평균 → (L,)
      - "BCL_feature"  : (B,F,L) where dim=1 is feature → (F,)
                          (LSTM BFL 입력 가정)
    """
    if attr.ndim != 3:
        raise RuntimeError(f"_aggregate_to_features expects 3D, got {attr.shape}")
    a = np.abs(attr)
    if mode == "BCL_channel":
        return a.mean(axis=(0, 2))      # (C,)
    if mode == "BCL_seq":
        return a.mean(axis=(0, 1))      # (L,)
    if mode == "BCL_feature":
        return a.mean(axis=(0, 2))      # (F,) - dim=1이 feature
    raise ValueError(f"Unknown aggregation mode: {mode}")


def _topk(scores: np.ndarray, k: int,
          names: Optional[Sequence[str]] = None) -> List[Dict]:
    """중요도 점수에서 Top-K 항목을 dict 리스트로 반환."""
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    out = []
    for rank, idx in enumerate(order, 1):
        out.append({
            "rank": rank,
            "index": int(idx),
            "name": names[idx] if names is not None and idx < len(names) else f"f{idx}",
            "score": float(scores[idx]),
        })
    return out


# ---------------------------------------------------------------------------
# Task별 진입점
# ---------------------------------------------------------------------------

def explain_classifier(
    model: nn.Module,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    target_class: Optional[int] = None,
    top_k: int = 10,
) -> Dict:
    """CNN 분류 모델의 Top-K 중요 채널 산출.

    n_classes==1이면 sigmoid 확률, ≥2이면 target_class(없으면 0)의 확률을
    target으로 IG 수행. AI4I(짧은 길이) 입력은 채널이 1이고 길이가 의미를
    가지므로 'BCL_seq' 집계로 위치(=원래의 feature)별 중요도를 본다.
    """
    device = torch.device(config.get_device())
    n_classes = getattr(model, "n_classes", 1)
    if n_classes == 1:
        wrapped = _BinaryClsWrapper(model)
    else:
        tc = 0 if target_class is None else int(target_class)
        wrapped = _MultiClsWrapper(model, tc)

    attr = _ig_attribute(wrapped, X, device)        # (B, C, L)

    # 입력 길이 L > 1이면 위치(시퀀스) 축이 의미 단위 → 그것을 feature로 사용
    # 길이 L == 1 (vibration의 GAP 후가 아님 - 여기선 raw input이라 길이가 큼)
    if attr.shape[-1] > attr.shape[1]:
        # 길이 >> 채널 → 위치 기반 (예: AI4I (B,1,11), CWRU (B,1,1024))
        scores = _aggregate_to_features(attr, mode="BCL_seq")
    else:
        scores = _aggregate_to_features(attr, mode="BCL_channel")

    return {
        "task": "classification",
        "n_samples": int(len(X)),
        "top_k": _topk(scores, top_k, feature_names),
        "scores": scores.tolist(),
    }


def explain_regressor(
    model: nn.Module,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
) -> Dict:
    """LSTM RUL 회귀 모델의 Top-K 중요 피처 산출.

    입력 (B, F, L)에서 dim=1이 feature이므로 _aggregate(BCL_feature) 사용.
    BiLSTMRegressor의 input_format='BFL' 가정에 맞춤.
    """
    device = torch.device(config.get_device())
    wrapped = _RegressionWrapper(model)
    attr = _ig_attribute(wrapped, X, device)        # (B, F, L)
    scores = _aggregate_to_features(attr, mode="BCL_feature")
    return {
        "task": "regression",
        "n_samples": int(len(X)),
        "top_k": _topk(scores, top_k, feature_names),
        "scores": scores.tolist(),
    }


def explain_autoencoder(
    model: nn.Module,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
) -> Dict:
    """AE의 재구성 오차에 가장 기여한 입력 차원 산출.

    AE 입력은 (B, F) 2D이므로 attr도 (B, F). sample 평균만 취하면 (F,).
    """
    device = torch.device(config.get_device())
    wrapped = _AEWrapper(model)
    attr = _ig_attribute(wrapped, X, device)  # baseline=0, n_steps=32
    if attr.ndim != 2:
        raise RuntimeError(f"AE attribution expected 2D, got {attr.shape}")
    scores = np.abs(attr).mean(axis=0)              # (F,)
    return {
        "task": "anomaly_detection",
        "n_samples": int(len(X)),
        "top_k": _topk(scores, top_k, feature_names),
        "scores": scores.tolist(),
    }


# ---------------------------------------------------------------------------
# 디스패처
# ---------------------------------------------------------------------------

EXPLAINERS = {
    "classification":         explain_classifier,
    "binary_classification":  explain_classifier,
    "multiclass":             explain_classifier,
    "anomaly_detection":      explain_autoencoder,
    "regression":             explain_regressor,
}
