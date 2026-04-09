"""HybridPdM - Risk Score 계산.

3개 모델 출력을 통합한 Risk Score를 두 가지 융합 전략으로 제공한다.
PdM 도메인 특성상 False Negative(고장을 정상으로 오판) 비용이 매우 크므로
Noisy-OR 방식도 함께 제공한다.

【두 가지 융합 전략】

1) Weighted Sum (기본)
     R = w_f * P(failure) + w_a * Anomaly + w_r * (1 - RUL_norm)
   - 직관적·해석 용이.
   - 한 모델만 강하게 발화해도 다른 모델이 0이면 점수가 희석됨 → FN 위험.

2) Noisy-OR (FN 최소화용 보조)
     R = 1 - Π_i (1 - p_i)^{w_i}
   - "어느 한 모델이라도 위험하다고 보면 위험"이라는 OR 의미.
   - 한 모델만 강하게 발화해도 점수가 1에 가까워져 FN을 줄임.

【Risk 등급 매핑】
config.RISK_LEVELS의 내림차순 임계값을 순회하여 첫 매칭으로 등급 부여.

입력 정규화:
  - failure_prob:  CNN sigmoid/softmax 확률.       [0, 1]
  - anomaly_score: AE.anomaly_score() 결과.        [0, 1]
  - rul_norm:      RUL / rul_clip 으로 정규화한 값. [0, 1]
                   "1 - rul_norm" 형태로 반전하여 위험도화.
모두 numpy/스칼라 둘 다 지원.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np

import config

ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# 입력 검증
# ---------------------------------------------------------------------------

def _as_array(x: ArrayLike, name: str) -> np.ndarray:
    """입력을 1D float numpy로 변환하고 [0,1] 범위 검증."""
    a = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    # PdM 도메인에서 음수/1초과는 정규화 단계의 버그일 가능성이 높음
    if (a < 0).any() or (a > 1).any():
        raise ValueError(
            f"{name} must be in [0, 1], got min={a.min():.4f} max={a.max():.4f}. "
            "Did you forget to normalize? (sigmoid for prob, /rul_clip for RUL)"
        )
    return a


# ---------------------------------------------------------------------------
# 1) Weighted Sum (PRD 기본)
# ---------------------------------------------------------------------------

def weighted_sum(
    failure_prob: ArrayLike,
    anomaly_score: ArrayLike,
    rul_norm: ArrayLike,
    weights: Dict[str, float] = None,
) -> np.ndarray:
    """가중합 Risk Score.

    R = w_f * failure + w_a * anomaly + w_r * (1 - rul_norm)
    가중치 합이 1이 아니면 정규화하여 결과가 [0,1]에 머물도록 한다.
    """
    if weights is None:
        weights = config.RISK_WEIGHTS
    f = _as_array(failure_prob,  "failure_prob")
    a = _as_array(anomaly_score, "anomaly_score")
    r = _as_array(rul_norm,      "rul_norm")

    w_f = weights.get("failure", 0.5)
    w_a = weights.get("anomaly", 0.3)
    w_r = weights.get("rul",     0.2)
    w_sum = w_f + w_a + w_r
    if w_sum <= 0:
        raise ValueError(f"weights sum must be > 0, got {w_sum}")
    w_f, w_a, w_r = w_f / w_sum, w_a / w_sum, w_r / w_sum

    return (w_f * f + w_a * a + w_r * (1.0 - r)).astype(np.float32)


# ---------------------------------------------------------------------------
# 2) Noisy-OR (FN 최소화 보조)
# ---------------------------------------------------------------------------

def noisy_or(
    failure_prob: ArrayLike,
    anomaly_score: ArrayLike,
    rul_norm: ArrayLike,
    weights: Dict[str, float] = None,
) -> np.ndarray:
    """가중치가 부여된 Noisy-OR 융합.

      R = 1 - (1 - f)^{w_f} * (1 - a)^{w_a} * (1 - (1-r))^{w_r}

    weight는 [0,1]로 클램프하지 않고 그대로 지수에 사용한다 (작을수록 영향↓).
    한 모델만 강하게 발화해도 R이 1에 가까워져 FN 위험이 크게 줄어든다.

    수치 안정성: 1-p가 정확히 0이 되는 경우를 피하기 위해 1e-9 clip.
    """
    if weights is None:
        weights = config.RISK_WEIGHTS
    f = _as_array(failure_prob,  "failure_prob")
    a = _as_array(anomaly_score, "anomaly_score")
    r = _as_array(rul_norm,      "rul_norm")
    rul_risk = 1.0 - r

    eps = 1e-9
    one_minus = lambda v: np.clip(1.0 - v, eps, 1.0)

    w_f = float(weights.get("failure", 0.5))
    w_a = float(weights.get("anomaly", 0.3))
    w_r = float(weights.get("rul",     0.2))

    prod = (one_minus(f) ** w_f) * (one_minus(a) ** w_a) * (one_minus(rul_risk) ** w_r)
    return (1.0 - prod).astype(np.float32)


# ---------------------------------------------------------------------------
# 3) 통합 인터페이스
# ---------------------------------------------------------------------------

def compute_risk(
    failure_prob: ArrayLike,
    anomaly_score: ArrayLike,
    rul_norm: ArrayLike,
    fusion: str = "max",
) -> np.ndarray:
    """Risk Score 계산 진입점.

    fusion:
      - "weighted" : 가중합만 사용 (해석성 우선)
      - "noisy_or" : Noisy-OR만 사용 (FN 최소화 우선)
      - "max"      : 두 방식의 최댓값 (보수적, PdM 기본 권장)
    """
    if fusion == "weighted":
        return weighted_sum(failure_prob, anomaly_score, rul_norm)
    if fusion == "noisy_or":
        return noisy_or(failure_prob, anomaly_score, rul_norm)
    if fusion == "max":
        ws = weighted_sum(failure_prob, anomaly_score, rul_norm)
        nor = noisy_or(failure_prob, anomaly_score, rul_norm)
        return np.maximum(ws, nor).astype(np.float32)
    raise ValueError(f"Unknown fusion '{fusion}'. Use 'weighted'|'noisy_or'|'max'.")


# ---------------------------------------------------------------------------
# 등급 매핑
# ---------------------------------------------------------------------------

def to_risk_level(score: ArrayLike) -> Union[str, List[str]]:
    """Risk Score 값을 등급 문자열로 변환.

    내림차순 RISK_LEVELS를 순회하여 첫 매칭. 스칼라/배열 모두 지원.
    """
    levels: List[Tuple[str, float]] = sorted(
        config.RISK_LEVELS, key=lambda x: -x[1]
    )

    def _one(v: float) -> str:
        for name, thr in levels:
            if v >= thr:
                return name
        return levels[-1][0]

    arr = np.atleast_1d(np.asarray(score, dtype=np.float64))
    out = [_one(float(v)) for v in arr]
    if np.isscalar(score) or np.ndim(score) == 0:
        return out[0]
    return out
