"""HybridPdM - 평가 모듈.

세 task에 맞춰 표준 메트릭을 계산한다. AE는 percentile grid search를 수행해
F1을 최대화하는 임계값을 찾는다.

각 evaluate_* 함수는 다음을 반환한다:
  - dict 형태의 메트릭 (best_threshold 등 포함)
  - 평가 후 모델은 그대로 반환 (in-place로 threshold가 갱신될 수 있음)

설계:
  · DataLoader를 거치지 않고 한 번에 텐서로 추론한다 (test set은 보통 작음).
    너무 큰 경우를 대비해 _batched_infer 헬퍼로 청크 추론.
  · sklearn.metrics를 활용 (binary/macro/regression 메트릭).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

import config


# ---------------------------------------------------------------------------
# 공통 추론 헬퍼
# ---------------------------------------------------------------------------

@torch.no_grad()
def _batched_infer(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    fn=None,
) -> np.ndarray:
    """X를 batch 단위로 model.forward (또는 fn(model, x))에 통과시켜 결과를
    numpy로 모아 반환한다.

    fn 인자가 주어지면 해당 콜러블이 호출된다 → AE의 reconstruction_error 등
    forward 외 메서드를 사용할 때 활용.
    """
    model.eval()
    out_chunks: List[np.ndarray] = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        if fn is None:
            yb = model(xb)
        else:
            yb = fn(model, xb)
        out_chunks.append(yb.detach().cpu().numpy())
    if not out_chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(out_chunks, axis=0)


# ---------------------------------------------------------------------------
# 1) 분류 평가
# ---------------------------------------------------------------------------

def evaluate_classifier(
    name: str,
    data: Dict,
    model: nn.Module,
    decision_threshold: float = 0.35,
) -> Dict:
    """이진/다중 분류 평가.

    이진(n_classes=1):
        val set에서 F1-best threshold를 grid search로 찾아 test에 적용한다.
        val set이 없거나 비어 있으면 인자 decision_threshold로 fallback.
        탐색 그리드: 0.05 ~ 0.95, 0.05 step (총 19개).
    다중(n_classes≥2): argmax 예측. macro precision/recall/F1 + accuracy.
    """
    device = torch.device(config.get_device())
    model.to(device)
    n_classes = getattr(model, "n_classes", 1)

    logits = _batched_infer(model, data["X_test"], device)
    y_true = np.asarray(data["y_test"])

    if n_classes == 1:
        # ── val set에서 F1-best threshold 탐색 ──────────────────────────
        used_threshold = float(decision_threshold)
        val_grid: List[Dict] = []
        val_best = {"threshold": None, "f1": -1.0}
        X_val = data.get("X_val")
        y_val = data.get("y_val")
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_logits = _batched_infer(model, X_val, device).reshape(-1)
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            y_val_i = np.asarray(y_val).astype(np.int32)
            for thr in np.arange(0.05, 1.0, 0.05):
                yp = (val_probs > thr).astype(np.int32)
                f1v = float(f1_score(y_val_i, yp, zero_division=0))
                val_grid.append({"threshold": float(thr), "val_f1": f1v})
                if f1v > val_best["f1"]:
                    val_best = {"threshold": float(thr), "f1": f1v}
            if val_best["threshold"] is not None:
                used_threshold = val_best["threshold"]

        # ── test 평가 ────────────────────────────────────────────────────
        logits = logits.reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))   # 수치 안정성을 위해 직접 sigmoid
        y_pred = (probs > used_threshold).astype(np.int32)
        y_true_i = y_true.astype(np.int32)
        return {
            "name": name,
            "task": "binary_classification",
            "decision_threshold": used_threshold,
            "decision_threshold_source": "val_grid_search" if val_best["threshold"] is not None else "fallback_arg",
            "val_f1_at_best": val_best["f1"],
            "val_grid": val_grid,
            "accuracy":  float(accuracy_score(y_true_i, y_pred)),
            "precision": float(precision_score(y_true_i, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true_i, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true_i, y_pred, zero_division=0)),
            "n_test":    int(len(y_true)),
        }

    # 다중 분류
    y_pred = logits.argmax(axis=1).astype(np.int32)
    y_true_i = y_true.astype(np.int32)
    return {
        "name": name,
        "task": "multiclass_classification",
        "n_classes": n_classes,
        "accuracy":  float(accuracy_score(y_true_i, y_pred)),
        "precision": float(precision_score(y_true_i, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true_i, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true_i, y_pred, average="macro", zero_division=0)),
        "n_test":    int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# 2) Autoencoder 평가 (percentile grid search)
# ---------------------------------------------------------------------------

def evaluate_autoencoder(
    name: str,
    data: Dict,
    model: nn.Module,
    percentile_grid: Optional[List[int]] = None,
    use_mahalanobis: bool = False,
) -> Dict:
    """AE 평가 + 임계값 grid search.

    절차:
      1) train(정상)에서 점수 분포를 구하고 각 percentile을 임계값 후보로 만든다.
         · use_mahalanobis=False : reconstruction_error (기본)
         · use_mahalanobis=True  : combined_score (recon z + mahal z)
      2) val set에서 각 임계값으로 F1을 계산해 best 선택.
      3) 선택된 임계값을 model.set_threshold()로 갱신한 뒤 test 메트릭 보고.
    """
    if percentile_grid is None:
        percentile_grid = config.AE_CFG["threshold_grid"]

    device = torch.device(config.get_device())
    model.to(device)

    # use_mahalanobis 분기: 점수 함수만 갈아끼움
    if use_mahalanobis:
        score_fn = lambda m, x: m.combined_score(x)
        score_name = "combined_score(recon_z + mahal_z)"
    else:
        score_fn = lambda m, x: m.reconstruction_error(x)
        score_name = "reconstruction_error"

    # ── 1) 정상 train 점수 분포 ─────────────────────────────────────────
    score_train = _batched_infer(model, data["X_train"], device, fn=score_fn)
    if score_train.size == 0:
        raise RuntimeError("evaluate_autoencoder: empty X_train")

    # ── 2) val set에서 percentile별 F1 ─────────────────────────────────
    score_val = _batched_infer(model, data["X_val"], device, fn=score_fn)
    y_val = np.asarray(data["y_val"]).astype(np.int32)

    grid_results = []
    best = {"percentile": None, "threshold": None, "f1": -1.0}
    for p in percentile_grid:
        thr = float(np.percentile(score_train, p))
        y_pred = (score_val > thr).astype(np.int32)
        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        grid_results.append({"percentile": p, "threshold": thr, "val_f1": f1})
        if f1 > best["f1"]:
            best = {"percentile": p, "threshold": thr, "f1": f1}

    # ── 3) test 평가 ────────────────────────────────────────────────────
    if best["threshold"] is None:
        raise RuntimeError("evaluate_autoencoder: percentile_grid is empty")
    # Mahalanobis 모드일 때 self.threshold는 combined 스케일이라 의미가 다르지만
    # state_dict 보존을 위해 동일 buffer를 갱신해 둔다.
    model.set_threshold(best["threshold"])

    score_test = _batched_infer(model, data["X_test"], device, fn=score_fn)
    y_test = np.asarray(data["y_test"]).astype(np.int32)
    y_pred_test = (score_test > best["threshold"]).astype(np.int32)

    return {
        "name": name,
        "task": "anomaly_detection",
        "score_fn": score_name,
        "use_mahalanobis": bool(use_mahalanobis),
        "best_percentile": best["percentile"],
        "best_threshold":  best["threshold"],
        "val_f1_at_best":  best["f1"],
        "test_accuracy":   float(accuracy_score(y_test, y_pred_test)),
        "test_precision":  float(precision_score(y_test, y_pred_test, zero_division=0)),
        "test_recall":     float(recall_score(y_test, y_pred_test, zero_division=0)),
        "test_f1":         float(f1_score(y_test, y_pred_test, zero_division=0)),
        "grid": grid_results,
        "n_test": int(len(y_test)),
    }


# ---------------------------------------------------------------------------
# 3) RUL 회귀 평가
# ---------------------------------------------------------------------------

def evaluate_regressor(
    name: str,
    data: Dict,
    model: nn.Module,
) -> Dict:
    """LSTM RUL 회귀 평가. RMSE / MAE 보고.

    meta["rul_norm"]=True인 경우(N-CMAPSS) 예측값과 타깃을 원래 cycle 스케일로
    역정규화한 뒤 메트릭을 계산한다. → RMSE/MAE 단위가 cycle이 되어 해석 가능.
    """
    device = torch.device(config.get_device())
    model.to(device)

    pred = _batched_infer(model, data["X_test"], device).reshape(-1)
    y    = np.asarray(data["y_test"]).reshape(-1).astype(np.float32)

    # RUL 타깃이 정규화된 경우 원래 스케일로 복원
    meta = data.get("meta", {})
    rul_norm = meta.get("rul_norm", False)
    rul_clip = meta.get("rul_clip", 1.0)
    if rul_norm:
        pred = pred * rul_clip
        y    = y * rul_clip

    mse  = float(mean_squared_error(y, pred))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y, pred))
    r2   = float(r2_score(y, pred))
    return {
        "name": name,
        "task": "regression",
        "rmse": rmse,
        "mae":  mae,
        "mse":  mse,
        "r2":   r2,
        "n_test": int(len(y)),
        "rul_norm": bool(rul_norm),
    }


# ---------------------------------------------------------------------------
# 4) GBDT 분류 평가 (sklearn)
# ---------------------------------------------------------------------------

def evaluate_gbdt_classifier(
    name: str,
    data: Dict,
    model,
) -> Dict:
    """sklearn GBDT 평가 + val threshold grid search.

    val proba에서 thr ∈ [0.05, 0.95] step 0.05로 F1-best 탐색.
    val에 단일 클래스만 있으면 thr=0.5 fallback.
    test에 best thr 적용해 표준 메트릭 보고.
    """
    proba_val  = model.predict_proba(data["X_val"])[:, 1]
    proba_test = model.predict_proba(data["X_test"])[:, 1]
    y_val  = np.asarray(data["y_val"]).astype(np.int32)
    y_test = np.asarray(data["y_test"]).astype(np.int32)

    used_thr = 0.5
    val_grid: List[Dict] = []
    val_best = {"threshold": None, "f1": -1.0}
    if np.unique(y_val).size >= 2:
        for thr in np.arange(0.05, 1.0, 0.05):
            yp = (proba_val > thr).astype(np.int32)
            f1v = float(f1_score(y_val, yp, zero_division=0))
            val_grid.append({"threshold": float(thr), "val_f1": f1v})
            if f1v > val_best["f1"]:
                val_best = {"threshold": float(thr), "f1": f1v}
        if val_best["threshold"] is not None:
            used_thr = val_best["threshold"]

    y_pred = (proba_test > used_thr).astype(np.int32)
    return {
        "name": name,
        "task": "gbdt_binary",
        "decision_threshold": float(used_thr),
        "decision_threshold_source": (
            "val_grid_search" if val_best["threshold"] is not None else "fallback_0.5"
        ),
        "val_f1_at_best": val_best["f1"],
        "val_grid": val_grid,
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "n_test":    int(len(y_test)),
    }


# ---------------------------------------------------------------------------
# 디스패처
# ---------------------------------------------------------------------------

EVALUATORS = {
    "classification":         evaluate_classifier,
    "binary_classification":  evaluate_classifier,
    "multiclass":             evaluate_classifier,
    "anomaly_detection":      evaluate_autoencoder,
    "regression":             evaluate_regressor,
    "gbdt_binary":            evaluate_gbdt_classifier,
}
