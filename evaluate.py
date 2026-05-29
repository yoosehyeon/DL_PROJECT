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

    # R² 음수 자동 경고 (P3-2):
    # R² < 0 은 "모델이 단순 평균 예측보다도 못함"을 의미.
    # 보통 test 분포의 분산이 비정상적으로 작거나(분포 왜곡), 모델이 train/test
    # 분포 shift에 실패했을 때 발생. RMSE 만 보면 놓치기 쉬워 명시적 경고 출력.
    warnings_list = []
    y_std = float(np.std(y))
    if r2 < 0:
        # ASCII-only 메시지: Windows cp949 콘솔에서 unicode dash 등이 깨지지 않도록 한다.
        msg = (
            f"R2={r2:.3f} (negative) - test std={y_std:.3f} too small, "
            f"mean-prediction baseline dominates. RMSE={rmse:.2f} alone is misleading. "
            f"Consider expanding sampling (e.g. max_units_test) in config for '{name}'."
        )
        warnings_list.append({"code": "negative_r2", "message": msg})
        print(f"[evaluate_regressor:WARN] {name} -> {msg}")

    out = {
        "name": name,
        "task": "regression",
        "rmse": rmse,
        "mae":  mae,
        "mse":  mse,
        "r2":   r2,
        "y_test_std": y_std,
        "n_test": int(len(y)),
        "rul_norm": bool(rul_norm),
    }
    if warnings_list:
        out["warnings"] = warnings_list
    return out


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
# 5) CNN + GBDT 스태킹 앙상블 (AI4I 전용)
# ---------------------------------------------------------------------------

def evaluate_ai4i_stacking(
    cnn_model: nn.Module,
    gbdt_model,
    data_cnn: Dict,
    data_gbdt: Dict,
    cnn_weight: Optional[float] = None,
) -> Dict:
    """AI4I CNN과 GBDT의 확률 출력을 가중 평균으로 앙상블한다.

    절차:
      1) val set에서 두 모델 확률 산출 → 가중치 grid + threshold grid 동시 탐색
         (cnn_weight=None일 때 [0.0, 0.1, ..., 1.0] 11개 grid 자동 탐색)
      2) val F1을 최대화하는 (w_cnn, threshold) 채택
      3) test set에 적용하여 표준 메트릭 보고

    가정: data_cnn["X_val/test"]와 data_gbdt["X_val/test"]는 동일한 split
          (load_ai4i_cnn → load_ai4i_gbdt 가 같은 SEED로 split하므로 성립).
    """
    device = torch.device(config.get_device())
    cnn_model.to(device)
    cnn_model.eval()

    # ── CNN 확률 ─────────────────────────────────────────────────────
    val_logits  = _batched_infer(cnn_model, data_cnn["X_val"],  device).reshape(-1)
    test_logits = _batched_infer(cnn_model, data_cnn["X_test"], device).reshape(-1)
    val_p_cnn   = 1.0 / (1.0 + np.exp(-val_logits))
    test_p_cnn  = 1.0 / (1.0 + np.exp(-test_logits))

    # ── GBDT 확률 ────────────────────────────────────────────────────
    val_p_gbdt  = gbdt_model.predict_proba(data_gbdt["X_val"])[:,  1]
    test_p_gbdt = gbdt_model.predict_proba(data_gbdt["X_test"])[:, 1]

    y_val  = np.asarray(data_cnn["y_val"]).astype(np.int32)
    y_test = np.asarray(data_cnn["y_test"]).astype(np.int32)

    # ── 가중치 grid (auto) ───────────────────────────────────────────
    if cnn_weight is None:
        w_grid = np.arange(0.0, 1.01, 0.1)
    else:
        w_grid = np.array([float(cnn_weight)])

    best = {"w_cnn": None, "threshold": None, "val_f1": -1.0}
    per_weight_best: List[Dict] = []
    for w in w_grid:
        val_p = w * val_p_cnn + (1.0 - w) * val_p_gbdt
        local_best = {"threshold": None, "val_f1": -1.0}
        for thr in np.arange(0.05, 1.0, 0.05):
            yp = (val_p > thr).astype(np.int32)
            f1v = float(f1_score(y_val, yp, zero_division=0))
            if f1v > local_best["val_f1"]:
                local_best = {"threshold": float(thr), "val_f1": f1v}
        per_weight_best.append({
            "w_cnn": float(w),
            "best_threshold": local_best["threshold"],
            "best_val_f1": local_best["val_f1"],
        })
        if local_best["val_f1"] > best["val_f1"]:
            best = {
                "w_cnn": float(w),
                "threshold": local_best["threshold"],
                "val_f1": local_best["val_f1"],
            }

    if best["w_cnn"] is None:
        raise RuntimeError("evaluate_ai4i_stacking: val grid search produced no result")

    # ── test 적용 ────────────────────────────────────────────────────
    w_cnn = best["w_cnn"]
    used_thr = best["threshold"]
    test_p = w_cnn * test_p_cnn + (1.0 - w_cnn) * test_p_gbdt
    y_pred = (test_p > used_thr).astype(np.int32)

    return {
        "name": "ai4i_stack",
        "task": "ensemble_binary",
        "members": ["ai4i_cnn", "ai4i_gbdt"],
        "weights": {"cnn": w_cnn, "gbdt": 1.0 - w_cnn},
        "decision_threshold": used_thr,
        "val_f1_at_best": best["val_f1"],
        "weight_grid": per_weight_best,
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "n_test":    int(len(y_test)),
    }


# ---------------------------------------------------------------------------
# 6) N-way 스태킹 앙상블 (AI4I - CNN + GBDT + CatBoost 등 임의 조합)
# ---------------------------------------------------------------------------

def evaluate_ai4i_stacking_nway(
    members: Dict[str, tuple],
    weight_step: float = 0.1,
) -> Dict:
    """임의 개수의 AI4I 모델을 가중 평균으로 앙상블한다.

    파라미터:
      members      : {name: (model, data_dict)} — 각 모델은 .pt/.pkl 어느 쪽이든 OK
      weight_step  : 가중치 grid step (기본 0.1, 즉 11개 후보 weight)

    절차:
      1) 각 멤버에 대해 val/test 확률 추출
         - nn.Module: sigmoid(logit)
         - sklearn/CatBoost: predict_proba[:, 1]
      2) 가중치 grid (Dirichlet 격자) × threshold grid 로 (w_vec, thr) 동시 탐색
      3) val F1 최대 조합을 채택, test 메트릭 보고

    가중치 정규화: w_vec 합이 1이 되도록 자동 정규화. 합이 0인 후보는 skip.
    """
    if len(members) < 2:
        raise ValueError("evaluate_ai4i_stacking_nway requires >= 2 members")

    device = torch.device(config.get_device())
    member_names = list(members.keys())

    val_probs: Dict[str, np.ndarray] = {}
    test_probs: Dict[str, np.ndarray] = {}
    y_val_ref: Optional[np.ndarray] = None
    y_test_ref: Optional[np.ndarray] = None

    for name, (model, data) in members.items():
        # nn.Module 분기: sigmoid(logit). 그 외(sklearn/CatBoost)는 predict_proba.
        if isinstance(model, nn.Module):
            model.to(device).eval()
            val_logits  = _batched_infer(model, data["X_val"],  device).reshape(-1)
            test_logits = _batched_infer(model, data["X_test"], device).reshape(-1)
            val_probs[name]  = 1.0 / (1.0 + np.exp(-val_logits))
            test_probs[name] = 1.0 / (1.0 + np.exp(-test_logits))
        else:
            val_probs[name]  = model.predict_proba(data["X_val"])[:,  1]
            test_probs[name] = model.predict_proba(data["X_test"])[:, 1]

        # 정답 라벨 일치성 확인 (모든 멤버가 같은 split 가정)
        y_val_cur  = np.asarray(data["y_val"]).astype(np.int32)
        y_test_cur = np.asarray(data["y_test"]).astype(np.int32)
        if y_val_ref is None:
            y_val_ref, y_test_ref = y_val_cur, y_test_cur
        else:
            if not (np.array_equal(y_val_ref, y_val_cur) and
                    np.array_equal(y_test_ref, y_test_cur)):
                raise RuntimeError(
                    f"stacking members have mismatched labels — "
                    f"동일한 split (같은 SEED) 사용해야 함"
                )

    # 가중치 grid (Dirichlet-style): 각 weight ∈ {0, 0.1, ..., 1.0}, 합 정규화
    candidates = list(np.round(np.arange(0.0, 1.0 + 1e-9, weight_step), 2))
    threshold_grid = list(np.round(np.arange(0.05, 1.0, 0.05), 2))

    n_members = len(member_names)
    best = {"weights": None, "threshold": None, "val_f1": -1.0}
    n_eval = 0

    def _iter_weights(remain: int, current: list):
        """n_members 길이의 weight 후보 (전수 탐색)."""
        if remain == 1:
            yield current + [0.0]   # placeholder, normalize 단계에서 무관
            return
        for w in candidates:
            yield from _iter_weights(remain - 1, current + [w])

    for w_partial in _iter_weights(n_members, []):
        # 마지막 위치만 candidates 순회 (전수 탐색)
        for last in candidates:
            w = np.array(w_partial[:-1] + [last], dtype=np.float64)
            s = w.sum()
            if s <= 0:
                continue
            w = w / s   # 정규화
            # val 가중 평균
            val_blend = sum(w[i] * val_probs[name] for i, name in enumerate(member_names))
            for thr in threshold_grid:
                yp = (val_blend > thr).astype(np.int32)
                f1v = float(f1_score(y_val_ref, yp, zero_division=0))
                n_eval += 1
                if f1v > best["val_f1"]:
                    best = {
                        "weights": {name: float(w[i]) for i, name in enumerate(member_names)},
                        "threshold": float(thr),
                        "val_f1": f1v,
                    }

    if best["weights"] is None:
        raise RuntimeError("evaluate_ai4i_stacking_nway: grid search produced no result")

    # test 적용
    test_blend = sum(best["weights"][name] * test_probs[name] for name in member_names)
    y_pred = (test_blend > best["threshold"]).astype(np.int32)

    return {
        "name": "ai4i_stack_nway",
        "task": "ensemble_binary",
        "members": member_names,
        "weights": best["weights"],
        "decision_threshold": best["threshold"],
        "val_f1_at_best": best["val_f1"],
        "n_combinations_evaluated": n_eval,
        "accuracy":  float(accuracy_score(y_test_ref, y_pred)),
        "precision": float(precision_score(y_test_ref, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test_ref, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test_ref, y_pred, zero_division=0)),
        "n_test":    int(len(y_test_ref)),
    }


# ---------------------------------------------------------------------------
# 7) C-MAPSS Multi-Window 회귀 앙상블 (Phase 1-3)
# ---------------------------------------------------------------------------

def evaluate_cmapss_multiwindow_ensemble(
    members: Dict[str, tuple],
) -> Dict:
    """C-MAPSS multi-window LSTM 앙상블 (예측 평균).

    파라미터:
      members : {name: (model, data)} — 각각 다른 window 길이로 학습된 LSTM

    절차:
      1) 각 모델로 test 추론
      2) test set이 모델마다 다를 수 있으므로 (window별 마지막 window 만 선택 →
         실제로 test 엔진 수가 동일하므로 비교 가능)
      3) 단순 평균 / 가중 평균 (val RMSE 역수 기반) 두 가지 보고
    """
    if len(members) < 2:
        raise ValueError("multi-window ensemble requires >= 2 members")

    device = torch.device(config.get_device())
    member_preds: Dict[str, np.ndarray] = {}
    member_val_rmse: Dict[str, float] = {}
    y_test_ref: Optional[np.ndarray] = None

    for name, (model, data) in members.items():
        model.to(device).eval()
        meta = data.get("meta", {})
        rul_norm = meta.get("rul_norm", False)
        rul_clip = meta.get("rul_clip", 1.0)

        # test 예측
        pred = _batched_infer(model, data["X_test"], device).reshape(-1)
        y    = np.asarray(data["y_test"]).reshape(-1).astype(np.float32)
        if rul_norm:
            pred = pred * rul_clip
            y    = y * rul_clip
        member_preds[name] = pred

        # val RMSE (가중 평균용)
        val_pred = _batched_infer(model, data["X_val"], device).reshape(-1)
        val_y    = np.asarray(data["y_val"]).reshape(-1).astype(np.float32)
        if rul_norm:
            val_pred = val_pred * rul_clip
            val_y    = val_y * rul_clip
        member_val_rmse[name] = float(np.sqrt(mean_squared_error(val_y, val_pred)))

        if y_test_ref is None:
            y_test_ref = y
        elif len(y) == len(y_test_ref):
            # 엔진 단위 마지막 window 선택은 동일하므로 길이 같으면 OK
            pass
        else:
            raise RuntimeError(
                f"multi-window members have mismatched test size: "
                f"{name}={len(y)} vs ref={len(y_test_ref)}"
            )

    # 단순 평균
    simple_avg = np.mean([member_preds[n] for n in members], axis=0)
    rmse_simple = float(np.sqrt(mean_squared_error(y_test_ref, simple_avg)))
    mae_simple  = float(mean_absolute_error(y_test_ref, simple_avg))
    r2_simple   = float(r2_score(y_test_ref, simple_avg))

    # 가중 평균 (val RMSE 역수 정규화)
    inv_rmse = {n: 1.0 / (member_val_rmse[n] + 1e-6) for n in members}
    total = sum(inv_rmse.values())
    weights = {n: inv_rmse[n] / total for n in members}
    weighted = sum(weights[n] * member_preds[n] for n in members)
    rmse_weighted = float(np.sqrt(mean_squared_error(y_test_ref, weighted)))
    mae_weighted  = float(mean_absolute_error(y_test_ref, weighted))
    r2_weighted   = float(r2_score(y_test_ref, weighted))

    return {
        "name": "cmapss_multiwindow_ensemble",
        "task": "regression_ensemble",
        "members": list(members.keys()),
        "member_val_rmse": member_val_rmse,
        "weights_for_weighted_avg": weights,
        "simple_avg":   {"rmse": rmse_simple,   "mae": mae_simple,   "r2": r2_simple},
        "weighted_avg": {"rmse": rmse_weighted, "mae": mae_weighted, "r2": r2_weighted},
        "n_test": int(len(y_test_ref)),
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
