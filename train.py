"""HybridPdM - 학습 모듈.

세 가지 학습 루프(분류/이상탐지/회귀)를 통합하여 제공한다.
모든 학습 함수는 다음 규약을 따른다:
  - 입력: data dict(load_*에서 반환), 모델, 설정 dict
  - 출력: 학습 메트릭 dict (best_val, best_epoch, history 등)
  - 부수효과: best 가중치를 config.CHECKPOINT_DIR/<name>.pt 로 저장,
              메트릭은 같은 이름의 .json으로 저장

설계 포인트:
  · EarlyStopping: val 손실이 patience 동안 개선되지 않으면 종료.
    종료 시점에 best_state로 모델을 복원하여 마지막 epoch의 과적합 가중치를
    사용하지 않도록 한다.
  · 손실 평균: DataLoader의 마지막 partial batch는 길이가 다르므로
    "배치 평균의 평균"이 아니라 "샘플 가중 평균"으로 계산한다.
  · DataLoader: worker_init_fn으로 워커별 시드 고정 (재현성).
  · 체크포인트: state_dict는 .pt, 메트릭은 .json으로 분리 저장.
"""
from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config

# ---------------------------------------------------------------------------
# 실행별 식별자 (run_id)
# ---------------------------------------------------------------------------
# main.py가 학습 시작 전 set_run_id()를 호출하여 설정한다.
# 설정된 경우 체크포인트 파일명에 포함되어 실행별 결과가 누적 보존된다.
# 예: ai4i_cnn_20250409_153022.pt

_RUN_ID: str = ""


def set_run_id(run_id: str) -> None:
    """run_id를 설정. main.py에서 학습 시작 전 한 번 호출."""
    global _RUN_ID
    _RUN_ID = run_id


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

@dataclass
class EarlyStopping:
    """검증 손실 기반 조기 종료기.

    patience epoch 동안 best_score 대비 min_delta 이상 개선이 없으면 stop.
    best_state는 model.state_dict()의 deepcopy를 보관하여 학습 종료 시
    restore() 호출로 복원한다.
    """
    patience: int = 10
    min_delta: float = 0.0
    best_score: float = float("inf")
    best_epoch: int = -1
    counter: int = 0
    should_stop: bool = False
    best_state: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """이번 epoch 점수를 입력. 개선되었으면 True 반환."""
        improved = score < (self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            # CPU 사본을 보관해 GPU 메모리 점유를 최소화
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

    def restore(self, model: nn.Module) -> None:
        """best_state로 모델 가중치 복원. best_state가 없으면 경고."""
        if self.best_state is None:
            warnings.warn(
                "EarlyStopping.restore() called but best_state is None. "
                "Model weights left as-is. Did training run for 0 epochs?",
                UserWarning,
            )
            return
        model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# DataLoader 헬퍼
# ---------------------------------------------------------------------------

def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    y_dtype: torch.dtype,
) -> DataLoader:
    """numpy 입력을 TensorDataset → DataLoader로 변환.

    y_dtype을 명시받는 이유:
      - 분류(BCE): float32  / 다중 분류(CE): long  / 회귀: float32
      - data_pipeline의 y는 task별로 dtype이 다르므로 호출자가 명시한다.
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=y_dtype)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,                  # Windows 호환성 우선 (worker spawn 비용↑)
        worker_init_fn=config.worker_init_fn,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# 공통 학습 루프 (지도학습)
# ---------------------------------------------------------------------------

def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
) -> float:
    """한 epoch을 돌고 샘플 가중 평균 손실을 반환.

    optimizer가 주어지면 학습 모드, 아니면 평가 모드.
    각 배치 손실에 batch_size를 곱해 합산 후 전체 sample 수로 나눈다 →
    마지막 partial batch가 있어도 정확한 평균이 계산된다.
    grad_clip > 0 이면 clip_grad_norm_으로 그래디언트 폭발을 방지한다.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_n = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_n += bs
    return total_loss / max(total_n, 1)


def _train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int,
    lr: float,
    weight_decay: float,
    early_stop_patience: int,
    device: torch.device,
    grad_clip: float = 0.0,
    use_lr_scheduler: bool = False,
) -> Dict:
    """공통 지도학습 루프 (분류/회귀 공용).

    grad_clip > 0 이면 각 배치에서 clip_grad_norm_(max_norm=grad_clip)을 적용한다.
    use_lr_scheduler=True 이면 ReduceLROnPlateau(factor=0.5, patience=5)로
    val 손실이 5 epoch 개선 없을 때 LR을 절반으로 줄인다.
    스케줄러 patience(5) < EarlyStopping patience(10) → LR 감소 후 한 번 더 기회.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )
    stopper = EarlyStopping(patience=early_stop_patience)
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        train_loss = _epoch_pass(model, train_loader, loss_fn, device, optimizer, grad_clip)
        val_loss   = _epoch_pass(model, val_loader,   loss_fn, device, optimizer=None)
        cur_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": cur_lr})
        stopper.step(val_loss, model, epoch)
        if scheduler is not None:
            scheduler.step(val_loss)  # stopper.step() 이후 호출 (best state 캡처 선행)
        if stopper.should_stop:
            break

    # best state 복원 (없으면 경고만)
    stopper.restore(model)
    return {
        "best_val": stopper.best_score,
        "best_epoch": stopper.best_epoch,
        "stopped_at": history[-1]["epoch"] if history else 0,
        "history": history,
    }


# ---------------------------------------------------------------------------
# 체크포인트 저장
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """JSON 직렬화 가능한 형태로 재귀 변환.

    torch.Tensor / numpy 스칼라/배열 / float('inf')·nan 등을 모두 처리.
    무한대/NaN은 None으로 치환 (JSON 표준은 inf/nan을 허용하지 않음).
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return _json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, float):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return obj
    return obj


def _save_checkpoint(name: str, model: nn.Module, metrics: Dict) -> Tuple[Path, Path]:
    """가중치(.pt)와 메트릭(.json)을 분리 저장.

    _RUN_ID가 설정되어 있으면 파일명에 포함하여 실행별 결과를 누적 보존한다.
    예: ai4i_cnn_20250409_153022.pt / ai4i_cnn_20250409_153022.json
    """
    suffix = f"_{_RUN_ID}" if _RUN_ID else ""
    pt_path   = config.CHECKPOINT_DIR / f"{name}{suffix}.pt"
    json_path = config.CHECKPOINT_DIR / f"{name}{suffix}.json"
    torch.save(model.state_dict(), pt_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2, ensure_ascii=False)
    return pt_path, json_path


# ---------------------------------------------------------------------------
# 1) CNN 분류 학습
# ---------------------------------------------------------------------------

def train_cnn_classifier(
    name: str,
    data: Dict,
    model: nn.Module,
    cfg: Dict = config.CNN_CFG,
) -> Dict:
    """1D-CNN 분류 학습 (이진/다중 자동 분기).

    분기 기준: model.n_classes
      · 1  → BCEWithLogitsLoss, y_dtype=float32, target shape=(B,)
             모델 forward 출력 (B,1)을 view(-1)로 맞춰 BCE에 전달.
      · ≥2 → CrossEntropyLoss, y_dtype=long, target shape=(B,)
    """
    device = torch.device(config.get_device())
    model.to(device)

    n_classes = getattr(model, "n_classes", 1)
    if n_classes == 1:
        # 클래스 불균형 보정 손실 선택:
        #   · use_focal_loss=False: BCE + pos_weight=n_neg/n_pos
        #   · use_focal_loss=True : Focal BCE (alpha, gamma)
        # 두 손실은 배타 사용. Focal은 alpha로 클래스 가중을 자체 처리하므로
        # pos_weight를 함께 적용하면 가중이 이중 적용되어 학습이 불안정해진다.
        y_tr_np = np.asarray(data["y_train"])
        n_neg = float((y_tr_np == 0).sum())
        n_pos = float((y_tr_np == 1).sum())

        if cfg.get("use_focal_loss", False):
            alpha = float(cfg.get("focal_alpha", 0.75))
            gamma = float(cfg.get("focal_gamma", 2.0))

            class _FocalBCELoss(nn.Module):
                """이진 Focal Loss.

                FL = -α_t · (1 - p_t)^γ · log(p_t)
                  where p_t = p if y=1 else 1-p, α_t = α if y=1 else 1-α
                BCE-with-logits를 reduction='none'으로 받아 modulating factor만
                추가로 곱한다. 수치 안정성은 BCE 내부 logsumexp 트릭에 의존.
                """
                def __init__(self, alpha: float, gamma: float):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma

                def forward(self, logits, target):
                    logits = logits.view(-1)
                    target = target.float().view(-1)
                    bce = F.binary_cross_entropy_with_logits(
                        logits, target, reduction="none"
                    )
                    p = torch.sigmoid(logits)
                    p_t = p * target + (1.0 - p) * (1.0 - target)
                    alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
                    focal = alpha_t * (1.0 - p_t) ** self.gamma * bce
                    return focal.mean()

            loss_fn = _FocalBCELoss(alpha, gamma)
        else:
            pw_val = n_neg / max(n_pos, 1.0)
            pw = torch.tensor([pw_val], dtype=torch.float32, device=device)

            class _BCEAdapter(nn.Module):
                def __init__(self, pos_weight: torch.Tensor):
                    super().__init__()
                    self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                def forward(self, logits, target):
                    return self.bce(logits.view(-1), target.float())

            loss_fn = _BCEAdapter(pw)

        y_dtype = torch.float32
    else:
        loss_fn = nn.CrossEntropyLoss()
        y_dtype = torch.long

    train_loader = _make_loader(
        data["X_train"], data["y_train"], cfg["batch_size"], shuffle=True, y_dtype=y_dtype
    )
    val_loader = _make_loader(
        data["X_val"], data["y_val"], cfg["batch_size"], shuffle=False, y_dtype=y_dtype
    )

    metrics = _train_supervised(
        model, train_loader, val_loader, loss_fn,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        early_stop_patience=cfg.get("early_stop_patience", 10),
        device=device,
    )
    metrics.update({
        "name": name,
        "task": "classification",
        "n_classes": n_classes,
        "decision_threshold": cfg.get("decision_threshold", 0.5),
    })
    pt, js = _save_checkpoint(name, model, metrics)
    metrics["checkpoint"] = str(pt)
    metrics["metrics_json"] = str(js)
    return metrics


# ---------------------------------------------------------------------------
# 2) Autoencoder 학습
# ---------------------------------------------------------------------------

class _MSEReconLoss(nn.Module):
    """AE용 재구성 손실. forward(model_out, target=input)을 받지만,
    AE는 self-supervised이므로 _epoch_pass에서 yb를 xb로 치환해 호출한다."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, recon, target):
        return self.mse(recon, target)


def _ae_epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    """AE 전용 epoch 루프. target을 입력 자체로 사용."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_n = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            loss = loss_fn(out, xb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_n += bs
    return total_loss / max(total_n, 1)


def train_autoencoder(
    name: str,
    data: Dict,
    model: nn.Module,
    cfg: Dict = config.AE_CFG,
) -> Dict:
    """Denoising AE 학습 → fit_threshold(percentile=95) 까지 일괄 처리.

    contract: data["X_train"]은 정상(label=0) 샘플로만 구성되어 있어야 한다.
              data_pipeline.load_*_ae가 이를 보장하므로 assert로 강제한다.
              만약 어긋나면 AE가 이상까지 학습하여 임계값이 무의미해진다.
    """
    # ── contract enforcement ──────────────────────────────────────────
    y_tr = np.asarray(data["y_train"])
    assert (y_tr == 0).all(), (
        f"train_autoencoder contract violation: X_train must contain only "
        f"normal samples (label==0), got non-zero labels in y_train. "
        f"Use data_pipeline.load_*_ae which filters to normal-only."
    )

    device = torch.device(config.get_device())
    model.to(device)

    train_loader = _make_loader(
        data["X_train"], data["y_train"],
        cfg["batch_size"], shuffle=True, y_dtype=torch.float32,
    )
    # val: AE 학습 동안의 loss 추적용. 정상/이상 모두 포함되지만 MSE 트렌드만
    # 본다. (실제 임계값 평가는 evaluate.py에서 수행)
    val_loader = _make_loader(
        data["X_val"], data["y_val"],
        cfg["batch_size"], shuffle=False, y_dtype=torch.float32,
    )

    loss_fn = _MSEReconLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    stopper = EarlyStopping(patience=cfg.get("early_stop_patience", 10))
    history: List[Dict] = []

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = _ae_epoch_pass(model, train_loader, loss_fn, device, optimizer)
        val_loss   = _ae_epoch_pass(model, val_loader,   loss_fn, device, optimizer=None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        stopper.step(val_loss, model, epoch)
        if stopper.should_stop:
            break

    stopper.restore(model)

    # 학습 종료 후 정상 데이터 기반 초기 임계값 설정
    # device 일치를 위해 numpy → tensor 변환은 fit_threshold 내부에서 수행.
    X_tr_t = torch.as_tensor(data["X_train"], dtype=torch.float32)
    init_thr = model.fit_threshold(
        X_tr_t,
        percentile=95.0,
        batch_size=cfg["batch_size"],
    )
    # Mahalanobis 통계 적합 (combined_score 사용 가능하게 함).
    # 평가 단계에서 use_mahalanobis 플래그로 활용 여부를 결정.
    model.fit_mahalanobis(X_tr_t, batch_size=cfg["batch_size"])

    metrics = {
        "name": name,
        "task": "anomaly_detection",
        "best_val": stopper.best_score,
        "best_epoch": stopper.best_epoch,
        "stopped_at": history[-1]["epoch"] if history else 0,
        "init_threshold_percentile": 95.0,
        "init_threshold_value": init_thr,
        "history": history,
    }
    pt, js = _save_checkpoint(name, model, metrics)
    metrics["checkpoint"] = str(pt)
    metrics["metrics_json"] = str(js)
    return metrics


# ---------------------------------------------------------------------------
# 3) LSTM 회귀 학습
# ---------------------------------------------------------------------------

class _MSERegressLoss(nn.Module):
    """LSTM 회귀용 손실. 출력 (B,)와 target (B,)을 명시적 float32로 비교."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred.view(-1), target.float().view(-1))


class _HuberRegressLoss(nn.Module):
    """Huber(Smooth L1) 회귀 손실.

    MSE보다 큰 오차에 덜 민감하여 RUL 이상치(긴 잔존수명 구간)에 의한
    loss 폭발을 방지한다. delta 이하 오차는 MSE, 이상은 MAE로 동작.
    """
    def __init__(self, delta: float = 5.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, pred, target):
        return self.huber(pred.view(-1), target.float().view(-1))


def train_lstm_regressor(
    name: str,
    data: Dict,
    model: nn.Module,
    cfg: Dict = config.LSTM_CFG,
) -> Dict:
    """BiLSTM RUL 회귀 학습.

    y_dtype을 float32로 명시 — data_pipeline에서 이미 float32지만, 추후
    정수형 RUL을 다루는 데이터셋이 추가되어도 학습에 문제가 없도록 강제.
    """
    device = torch.device(config.get_device())
    model.to(device)

    train_loader = _make_loader(
        data["X_train"], data["y_train"],
        cfg["batch_size"], shuffle=True, y_dtype=torch.float32,
    )
    val_loader = _make_loader(
        data["X_val"], data["y_val"],
        cfg["batch_size"], shuffle=False, y_dtype=torch.float32,
    )

    if cfg.get("use_huber_loss", False):
        loss_fn = _HuberRegressLoss(delta=cfg.get("huber_delta", 5.0))
    else:
        loss_fn = _MSERegressLoss()
    metrics = _train_supervised(
        model, train_loader, val_loader, loss_fn,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        early_stop_patience=cfg.get("early_stop_patience", 10),
        device=device,
        grad_clip=cfg.get("grad_clip", 1.0),              # BiLSTM gradient explosion 방지
        use_lr_scheduler=cfg.get("use_lr_scheduler", True),  # ReduceLROnPlateau
    )
    metrics.update({
        "name": name,
        "task": "regression",
        "rul_clip": cfg.get("rul_clip"),
        "window": cfg.get("window"),
    })
    pt, js = _save_checkpoint(name, model, metrics)
    metrics["checkpoint"] = str(pt)
    metrics["metrics_json"] = str(js)
    return metrics


# ---------------------------------------------------------------------------
# 4) GBDT 분류 학습 (sklearn — torch와 별도 경로)
# ---------------------------------------------------------------------------

def train_gbdt_classifier(
    name: str,
    data: Dict,
    model,
    cfg: Optional[Dict] = None,
) -> Dict:
    """sklearn HistGradientBoostingClassifier 학습.

    torch 모델이 아니므로 _make_loader / _epoch_pass를 우회한다.
    클래스 불균형은 sample_weight로 보정 (n_neg/n_pos를 양성에 부여).
    체크포인트는 pickle (.pkl). torch.save가 아닌 stdlib pickle 사용.
    cfg 인자는 호환성을 위해 받지만 sklearn 모델 자체에 hparams가 내장돼 있어
    smoke 모드에서는 무시된다.
    """
    import pickle

    y_tr = np.asarray(data["y_train"]).astype(np.int32)
    n_neg = float((y_tr == 0).sum())
    n_pos = float((y_tr == 1).sum())
    pos_weight = n_neg / max(n_pos, 1.0)
    sample_weight = np.where(y_tr == 1, pos_weight, 1.0).astype(np.float32)

    model.fit(data["X_train"], y_tr, sample_weight=sample_weight)

    # val 손실 트래킹용 (참고치 — sklearn은 자체 early-stop 보유)
    y_val = np.asarray(data["y_val"]).astype(np.int32)
    val_proba = model.predict_proba(data["X_val"])[:, 1]
    val_logloss = float(
        -np.mean(
            y_val * np.log(np.clip(val_proba, 1e-7, 1.0))
            + (1 - y_val) * np.log(np.clip(1 - val_proba, 1e-7, 1.0))
        )
    )

    metrics: Dict = {
        "name": name,
        "task": "gbdt_binary",
        "n_train": int(len(y_tr)),
        "n_pos_train": int(n_pos),
        "n_neg_train": int(n_neg),
        "pos_weight": pos_weight,
        "val_logloss": val_logloss,
        "best_val": val_logloss,
        "best_epoch": getattr(model, "n_iter_", -1),
    }

    # 체크포인트: pickle (sklearn 표준)
    suffix = f"_{_RUN_ID}" if _RUN_ID else ""
    pkl_path  = config.CHECKPOINT_DIR / f"{name}{suffix}.pkl"
    json_path = config.CHECKPOINT_DIR / f"{name}{suffix}.json"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2, ensure_ascii=False)

    metrics["checkpoint"] = str(pkl_path)
    metrics["metrics_json"] = str(json_path)
    return metrics


# ---------------------------------------------------------------------------
# 디스패처
# ---------------------------------------------------------------------------

TRAINERS: Dict[str, Callable] = {
    "classification":    train_cnn_classifier,
    "anomaly_detection": train_autoencoder,
    "regression":        train_lstm_regressor,
    "gbdt_binary":       train_gbdt_classifier,
}
